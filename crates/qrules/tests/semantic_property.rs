//! Applying a rewrite rule must never change what a circuit computes.
//!
//! This is the property the whole optimizer rests on, and the reference suite never
//! tested it: `OptimizerTest.java` compared output QASM against hand-written strings, so
//! it could only catch regressions in cases someone had already thought of.
//!
//! Here real rules from `rules/*.txt` are applied to real circuits from `benchmarks/`,
//! and each result is checked by building both unitaries. Anything the matcher, the
//! convexity check, the angle binding, or the splicing gets wrong shows up as a distance.

use qcircuit::{qasm, Dag, GateRegistry, QubitId};
use qrules::legacy::{self, LoadOptions};
use qrules::{apply_rule, ApplyOptions, Rule};
use qsemantics::{phase_invariant_distance, Unitary};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const TOL: f64 = 1e-9;
/// Circuits wider than this are too large to build a dense unitary for.
const MAX_QUBITS: usize = 10;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<name> has a grandparent")
        .to_path_buf()
}

fn reg() -> GateRegistry {
    GateRegistry::with_builtins()
}

fn same_semantics(a: &Dag, b: &Dag) -> f64 {
    let mut qs: Vec<QubitId> = a.qubits().to_vec();
    for q in b.qubits() {
        if !qs.contains(q) {
            qs.push(q.clone());
        }
    }
    qs.sort();
    let ua = Unitary::from_dag_over(a, qs.clone(), &reg()).unwrap();
    let ub = Unitary::from_dag_over(b, qs, &reg()).unwrap();
    phase_invariant_distance(&ua, &ub)
}

/// Load a bounded number of rules from a rule file.
fn load_rules(file: &str, limit: usize) -> Vec<Rule> {
    let path = repo_root().join("rules").join(file);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let head: String = text.lines().take(limit).collect::<Vec<_>>().join("\n");
    let report = legacy::load_str(&head, &LoadOptions::default(), &reg());
    report.rules
}

fn load_circuits(dirs: &[&str], limit: usize) -> Vec<(String, Dag)> {
    let mut out = Vec::new();
    for d in dirs {
        out.extend(load_circuits_from(d, limit.saturating_sub(out.len())));
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn load_circuits_from(dir: &str, limit: usize) -> Vec<(String, Dag)> {
    if limit == 0 {
        return Vec::new();
    }
    let path = repo_root().join("benchmarks").join(dir);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "qasm"))
        .collect();
    files.sort();
    files
        .into_iter()
        .filter_map(|p| {
            let src = std::fs::read_to_string(&p).ok()?;
            let dag = qasm::parse(&src).ok()?;
            if dag.num_qubits() > MAX_QUBITS {
                return None;
            }
            let name = p.file_name()?.to_string_lossy().into_owned();
            Some((name, dag))
        })
        .take(limit)
        .collect()
}

struct Stats {
    attempts: usize,
    fired: usize,
}

/// Apply every rule to every circuit and check semantics are preserved.
fn sweep(rule_file: &str, bench_dirs: &[&str], rule_limit: usize, circuit_limit: usize) -> Stats {
    let rules = load_rules(rule_file, rule_limit);
    assert!(!rules.is_empty(), "no rules loaded from {rule_file}");
    let circuits = load_circuits(bench_dirs, circuit_limit);
    assert!(
        !circuits.is_empty(),
        "no circuits loaded from {bench_dirs:?}"
    );

    let mut stats = Stats {
        attempts: 0,
        fired: 0,
    };
    let mut failures: Vec<String> = Vec::new();

    for (name, circuit) in &circuits {
        for rule in &rules {
            stats.attempts += 1;
            let mut rng = ChaCha8Rng::seed_from_u64(stats.attempts as u64);
            let Some(out) = apply_rule(circuit, rule, &ApplyOptions::default(), &mut rng) else {
                continue;
            };
            stats.fired += 1;

            if !out.dag.is_acyclic() {
                failures.push(format!("{name} + `{}`: produced a cycle", rule.id));
                continue;
            }
            if out.dag.num_qubits() != circuit.num_qubits() {
                failures.push(format!(
                    "{name} + `{}`: qubit count changed {} -> {}",
                    rule.id,
                    circuit.num_qubits(),
                    out.dag.num_qubits()
                ));
                continue;
            }
            let d = same_semantics(circuit, &out.dag);
            if d > TOL {
                failures.push(format!(
                    "{name} + `{}`: semantics changed by {d:.3e} ({} applications)",
                    rule.id, out.applications
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} rule applications changed semantics:\n{}",
        failures.len(),
        stats.fired,
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    stats
}

// A six-gate pattern needing exact wire adjacency is a demanding thing to find, so a
// sweep must cover a lot of rules to fire often enough to mean anything. The full sweeps
// are `#[ignore]`d because they are slow in a debug build; CI runs them in release with
// `--include-ignored`. The smoke test below runs every time and covers all four gate
// sets, so a matcher that stopped working entirely would still be caught immediately.

/// Every gate set, enough rules to fire, cheap enough to run on every `cargo test`.
#[test]
fn all_gate_sets_smoke() {
    let cases: [(&str, &[&str], usize, usize); 4] = [
        ("rules_q3_s6_nam.txt", &["nam_rz"], 6_000, 4),
        ("rules_q3_s6_ibmnew.txt", &["ibmnew"], 6_000, 4),
        (
            "rules_q3_s6_cliffordt.txt",
            &["nam_t_tdg", "benchmarksCCZ"],
            6_000,
            4,
        ),
        ("rules_q3_s3_ion.txt", &["ion"], 4_000, 3),
    ];
    let mut total = 0;
    for (rules, bench, rule_limit, circuits) in cases {
        let s = sweep(rules, bench, rule_limit, circuits);
        println!("{bench:?}: {} fired of {} attempts", s.fired, s.attempts);
        total += s.fired;
    }
    assert!(
        total > 20,
        "only {total} applications fired across all gate sets"
    );
}

#[test]
#[ignore = "full corpus; run with --include-ignored, ideally in release"]
fn nam_rules_preserve_semantics_on_nam_circuits() {
    let s = sweep("rules_q3_s6_nam.txt", &["nam_rz"], 40_000, 12);
    println!(
        "nam: {} applications fired of {} attempts",
        s.fired, s.attempts
    );
    assert!(s.fired > 100, "only {} applications fired", s.fired);
}

#[test]
#[ignore = "full corpus; run with --include-ignored, ideally in release"]
fn ibmnew_rules_preserve_semantics_on_ibmnew_circuits() {
    let s = sweep("rules_q3_s6_ibmnew.txt", &["ibmnew"], 32_000, 12);
    println!(
        "ibmn: {} applications fired of {} attempts",
        s.fired, s.attempts
    );
    assert!(s.fired > 50, "only {} applications fired", s.fired);
}

#[test]
#[ignore = "full corpus; run with --include-ignored, ideally in release"]
fn cliffordt_rules_preserve_semantics_on_cliffordt_circuits() {
    // Only two nam_t_tdg circuits are small enough to build a dense unitary for, so the
    // Clifford+T-heavy CCZ benchmarks are drawn on as well.
    let s = sweep(
        "rules_q3_s6_cliffordt.txt",
        &["nam_t_tdg", "benchmarksCCZ"],
        64_000,
        12,
    );
    println!(
        "cliffordt: {} applications fired of {} attempts",
        s.fired, s.attempts
    );
    assert!(s.fired > 50, "only {} applications fired", s.fired);
}

#[test]
#[ignore = "full corpus; run with --include-ignored, ideally in release"]
fn ion_rules_preserve_semantics_on_ion_circuits() {
    let s = sweep("rules_q3_s3_ion.txt", &["ion"], 12_000, 8);
    println!(
        "ion: {} applications fired of {} attempts",
        s.fired, s.attempts
    );
    assert!(s.fired > 100, "only {} applications fired", s.fired);
}

/// Chains of rewrites must also preserve semantics: errors that only appear after a
/// circuit has already been rewritten once would slip past a single-application sweep.
#[test]
#[ignore = "full corpus; run with --include-ignored, ideally in release"]
fn repeated_rewriting_preserves_semantics() {
    let rules = load_rules("rules_q3_s6_nam.txt", 40_000);
    let circuits = load_circuits(&["nam_rz"], 6);
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let mut chains = 0;

    for (name, original) in &circuits {
        let mut current = original.clone();
        let mut order: Vec<usize> = (0..rules.len()).collect();
        order.shuffle(&mut rng);

        let mut applied = 0;
        for i in order {
            if applied >= 25 {
                break;
            }
            let Some(out) = apply_rule(&current, &rules[i], &ApplyOptions::default(), &mut rng)
            else {
                continue;
            };
            applied += 1;
            assert!(
                out.dag.is_acyclic(),
                "{name}: cycle after {applied} rewrites"
            );
            current = out.dag;
        }

        if applied == 0 {
            continue;
        }
        chains += 1;
        let d = same_semantics(original, &current);
        assert!(
            d <= TOL,
            "{name}: semantics drifted by {d:.3e} after {applied} rewrites"
        );
        println!(
            "{name}: {applied} chained rewrites, {} gates -> {}",
            original.gate_count(),
            current.gate_count()
        );
    }
    assert!(chains > 0, "no chain ever got started");
}

/// Rewriting must never break the wire invariant.
#[test]
#[ignore = "full corpus; run with --include-ignored, ideally in release"]
fn rewriting_keeps_wires_well_formed() {
    let rules = load_rules("rules_q3_s6_nam.txt", 20_000);
    let circuits = load_circuits(&["nam_rz"], 4);
    let mut checked = 0;

    for (name, circuit) in &circuits {
        for (i, rule) in rules.iter().enumerate() {
            let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
            let Some(out) = apply_rule(circuit, rule, &ApplyOptions::default(), &mut rng) else {
                continue;
            };
            checked += 1;
            for q in out.dag.qubits() {
                let sink = out.dag.sink(q).unwrap();
                let mut n = out.dag.source(q).unwrap();
                let mut steps = 0;
                while n != sink {
                    n = out
                        .dag
                        .next_on(n, q)
                        .unwrap_or_else(|| panic!("{name}: wire {q} breaks"));
                    steps += 1;
                    assert!(
                        steps <= out.dag.gate_count() + 1,
                        "{name}: wire {q} loops after `{}`",
                        rule.id
                    );
                }
            }
        }
    }
    assert!(checked > 0);
    println!("wire invariant checked on {checked} rewrites");
}

/// Structural invariants, on circuits of every size.
///
/// Building a dense unitary caps the semantic sweeps at ten qubits, which excludes most
/// of the benchmark corpus. These checks scale to any width: after a rewrite the graph
/// must stay acyclic, every wire must still run unbroken from source to sink, the qubit
/// set must be unchanged, and the gate count must move by exactly the rule's size delta
/// times the number of applications.
///
/// The last of those is a surprisingly sharp check — a splice that drops or duplicates a
/// gate shows up immediately.
#[test]
fn rewriting_preserves_structure_on_all_circuit_sizes() {
    let _reg = reg();
    let mut rules: Vec<Rule> = Vec::new();
    for f in [
        "rules_q3_s6_nam.txt",
        "rules_q3_s6_ibmnew.txt",
        "rules_q3_s6_cliffordt.txt",
        "rules_q3_s3_ion.txt",
    ] {
        rules.extend(load_rules(f, 4_000));
    }
    assert!(!rules.is_empty());

    let root = repo_root().join("benchmarks");
    let mut circuits: Vec<(String, Dag)> = Vec::new();
    for dir in std::fs::read_dir(&root).unwrap().flatten() {
        if !dir.path().is_dir() {
            continue;
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "qasm"))
            .collect();
        files.sort();
        for p in files.into_iter().take(4) {
            let src = std::fs::read_to_string(&p).unwrap();
            if let Ok(dag) = qasm::parse(&src) {
                circuits.push((p.file_name().unwrap().to_string_lossy().into_owned(), dag));
            }
        }
    }
    assert!(circuits.len() > 20, "expected a broad circuit sample");

    let mut fired = 0usize;
    let mut widest = 0usize;
    for (name, circuit) in &circuits {
        widest = widest.max(circuit.num_qubits());
        for (i, rule) in rules.iter().enumerate() {
            let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
            let Some(out) = apply_rule(circuit, rule, &ApplyOptions::default(), &mut rng) else {
                continue;
            };
            fired += 1;

            assert!(out.dag.is_acyclic(), "{name} + `{}`: cycle", rule.id);
            assert_eq!(
                out.dag.qubits().len(),
                circuit.qubits().len(),
                "{name} + `{}`: qubit set changed",
                rule.id
            );

            let expected =
                circuit.gate_count() as isize + rule.size_delta() * out.applications as isize;
            assert_eq!(
                out.dag.gate_count() as isize,
                expected,
                "{name} + `{}`: gate count {} after {} applications, expected {expected}",
                rule.id,
                out.dag.gate_count(),
                out.applications
            );

            for q in out.dag.qubits() {
                let sink = out.dag.sink(q).unwrap();
                let mut n = out.dag.source(q).unwrap();
                let mut steps = 0;
                while n != sink {
                    n = out
                        .dag
                        .next_on(n, q)
                        .unwrap_or_else(|| panic!("{name}: wire {q} breaks"));
                    steps += 1;
                    assert!(steps <= out.dag.gate_count() + 1, "{name}: wire {q} loops");
                }
            }

            // Every gate must still name only qubits the circuit declares.
            for idx in out.dag.gate_indices() {
                for q in &out.dag.gate(idx).qubits {
                    assert!(
                        out.dag.qubits().contains(q),
                        "{name} + `{}`: gate names undeclared qubit {q}",
                        rule.id
                    );
                }
            }
        }
    }
    println!(
        "structural: {fired} rewrites across {} circuits, widest {widest} qubits",
        circuits.len()
    );
    assert!(fired > 100, "only {fired} rewrites exercised");
}
