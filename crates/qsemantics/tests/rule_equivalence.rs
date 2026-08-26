//! Every shipped rewrite rule must be semantics-preserving.
//!
//! This is the single highest-value test in the suite. Each line of `rules/*.txt` claims
//! that two circuits are equivalent; this checks the claim by building both sides'
//! unitaries and comparing them up to global phase. Symbolic angles are bound to random
//! concrete values, drawn from a fixed seed so failures reproduce.
//!
//! One test exercises the QASM parser, the gate registry's unitaries, the operand-order
//! convention, the angle-expression evaluator, and the rule-file format simultaneously,
//! over 177,380 rules. The reference Java suite checked none of this: its rule files were
//! trusted outputs of a synthesizer that was itself only spot-checked by two assertions.

use qcircuit::{qasm, GateRegistry, QubitId};
use qsemantics::{phase_invariant_distance, Unitary};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Rules hold at every angle, so any tolerance must absorb only floating-point error.
const TOL: f64 = 1e-9;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<name> has a grandparent")
        .to_path_buf()
}

fn rule_files() -> Vec<PathBuf> {
    let root = repo_root().join("rules");
    let mut out: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("no rules directory at {}: {e}", root.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    out.sort();
    out
}

/// A tiny deterministic PRNG, so this test needs no rand dependency and always reproduces.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct Outcome {
    checked: usize,
    skipped_symbolic: usize,
    failures: Vec<String>,
}

fn check_file(path: &Path, registry: &GateRegistry, samples: usize) -> Outcome {
    let text = std::fs::read_to_string(path).unwrap();
    let mut out = Outcome {
        checked: 0,
        skipped_symbolic: 0,
        failures: Vec::new(),
    };

    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(" | ").collect();
        if parts.len() < 2 {
            out.failures.push(format!(
                "{}:{}: malformed rule (no ` | ` separator)",
                path.display(),
                lineno + 1
            ));
            continue;
        }
        let (lhs_src, rhs_src) = (parts[0], parts[1]);

        // The symbolic hole `symb q` stands for an arbitrary sub-circuit; those rules are
        // checked by the constraint machinery in qrules, not here.
        if lhs_src.contains("symb q") || rhs_src.contains("symb q") {
            out.skipped_symbolic += 1;
            continue;
        }

        let lhs = match qasm::parse(lhs_src) {
            Ok(d) => d,
            Err(e) => {
                out.failures
                    .push(format!("{}:{}: lhs parse: {e}", path.display(), lineno + 1));
                continue;
            }
        };
        let rhs = match qasm::parse(rhs_src) {
            Ok(d) => d,
            Err(e) => {
                out.failures
                    .push(format!("{}:{}: rhs parse: {e}", path.display(), lineno + 1));
                continue;
            }
        };

        // Both sides must be built over the same qubit ordering.
        let mut qubits: Vec<QubitId> = lhs.qubits().to_vec();
        for q in rhs.qubits() {
            if !qubits.contains(q) {
                qubits.push(q.clone());
            }
        }
        qubits.sort();

        // Collect the free angle variables appearing on either side.
        let mut vars: Vec<String> = Vec::new();
        for dag in [&lhs, &rhs] {
            for idx in dag.gate_indices() {
                for p in &dag.gate(idx).params {
                    for v in p.free_vars() {
                        if !vars.iter().any(|x| x == v) {
                            vars.push(v.to_string());
                        }
                    }
                }
            }
        }

        // A seed derived from the rule text keeps each rule's sample independent but
        // reproducible.
        let mut rng = Rng(line.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        }));

        let mut worst = 0.0f64;
        let mut error: Option<String> = None;
        for _ in 0..samples.max(1) {
            let env: BTreeMap<String, f64> = vars
                .iter()
                .map(|v| (v.clone(), rng.next_f64() * 8.0 - 4.0))
                .collect();
            let lookup = |name: &str| env.get(name).copied();

            let a = match Unitary::from_dag_over_with_env(&lhs, qubits.clone(), registry, &lookup) {
                Ok(u) => u,
                Err(e) => {
                    error = Some(format!("lhs build: {e}"));
                    break;
                }
            };
            let b = match Unitary::from_dag_over_with_env(&rhs, qubits.clone(), registry, &lookup) {
                Ok(u) => u,
                Err(e) => {
                    error = Some(format!("rhs build: {e}"));
                    break;
                }
            };
            worst = worst.max(phase_invariant_distance(&a, &b));
        }

        if let Some(e) = error {
            out.failures
                .push(format!("{}:{}: {e}", path.display(), lineno + 1));
            continue;
        }

        out.checked += 1;
        if worst > TOL {
            out.failures.push(format!(
                "{}:{}: sides differ by {worst:.3e}\n    lhs: {lhs_src}\n    rhs: {rhs_src}",
                path.display(),
                lineno + 1
            ));
        }
    }
    out
}

/// Check every plain rewrite rule in every shipped rule file.
///
/// Runs the full corpus. In debug builds this takes a few minutes, so it is marked
/// `#[ignore]`; CI runs it in release with `--include-ignored`. The sampled variant below
/// runs by default and covers every file.
#[test]
#[ignore = "full corpus; run with --include-ignored, ideally in release"]
fn all_rules_preserve_semantics() {
    let registry = GateRegistry::with_builtins();
    let mut checked = 0;
    let mut skipped = 0;
    let mut failures: Vec<String> = Vec::new();

    for path in rule_files() {
        let o = check_file(&path, &registry, 3);
        checked += o.checked;
        skipped += o.skipped_symbolic;
        failures.extend(o.failures);
    }

    println!("checked {checked} rules, skipped {skipped} symbolic");
    report(&failures, checked);
}

/// A fast subset: the first 500 rules of every shipped rule file.
///
/// Cheap enough to run on every `cargo test`, while still touching every gate set.
#[test]
fn sampled_rules_preserve_semantics() {
    let registry = GateRegistry::with_builtins();
    let mut checked = 0;
    let mut failures: Vec<String> = Vec::new();
    let tmp = std::env::temp_dir().join(format!("guoq-rule-sample-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    for path in rule_files() {
        let text = std::fs::read_to_string(&path).unwrap();
        let head: String = text.lines().take(500).collect::<Vec<_>>().join("\n");
        let sample = tmp.join(path.file_name().unwrap());
        std::fs::write(&sample, head).unwrap();
        let o = check_file(&sample, &registry, 2);
        checked += o.checked;
        failures.extend(
            o.failures
                .into_iter()
                .map(|f| f.replace(&tmp.display().to_string(), "rules")),
        );
    }
    std::fs::remove_dir_all(&tmp).ok();

    println!("checked {checked} sampled rules");
    assert!(checked >= 3000, "expected a broad sample, saw {checked}");
    report(&failures, checked);
}

fn report(failures: &[String], checked: usize) {
    if failures.is_empty() {
        return;
    }
    let shown: Vec<&String> = failures.iter().take(20).collect();
    panic!(
        "{} of {checked} rules are not semantics-preserving:\n{}\n{}",
        failures.len(),
        shown
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        if failures.len() > shown.len() {
            format!("... and {} more", failures.len() - shown.len())
        } else {
            String::new()
        }
    );
}

/// A rule that is deliberately wrong must be caught, so the checker is not vacuous.
#[test]
fn the_checker_rejects_a_bad_rule() {
    let dir = std::env::temp_dir().join(format!("guoq-badrule-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.txt");
    std::fs::write(
        &path,
        // True.
        "h q0; h q0; | \n\
         cx q0, q1; cx q0, q1; | \n\
         rz(theta1) q0; rz(theta2) q0; | rz((theta1+theta2)) q0;\n\
         // false below\n\
         h q0; | x q0;\n\
         cx q0, q1; | cx q1, q0;\n\
         rz(theta1) q0; | rz(theta2) q0;\n",
    )
    .unwrap();

    let o = check_file(&path, &GateRegistry::with_builtins(), 3);
    std::fs::remove_dir_all(&dir).ok();

    // The `// false below` line is malformed, plus the three genuinely wrong rules.
    assert_eq!(
        o.failures.len(),
        4,
        "expected 4 failures, got: {:#?}",
        o.failures
    );
    assert!(o
        .failures
        .iter()
        .any(|f| f.contains("lhs: h q0;") && f.contains("rhs: x q0;")));
    assert!(o
        .failures
        .iter()
        .any(|f| f.contains("lhs: cx q0, q1;") && f.contains("rhs: cx q1, q0;")));
    assert!(o.failures.iter().any(|f| f.contains("malformed rule")));
    // Six well-formed rules are checked (three true, three false); the malformed line
    // is reported but not counted as checked.
    assert_eq!(o.checked, 6);
}
