//! Symbolic rules must preserve semantics too.
//!
//! A symbolic rule moves gates across an arbitrary sub-circuit, licensed by a classical
//! constraint on that sub-circuit's action. Whether the constraint check and the region
//! derivation are right is not obvious from reading them, so this applies real symbolic
//! rules to real circuits and checks the unitary is unchanged.
//!
//! The reference suite had no symbolic-rule tests of any kind beyond five hand-written
//! `applySymbRule` cases with fixed expected strings.

use qcircuit::{qasm, Dag, GateRegistry, QubitId};
use qrules::legacy::{self, LoadOptions};
use qrules::symbolic::{apply_symbolic, find_symbolic, SymbolicLimits};
use qrules::SymbolicRule;
use qsemantics::{phase_invariant_distance, Unitary};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const TOL: f64 = 1e-9;
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

fn distance(a: &Dag, b: &Dag) -> f64 {
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

/// How many rules of each shipped file the load-time identity check refuses.
///
/// These are not parse failures: they are shipped rules whose claimed permutations make
/// the two sides unequal, checked numerically over the boundary. They never fired in
/// the reference either -- its `+`/`-` pattern filter happened to set every one aside
/// before use. The nine in `nam`/`ibmnew` are the `x q1; rz(..) q1;
/// symb q` family, which no permutation validates; the 504 in `ibm` are bad `u2`/`u3`
/// rules. The port checks the identity instead of relying on an accidental filter, so it
/// keeps the 36 `ibm` rules the filter would also have discarded that are actually fine.
fn expected_invalid(file: &str) -> usize {
    match file {
        "rules_q3_s3_nam_symb.txt" | "rules_q3_s3_ibmnew_symb.txt" => 9,
        "rules_q3_s3_ibm_symb.txt" => 504,
        _ => 0,
    }
}

fn load_symbolic(file: &str) -> Vec<SymbolicRule> {
    let path = repo_root().join("rules").join(file);
    let text = std::fs::read_to_string(&path).unwrap();
    let report = legacy::load_str(&text, &LoadOptions::default(), &reg());
    let (rules, rejected) = legacy::parse_symbolic(&report.symbolic_lines, &reg());
    assert_eq!(
        rejected.len(),
        expected_invalid(file),
        "{}: {} of {} symbolic rules rejected: {:?}",
        file,
        rejected.len(),
        report.symbolic_lines.len(),
        &rejected[..rejected.len().min(3)]
    );
    rules
}

fn load_circuits(dir: &str, limit: usize) -> Vec<(String, Dag)> {
    let path = repo_root().join("benchmarks").join(dir);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "qasm"))
        .collect();
    files.sort();
    files
        .into_iter()
        .filter_map(|p| {
            let dag = qasm::parse(&std::fs::read_to_string(&p).ok()?).ok()?;
            if dag.num_qubits() > MAX_QUBITS {
                return None;
            }
            Some((p.file_name()?.to_string_lossy().into_owned(), dag))
        })
        .take(limit)
        .collect()
}

/// Every symbolic rule in every shipped file must parse into the width-generic form.
#[test]
fn all_shipped_symbolic_rules_parse() {
    let files = [
        "rules_q3_s3_nam_symb.txt",
        "rules_q3_s3_ibmnew_symb.txt",
        "rules_q3_s3_ibm_symb.txt",
        "rules_q3_s3_ion_symb.txt",
        "rules_q3_s3_rigetti_symb.txt",
        "rules_q3_s3_cliffordt_symb.txt",
    ];
    let mut total = 0;
    for f in files {
        let rules = load_symbolic(f);
        assert!(!rules.is_empty(), "{f} produced no symbolic rules");
        // Boundaries come from the rule's own halves, not from the literals "q0"/"q1".
        for r in &rules {
            assert_eq!(
                r.boundary.len(),
                r.constraints[0].width(),
                "{f}: boundary and constraint width disagree in `{}`",
                r.id
            );
            assert!(!r.boundary.is_empty());
        }
        total += rules.len();
        println!("{f}: {} rules", rules.len());
    }
    assert!(total > 2000, "expected the full corpus, saw {total}");
}

struct Stats {
    attempts: usize,
    fired: usize,
}

fn sweep(rule_file: &str, bench_dir: &str, circuits: usize) -> Stats {
    let rules = load_symbolic(rule_file);
    let circuits = load_circuits(bench_dir, circuits);
    assert!(!circuits.is_empty(), "no circuits from {bench_dir}");
    let limits = SymbolicLimits::default();
    let registry = reg();

    let mut stats = Stats {
        attempts: 0,
        fired: 0,
    };
    let mut failures = Vec::new();

    for (name, circuit) in &circuits {
        for (i, rule) in rules.iter().enumerate() {
            stats.attempts += 1;
            let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
            let Some(out) = apply_symbolic(circuit, rule, &limits, &registry, &mut rng) else {
                continue;
            };
            stats.fired += 1;

            if !out.is_acyclic() {
                failures.push(format!("{name} + `{}`: cycle", rule.id));
                continue;
            }
            let expected = circuit.gate_count() as isize + rule.size_delta();
            if out.gate_count() as isize != expected {
                failures.push(format!(
                    "{name} + `{}`: gate count {} expected {expected}",
                    rule.id,
                    out.gate_count()
                ));
                continue;
            }
            let d = distance(circuit, &out);
            if d > TOL {
                failures.push(format!(
                    "{name} + `{}`: semantics changed by {d:.3e}",
                    rule.id
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} symbolic applications were wrong:\n{}",
        failures.len(),
        stats.fired,
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    stats
}

#[test]
fn nam_symbolic_rules_preserve_semantics() {
    let s = sweep("rules_q3_s3_nam_symb.txt", "nam_rz", 8);
    println!("nam symb: {} fired of {} attempts", s.fired, s.attempts);
    assert!(s.fired > 0, "no symbolic rule ever applied");
}

#[test]
fn ibmnew_symbolic_rules_preserve_semantics() {
    let s = sweep("rules_q3_s3_ibmnew_symb.txt", "ibmnew", 8);
    println!("ibmn symb: {} fired of {} attempts", s.fired, s.attempts);
    assert!(s.fired > 0, "no symbolic rule ever applied");
}

#[test]
fn cliffordt_symbolic_rules_preserve_semantics() {
    let s = sweep("rules_q3_s3_cliffordt_symb.txt", "nam_t_tdg", 6);
    println!(
        "cliffordt symb: {} fired of {} attempts",
        s.fired, s.attempts
    );
}

/// Ion symbolic rules are expected to fire rarely or never on ion circuits: the gate set
/// is `rx, ry, rz, rxx`, and `rx`/`ry` put the boundary into superposition, so a region
/// containing them satisfies no basis permutation. The sweep still runs, to prove that
/// nothing *wrong* is applied.
#[test]
fn ion_symbolic_rules_preserve_semantics() {
    let s = sweep("rules_q3_s3_ion_symb.txt", "ion", 6);
    println!("ion symb: {} fired of {} attempts", s.fired, s.attempts);
    assert!(s.attempts > 1000, "the sweep should at least have tried");
}

/// The local soundness check must not be vacuous.
///
/// This rule carries a genuine constraint but a replacement that is simply wrong: it
/// claims `h q1` before the hole equals `x q1` after it. The constraint machinery has no
/// way to notice, so if this ever applies, the safety net is not working.
#[test]
fn a_deliberately_wrong_symbolic_rule_never_applies() {
    let identity_constraint = "[{[false, false]=[false, false], [true, false]=[true, false], \
                               [false, true]=[false, true], [true, true]=[true, true]}]";
    // Finds `h q0 ... h q1` and claims it equals `h q0; x q1` before the hole. The claim
    // is false for its one listed permutation, and since every constraint is now checked
    // numerically at parse time, the rule is refused before it can ever be tried.
    let line = format!("h q0; x q1; symb q; | h q0; symb q; h q1; | {identity_constraint}");
    let err = SymbolicRule::parse_legacy_with_builtins(&line).unwrap_err();
    assert!(
        err.to_string()
            .contains("no constraint makes the rule an identity"),
        "expected the identity check to refuse the rule, got: {err}"
    );
}

/// ...and a correct rule of the same shape *does* apply, so the previous test is not
/// passing merely because nothing ever matches.
#[test]
fn a_correct_symbolic_rule_of_the_same_shape_does_apply() {
    let identity_constraint = "[{[false, false]=[false, false], [true, false]=[true, false], \
                               [false, true]=[false, true], [true, true]=[true, true]}]";
    // Moving an rz back across a region that acts as the identity on the boundary is
    // sound: the boundary bits are unchanged in every branch, so the rotation's phase is
    // the same whichever side it sits on.
    let line = format!(
        "h q0; rz(theta1) q1; symb q; | h q0; symb q; rz(theta1) q1; | {identity_constraint}"
    );
    let rule = SymbolicRule::parse_legacy_with_builtins(&line).unwrap();

    let limits = SymbolicLimits::default();
    let registry = reg();
    let mut applied = 0;
    for (_, circuit) in load_circuits("nam_rz", 8) {
        for seed in 0..8u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            if let Some(out) = apply_symbolic(&circuit, &rule, &limits, &registry, &mut rng) {
                applied += 1;
                assert!(out.is_acyclic());
                let d = distance(&circuit, &out);
                assert!(d <= TOL, "semantics changed by {d:.3e}");
            }
        }
    }
    assert!(applied > 0, "the control rule never applied");
}

/// A pattern extending past the end of a circuit must find nothing rather than panic;
/// these shapes are exactly the ones that end mid-pattern.
#[test]
fn no_panic_on_circuits_that_end_mid_pattern() {
    let rules = load_symbolic("rules_q3_s3_nam_symb.txt");
    let limits = SymbolicLimits::default();
    let registry = reg();
    let shapes = [
        "",
        "h q0;",
        "rz(0.5) q0;",
        "rz(0.5) q0; h q1;",
        "h q1; rz(0.5) q0;",
        "rz(0.5) q0; cx q0, q1;",
        "cx q0, q1; h q1;",
        "rz(0.5) q0; cx q0, q1; h q1;",
        "x q0;",
    ];
    for src in shapes {
        let dag = qasm::parse(src).unwrap();
        for (i, rule) in rules.iter().enumerate().take(60) {
            let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
            // Must answer, not panic, whatever the circuit's shape.
            let _ = find_symbolic(&dag, rule, &limits, &registry, &mut rng);
            let _ = apply_symbolic(&dag, rule, &limits, &registry, &mut rng);
        }
    }
}

/// Tightening the limits must only ever reduce what applies, never change a result.
#[test]
fn limits_only_restrict() {
    let rules = load_symbolic("rules_q3_s3_nam_symb.txt");
    let circuits = load_circuits("nam_rz", 4);
    let registry = reg();
    let generous = SymbolicLimits {
        max_qubits: 7,
        max_gates: 10,
        ..Default::default()
    };
    let tight = SymbolicLimits {
        max_qubits: 2,
        max_gates: 1,
        ..Default::default()
    };
    let mut loose_hits = 0;
    let mut tight_hits = 0;
    for (_, circuit) in &circuits {
        for (i, rule) in rules.iter().enumerate() {
            let mut a = ChaCha8Rng::seed_from_u64(i as u64);
            let mut b = ChaCha8Rng::seed_from_u64(i as u64);
            if apply_symbolic(circuit, rule, &generous, &registry, &mut a).is_some() {
                loose_hits += 1;
            }
            if apply_symbolic(circuit, rule, &tight, &registry, &mut b).is_some() {
                tight_hits += 1;
            }
        }
    }
    assert!(
        tight_hits <= loose_hits,
        "tight limits admitted more matches ({tight_hits}) than generous ones ({loose_hits})"
    );
    println!("limits: {loose_hits} generous, {tight_hits} tight");
}

/// Some shipped symbolic rules are not identities under any of their listed
/// permutations; loading must refuse them.
///
/// This is the first `x q1` rule from `rules_q3_s3_nam_symb.txt`, verbatim. No
/// permutation of the two boundary bits makes `symb; rz(t) q1` equal
/// `x q1; rz(t) q1; symb` -- the two sides disagree about whether the hole sees the
/// flipped bit -- and checking all 24 confirms it. The reference never applied these
/// nine (nor `ibmnew`'s nine, nor 504 of `ibm`'s 1,152): its `+`/`-` pattern filter
/// happened to discard them before use. The port validates instead of filtering, so the
/// invalid rules are refused by name and the valid compound-angle rules survive.
#[test]
fn false_constraints_are_dropped_at_load() {
    let text = std::fs::read_to_string(repo_root().join("rules/rules_q3_s3_nam_symb.txt")).unwrap();
    let x_lines: Vec<&str> = text
        .lines()
        .filter(|l| {
            l.split('|')
                .nth(1)
                .is_some_and(|find| find.trim().starts_with("x q"))
        })
        .collect();
    assert!(
        !x_lines.is_empty(),
        "the known-bad rule is no longer in the file"
    );
    for line in x_lines {
        let err = SymbolicRule::parse_legacy_with_builtins(line);
        assert!(
            err.is_err(),
            "an invalid shipped rule was accepted: `{}`",
            &line[..60]
        );
    }
}

/// A symbolic rule whose halves contain a `cx` -- which the reference synthesizer could
/// never produce -- matches and preserves semantics, including against a region that
/// carries basis-dependent phases.
///
/// The rule moves `rz(theta1) q0` across both the hole and a `cx q0, q1`, licensed by
/// permutations that preserve bit `q0`. Its before-halves differ by a diagonal
/// (`rz` against nothing), so it is phase-safe: the `t` in the region is fine.
#[test]
fn a_rule_with_cx_in_the_halves_applies_soundly() {
    // Constraints: identity, and cx(q0 -> q1); both preserve bit q0.
    let line = "symb q; cx q0, q1; rz(theta1) q0; | rz(theta1) q0; symb q; cx q0, q1; | \
                [{[false, false]=[false, false], [true, false]=[true, false], \
                [false, true]=[false, true], [true, true]=[true, true]}, \
                {[false, false]=[false, false], [true, false]=[true, true], \
                [false, true]=[false, true], [true, true]=[true, false]}]";
    let rule = SymbolicRule::parse_legacy_with_builtins(line).unwrap();
    assert!(rule.phase_safe, "rz-only before-halves are diagonal");
    let registry = reg();
    let limits = SymbolicLimits::default();

    // Region `t b; cx a, b`: the support is the cx permutation, and the `t` adds a
    // basis-dependent phase that only the phase-safe tier may accept.
    let dag = qasm::parse("rz(0.6) a; t b; cx a, b; cx a, b; h b;").unwrap();
    let mut fired = false;
    for seed in 0..16u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        if let Some(out) = apply_symbolic(&dag, &rule, &limits, &registry, &mut rng) {
            fired = true;
            let d = distance(&dag, &out);
            assert!(d < TOL, "cx-in-half rule broke semantics: {d:.3e}");
        }
    }
    assert!(fired, "the cx-in-half rule never applied");
}

/// A boundary qubit that no half touches is bound through the region.
///
/// `x q1; symb q; x q1 | symb q` with the constraint "X on q0": every gate the rule
/// names sits on `q1`, and `q0` exists only in the constraint, standing for whichever
/// wire the region flips. Before free-boundary binding this rule could never match;
/// 540 of the 1,152 shipped `ibm` rules have this shape.
///
/// The rule's before-halves differ by an `x`, which is not diagonal, so it is *not*
/// phase-safe: the region must be the bare permutation. The constraint is "flip both
/// bits", which commutes with the halves' `x q1`, and the good circuit's region is
/// exactly `x a; x b`; the bad one adds a `t a`, whose basis-dependent phase really does
/// break the rewrite, and must be refused.
#[test]
fn a_free_boundary_qubit_binds_through_the_region() {
    // The searched side is the second field, as with plain rules.
    let line = "symb q; | x q1; symb q; x q1; | \
                [{[false, false]=[true, true], [true, false]=[false, true], \
                [false, true]=[true, false], [true, true]=[false, false]}]";
    let rule = SymbolicRule::parse_legacy_with_builtins(line).unwrap();
    assert!(
        !rule.phase_safe,
        "x-against-nothing before-halves are not diagonal"
    );
    let registry = reg();
    let limits = SymbolicLimits::default();

    // The region between the half-matched `x b` gates is `cx b,a; x b; cx b,a`, which
    // computes exactly (a, b) -> (not a, not b): the flip-both permutation, connected to
    // the halves through wire b so the causal region derivation can see all of it.
    let good = qasm::parse("x b; cx b, a; x b; cx b, a; x b; h a;").unwrap();
    let mut fired = false;
    for seed in 0..16u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        if let Some(out) = apply_symbolic(&good, &rule, &limits, &registry, &mut rng) {
            fired = true;
            let d = distance(&good, &out);
            assert!(d < TOL, "free-boundary rule broke semantics: {d:.3e}");
            assert!(
                out.gate_count() + 2 == good.gate_count(),
                "the two x q1 gates should be gone, got {} from {}",
                out.gate_count(),
                good.gate_count()
            );
        }
    }
    assert!(fired, "the free-boundary rule never applied");

    // `t a` puts a basis-dependent phase on the free boundary qubit; accepting it would
    // rewrite to something inequivalent, so the exact tier must refuse the region.
    let bad = qasm::parse("x b; cx b, a; t a; x b; cx b, a; x b; h a;").unwrap();
    for seed in 0..16u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        assert!(
            apply_symbolic(&bad, &rule, &limits, &registry, &mut rng).is_none(),
            "a phase-carrying region was accepted by a non-phase-safe rule"
        );
    }
}

/// A shipped rule whose find side *starts at the hole* now matches.
///
/// `rz((t1+t2)) q0; symb q; h q1; | symb q; h q1; rz((t1+t2)) q0;` is the first rule of
/// the nam symbolic file, and its searched side has no gates before the hole, so the
/// two-anchor matcher could never seed it: 77 of nam's 136 symbolic rules share that
/// shape, 852 of ion's 1,126. The region is grown causally from the one half instead.
/// Its constraints include the `cx q0 -> q1` permutation, so a `cx a, b` region
/// commutes the trailing `rz` back across the hole.
#[test]
fn a_hole_first_rule_matches_by_growing_the_region() {
    let rules = load_symbolic("rules_q3_s3_nam_symb.txt");
    let rule = &rules[0];
    assert!(rule.find_before.is_empty(), "expected the hole-first rule");

    let registry = reg();
    let limits = SymbolicLimits::default();
    let dag = qasm::parse("cx a, b; h b; rz(0.8) a; t b;").unwrap();
    let mut fired = false;
    for seed in 0..16u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        if let Some(out) = apply_symbolic(&dag, rule, &limits, &registry, &mut rng) {
            fired = true;
            let d = distance(&dag, &out);
            assert!(d < TOL, "hole-first rule broke semantics: {d:.3e}");
        }
    }
    assert!(fired, "the hole-first rule never applied");
}
