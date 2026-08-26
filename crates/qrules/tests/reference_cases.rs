//! The rule-application cases from the reference Java suite, ported.
//!
//! `OptimizerTest.java` asserted on exact QASM strings, which pins the topological order
//! the reference's printer happened to produce. These check the two things that actually
//! matter — did the rule fire, and is the result the same unitary — so they are robust to
//! a different but equally valid gate ordering.

use qcircuit::{qasm, GateRegistry, QubitId};
use qrules::{apply_rule, ApplyOptions, Rule};
use qsemantics::{phase_invariant_distance, Unitary};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn reg() -> GateRegistry {
    GateRegistry::with_builtins()
}

fn opts() -> ApplyOptions {
    ApplyOptions {
        apply_once: false,
        shuffle: false,
        ..Default::default()
    }
}

/// Apply `find | replace` to `circuit`; returns the resulting circuit.
fn apply(circuit: &str, find: &str, replace: &str) -> qcircuit::Dag {
    let dag = qasm::parse(circuit).unwrap();
    let rule = Rule::new(find, replace).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    match apply_rule(&dag, &rule, &opts(), &mut rng) {
        Some(r) => r.dag,
        None => dag,
    }
}

fn union_qubits(a: &qcircuit::Dag, b: &qcircuit::Dag) -> Vec<QubitId> {
    let mut qs: Vec<QubitId> = a.qubits().to_vec();
    for q in b.qubits() {
        if !qs.contains(q) {
            qs.push(q.clone());
        }
    }
    qs.sort();
    qs
}

/// Assert two circuits implement the same unitary.
fn assert_same_semantics(got: &qcircuit::Dag, want_src: &str) {
    let want = qasm::parse(want_src).unwrap();
    let qs = union_qubits(got, &want);
    let a = Unitary::from_dag_over(got, qs.clone(), &reg()).unwrap();
    let b = Unitary::from_dag_over(&want, qs, &reg()).unwrap();
    let d = phase_invariant_distance(&a, &b);
    assert!(
        d < 1e-9,
        "circuits differ by {d:.3e}\n  got:  {}\n  want: {want_src}",
        qasm::to_qasm_with(got, &qasm::PrintOptions::bare())
            .lines()
            .collect::<Vec<_>>()
            .join(" ")
    );
}

fn gate_names(dag: &qcircuit::Dag) -> Vec<String> {
    let mut v: Vec<String> = dag
        .topological_gates()
        .into_iter()
        .map(|i| {
            let g = dag.gate(i);
            format!("{} {}", g.gate, g.qubits.join(","))
        })
        .collect();
    v.sort();
    v
}

/// Assert the rule did not fire.
fn assert_unchanged(circuit: &str, find: &str, replace: &str) {
    let before = qasm::parse(circuit).unwrap();
    let after = apply(circuit, find, replace);
    assert_eq!(
        gate_names(&before),
        gate_names(&after),
        "rule `{find} | {replace}` should not have matched"
    );
}

#[test]
fn rule_01_cancels_adjacent_hadamards() {
    let got = apply("h q1; h q2; h q2; x q2;", "h q0; h q0;", "");
    assert_eq!(got.gate_count(), 2);
    assert_same_semantics(&got, "h q1; x q2;");
}

#[test]
fn rule_02_reorders_a_cx_triple() {
    let got = apply(
        "x q0; x q1; cx q0,q1; cx q2,q0; cx q2,q1;",
        "cx q0,q1; cx q2,q0; cx q2,q1;",
        "cx q2,q0; cx q0,q1;",
    );
    assert_eq!(got.gate_count(), 4);
    assert_same_semantics(&got, "x q0; x q1; cx q2,q0; cx q0,q1;");
}

#[test]
fn rule_03_commutes_two_cx_sharing_a_control() {
    let got = apply(
        "t q2; cx q2,q1; cx q2,q0; cx q3,q1;",
        "cx q2,q1; cx q2,q0;",
        "cx q2,q0; cx q2,q1;",
    );
    assert_eq!(got.gate_count(), 4);
    assert_same_semantics(&got, "t q2; cx q2,q0; cx q2,q1; cx q3,q1;");
}

#[test]
fn rule_04_does_not_match_a_different_shape() {
    assert_unchanged(
        "x q0; x q1; cx q0,q1; cx q2,q0; cx q2,q1;",
        "x q3; x q1; cx q3,q1; cx q2,q4; cx q2,q1;",
        "",
    );
}

#[test]
fn rule_05_does_not_match_across_different_controls() {
    assert_unchanged(
        "t q4; cx q2,q4; cx q2,q6; tdg q4; cx q3,q5; cx q3,q4;",
        "t q1; cx q0,q1; tdg q1; cx q0,q1;",
        "cx q0,q1; tdg q1; cx q0,q1; t q1;",
    );
}

#[test]
fn rule_06_does_not_match_when_operands_disagree() {
    assert_unchanged(
        "s q2; cx q1,q2; cx q2,q3; tdg q3; cx q1,q3;",
        "s q0; cx q2,q0; cx q2,q1;",
        "",
    );
}

#[test]
fn rule_07_cancels_a_cx_pair_with_swapped_operands() {
    let got = apply(
        "x q1; x q0; cx q0,q1; cx q1,q0; x q1; x q0;",
        "cx q0,q1; cx q1,q0;",
        "",
    );
    assert_eq!(got.gate_count(), 4);
    assert_same_semantics(&got, "x q1; x q0; x q1; x q0;");
}

/// Adjacency: an intervening gate on a shared wire blocks the match.
#[test]
fn rule_08_does_not_match_across_an_intervening_gate() {
    assert_unchanged(
        "x q1; x q0; cx q0,q1; h q1; cx q1,q0; x q1; x q0;",
        "cx q0,q1; cx q1,q0;",
        "",
    );
}

/// Convexity: every pattern gate is adjacent on its wires, but a path runs out of the
/// match and back into it through `cx q2,q3; cx q3,q1`. Replacing would have no single
/// valid insertion point.
///
/// This is the case the reference's `checkLCA` existed to catch. That check only
/// considered gates for which `isCX()` held; here convexity is tested directly, so the
/// same rejection follows from a general rule rather than a special case.
#[test]
fn rule_09_rejects_a_non_convex_match() {
    assert_unchanged(
        "h q0; cx q2,q0; h q0; cx q2,q3; cx q3,q1; cx q0,q1;",
        "h q0; cx q2,q0; h q0; cx q0,q1;",
        "cx q0,q1; h q0; cx q2,q0; h q0;",
    );
}

#[test]
fn rule_10_matches_a_whole_circuit_including_a_three_qubit_gate() {
    let circuit =
        "t q13; cx q13,q16; tdg q16; cx q14,q16; t q16; h q16; ccz q9,q16,q15; t q14; h q9; h q16;";
    let got = apply(circuit, circuit, "");
    assert_eq!(got.gate_count(), 0);
}

#[test]
fn rule_11_applies_only_the_valid_disjoint_match() {
    let got = apply(
        "cx q0,q3; cx q3,q4; t q4; tdg q3; cx q0,q4; cx q2,q3; cx q0,q3; cx q2,q4; cx q2,q1; tdg q4;",
        "cx q2,q0; cx q1,q0",
        "cx q1,q0; cx q2,q0;",
    );
    // Gate count is preserved by this rule; the point is that a second, invalid match is
    // not taken.
    assert_eq!(got.gate_count(), 10);
    assert_same_semantics(
        &got,
        "cx q0,q3; cx q3,q4; t q4; tdg q3; cx q2,q3; cx q2,q4; cx q0,q4; cx q2,q1; tdg q4; cx q0,q3;",
    );
}

/// Rotation merging to the identity.
///
/// The reference expected an empty circuit here, because its *parser* discarded any gate
/// whose angles were multiples of `4*PI` — so `rz(0)` vanished the moment the rewritten
/// circuit was re-parsed. This port keeps the parser semantics-neutral and makes identity
/// removal an explicit pass, so the merged `rz(0)` is present until asked to go.
#[test]
fn rule_12_merges_rotations_to_identity() {
    use std::f64::consts::PI;
    let got = apply(
        &format!("rz({}) q0; rz({}) q0;", PI / 2.0, -PI / 2.0),
        "rz(theta1) q0; rz(theta2) q0;",
        "rz((theta1+theta2)) q0;",
    );
    assert_eq!(got.gate_count(), 1);
    assert_eq!(
        got.gate(got.topological_gates()[0]).params[0].eval(),
        Some(0.0)
    );

    let mut cleaned = got.clone();
    cleaned.drop_identity_gates(&reg());
    assert_eq!(cleaned.gate_count(), 0);
}

#[test]
fn rotation_merging_preserves_semantics_at_arbitrary_angles() {
    let got = apply(
        "rz(0.3) q0; rz(1.4) q0;",
        "rz(theta1) q0; rz(theta2) q0;",
        "rz((theta1+theta2)) q0;",
    );
    assert_eq!(got.gate_count(), 1);
    assert_same_semantics(&got, "rz(1.7) q0;");
}
