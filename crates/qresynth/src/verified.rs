//! Resynthesis with the error measured by the optimizer.
//!
//! This is the module the whole "compute the distance ourselves" change lives in.
//!
//! # What the reference did
//!
//! `Optimizer.applyResynth` picked a partition, sent it to a backend with
//! `Params.EPSILON / Params.MAX_RESYNTH_ALLOWED` as the epsilon, and spliced whatever came
//! back into the circuit. Nothing measured the result. Three consequences:
//!
//! - The epsilon meant different things to BQSKit and Synthetiq, so the same number bought
//!   different accuracy depending on which was configured.
//! - `MAX_RESYNTH_ALLOWED` is `-1` for "no limit", the value the README documents, which
//!   makes the per-call epsilon **negative**.
//! - A backend that returned something wrong — or the input circuit unchanged, which both
//!   `Bqskit.socket` and `Synthetiq.socket` do on any HTTP error — was indistinguishable
//!   from one that worked.
//!
//! # What happens here
//!
//! The partition's unitary is built before the call and the result's unitary after it, and
//! the two are compared with [`qsemantics::phase_invariant_distance`]. A result is
//! accepted only if it fits in the *remaining* budget, which is tracked per candidate
//! rather than divided up front. Anything the backend gets wrong shows up as a distance.

use anyhow::Result;
use rand::Rng;

use qcircuit::{Dag, GateRegistry};
use qsemantics::{phase_invariant_distance, DistanceReport, Unitary};

use crate::backend::{Backend, Request};
use crate::partition::{replace_partition, PartitionLimits, PartitionPicker, RandomPartition};

/// A resynthesis that was measured and accepted.
#[derive(Debug, Clone)]
pub struct Accepted {
    /// The whole circuit, with the partition replaced.
    pub dag: Dag,
    /// The measured distance between the old block and the new one.
    pub error: f64,
    /// The full measurement, for logging.
    pub report: DistanceReport,
    pub gates_before: usize,
    pub gates_after: usize,
}

/// Why a resynthesis attempt produced nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum Rejected {
    /// No partition could be chosen.
    NoPartition,
    /// The backend declined.
    BackendDeclined,
    /// The result could not be parsed or mapped back onto the circuit.
    Unusable(String),
    /// The result was measured and costs more than the remaining budget.
    OverBudget { error: f64, budget: f64 },
    /// The result was not an improvement.
    NoImprovement,
}

/// Outcome of one attempt.
#[derive(Debug, Clone)]
pub enum Outcome {
    Accepted(Box<Accepted>),
    Rejected(Rejected),
}

/// Runs a backend and checks what it returns.
pub struct VerifiedResynthesizer<'a> {
    pub backend: &'a dyn Backend,
    pub target_gate_set: String,
    pub limits: PartitionLimits,
    /// Accept a result only if it does not make the block larger.
    pub require_improvement: bool,
    /// The accuracy asked of the backend per call; `None` asks for the whole remaining
    /// budget.
    ///
    /// An even slice of the total budget (`epsilon / N`, the reference's split) spreads
    /// the spending so one early sloppy result cannot eat the allowance that hundreds of
    /// later calls could have shared. It is a request, never a bound: acceptance is
    /// always the measurement against the remaining budget, and a call may come back
    /// better or worse than its slice.
    pub epsilon_hint: Option<f64>,
}

impl<'a> VerifiedResynthesizer<'a> {
    pub fn new(backend: &'a dyn Backend, target_gate_set: impl Into<String>) -> Self {
        Self {
            backend,
            target_gate_set: target_gate_set.into(),
            limits: PartitionLimits::default(),
            require_improvement: false,
            epsilon_hint: None,
        }
    }

    /// Try to resynthesize part of `dag`, spending at most `budget` error.
    pub fn attempt<R: Rng + ?Sized>(
        &self,
        dag: &Dag,
        budget: f64,
        registry: &GateRegistry,
        rng: &mut R,
    ) -> Result<Outcome> {
        let Some(partition) = RandomPartition.pick(dag, &self.limits, rng) else {
            return Ok(Outcome::Rejected(Rejected::NoPartition));
        };
        let block = partition.to_circuit(dag);

        // The block's unitary, before anyone touches it.
        let order: Vec<qcircuit::QubitId> = block.qubits().to_vec();
        let before = match Unitary::from_dag_over(&block, order.clone(), registry) {
            Ok(u) => u,
            Err(e) => return Ok(Outcome::Rejected(Rejected::Unusable(e.to_string()))),
        };

        let request = Request {
            circuit: block.clone(),
            target_gate_set: self.target_gate_set.clone(),
            // The configured slice, but never more than remains: asking for less
            // accuracy than the remaining budget would invite a result that cannot be
            // accepted.
            epsilon_hint: self
                .epsilon_hint
                .map_or(budget, |slice| slice.min(budget))
                .max(0.0),
        };
        let Some(result) = self.backend.run(&request, registry)? else {
            return Ok(Outcome::Rejected(Rejected::BackendDeclined));
        };

        // Measure. This is the step the reference never performed.
        let after = match Unitary::from_dag_over(&result, order, registry) {
            Ok(u) => u,
            Err(e) => return Ok(Outcome::Rejected(Rejected::Unusable(e.to_string()))),
        };
        let report = DistanceReport::measure(&before, &after);
        let error = report.phase_invariant;

        if !error.is_finite() || error > budget {
            return Ok(Outcome::Rejected(Rejected::OverBudget { error, budget }));
        }
        if self.require_improvement && result.gate_count() > block.gate_count() {
            return Ok(Outcome::Rejected(Rejected::NoImprovement));
        }

        // Map the backend's register naming back onto the circuit's own qubits.
        let mut mapped = Dag::new(partition.qubits.clone());
        for idx in result.topological_gates() {
            let op = result.gate(idx);
            let mut qubits = Vec::with_capacity(op.qubits.len());
            for q in &op.qubits {
                match partition.rename_back(q) {
                    Some(name) => qubits.push(name),
                    None => {
                        return Ok(Outcome::Rejected(Rejected::Unusable(format!(
                            "backend returned unknown qubit `{q}`"
                        ))))
                    }
                }
            }
            mapped.push_gate(qcircuit::GateOp {
                gate: op.gate.clone(),
                qubits,
                params: op.params.clone(),
            });
        }

        let Some(spliced) = replace_partition(dag, &partition, &mapped) else {
            return Ok(Outcome::Rejected(Rejected::Unusable(
                "replacement could not be spliced in".into(),
            )));
        };

        Ok(Outcome::Accepted(Box::new(Accepted {
            dag: spliced,
            error,
            report,
            gates_before: block.gate_count(),
            gates_after: result.gate_count(),
        })))
    }
}

/// Compare two whole circuits, for `--verify-final`.
pub fn circuit_distance(
    a: &Dag,
    b: &Dag,
    registry: &GateRegistry,
    max_qubits: usize,
) -> Option<f64> {
    let mut qubits: Vec<qcircuit::QubitId> = a.qubits().to_vec();
    for q in b.qubits() {
        if !qubits.contains(q) {
            qubits.push(q.clone());
        }
    }
    qubits.sort();
    if qubits.len() > max_qubits {
        return None;
    }
    let ua = Unitary::from_dag_over(a, qubits.clone(), registry).ok()?;
    let ub = Unitary::from_dag_over(b, qubits, registry).ok()?;
    Some(phase_invariant_distance(&ua, &ub))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    /// A backend that returns whatever it is told to, ignoring the input.
    struct Canned(Option<String>);

    impl Backend for Canned {
        fn run(&self, _r: &Request, _reg: &GateRegistry) -> Result<Option<Dag>> {
            Ok(self.0.as_ref().map(|s| qasm::parse(s).unwrap()))
        }
        fn name(&self) -> &str {
            "canned"
        }
    }

    /// A backend that hands the input straight back, which is what the reference's HTTP
    /// clients do on any error.
    struct Echo;

    impl Backend for Echo {
        fn run(&self, r: &Request, _reg: &GateRegistry) -> Result<Option<Dag>> {
            Ok(Some(r.circuit.clone()))
        }
        fn name(&self) -> &str {
            "echo"
        }
    }

    #[test]
    fn a_declining_backend_is_reported() {
        let v = VerifiedResynthesizer::new(&NoBackendRef, "none");
        let d = qasm::parse("h a; cx a, b;").unwrap();
        match v.attempt(&d, 1.0, &reg(), &mut rng(0)).unwrap() {
            Outcome::Rejected(Rejected::BackendDeclined) => {}
            other => panic!("expected a decline, got {other:?}"),
        }
    }

    struct NoBackendRef;
    impl Backend for NoBackendRef {
        fn run(&self, _r: &Request, _reg: &GateRegistry) -> Result<Option<Dag>> {
            Ok(None)
        }
        fn name(&self) -> &str {
            "none"
        }
    }

    #[test]
    fn a_gate_free_circuit_has_no_partition() {
        let v = VerifiedResynthesizer::new(&Echo, "none");
        let d = qasm::parse("qreg q[3];").unwrap();
        match v.attempt(&d, 1.0, &reg(), &mut rng(0)).unwrap() {
            Outcome::Rejected(Rejected::NoPartition) => {}
            other => panic!("expected no partition, got {other:?}"),
        }
    }

    /// An echoing backend costs nothing and changes nothing: the measurement must say so.
    #[test]
    fn an_echoing_backend_measures_zero_error() {
        let v = VerifiedResynthesizer::new(&Echo, "none");
        let d = qasm::parse("h a; cx a, b; t b;").unwrap();
        match v.attempt(&d, 1.0, &reg(), &mut rng(3)).unwrap() {
            Outcome::Accepted(a) => {
                assert!(a.error < 1e-12, "echo should measure zero, got {}", a.error);
                assert_eq!(a.gates_before, a.gates_after);
                assert_eq!(a.dag.structural_hash(), d.structural_hash());
            }
            other => panic!("expected acceptance, got {other:?}"),
        }
    }

    /// The headline behaviour: a backend that returns something wrong is caught, however
    /// confidently it was returned.
    #[test]
    fn a_faulty_backend_is_rejected() {
        // Whatever the partition, this replacement is not equivalent to it.
        let backend = Canned(Some("x q[0];".into()));
        let mut v = VerifiedResynthesizer::new(&backend, "none");
        v.limits = PartitionLimits {
            max_qubits: 1,
            max_gates: 4,
        };
        let d = qasm::parse("h a; t a; h b;").unwrap();

        let mut rejected = 0;
        let mut accepted = 0;
        for seed in 0..20 {
            match v.attempt(&d, 1e-8, &reg(), &mut rng(seed)).unwrap() {
                Outcome::Rejected(Rejected::OverBudget { error, budget }) => {
                    assert!(error > budget);
                    rejected += 1;
                }
                Outcome::Accepted(a) => {
                    // Only acceptable if it really was equivalent.
                    assert!(a.error <= 1e-8);
                    accepted += 1;
                }
                other => panic!("unexpected outcome {other:?}"),
            }
        }
        assert!(rejected > 0, "a wrong result was never rejected");
        let _ = accepted;
    }

    /// The budget is a ceiling, checked against the measurement.
    #[test]
    fn the_budget_is_enforced() {
        // `rz(1e-6)` on top of the block: a small but real error.
        let d = qasm::parse("rz(0.5) a;").unwrap();
        let backend = Canned(Some("rz(0.500001) q[0];".into()));
        let mut v = VerifiedResynthesizer::new(&backend, "none");
        v.limits = PartitionLimits {
            max_qubits: 1,
            max_gates: 4,
        };

        // A rotation error of eps on a single qubit measures eps / sqrt(2); see
        // `distance_scales_with_partition_width` for why.
        let expected = 1e-6 / std::f64::consts::SQRT_2;

        // Generous budget: accepted, with the error measured rather than assumed.
        match v.attempt(&d, 1e-3, &reg(), &mut rng(0)).unwrap() {
            Outcome::Accepted(a) => {
                assert!((a.error - expected).abs() < 1e-9, "measured {}", a.error);
            }
            other => panic!("expected acceptance, got {other:?}"),
        }

        // Tight budget: refused, with the numbers reported.
        match v.attempt(&d, 1e-9, &reg(), &mut rng(0)).unwrap() {
            Outcome::Rejected(Rejected::OverBudget { error, budget }) => {
                assert!((error - expected).abs() < 1e-9);
                assert_eq!(budget, 1e-9);
            }
            other => panic!("expected a budget rejection, got {other:?}"),
        }
    }

    /// The measured error depends on how wide the partition is.
    ///
    /// `phase_invariant_distance` is a Frobenius norm. A rotation error `eps` puts a
    /// deviation of about `eps/2` on each of the `d = 2^n` diagonal entries, so the norm
    /// is `sqrt(d) * eps/2 = eps * 2^(n/2 - 1)`: `eps/sqrt(2)` on one qubit, `eps` on two,
    /// `eps*sqrt(2)` on three. The operator-norm distance it bounds is `eps/2` whatever
    /// the width, so the bound loosens by exactly `sqrt(d)` as blocks grow — a factor of
    /// 2.8 at the default three-qubit limit.
    ///
    /// That direction is the safe one: a loose *upper* bound rejects some results that
    /// would have been affordable, rather than accepting ones that are not. Tightening it
    /// would need the eigenphases of `U^dagger V`, hence an eigensolver, which is not
    /// worth a LAPACK dependency for a factor of three.
    #[test]
    fn distance_scales_with_partition_width() {
        use qsemantics::Unitary;
        let eps = 1e-6;
        let exact = qasm::parse("rz(0.5) q0;").unwrap();
        let approx = qasm::parse(&format!("rz({}) q0;", 0.5 + eps)).unwrap();
        for n in 1..=4usize {
            let qubits: Vec<qcircuit::QubitId> =
                (0..n).map(|i| qcircuit::intern(&format!("q{i}"))).collect();
            let a = Unitary::from_dag_over(&exact, qubits.clone(), &reg()).unwrap();
            let b = Unitary::from_dag_over(&approx, qubits, &reg()).unwrap();
            let measured = phase_invariant_distance(&a, &b);
            let expected = eps * 2f64.powf(n as f64 / 2.0 - 1.0);
            assert!(
                (measured - expected).abs() < expected * 1e-3,
                "n={n}: measured {measured:.3e}, expected {expected:.3e}"
            );
        }
    }

    /// A zero or negative budget accepts only an exact result.
    ///
    /// The reference reached a negative epsilon whenever `--max-resynth-allowed -1` was
    /// passed, the value its own README documents for "no limit".
    #[test]
    fn a_nonpositive_budget_admits_only_exact_results() {
        let d = qasm::parse("rz(0.5) a;").unwrap();
        let mut v = VerifiedResynthesizer::new(&Echo, "none");
        v.limits = PartitionLimits {
            max_qubits: 1,
            max_gates: 4,
        };
        // Echo is exact, so zero budget is enough.
        assert!(matches!(
            v.attempt(&d, 0.0, &reg(), &mut rng(0)).unwrap(),
            Outcome::Accepted(_)
        ));

        let backend = Canned(Some("rz(0.6) q[0];".into()));
        let mut v = VerifiedResynthesizer::new(&backend, "none");
        v.limits = PartitionLimits {
            max_qubits: 1,
            max_gates: 4,
        };
        assert!(matches!(
            v.attempt(&d, 0.0, &reg(), &mut rng(0)).unwrap(),
            Outcome::Rejected(Rejected::OverBudget { .. })
        ));
        // And a negative budget rejects even an exact result rather than doing something
        // undefined.
        assert!(matches!(
            v.attempt(&d, -1.0, &reg(), &mut rng(0)).unwrap(),
            Outcome::Rejected(Rejected::OverBudget { .. })
        ));
    }

    /// The backend is asked for the configured slice of the budget, not the whole
    /// remainder — until less than a slice remains, when asking for more accuracy than
    /// could be accepted is the only honest request left.
    #[test]
    fn the_backend_is_hinted_the_slice_but_never_more_than_remains() {
        struct HintProbe(std::sync::Mutex<Vec<f64>>);
        impl Backend for HintProbe {
            fn run(&self, r: &Request, _reg: &GateRegistry) -> Result<Option<Dag>> {
                self.0.lock().unwrap().push(r.epsilon_hint);
                Ok(None)
            }
            fn name(&self) -> &str {
                "probe"
            }
        }
        let probe = HintProbe(std::sync::Mutex::new(Vec::new()));
        let mut v = VerifiedResynthesizer::new(&probe, "none");
        v.epsilon_hint = Some(1e-10);
        let d = qasm::parse("h a; t a;").unwrap();

        // Plenty of budget: the slice is what gets asked for.
        v.attempt(&d, 1e-8, &reg(), &mut rng(0)).unwrap();
        // Less than a slice left: the remainder is.
        v.attempt(&d, 4e-11, &reg(), &mut rng(0)).unwrap();
        // No slice configured: the remaining budget, as before.
        v.epsilon_hint = None;
        v.attempt(&d, 1e-8, &reg(), &mut rng(0)).unwrap();

        let hints = probe.0.lock().unwrap();
        assert_eq!(hints.as_slice(), &[1e-10, 4e-11, 1e-8]);
    }

    #[test]
    fn a_result_naming_unknown_qubits_is_refused() {
        let backend = Canned(Some("h q[9];".into()));
        let mut v = VerifiedResynthesizer::new(&backend, "none");
        v.limits = PartitionLimits {
            max_qubits: 1,
            max_gates: 2,
        };
        let d = qasm::parse("h a;").unwrap();
        match v.attempt(&d, 1.0, &reg(), &mut rng(0)).unwrap() {
            Outcome::Rejected(Rejected::Unusable(_))
            | Outcome::Rejected(Rejected::OverBudget { .. }) => {}
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    #[test]
    fn require_improvement_refuses_larger_results() {
        let backend = Canned(Some("h q[0]; h q[0]; h q[0];".into()));
        let mut v = VerifiedResynthesizer::new(&backend, "none");
        v.limits = PartitionLimits {
            max_qubits: 1,
            max_gates: 1,
        };
        v.require_improvement = true;
        let d = qasm::parse("h a;").unwrap();
        // h;h;h == h, so the error is zero, but it is three gates instead of one.
        match v.attempt(&d, 1.0, &reg(), &mut rng(0)).unwrap() {
            Outcome::Rejected(Rejected::NoImprovement) => {}
            other => panic!("expected NoImprovement, got {other:?}"),
        }
    }

    #[test]
    fn accepted_results_keep_the_circuit_well_formed() {
        let v = VerifiedResynthesizer::new(&Echo, "none");
        let d = qasm::parse("h a; cx a, b; t b; cx b, c; h c; x a;").unwrap();
        for seed in 0..25 {
            if let Outcome::Accepted(a) = v.attempt(&d, 1.0, &reg(), &mut rng(seed)).unwrap() {
                assert!(a.dag.is_acyclic(), "seed {seed} produced a cycle");
                assert_eq!(a.dag.num_qubits(), d.num_qubits());
                let dist = circuit_distance(&d, &a.dag, &reg(), 12).unwrap();
                assert!(dist < 1e-9, "seed {seed}: circuit changed by {dist:.3e}");
            }
        }
    }

    #[test]
    fn circuit_distance_declines_wide_circuits() {
        let a = qcircuit::Dag::new((0..20).map(|i| format!("q{i}")));
        let b = a.clone();
        assert!(circuit_distance(&a, &b, &reg(), 12).is_none());
    }
}
