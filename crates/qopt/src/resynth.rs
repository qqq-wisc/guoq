//! The interface the search uses to reach a resynthesis backend.
//!
//! The backends themselves live in `qresynth` (milestone 6). What matters here is the
//! contract: a backend hands back a circuit *and the error it actually introduced*, as
//! measured by the optimizer rather than claimed by the tool.

use qcircuit::{Dag, GateRegistry};

/// A resynthesis result the optimizer has already verified.
#[derive(Debug, Clone)]
pub struct ResynthOutcome {
    /// The circuit with one partition replaced.
    pub dag: Dag,
    /// The distance between the original partition and its replacement, measured by
    /// [`qsemantics::phase_invariant_distance`].
    pub error: f64,
}

/// Something that can replace part of a circuit with an equivalent.
pub trait Resynthesizer {
    /// Try to improve `dag`, spending no more than `budget` error.
    ///
    /// Returns `None` if nothing was produced, or if what came back would cost more than
    /// `budget`. Implementations must measure the error themselves; the budget is a hard
    /// ceiling, not a hint passed on to a backend and forgotten.
    fn resynthesize(
        &self,
        dag: &Dag,
        budget: f64,
        registry: &GateRegistry,
        rng: &mut dyn RngCore,
    ) -> Option<ResynthOutcome>;

    /// A name for logs.
    fn name(&self) -> &str;
}

/// Object-safe RNG handle, so `Resynthesizer` can be a trait object.
pub trait RngCore: rand::RngCore {}

impl<T: rand::RngCore> RngCore for T {}

/// A backend that never does anything, for `-resynth NONE`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullResynthesizer;

impl Resynthesizer for NullResynthesizer {
    fn resynthesize(
        &self,
        _dag: &Dag,
        _budget: f64,
        _registry: &GateRegistry,
        _rng: &mut dyn RngCore,
    ) -> Option<ResynthOutcome> {
        None
    }

    fn name(&self) -> &str {
        "none"
    }
}

/// Adapts a [`qresynth::VerifiedResynthesizer`] to the search's interface.
///
/// The measurement and the budget check happen inside `qresynth`; this only carries the
/// verdict across. What reaches the search is a circuit and the error that was *measured*
/// for it, never a number a backend claimed.
pub struct VerifiedBackend<'a> {
    pub inner: qresynth::VerifiedResynthesizer<'a>,
    /// Counts of each rejection reason, for logging why resynthesis is not helping.
    pub stats: std::sync::Mutex<ResynthStats>,
}

/// Why resynthesis attempts did not produce anything.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResynthStats {
    pub attempts: usize,
    pub accepted: usize,
    pub no_partition: usize,
    pub backend_declined: usize,
    pub unusable: usize,
    pub over_budget: usize,
    pub errored: usize,
    /// Total measured error of accepted results.
    pub total_error: f64,
}

impl<'a> VerifiedBackend<'a> {
    pub fn new(inner: qresynth::VerifiedResynthesizer<'a>) -> Self {
        Self {
            inner,
            stats: std::sync::Mutex::new(ResynthStats::default()),
        }
    }

    pub fn stats(&self) -> ResynthStats {
        *self.stats.lock().expect("stats mutex")
    }
}

impl Resynthesizer for VerifiedBackend<'_> {
    fn resynthesize(
        &self,
        dag: &Dag,
        budget: f64,
        registry: &GateRegistry,
        rng: &mut dyn RngCore,
    ) -> Option<ResynthOutcome> {
        let mut stats = self.stats.lock().expect("stats mutex");
        stats.attempts += 1;
        let outcome = self.inner.attempt(dag, budget, registry, rng);
        match outcome {
            Ok(qresynth::Outcome::Accepted(a)) => {
                stats.accepted += 1;
                stats.total_error += a.error;
                Some(ResynthOutcome {
                    dag: a.dag,
                    error: a.error,
                })
            }
            Ok(qresynth::Outcome::Rejected(r)) => {
                match r {
                    qresynth::Rejected::NoPartition => stats.no_partition += 1,
                    qresynth::Rejected::BackendDeclined => stats.backend_declined += 1,
                    qresynth::Rejected::Unusable(_) => stats.unusable += 1,
                    qresynth::Rejected::OverBudget { .. } => stats.over_budget += 1,
                    qresynth::Rejected::NoImprovement => stats.unusable += 1,
                }
                None
            }
            Err(_) => {
                stats.errored += 1;
                None
            }
        }
    }

    fn name(&self) -> &str {
        self.inner.backend.name()
    }
}

/// A backend that returns a fixed circuit at a fixed cost. For tests.
#[derive(Debug, Clone)]
pub struct FixedResynthesizer {
    pub result: Dag,
    pub error: f64,
}

impl Resynthesizer for FixedResynthesizer {
    fn resynthesize(
        &self,
        _dag: &Dag,
        budget: f64,
        _registry: &GateRegistry,
        _rng: &mut dyn RngCore,
    ) -> Option<ResynthOutcome> {
        // Even a stub must respect the budget, so tests exercise the same path.
        if self.error > budget {
            return None;
        }
        Some(ResynthOutcome {
            dag: self.result.clone(),
            error: self.error,
        })
    }

    fn name(&self) -> &str {
        "fixed"
    }
}

/// Helper so `&mut ChaCha8Rng` can be passed where `&mut dyn RngCore` is wanted.
pub fn as_dyn<R: rand::RngCore>(rng: &mut R) -> &mut dyn RngCore {
    rng
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// The adapter must report the *measured* error, and count why attempts failed.
    #[test]
    fn the_adapter_reports_measured_error_and_reasons() {
        struct Echo;
        impl qresynth::Backend for Echo {
            fn run(
                &self,
                r: &qresynth::Request,
                _reg: &GateRegistry,
            ) -> anyhow::Result<Option<Dag>> {
                Ok(Some(r.circuit.clone()))
            }
            fn name(&self) -> &str {
                "echo"
            }
        }

        let echo = Echo;
        let inner = qresynth::VerifiedResynthesizer::new(&echo, "none");
        let backend = VerifiedBackend::new(inner);
        let reg = GateRegistry::with_builtins();
        let mut rng = ChaCha8Rng::seed_from_u64(4);

        let dag = qasm::parse("h a; cx a, b; t b;").unwrap();
        let out = backend
            .resynthesize(&dag, 1.0, &reg, as_dyn(&mut rng))
            .expect("echo always succeeds");
        assert!(out.error < 1e-12, "echo should measure zero error");
        assert_eq!(backend.name(), "echo");

        let s = backend.stats();
        assert_eq!(s.attempts, 1);
        assert_eq!(s.accepted, 1);

        // A circuit with no gates has nothing to partition.
        let empty = qasm::parse("qreg q[2];").unwrap();
        assert!(backend
            .resynthesize(&empty, 1.0, &reg, as_dyn(&mut rng))
            .is_none());
        assert_eq!(backend.stats().no_partition, 1);
    }

    #[test]
    fn null_backend_never_produces_anything() {
        let dag = qasm::parse("h a;").unwrap();
        let reg = GateRegistry::with_builtins();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert!(NullResynthesizer
            .resynthesize(&dag, 1.0, &reg, as_dyn(&mut rng))
            .is_none());
        assert_eq!(NullResynthesizer.name(), "none");
    }

    #[test]
    fn a_backend_must_respect_the_budget() {
        let dag = qasm::parse("h a;").unwrap();
        let reg = GateRegistry::with_builtins();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let backend = FixedResynthesizer {
            result: qasm::parse("x a;").unwrap(),
            error: 0.5,
        };
        assert!(backend
            .resynthesize(&dag, 1.0, &reg, as_dyn(&mut rng))
            .is_some());
        assert!(
            backend
                .resynthesize(&dag, 0.1, &reg, as_dyn(&mut rng))
                .is_none(),
            "a result costing more than the budget must be refused"
        );
        // Exactly on budget is affordable.
        assert!(backend
            .resynthesize(&dag, 0.5, &reg, as_dyn(&mut rng))
            .is_some());
    }
}
