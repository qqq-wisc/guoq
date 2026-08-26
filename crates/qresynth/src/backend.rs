//! The resynthesis backend interface.

use qcircuit::{Dag, GateRegistry};

/// What a backend is asked for.
#[derive(Debug, Clone)]
pub struct Request {
    /// The block to resynthesize, over `q[0]..q[k-1]`.
    pub circuit: Dag,
    /// The gate set the result should be expressed in, by the backend's own name for it.
    pub target_gate_set: String,
    /// An error the backend may aim for.
    ///
    /// A *hint*, not a contract. Backends define error differently from one another —
    /// BQSKit's `synthesis_epsilon` bounds a per-pass Hilbert-Schmidt residual, Synthetiq's
    /// `-eps` means something else — and neither is checked by the tool that reports it.
    /// The optimizer measures what it actually got; this only stops a backend from
    /// working harder than it needs to.
    pub epsilon_hint: f64,
}

/// Something that can rewrite a small circuit into an equivalent one.
///
/// Backends return a circuit and nothing else. They are not asked how accurate it is,
/// because the answer would not be comparable between them and could not be checked.
pub trait Backend: Send + Sync {
    /// Resynthesize, or return `None` if this backend cannot help with this request.
    fn run(&self, request: &Request, registry: &GateRegistry) -> anyhow::Result<Option<Dag>>;

    fn name(&self) -> &str;

    /// Release any resources, such as a worker process.
    fn shutdown(&self) {}
}

/// A backend that declines everything, for `-resynth NONE`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoBackend;

impl Backend for NoBackend {
    fn run(&self, _request: &Request, _registry: &GateRegistry) -> anyhow::Result<Option<Dag>> {
        Ok(None)
    }

    fn name(&self) -> &str {
        "none"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;

    #[test]
    fn the_null_backend_declines() {
        let req = Request {
            circuit: qasm::parse("h q[0];").unwrap(),
            target_gate_set: "none".into(),
            epsilon_hint: 1e-8,
        };
        let out = NoBackend.run(&req, &GateRegistry::with_builtins()).unwrap();
        assert!(out.is_none());
        assert_eq!(NoBackend.name(), "none");
    }
}
