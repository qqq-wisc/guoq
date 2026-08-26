//! Unitary semantics for quantum circuits.
//!
//! Provides the ability to build a circuit's unitary matrix and to measure how far apart
//! two unitaries are. The latter is what lets the optimizer check resynthesis results
//! itself rather than trusting a backend's own error claim — see [`distance`].

pub mod distance;
pub mod unitary;

pub use distance::{
    equivalent_exact, equivalent_up_to_phase, hs_distance, normalized_overlap, overlap,
    phase_invariant_distance, trace_overlap, DistanceReport,
};
pub use unitary::{ParamEnv, Unitary, DEFAULT_MAX_QUBITS};
