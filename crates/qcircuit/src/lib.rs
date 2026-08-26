//! Quantum circuit intermediate representation.
//!
//! This crate holds everything that is true about a circuit independent of what one wants
//! to *do* with it: the gate registry, the DAG, and OpenQASM parsing and printing.
//!
//! The organising idea is that **gate arity is data, not code**. A gate declares how many
//! qubits it takes; edges record which operand slot each wire occupies. Nothing here
//! enumerates "one-qubit gate", "two-qubit gate", "three-qubit gate" as separate cases,
//! which is what made the reference Java implementation hard to extend past `ccz`.

pub mod angle;
pub mod dag;
pub mod embed;
pub mod error;
pub mod gate;
pub mod gateset;
pub mod qasm;

pub use angle::{angles_equivalent, normalize_angle, AngleExpr, ANGLE_EPS, FULL_PERIOD};
pub use dag::{
    intern, is_global_phase, Dag, Edge, GateName, GateOp, Node, NodeIndex, QasmHeader, QubitId,
};
pub use error::{CircuitError, Result};
pub use gate::{is_t_gate, Builtin, CompositeStep, GateDef, GateRegistry, GateSemantics};
pub use gateset::{GateSet, GateSetLibrary};
pub use qasm::{parse, to_qasm};
