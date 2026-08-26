//! Error types for circuit construction and parsing.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, CircuitError>;

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("unknown gate `{0}`")]
    UnknownGate(String),

    #[error("gate `{gate}` takes {expected} parameter(s), got {got}")]
    ParamCount {
        gate: String,
        expected: usize,
        got: usize,
    },

    #[error("gate `{gate}` takes {expected} qubit(s), got {got}")]
    Arity {
        gate: String,
        expected: usize,
        got: usize,
    },

    #[error("gate `{gate}` repeats qubit `{qubit}`")]
    DuplicateQubit { gate: String, qubit: String },

    #[error("could not resolve parameter `{expr}` of composite gate `{gate}`")]
    UnresolvedParam { gate: String, expr: String },

    #[error("parse error at line {line}, column {col}: {msg}")]
    Parse {
        line: usize,
        col: usize,
        msg: String,
    },

    #[error("circuit has {qubits} qubits, too many to build a {} x {} unitary (limit {limit})",
            1u64 << qubits, 1u64 << qubits)]
    TooManyQubits { qubits: usize, limit: usize },

    #[error("{0}")]
    Other(String),
}
