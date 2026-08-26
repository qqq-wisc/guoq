//! Errors from rule parsing and application.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, RuleError>;

#[derive(Debug, Error)]
pub enum RuleError {
    #[error("circuit error: {0}")]
    Parse(#[from] qcircuit::CircuitError),

    #[error("pattern is disconnected, so a replacement has no single insertion point: `{0}`")]
    Disconnected(String),

    #[error("malformed rule line: {0}")]
    Malformed(String),

    #[error("replacement uses qubit `{qubit}` that the pattern does not bind: `{rule}`")]
    UnboundQubit { qubit: String, rule: String },

    #[error("replacement uses angle `{angle}` that the pattern does not bind: `{rule}`")]
    UnboundAngle { angle: String, rule: String },

    #[error("io error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}
