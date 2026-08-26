//! Command-line interface for GUOQ.

pub mod args;
pub mod log;
pub mod run;
pub mod synth;

pub use args::Cli;
pub use log::{IterationInfo, Logger};
pub use run::{optimize, BackendChoice, RunOutcome};
