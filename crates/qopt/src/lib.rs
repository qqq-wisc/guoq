//! Circuit optimization: cost models, search strategies, and resynthesis orchestration.

pub mod candidate;
pub mod config;
pub mod cost;
pub mod eligibility;
pub mod reduce;
pub mod resynth;
pub mod sampling;
pub mod search;
pub mod transform;
pub mod windowed;

pub use candidate::{Candidate, Step};
pub use config::{SearchConfig, Strategy};
pub use cost::{CostKey, CostModel, GateCounts, Objective};
pub use resynth::{
    NullResynthesizer, ResynthOutcome, ResynthStats, Resynthesizer, VerifiedBackend,
};
pub use search::{
    acceptance_probability, metropolis_acceptance, AcceptanceRule, Search, SearchResult,
};
pub use transform::{Transformation, TransformationSet};
pub use windowed::{
    optimize_windowed, WindowConfig, WindowProgress, WindowProgressFn, WindowReport,
};
