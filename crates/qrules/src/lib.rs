//! Rewrite rules: representation, matching, and application.

pub mod constraint;
pub mod error;
pub mod legacy;
pub mod matcher;
pub mod pattern;
pub mod rewrite;
pub mod rule;
pub mod symbolic;

pub use constraint::{format_constraints, parse_constraints, BasisPermutation};
pub use error::{Result, RuleError};
pub use matcher::{Match, MatchContext, MatchIndex};
pub use pattern::{ParamShape, Pattern};
pub use rewrite::{
    apply_match_undoable, apply_rule, apply_rule_in_place, apply_rule_with_context,
    drop_identities_undoable, find_disjoint_matches, matches_anywhere, rollback,
    rollback_with_index, ApplyOptions, Rewritten, Undo,
};
pub use rule::{Rule, RuleKind};
pub use symbolic::{apply_symbolic, find_symbolic, SymbolicLimits, SymbolicMatch, SymbolicRule};
