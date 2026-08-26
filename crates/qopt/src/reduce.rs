//! Deterministic reduction: the harvest before the search.
//!
//! A freshly synthesized circuit — a Trotterized simulation, a gridsynth expansion — is
//! full of rewrites that are pure wins under the objective: adjacent cancellations,
//! `t t -> s`, pairs that annihilate. A stochastic search rediscovers each of them one
//! lucky sample at a time, which on a large circuit means most of the budget is spent
//! reaching a floor a deterministic pass reaches in seconds. [`Strategy::Reduce`] runs
//! that pass first; this module is the pass: every plain rule whose replacement is
//! strictly cheaper than its pattern under the active cost model is applied
//! exhaustively, in file order, until a full sweep fires nothing. Every application is
//! a strict improvement, so termination is a decreasing cost argument, not a hope.
//!
//! [`Strategy::Reduce`]: crate::Strategy::Reduce

use std::time::Instant;

use rand::SeedableRng;
use rustc_hash::FxHashMap;

use qcircuit::{Dag, GateRegistry};
use qrules::{ApplyOptions, MatchIndex};

use crate::eligibility::EligibilityIndex;
use crate::transform::{Transformation, TransformationSet};

/// Apply every strictly cost-reducing plain rule exhaustively until a sweep fires
/// nothing. Returns total applications; records per-rule counts into `applications`.
///
/// `reducing` says which transformation indices are strictly reducing plain rules; it
/// comes from [`EligibilityIndex::reducing_plain`], computed once per run, because
/// classifying 63k rules costs more than a whole window's slice of a tight budget.
///
/// Every application strictly decreases the circuit's key, which is a well-founded
/// order, so the fixpoint terminates without further argument.
#[allow(clippy::too_many_arguments)]
pub fn reducing_rule_fixpoint(
    dag: &mut Dag,
    index: &mut MatchIndex,
    transformations: &TransformationSet,
    eligibility: &EligibilityIndex,
    reducing: &[bool],
    registry: &GateRegistry,
    deadline: Option<Instant>,
    applications: &mut FxHashMap<String, usize>,
) -> u64 {
    let opts = ApplyOptions {
        apply_once: false,
        // Sites in index order, so the pass is deterministic.
        shuffle: false,
        deadline,
    };
    // Unused while `shuffle` is false, but the signature wants one.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let mut total = 0u64;
    loop {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        let mask = eligibility.circuit_mask(dag);
        let eligible = eligibility.eligible(mask);
        let mut fired = 0u64;
        for &i in eligible.iter() {
            if !reducing[i] {
                continue;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            let Some(Transformation::Plain(rule)) = transformations.get(i) else {
                continue;
            };
            let Some(mut undo) = qrules::apply_rule_in_place(dag, index, rule, &opts, &mut rng)
            else {
                continue;
            };
            // Same edit as the search's plain-rule path: replacements can contain
            // identities, and they are part of this rewrite, not a later cleanup.
            let inserted = undo.inserted();
            let gone = qrules::drop_identities_undoable(dag, &inserted, registry, &mut undo);
            index.apply_rewrite(dag, &gone, &[]);
            let n = undo.applications() as u64;
            fired += n;
            *applications.entry(rule.id.clone()).or_insert(0) += n as usize;
        }
        total += fired;
        if fired == 0 {
            break;
        }
    }
    total
}
