//! Search configuration.
//!
//! The reference kept this in `Params`: roughly thirty `public static` fields, several
//! mutated *during* the search (`Params.TEMPERATURE *= 1 - Params.COOLING_RATE`) while a
//! resynthesis thread read others. Here the configuration is immutable and shared, and
//! anything that changes as the search runs lives in the search's own state.

use std::path::PathBuf;
use std::time::Duration;

use crate::cost::Objective;

/// Which search to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Priority queue over candidates, expanding the best first.
    Beam,
    /// Single chain with Metropolis acceptance.
    Mcmc,
    /// Single chain with a cooling schedule.
    SimAnn,
    /// Priority queue whose members are accepted by Metropolis.
    BeamMcmc,
    /// Deterministic reduction to a fixpoint, then [`Strategy::BeamMcmc`] on what is
    /// left of the budget.
    ///
    /// A freshly synthesized circuit is full of rewrites that are pure wins — adjacent
    /// cancellations, `t t -> s`, pairs that annihilate — and a stochastic search
    /// rediscovers them one lucky sample at a time. This strategy harvests them first:
    /// exhaustive application of every strictly cost-reducing rule (see
    /// `crate::reduce`), repeated until nothing changes. The stochastic search then
    /// starts from that floor instead of spending its budget getting there.
    Reduce,
}

impl Strategy {
    pub const ALL: [Strategy; 5] = [
        Strategy::Beam,
        Strategy::Mcmc,
        Strategy::SimAnn,
        Strategy::BeamMcmc,
        Strategy::Reduce,
    ];

    pub fn as_flag(self) -> &'static str {
        match self {
            Strategy::Beam => "BEAM",
            Strategy::Mcmc => "MCMC",
            Strategy::SimAnn => "SIM_ANN",
            Strategy::BeamMcmc => "BEAM_MCMC",
            Strategy::Reduce => "REDUCE",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Strategy::ALL
            .into_iter()
            .find(|x| x.as_flag().eq_ignore_ascii_case(s))
    }
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_flag())
    }
}

/// Immutable search configuration.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub strategy: Strategy,
    pub objective: Objective,
    /// Cost of a two-qubit gate in one-qubit-gate units.
    pub fidelity_breakeven: i64,
    /// Largest number of candidates the queue keeps.
    pub queue_size: usize,
    /// Metropolis temperature, or the softmax temperature for queue selection.
    ///
    /// Its meaning depends on [`SearchConfig::acceptance`]: under the reference's
    /// cost-ratio rule it is an inverse temperature, so larger means *greedier*.
    pub temperature: f64,
    /// How a worse candidate's acceptance probability is computed.
    pub acceptance: crate::search::AcceptanceRule,
    /// Cooling rate for simulated annealing; zero means no cooling.
    pub cooling_rate: f64,
    /// Temperature for pruning the transformation set; zero prunes greedily.
    pub prune_temperature: f64,
    /// Transformations to sample per iteration; `None` uses all of them.
    pub num_transformations_sample: Option<usize>,
    /// Iterations to run before pruning starts.
    pub iters_before_prune: Option<usize>,
    /// Seconds to run before pruning starts.
    pub secs_before_prune: Option<u64>,
    /// Apply a rule at only one site per iteration rather than at every disjoint site.
    pub apply_once: bool,
    /// Remove gates a rewrite has turned into the identity.
    ///
    /// Rotation merging routinely produces one: `rz(theta1); rz(theta2)` collapses to
    /// `rz(theta1 + theta2)`, and when the angles cancel that is `rz(0)`. It contributes
    /// nothing but still counts against every gate-count objective.
    ///
    /// Removing identities belongs here, after a rewrite — not in the parser, where a
    /// dropped gate would corrupt rule patterns as they are read.
    pub normalize_identities: bool,
    /// Total error budget for resynthesis.
    ///
    /// This is the only thing that limits resynthesis: there is no call-count cap. The
    /// reference stopped after `MAX_RESYNTH_ALLOWED` calls; here that number only sizes
    /// the per-call accuracy hint (see `VerifiedResynthesizer::epsilon_hint`).
    pub epsilon: f64,
    /// Deterministic seed.
    pub seed: u64,
    /// Stop after this long. The reference relied entirely on an external `timeout`.
    pub time_limit: Option<Duration>,
    /// Stop after this many iterations.
    pub max_iters: Option<u64>,
    /// Where to write the running best solution.
    pub output_dir: PathBuf,
    /// Tag inserted into output file names.
    pub job_info: String,
    /// 0 quiet, 1 progress, 2 progress and config, 3 also rules applied.
    pub verbosity: u8,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            strategy: Strategy::BeamMcmc,
            objective: Objective::Fidelity,
            fidelity_breakeven: 1,
            queue_size: 1,
            temperature: 10.0,
            acceptance: crate::search::AcceptanceRule::CostRatio,
            cooling_rate: 0.0,
            prune_temperature: 0.0,
            num_transformations_sample: Some(1),
            iters_before_prune: None,
            secs_before_prune: None,
            apply_once: false,
            normalize_identities: true,
            epsilon: 1e-8,
            seed: 0,
            time_limit: None,
            max_iters: None,
            output_dir: PathBuf::from("."),
            job_info: String::new(),
            verbosity: 0,
        }
    }
}

impl SearchConfig {
    /// Path the running best solution is written to.
    pub fn latest_path(&self, circuit_name: &str) -> PathBuf {
        self.output_dir
            .join(format!("latest_sol_{}_{}", self.job_info, circuit_name))
    }

    /// Path the final solution is written to.
    pub fn final_path(&self, circuit_name: &str) -> PathBuf {
        self.output_dir
            .join(format!("optimized_{}_{}", self.job_info, circuit_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_flags_round_trip() {
        for s in Strategy::ALL {
            assert_eq!(Strategy::parse(s.as_flag()), Some(s));
            assert_eq!(Strategy::parse(&s.as_flag().to_lowercase()), Some(s));
        }
        assert_eq!(Strategy::parse("nope"), None);
    }

    /// The output naming the evaluation harness scrapes.
    #[test]
    fn output_paths_match_the_reference_convention() {
        let cfg = SearchConfig {
            output_dir: PathBuf::from("/out"),
            job_info: "cluster_7".into(),
            ..SearchConfig::default()
        };
        assert_eq!(
            cfg.latest_path("tof_3.qasm"),
            PathBuf::from("/out/latest_sol_cluster_7_tof_3.qasm")
        );
        assert_eq!(
            cfg.final_path("tof_3.qasm"),
            PathBuf::from("/out/optimized_cluster_7_tof_3.qasm")
        );
    }

    #[test]
    fn defaults_match_the_reference() {
        let d = SearchConfig::default();
        assert_eq!(d.strategy, Strategy::BeamMcmc);
        assert_eq!(d.objective, Objective::Fidelity);
        assert_eq!(d.queue_size, 1);
        assert_eq!(d.temperature, 10.0);
        assert_eq!(d.epsilon, 1e-8);
    }
}
