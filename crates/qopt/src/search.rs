//! The optimization search.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashSet;

use qcircuit::{Dag, GateRegistry, NodeIndex};
use qrules::{apply_rule, symbolic, ApplyOptions, SymbolicLimits};

use crate::candidate::{BestFirst, Candidate, WorstFirst};
use crate::config::{SearchConfig, Strategy};
use crate::cost::CostModel;
use crate::eligibility::EligibilityIndex;
use crate::resynth::Resynthesizer;
use crate::sampling::{sample_distinct, sample_uniform_distinct};
use crate::transform::{Transformation, TransformationSet};

/// How a worse candidate's acceptance probability is computed.
///
/// The two forms behave very differently at the same `--temperature`, so this is a
/// user-visible choice rather than an internal detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AcceptanceRule {
    /// `exp(-beta * candidate_cost / current_cost)`, the reference's formulation.
    ///
    /// `--temperature` is an inverse temperature here, which is why the reference's help
    /// text calls it "beta for mcmc". At its default of 10 the exponent is about `-10`
    /// whatever the candidate costs, so a worsening move is accepted with probability
    /// around `4e-5`: the search is near-greedy.
    #[default]
    CostRatio,
    /// `exp(-delta / temperature)`, textbook Metropolis.
    ///
    /// Scales with *how much* worse a candidate is, which the ratio form barely does. The
    /// same `--temperature` denotes a far more exploratory search under it, so retune when
    /// switching.
    CostDelta,
}

impl AcceptanceRule {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ratio" | "cost-ratio" => Some(AcceptanceRule::CostRatio),
            "delta" | "cost-delta" => Some(AcceptanceRule::CostDelta),
            _ => None,
        }
    }

    pub fn as_flag(self) -> &'static str {
        match self {
            AcceptanceRule::CostRatio => "ratio",
            AcceptanceRule::CostDelta => "delta",
        }
    }
}

/// Acceptance probability for a candidate, given the current cost.
///
/// A candidate no worse than the current one is always accepted.
///
/// # Why the rule is a choice
///
/// The ratio form is the reference's formulation, and its shipped defaults were tuned
/// against it; the delta form is the textbook rule. They are not interchangeable: the
/// delta scales with *how much* worse a candidate is, where the ratio barely does, and
/// switching changes what `--temperature 10` means by four orders of magnitude. So the
/// ratio form stays the default, computed in floating point throughout, and the delta
/// form is opt-in.
pub fn acceptance_probability(
    rule: AcceptanceRule,
    current_cost: f64,
    candidate_cost: f64,
    temperature: f64,
) -> f64 {
    if candidate_cost <= current_cost {
        return 1.0;
    }
    if temperature <= 0.0 {
        return 0.0;
    }
    let exponent = match rule {
        AcceptanceRule::CostRatio => {
            if current_cost == 0.0 {
                return 0.0;
            }
            -temperature * (candidate_cost / current_cost)
        }
        AcceptanceRule::CostDelta => -(candidate_cost - current_cost) / temperature,
    };
    exponent.exp().clamp(0.0, 1.0)
}

/// Textbook Metropolis acceptance, kept for callers that want it directly.
pub fn metropolis_acceptance(delta: f64, temperature: f64) -> f64 {
    if delta <= 0.0 {
        return 1.0;
    }
    if temperature <= 0.0 {
        return 0.0;
    }
    (-delta / temperature).exp().clamp(0.0, 1.0)
}

/// What a search produced.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best: Candidate,
    pub iterations: u64,
    pub elapsed: Duration,
    /// Seconds from the start until the best circuit was found.
    pub time_to_best: f64,
    /// How many times each transformation fired.
    pub applications: Vec<(String, usize)>,
}

/// Called whenever the search improves on its best circuit.
pub type ProgressFn<'a> = &'a mut dyn FnMut(&Candidate, f64);

/// Runs the optimization.
pub struct Search<'a> {
    config: &'a SearchConfig,
    model: CostModel,
    registry: &'a GateRegistry,
    transformations: &'a TransformationSet,
    resynth: &'a dyn Resynthesizer,
    limits: SymbolicLimits,
    rng: ChaCha8Rng,
    /// Which transformations can fire on a given circuit, so the draw skips the rest.
    eligibility: Arc<EligibilityIndex>,
    /// Mutable during the run; the reference kept these in `Params` statics.
    temperature: f64,
    prune_temperature: f64,
    applications: rustc_hash::FxHashMap<String, usize>,
    sequence: u64,
    /// When the wall-clock budget runs out, if it is time-bounded. Rule application on a
    /// large circuit can spend seconds inside a single exhaustive pass, so the deadline
    /// is handed down into it rather than only consulted between iterations.
    deadline: Option<Instant>,
}

impl<'a> Search<'a> {
    pub fn new(
        config: &'a SearchConfig,
        registry: &'a GateRegistry,
        transformations: &'a TransformationSet,
        resynth: &'a dyn Resynthesizer,
        limits: SymbolicLimits,
    ) -> Self {
        Self::with_eligibility(
            config,
            registry,
            transformations,
            resynth,
            limits,
            Arc::new(EligibilityIndex::build(transformations)),
        )
    }

    /// As [`new`], reusing an eligibility index built elsewhere.
    ///
    /// Building one scans every rule's pattern, which is once-per-run work that the
    /// windowed search would otherwise redo for every window.
    ///
    /// [`new`]: Search::new
    pub fn with_eligibility(
        config: &'a SearchConfig,
        registry: &'a GateRegistry,
        transformations: &'a TransformationSet,
        resynth: &'a dyn Resynthesizer,
        limits: SymbolicLimits,
        eligibility: Arc<EligibilityIndex>,
    ) -> Self {
        Self {
            model: CostModel::new(
                config.objective,
                config.fidelity_breakeven,
                registry.clone(),
            ),
            registry,
            transformations,
            resynth,
            limits,
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            eligibility,
            temperature: config.temperature,
            prune_temperature: config.prune_temperature,
            applications: rustc_hash::FxHashMap::default(),
            config,
            sequence: 0,
            deadline: None,
        }
    }

    pub fn model(&self) -> &CostModel {
        &self.model
    }

    fn next_sequence(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    /// Optimize `dag`, reporting improvements through `progress`.
    pub fn run(&mut self, dag: Dag, progress: ProgressFn<'_>) -> SearchResult {
        let start = Instant::now();
        self.deadline = self.config.time_limit.map(|l| start + l);
        let seq = self.next_sequence();
        // The whole-circuit pass runs exactly once, here. Thereafter each rewrite
        // normalizes the gates it introduced and nothing else, which is complete only
        // because the circuit it starts from is already free of identities: a gate no
        // rewrite touched cannot have become one. Skipping this would leave any identity
        // present in the input there for the whole run.
        let initial = Candidate::new(self.normalize(dag), &self.model, seq);
        let mut best = initial.clone();
        let mut time_to_best = 0.0;

        let iterations = match self.config.strategy {
            Strategy::Beam | Strategy::BeamMcmc => {
                self.run_queue(initial, &mut best, &mut time_to_best, start, progress)
            }
            Strategy::Mcmc | Strategy::SimAnn => {
                self.run_chain(initial, &mut best, &mut time_to_best, start, progress)
            }
            Strategy::Reduce => {
                let (reduced, applied) =
                    self.reduce_to_fixpoint(initial, &mut best, &mut time_to_best, start, progress);
                applied + self.run_queue(reduced, &mut best, &mut time_to_best, start, progress)
            }
        };

        let mut applications: Vec<(String, usize)> = self
            .applications
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        applications.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        SearchResult {
            best,
            iterations,
            elapsed: start.elapsed(),
            time_to_best,
            applications,
        }
    }

    fn out_of_time(&self, start: Instant, iterations: u64) -> bool {
        if let Some(limit) = self.config.time_limit {
            if start.elapsed() >= limit {
                return true;
            }
        }
        if let Some(max) = self.config.max_iters {
            if iterations >= max {
                return true;
            }
        }
        false
    }

    /// The deterministic phase of [`Strategy::Reduce`]: the reducing-rule fixpoint.
    ///
    /// Every step is a strict improvement under the model, so it terminates by a
    /// decreasing-cost argument; the deadline only bounds how long it may take to get
    /// there. See `crate::reduce`. Returns the reduced candidate and how many rewrites
    /// were applied.
    fn reduce_to_fixpoint(
        &mut self,
        initial: Candidate,
        best: &mut Candidate,
        time_to_best: &mut f64,
        start: Instant,
        progress: ProgressFn<'_>,
    ) -> (Candidate, u64) {
        let mut dag = (*initial.dag).clone();
        let mut index = (*initial.index).clone();
        // Memoized on the shared eligibility index: the windowed search calls this once
        // per window, and classifying the rule set costs more than a tight budget's
        // whole per-window slice.
        let reducing = self.eligibility.reducing_plain(
            self.transformations,
            &self.model,
            self.config.objective,
            self.config.fidelity_breakeven,
        );

        let total = crate::reduce::reducing_rule_fixpoint(
            &mut dag,
            &mut index,
            self.transformations,
            &self.eligibility,
            &reducing,
            self.registry,
            self.deadline,
            &mut self.applications,
        );

        let seq = self.next_sequence();
        let key = self.model.key(&dag);
        let candidate = Candidate {
            dag: Arc::new(dag),
            index: Arc::new(index),
            key,
            history: initial.history,
            accumulated_error: initial.accumulated_error,
            sequence: seq,
        };
        if candidate.key < best.key {
            *best = candidate.clone();
            *time_to_best = start.elapsed().as_secs_f64();
            progress(best, *time_to_best);
        }
        (candidate, total)
    }

    /// Queue-based search: `BEAM` and `BEAM_MCMC`.
    fn run_queue(
        &mut self,
        initial: Candidate,
        best: &mut Candidate,
        time_to_best: &mut f64,
        start: Instant,
        progress: ProgressFn<'_>,
    ) -> u64 {
        let mut queue: Vec<Candidate> = vec![initial.clone()];
        let mut seen: FxHashSet<u128> = FxHashSet::default();
        seen.insert(initial.dag.structural_hash());
        let mut iterations = 0u64;

        while !queue.is_empty() && !self.out_of_time(start, iterations) {
            iterations += 1;

            let mut current = self.dequeue(&mut queue);
            if BestFirst::compare(&current, best) == std::cmp::Ordering::Less {
                // A detached copy, not a shared one. `best` is the only alias that
                // outlives an iteration, and while it aliases, the circuit below cannot
                // be rewritten in place. Copying when the search improves is far cheaper
                // than copying every iteration.
                *best = current.detached();
                *time_to_best = start.elapsed().as_secs_f64();
                progress(best, *time_to_best);
            }

            let picks = self.select_transformations(best, iterations, start, &current.dag);
            // Reduce continues as BeamMcmc after its deterministic phase, so it accepts
            // the same way.
            let metropolis = matches!(self.config.strategy, Strategy::BeamMcmc | Strategy::Reduce);
            let mut admitted = 0usize;
            // When only one transformation is sampled -- the shipped default -- an
            // admitted rewrite can simply stay in the circuit. With several, the later
            // ones must still see the original, so an admitted one is banked as a copy.
            let more_picks = picks.len() > 1;
            // `true` once the circuit in hand *is* an admitted candidate rather than the
            // one the iteration started with.
            let mut current_holds_admitted = false;
            let registry = self.registry;
            let normalize = self.config.normalize_identities;

            for index in picks {
                // The per-iteration check at the top of the loop cannot see time spent
                // inside this one, and a pick on a large circuit is not cheap.
                if self.deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
                let Some(t) = self.transformations.get(index) else {
                    continue;
                };
                // A plain rule is rewritten *into* the current circuit rather than into
                // a copy of it, and rolled back if the candidate is not admitted. Copying
                // the circuit per attempt made a rejected move -- the common outcome --
                // cost O(circuit) for nothing, and rebuilding the match index afterwards
                // cost it a second time. Both are now O(gates the rewrite touched).
                if let Transformation::Plain(rule) = t {
                    let opts = ApplyOptions {
                        apply_once: self.config.apply_once,
                        shuffle: true,
                        deadline: self.deadline,
                    };
                    let id = rule.id.clone();
                    let undo = {
                        let (dag, idx) = current.parts_mut();
                        qrules::apply_rule_in_place(dag, idx, rule, &opts, &mut self.rng).map(
                            |mut u| {
                                if normalize {
                                    // Part of the same edit: a rollback that did not
                                    // restore what normalization removed would leave the
                                    // circuit short of gates.
                                    let ins = u.inserted();
                                    let gone = qrules::drop_identities_undoable(
                                        dag, &ins, registry, &mut u,
                                    );
                                    idx.apply_rewrite(dag, &gone, &[]);
                                }
                                u
                            },
                        )
                    };
                    let Some(undo) = undo else { continue };

                    // Beam search still rejects a revisit; a Metropolis walk still must
                    // not (see the note below). The hash is computed only when it will be
                    // consulted, which is what keeps this O(circuit) step off the default
                    // path entirely.
                    if !metropolis && !seen.insert(current.dag.structural_hash()) {
                        let (dag, idx) = current.parts_mut();
                        qrules::rollback_with_index(dag, idx, undo);
                        continue;
                    }

                    let before_key = current.key;
                    let seq = self.next_sequence();
                    current.rescore_in_place(&self.model, &id, 0.0, seq);
                    *self.applications.entry(id.clone()).or_insert(0) += 1;

                    let admit = if metropolis {
                        let p = acceptance_probability(
                            self.config.acceptance,
                            before_key.primary() as f64,
                            current.key.primary() as f64,
                            self.temperature,
                        );
                        p >= 1.0 || self.rng.gen::<f64>() <= p
                    } else {
                        current.key <= best.key
                    };

                    if admit {
                        admitted += 1;
                        // With more than one transformation sampled per iteration the
                        // others are still tried against the *original* circuit, so an
                        // admitted candidate is banked as a copy and the rewrite undone.
                        // With the default of one, nothing is copied at all.
                        if more_picks {
                            queue.push(current.detached());
                            let (dag, idx) = current.parts_mut();
                            qrules::rollback_with_index(dag, idx, undo);
                            current.key = before_key;
                        } else {
                            current_holds_admitted = true;
                        }
                    } else {
                        let (dag, idx) = current.parts_mut();
                        qrules::rollback_with_index(dag, idx, undo);
                        current.key = before_key;
                    }
                    continue;
                }

                let applied = self.apply(&current, t);
                let Some((next_dag, error)) = applied else {
                    continue;
                };
                // Only beam search rejects a revisit. A Metropolis walk *needs* to be
                // able to re-enter a state it has already visited -- that is how it
                // leaves a basin: downhill to a neighbour, back uphill through the state
                // it came from, out the other side. Rejecting revisits makes the walk
                // self-avoiding, and once every neighbour of the current state has been
                // visited nothing can ever be admitted again: the current candidate is
                // pushed back unchanged and the rest of the budget is burnt re-deriving
                // states that are all discarded. The reference declares a `seen` set in
                // `optimizeBeamMCMC` and then never reads it, which is what makes its
                // walk work; applying it here cost real quality. On
                // benchmarks/nam_rz/4mod5-v0_20.qasm this port used to reach 23 gates at
                // 0.19s and then not improve across the remaining 574k iterations; with
                // revisits allowed the same budget reaches 21.
                if !metropolis && !seen.insert(next_dag.structural_hash()) {
                    continue;
                }
                let seq = self.next_sequence();
                let candidate = current.derive(next_dag, &self.model, t.id(), error, seq);
                *self.applications.entry(t.id().to_string()).or_insert(0) += 1;

                let admit = if metropolis {
                    let p = acceptance_probability(
                        self.config.acceptance,
                        current.key.primary() as f64,
                        candidate.key.primary() as f64,
                        self.temperature,
                    );
                    p >= 1.0 || self.rng.gen::<f64>() <= p
                } else {
                    // Plain beam search keeps anything no worse than the best so far.
                    candidate.key <= best.key
                };
                if admit {
                    queue.push(candidate);
                    admitted += 1;
                }
            }

            // The current candidate goes back if nothing better came of it. Without this
            // the queue empties the first time a sampled transformation fails to fire,
            // and the search stops after a single iteration -- which is exactly what
            // happens with the reference's default of one sampled transformation per
            // iteration. The reference avoids it by re-adding the current circuit on
            // every Metropolis rejection.
            if current_holds_admitted || admitted == 0 {
                queue.push(current);
            }

            self.prune_queue(&mut queue);
            if self.config.cooling_rate > 0.0 {
                self.temperature *= 1.0 - self.config.cooling_rate;
                self.prune_temperature *= 1.0 - self.config.cooling_rate;
            }
        }
        iterations
    }

    /// Single-chain search: `MCMC` and `SIM_ANN`.
    fn run_chain(
        &mut self,
        initial: Candidate,
        best: &mut Candidate,
        time_to_best: &mut f64,
        start: Instant,
        progress: ProgressFn<'_>,
    ) -> u64 {
        let mut current = initial;
        let mut iterations = 0u64;

        while !self.out_of_time(start, iterations) {
            iterations += 1;
            if self.transformations.is_empty() {
                break;
            }
            let index = self.rng.gen_range(0..self.transformations.len());
            let Some(t) = self.transformations.get(index) else {
                continue;
            };
            let Some((next_dag, error)) = self.apply(&current, t) else {
                continue;
            };
            let seq = self.next_sequence();
            let candidate = current.derive(next_dag, &self.model, t.id(), error, seq);
            *self.applications.entry(t.id().to_string()).or_insert(0) += 1;

            let p = acceptance_probability(
                self.config.acceptance,
                current.key.primary() as f64,
                candidate.key.primary() as f64,
                self.temperature,
            );
            if p >= 1.0 || self.rng.gen::<f64>() <= p {
                current = candidate;
            }

            if BestFirst::compare(&current, best) == std::cmp::Ordering::Less {
                *best = current.clone();
                *time_to_best = start.elapsed().as_secs_f64();
                progress(best, *time_to_best);
            }

            // Simulated annealing cools; MCMC holds its temperature.
            if self.config.strategy == Strategy::SimAnn && self.config.cooling_rate > 0.0 {
                self.temperature *= 1.0 - self.config.cooling_rate;
                if self.temperature <= f64::EPSILON {
                    break;
                }
            }
        }
        iterations
    }

    /// Take the next candidate to expand.
    ///
    /// At temperature zero this is the best one; above zero it is drawn under a softmax
    /// over negated costs, so worse candidates are occasionally explored.
    fn dequeue(&mut self, queue: &mut Vec<Candidate>) -> Candidate {
        if queue.len() == 1 || self.temperature <= 0.0 {
            let mut best_i = 0;
            for i in 1..queue.len() {
                if BestFirst::compare(&queue[i], &queue[best_i]) == std::cmp::Ordering::Less {
                    best_i = i;
                }
            }
            return queue.swap_remove(best_i);
        }
        let weights: Vec<f64> = queue.iter().map(|c| -(c.key.primary() as f64)).collect();
        let probs = crate::sampling::softmax(&weights, self.temperature);
        let idx = crate::sampling::sample_index(&probs, &mut self.rng).unwrap_or(0);
        queue.swap_remove(idx)
    }

    /// Keep the queue within its configured size, discarding the worst.
    fn prune_queue(&self, queue: &mut Vec<Candidate>) {
        if queue.len() <= self.config.queue_size {
            return;
        }
        let mut heap: BinaryHeap<WorstFirst> = queue.drain(..).map(WorstFirst).collect();
        while heap.len() > self.config.queue_size {
            heap.pop();
        }
        queue.extend(heap.into_iter().map(|w| w.0));
    }

    /// Which transformations to try this iteration.
    fn select_transformations(
        &mut self,
        best: &Candidate,
        iterations: u64,
        start: Instant,
        current: &Dag,
    ) -> Vec<usize> {
        let n = self.transformations.len();
        if n == 0 {
            return Vec::new();
        }

        let pruning_active = {
            let iters_ok = self
                .config
                .iters_before_prune
                .is_none_or(|k| iterations as usize >= k);
            let secs_ok = self
                .config
                .secs_before_prune
                .is_none_or(|s| start.elapsed().as_secs() >= s);
            self.config.cooling_rate > 0.0 && iters_ok && secs_ok
        };

        // Plain beam search expands a candidate against every transformation; only the
        // Metropolis variants sample. The reference draws the same distinction --
        // `optimizeBeam` sets `rulesToUse = rules` and ignores
        // `NUM_TRANSFORMATIONS_SAMPLE`, which `optimizeBeamMCMC` honours.
        let sample = match self.config.strategy {
            Strategy::Beam => None,
            _ => self.config.num_transformations_sample,
        };
        if !pruning_active {
            return match sample {
                None => (0..n).collect(),
                // Draw only from transformations that could fire on this circuit. A rule
                // whose pattern names a gate the circuit does not contain cannot match,
                // and including it in the draw wastes the iteration; see
                // `crate::eligibility`.
                Some(k) => {
                    let mask = self.eligibility.circuit_mask(current);
                    let eligible = self.eligibility.eligible(mask);
                    if eligible.is_empty() {
                        return Vec::new();
                    }
                    sample_uniform_distinct(eligible.len(), k, &mut self.rng)
                        .into_iter()
                        .map(|i| eligible[i])
                        .collect()
                }
            };
        }

        // Score by how often each transformation has fired, and how often it appears in
        // the best circuit's history. `TransformationSet::get` owns the index arithmetic
        // this scoring needs.
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                let Some(t) = self.transformations.get(i) else {
                    return 0.0;
                };
                let id = t.id();
                let total = *self.applications.get(id).unwrap_or(&0);
                let in_best = best.count_applications(id);
                (total + 2 * in_best) as f64
            })
            .collect();
        let k = sample.unwrap_or(n).min(n);
        sample_distinct(&weights, k, self.prune_temperature, &mut self.rng)
    }

    /// Apply one transformation, returning the new circuit and the error it added.
    fn apply(&mut self, current: &Candidate, t: Transformation<'_>) -> Option<(Dag, f64)> {
        let (dag, error, touched) = self.apply_raw(current, t)?;
        // A transformation that reports which gates it introduced gets the cheap
        // normalization; one that does not -- symbolic rewriting, resynthesis -- falls
        // back to the whole-circuit scan.
        let dag = match touched {
            Some(t) => self.normalize_local(dag, &t),
            None => self.normalize(dag),
        };
        Some((dag, error))
    }

    /// Drop gates a rewrite turned into the identity.
    ///
    /// See [`SearchConfig::normalize_identities`]: rotation merging produces `rz(0)` all
    /// the time, and leaving it in the circuit inflates every gate-count objective.
    /// Normalize a whole circuit. Used for the initial circuit and for transformations
    /// that do not report what they changed.
    fn normalize(&self, mut dag: Dag) -> Dag {
        if self.config.normalize_identities {
            dag.drop_identity_gates(self.registry);
        }
        dag
    }

    /// Normalize only the gates a rewrite introduced.
    ///
    /// A rewrite leaves every gate outside its replaced spans untouched, so no other gate
    /// can have *become* an identity. Scanning the whole circuit for one is O(circuit)
    /// work per accepted move, which on a large circuit costs more than the move itself.
    fn normalize_local(&self, mut dag: Dag, touched: &[NodeIndex]) -> Dag {
        if self.config.normalize_identities {
            dag.drop_identity_gates_among(touched, self.registry);
        }
        dag
    }

    #[allow(clippy::type_complexity)]
    fn apply_raw(
        &mut self,
        current: &Candidate,
        t: Transformation<'_>,
    ) -> Option<(Dag, f64, Option<Vec<NodeIndex>>)> {
        match t {
            Transformation::Plain(rule) => {
                let opts = ApplyOptions {
                    apply_once: self.config.apply_once,
                    shuffle: true,
                    deadline: self.deadline,
                };
                apply_rule(&current.dag, rule, &opts, &mut self.rng)
                    .map(|r| (r.dag, 0.0, Some(r.inserted)))
            }
            Transformation::Symbolic(rule) => symbolic::apply_symbolic(
                &current.dag,
                rule,
                // Symbolic matching is the most expensive transformation there is -- a
                // nested seed loop with a reachability walk per pair -- so it gets the
                // clock, not just the limits.
                &SymbolicLimits {
                    deadline: self.deadline,
                    ..self.limits
                },
                self.registry,
                &mut self.rng,
            )
            .map(|d| (d, 0.0, None)),
            Transformation::Resynth => {
                // The remaining budget is the only gate: the reference also stopped
                // after MAX_RESYNTH_ALLOWED calls, but a call count is a proxy for the
                // spending the budget already measures directly, and it forbade calls
                // the ledger could still afford.
                let budget = self.config.epsilon - current.accumulated_error;
                if budget <= 0.0 {
                    return None;
                }
                let outcome = self.resynth.resynthesize(
                    &current.dag,
                    budget,
                    self.registry,
                    &mut self.rng,
                )?;
                Some((outcome.dag, outcome.error, None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::Objective;
    use crate::resynth::{FixedResynthesizer, NullResynthesizer};
    use qcircuit::qasm;
    use qrules::Rule;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn cancel_rules() -> TransformationSet {
        TransformationSet::new(
            vec![
                Rule::new("h q0; h q0;", "").unwrap(),
                Rule::new("x q0; x q0;", "").unwrap(),
                Rule::new("cx q0, q1; cx q0, q1;", "").unwrap(),
            ],
            Vec::new(),
            0,
        )
    }

    fn run(src: &str, cfg: SearchConfig) -> SearchResult {
        let registry = reg();
        let set = cancel_rules();
        let null = NullResynthesizer;
        let mut search = Search::new(&cfg, &registry, &set, &null, SymbolicLimits::default());
        let mut noop = |_: &Candidate, _: f64| {};
        search.run(qasm::parse(src).unwrap(), &mut noop)
    }

    fn config(strategy: Strategy) -> SearchConfig {
        SearchConfig {
            strategy,
            objective: Objective::Total,
            queue_size: 16,
            temperature: 1.0,
            num_transformations_sample: None,
            max_iters: Some(200),
            ..SearchConfig::default()
        }
    }

    /// Acceptance decays smoothly with how much worse the candidate is.
    #[test]
    fn metropolis_uses_real_division() {
        // A candidate that is better is always taken.
        assert_eq!(metropolis_acceptance(-1.0, 1.0), 1.0);
        assert_eq!(metropolis_acceptance(0.0, 1.0), 1.0);

        // A worse candidate's probability decays smoothly with how much worse it is.
        let p1 = metropolis_acceptance(1.0, 10.0);
        let p2 = metropolis_acceptance(2.0, 10.0);
        let p3 = metropolis_acceptance(20.0, 10.0);
        assert!(p1 > p2 && p2 > p3);
        assert!(p1 < 1.0 && p3 > 0.0);
        assert!((p1 - (-0.1f64).exp()).abs() < 1e-12);

        // The reference's expression, reproduced: `candidateSize / currentSize` on ints.
        let (candidate_size, current_size) = (11i64, 10i64);
        let truncated = candidate_size / current_size; // == 1
        assert_eq!(truncated, 1);
        let much_worse = 100i64 / current_size; // == 10
        assert_ne!(truncated, much_worse);
        // ...and for any candidate smaller than current it is 0, making the probability
        // identical no matter how much better the candidate is.
        assert_eq!(1i64 / 10, 0);
        assert_eq!(9i64 / 10, 0);
    }

    #[test]
    fn metropolis_edge_cases() {
        assert_eq!(metropolis_acceptance(5.0, 0.0), 0.0);
        assert_eq!(metropolis_acceptance(-5.0, 0.0), 1.0);
        assert!((0.0..=1.0).contains(&metropolis_acceptance(1e9, 1.0)));
        assert!(metropolis_acceptance(1e9, 1.0) >= 0.0);
    }

    #[test]
    fn every_strategy_reduces_a_reducible_circuit() {
        let src = "h a; h a; x b; x b; cx a, b; cx a, b; h c;";
        for strategy in Strategy::ALL {
            let r = run(src, config(strategy));
            assert!(
                r.best.dag.gate_count() < 7,
                "{strategy} made no progress: {} gates",
                r.best.dag.gate_count()
            );
            assert!(r.best.dag.is_acyclic(), "{strategy} produced a cycle");
        }
    }

    /// The default objective and strategy run out of the box.
    #[test]
    fn the_default_configuration_runs() {
        let cfg = SearchConfig {
            queue_size: 8,
            max_iters: Some(50),
            ..SearchConfig::default()
        };
        assert_eq!(cfg.strategy, Strategy::BeamMcmc);
        assert_eq!(cfg.objective, Objective::Fidelity);
        let r = run("h a; h a; cx a, b; cx a, b; x c; x c;", cfg);
        assert!(
            r.iterations > 1,
            "the search must get past its first insert"
        );
        assert!(r.best.dag.gate_count() <= 6);
    }

    #[test]
    fn search_is_deterministic_for_a_seed() {
        let src = "h a; h a; x b; x b; cx a, b; cx a, b; h c; h c;";
        for strategy in Strategy::ALL {
            let mut cfg = config(strategy);
            cfg.seed = 12345;
            let a = run(src, cfg.clone());
            let b = run(src, cfg);
            assert_eq!(
                a.best.dag.gate_count(),
                b.best.dag.gate_count(),
                "{strategy} is not deterministic"
            );
            assert_eq!(a.iterations, b.iterations, "{strategy}");
        }
    }

    #[test]
    fn the_result_never_exceeds_the_input_cost() {
        let src = "h a; cx a, b; t b; h a; x c;";
        for strategy in Strategy::ALL {
            let cfg = config(strategy);
            let registry = reg();
            let set = cancel_rules();
            let null = NullResynthesizer;
            let model = CostModel::new(cfg.objective, cfg.fidelity_breakeven, registry.clone());
            let input = qasm::parse(src).unwrap();
            let input_key = model.key(&input);
            let mut search = Search::new(&cfg, &registry, &set, &null, SymbolicLimits::default());
            let mut noop = |_: &Candidate, _: f64| {};
            let r = search.run(input, &mut noop);
            assert!(
                r.best.key <= input_key,
                "{strategy} returned a worse circuit than it was given"
            );
        }
    }

    #[test]
    fn progress_fires_on_improvement_only() {
        let cfg = config(Strategy::Beam);
        let registry = reg();
        let set = cancel_rules();
        let null = NullResynthesizer;
        let mut search = Search::new(&cfg, &registry, &set, &null, SymbolicLimits::default());
        let mut seen: Vec<usize> = Vec::new();
        {
            let mut cb = |c: &Candidate, _t: f64| seen.push(c.dag.gate_count());
            search.run(qasm::parse("h a; h a; x b; x b;").unwrap(), &mut cb);
        }
        assert!(!seen.is_empty(), "no progress was reported");
        for w in seen.windows(2) {
            assert!(
                w[1] < w[0],
                "progress reported without improvement: {seen:?}"
            );
        }
    }

    #[test]
    fn an_empty_transformation_set_terminates() {
        let cfg = config(Strategy::Mcmc);
        let registry = reg();
        let set = TransformationSet::default();
        let null = NullResynthesizer;
        let mut search = Search::new(&cfg, &registry, &set, &null, SymbolicLimits::default());
        let mut noop = |_: &Candidate, _: f64| {};
        let r = search.run(qasm::parse("h a;").unwrap(), &mut noop);
        assert_eq!(r.best.dag.gate_count(), 1);
    }

    /// A Metropolis walk has to be able to re-enter a state it has already visited.
    ///
    /// The transformation set here is a commutation and its inverse, so the reachable
    /// state space is exactly two circuits: `rz(0.3) a; cx a, b;` and its transpose.
    /// Neither is cheaper than the other, so every step is an accepted lateral move and
    /// the walk should keep stepping for the whole budget.
    ///
    /// Rejecting revisits — which is right for a beam and was wrongly applied to both
    /// strategies here — caps this at two firings: the second state is new, the third
    /// step lands back on the initial circuit, and from then on nothing can ever be
    /// admitted again. `BEAM` is expected to stop, and is asserted to, so the two
    /// behaviours stay distinguishable rather than one silently becoming the other.
    #[test]
    fn metropolis_may_revisit_a_state() {
        fn firings(strategy: Strategy) -> usize {
            let cfg = SearchConfig {
                queue_size: 1,
                max_iters: Some(200),
                ..config(strategy)
            };
            let registry = reg();
            let set = TransformationSet::new(
                vec![
                    Rule::new("rz(theta1) q0; cx q0, q1;", "cx q0, q1; rz(theta1) q0;").unwrap(),
                    Rule::new("cx q0, q1; rz(theta1) q0;", "rz(theta1) q0; cx q0, q1;").unwrap(),
                ],
                Vec::new(),
                0,
            );
            let null = NullResynthesizer;
            let mut search = Search::new(&cfg, &registry, &set, &null, SymbolicLimits::default());
            let mut noop = |_: &Candidate, _: f64| {};
            let r = search.run(qasm::parse("rz(0.3) a; cx a, b;").unwrap(), &mut noop);
            r.applications.iter().map(|(_, n)| n).sum()
        }

        let mcmc = firings(Strategy::BeamMcmc);
        assert!(
            mcmc > 100,
            "Metropolis walk froze after {mcmc} firings in 200 iterations"
        );
        assert!(
            firings(Strategy::Beam) <= 2,
            "beam search should still reject revisits"
        );
    }

    /// Resynthesis is one transformation among others here, with no separate lifecycle to
    /// get wrong: its results land in the circuit like any rewrite.
    #[test]
    fn resynthesis_results_are_consumed() {
        let cfg = SearchConfig {
            // Resynthesis every iteration rather than one slot in the rule set, so what
            // is being tested is the consumption and not the sampler's luck.
            max_iters: Some(20),
            ..config(Strategy::BeamMcmc)
        };
        let registry = reg();
        let set = TransformationSet::new(Vec::new(), Vec::new(), 1);
        let backend = FixedResynthesizer {
            result: qasm::parse("h a;").unwrap(),
            error: 0.0,
        };
        let mut search = Search::new(&cfg, &registry, &set, &backend, SymbolicLimits::default());
        let mut noop = |_: &Candidate, _: f64| {};
        let r = search.run(qasm::parse("h a; x a; h a;").unwrap(), &mut noop);

        assert_eq!(
            r.best.dag.gate_count(),
            1,
            "the resynthesis result never reached the circuit"
        );
        assert!(
            r.applications
                .iter()
                .any(|(id, n)| id == "resynth" && *n > 0),
            "resynthesis never fired: {:?}",
            r.applications
        );
    }

    /// The budget is the only thing that limits resynthesis. The reference also stopped
    /// after `MAX_RESYNTH_ALLOWED` calls; a lineage here keeps resynthesizing past any
    /// call count while its spending fits, and stops when — and only when — the next
    /// charge no longer does.
    #[test]
    fn resynthesis_is_limited_by_budget_not_call_count() {
        let registry = reg();
        let set = TransformationSet::new(Vec::new(), Vec::new(), 1);
        let input = qasm::parse("h a; x a; h a;").unwrap();
        let mut noop = |_: &Candidate, _: f64| {};

        // Free calls: far more of them than the reference's default cap of 100.
        let cfg = SearchConfig {
            max_iters: Some(150),
            queue_size: 1,
            ..config(Strategy::BeamMcmc)
        };
        let backend = FixedResynthesizer {
            result: qasm::parse("h a;").unwrap(),
            error: 0.0,
        };
        let mut search = Search::new(&cfg, &registry, &set, &backend, SymbolicLimits::default());
        let r = search.run(input.clone(), &mut noop);
        let fired: usize = r
            .applications
            .iter()
            .filter(|(id, _)| id == "resynth")
            .map(|(_, n)| *n)
            .sum();
        assert!(
            fired > 100,
            "only {fired} calls; a call cap is still in force"
        );

        // Costed calls: each charges 1e-9 against an epsilon of 3.5e-9. This double
        // honours the trait's contract — the budget is a hard ceiling — so it records
        // what it was offered and declines what no longer fits, the way the verified
        // resynthesizer does.
        struct Metered {
            result: Dag,
            error: f64,
            offered: std::sync::Mutex<Vec<f64>>,
        }
        impl crate::resynth::Resynthesizer for Metered {
            fn resynthesize(
                &self,
                _dag: &Dag,
                budget: f64,
                _registry: &GateRegistry,
                _rng: &mut dyn crate::resynth::RngCore,
            ) -> Option<crate::resynth::ResynthOutcome> {
                self.offered.lock().unwrap().push(budget);
                (self.error <= budget).then(|| crate::resynth::ResynthOutcome {
                    dag: self.result.clone(),
                    error: self.error,
                })
            }
            fn name(&self) -> &str {
                "resynth"
            }
        }
        let cfg = SearchConfig {
            epsilon: 3.5e-9,
            max_iters: Some(50),
            queue_size: 1,
            ..config(Strategy::BeamMcmc)
        };
        let backend = Metered {
            result: qasm::parse("h a;").unwrap(),
            error: 1e-9,
            offered: std::sync::Mutex::new(Vec::new()),
        };
        let mut search = Search::new(&cfg, &registry, &set, &backend, SymbolicLimits::default());
        let r = search.run(input, &mut noop);
        let offered = backend.offered.lock().unwrap();
        assert!(
            offered.iter().all(|&b| b > 0.0 && b <= 3.5e-9),
            "a call was offered a nonpositive or over-total budget: {offered:?}"
        );
        assert!(
            offered.iter().any(|&b| b < 1e-9),
            "the search never ran a lineage's budget down to a decline, so the \
             budget-not-count stop was not exercised: {offered:?}"
        );
        assert!(
            r.best.accumulated_error <= 3.5e-9,
            "the lineage overspent: {:.3e}",
            r.best.accumulated_error
        );
    }

    /// Normalization is incremental, and that is only *complete* if the circuit starts
    /// normalized.
    ///
    /// After a rewrite only the gates it introduced can have become identities, so the
    /// search checks only those -- which is what keeps a move from costing O(circuit).
    /// The induction needs a base case: an identity already present in the *input* is
    /// touched by no rewrite, so nothing would ever look at it. This pins both halves:
    /// an input identity is gone, and so is one a rewrite produces.
    #[test]
    fn no_identity_gate_survives_the_search() {
        use std::f64::consts::PI;
        let cfg = SearchConfig {
            max_iters: Some(50),
            ..config(Strategy::BeamMcmc)
        };
        let registry = reg();
        // `rz(a); rz(b)` fuses to `rz(a+b)`, so the search can *make* an identity as well
        // as inherit one: the two rz gates on `b` sum to 4*pi.
        let set = TransformationSet::new(
            vec![Rule::new("rz(theta1) q0; rz(theta2) q0;", "rz((theta1+theta2)) q0;").unwrap()],
            Vec::new(),
            0,
        );
        let null = NullResynthesizer;
        let mut search = Search::new(&cfg, &registry, &set, &null, SymbolicLimits::default());
        let mut noop = |_: &Candidate, _: f64| {};
        let src = format!("rz(0) a; h a; rz({0}) b; rz({0}) b; x c;", 2.0 * PI);
        let r = search.run(qasm::parse(&src).unwrap(), &mut noop);

        for i in r.best.dag.gate_indices() {
            let op = r.best.dag.gate(i);
            assert!(
                !qcircuit::is_global_phase(op, &registry),
                "identity gate `{}` survived the search",
                op.gate
            );
        }
        // The real gates are untouched.
        assert_eq!(r.best.dag.gate_count(), 2, "expected just `h a` and `x c`");
    }

    /// A search over a symbolic-only transformation set runs like any other.
    #[test]
    fn symbolic_only_set_runs() {
        let c = "[{[false, false]=[false, false], [true, false]=[true, false], \
                 [false, true]=[false, true], [true, true]=[true, true]}]";
        let rule = qrules::SymbolicRule::parse_legacy_with_builtins(&format!(
            "h q0; rz(theta1) q1; symb q; | h q0; symb q; rz(theta1) q1; | {c}"
        ))
        .unwrap();
        let cfg = config(Strategy::Beam);
        let registry = reg();
        let set = TransformationSet::new(Vec::new(), vec![rule], 0);
        let null = NullResynthesizer;
        let mut search = Search::new(&cfg, &registry, &set, &null, SymbolicLimits::default());
        let mut noop = |_: &Candidate, _: f64| {};
        let r = search.run(
            qasm::parse("h a; rz(0.3) b; cx a, b; x b;").unwrap(),
            &mut noop,
        );
        assert!(r.iterations > 0);
    }

    #[test]
    fn iteration_and_time_limits_are_respected() {
        let mut cfg = config(Strategy::Mcmc);
        cfg.max_iters = Some(37);
        let r = run("h a; cx a, b;", cfg);
        assert!(r.iterations <= 37, "ran {} iterations", r.iterations);

        let mut cfg = config(Strategy::Mcmc);
        cfg.max_iters = None;
        cfg.time_limit = Some(Duration::from_millis(60));
        let r = run("h a; cx a, b;", cfg);
        assert!(r.elapsed < Duration::from_secs(5));
    }

    #[test]
    fn queue_size_is_respected() {
        let mut cfg = config(Strategy::Beam);
        cfg.queue_size = 3;
        cfg.max_iters = Some(40);
        // A large reducible circuit so the queue would otherwise grow.
        let src = "h a; h a; h b; h b; h c; h c; x a; x a; x b; x b;";
        let r = run(src, cfg);
        assert!(r.best.dag.gate_count() < 10);
    }

    #[test]
    fn applications_are_counted_and_ranked() {
        let cfg = config(Strategy::Beam);
        let r = run("h a; h a; h b; h b; x c; x c;", cfg);
        assert!(!r.applications.is_empty());
        for w in r.applications.windows(2) {
            assert!(w[0].1 >= w[1].1, "applications are not sorted by count");
        }
    }

    #[test]
    fn history_records_the_path_to_the_best() {
        let cfg = config(Strategy::Beam);
        let r = run("h a; h a; x b; x b;", cfg);
        assert_eq!(r.best.dag.gate_count(), 0);
        assert!(!r.best.history.is_empty());
        assert_eq!(r.best.history.last().unwrap().gate_count, 0);
        assert_eq!(r.best.accumulated_error, 0.0);
    }
}
