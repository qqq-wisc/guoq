//! Optimizing a large circuit one window at a time.
//!
//! # Why a flat search stops working
//!
//! Making a move cheap is necessary for a large circuit and not sufficient. A single
//! Metropolis chain over a million gates improves one site per step, and the signal it
//! steers by — the change in total cost — is a part in a million. The walk is not slow,
//! it is *undirected*: almost every move is accepted, because almost every move is
//! nearly free relative to the whole, and the search stops being a search.
//!
//! Windowing fixes the ratio rather than the speed. A few hundred gates is the size the
//! existing search was tuned for and where its acceptance rule discriminates, so a large
//! circuit is optimized as a sequence of small ones.
//!
//! # Why a window can be cut out and put back
//!
//! A window here is a run of consecutive positions in a topological order, which is
//! always convex: every edge runs from a lower position to a higher one, so a path that
//! leaves the run above it can only continue upward and can never re-enter. A convex set
//! of gates is a contiguous factor of the circuit, so replacing it with anything
//! equivalent leaves the whole circuit equivalent — the same argument that licenses
//! resynthesis on a partition, and the reason `Dag::is_convex` is load-bearing throughout
//! this codebase.
//!
//! Rewrite rules are context-free unitary identities, so optimizing a window in isolation
//! is sound regardless of what surrounds it.
//!
//! # Seams
//!
//! A gate near a window boundary only ever sees half its neighbourhood. Rounds therefore
//! shift the boundaries by half a window, so a gate that was at a seam sits in the
//! interior next time.

use std::time::Instant;

use rayon::prelude::*;

use qcircuit::{Dag, GateRegistry, NodeIndex};
use qrules::SymbolicLimits;

use crate::candidate::Candidate;
use crate::config::SearchConfig;
use crate::cost::CostModel;
use crate::resynth::Resynthesizer;
use crate::search::Search;
use crate::transform::TransformationSet;

/// How to slice a circuit into windows.
#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    /// Gates per window. The search's own tuning is what sets a sensible value: a few
    /// hundred is the scale at which its acceptance rule still discriminates.
    pub gates: usize,
    /// How many passes to make, each with boundaries shifted by half a window.
    pub rounds: usize,
    /// Stop starting new windows once this instant has passed.
    pub deadline: Option<Instant>,
    /// Optimize the windows of a round concurrently.
    ///
    /// Sound because the windows of a round are disjoint sets of gates and each is
    /// optimized as a *detached* sub-circuit; nothing is shared but read-only rules and
    /// the registry. Reproducible because a window's RNG stream is derived from its
    /// position rather than from how many windows preceded it, so the result does not
    /// depend on the order threads happen to finish in, and because splicing stays
    /// sequential and in position order.
    ///
    /// That reproducibility is over the same *set of completed windows*. Under a
    /// wall-clock deadline the two do not agree, and should not: the parallel run
    /// finishes more windows before the clock stops it, which is the entire point.
    pub parallel: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        // Swept, not guessed; see docs/SCALING.md for the table. 192 sits in the band
        // where a window is small enough for the acceptance rule to discriminate and
        // large enough that the per-window setup is not most of the work, and one round
        // spends the budget on optimizing rather than on re-visiting with shifted seams.
        Self {
            gates: 192,
            rounds: 1,
            deadline: None,
            parallel: true,
        }
    }
}

/// One window's outcome: the gates it covered, what the search made of them, and whether
/// that is an improvement worth splicing back.
type WindowOutcome = (Vec<NodeIndex>, std::sync::Arc<Dag>, bool);

/// What a windowed run did.
#[derive(Debug, Clone, Default)]
pub struct WindowReport {
    /// Windows optimized.
    pub windows: usize,
    /// Windows whose optimized form was kept.
    pub improved: usize,
    /// Gate count before and after.
    pub before: usize,
    pub after: usize,
}

/// A mid-run snapshot of a windowed optimization, for [`WindowProgressFn`].
///
/// From the outside a windowed run used to be silent between start and finish, which
/// made a sixty-second run and a hung process look identical; per-window improvement
/// callbacks are the wrong replacement — a thousand windows each improving dozens of
/// times, concurrently and out of order, is noise. This is the coarse signal instead.
#[derive(Debug, Clone, Copy)]
pub struct WindowProgress {
    /// The current round, 1-based, of [`WindowConfig::rounds`].
    pub round: usize,
    pub rounds: usize,
    /// Windows spliced so far this round, of `total` in the round.
    pub done: usize,
    pub total: usize,
    /// Windows whose optimized form was kept, across all rounds so far.
    pub improved: usize,
    /// Gate count when the run started, and now.
    pub gates_start: usize,
    pub gates_now: usize,
}

/// Called with the circuit's current state as a windowed run advances.
///
/// The circuit reference is always *consistent* — results are spliced in as they arrive,
/// in position order — so the callback may serialize it as a valid best-so-far. Calls
/// are rate-limited to one per couple of seconds plus one at each round's end, and are
/// made under the splice lock, so the callback should stay cheap-ish; writing a file is
/// fine, optimizing dinner plans is not.
pub type WindowProgressFn<'a> = &'a (dyn Fn(&Dag, &WindowProgress) + Sync);

/// Least time between two progress callbacks within a round.
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// Optimize `dag` window by window, in place.
///
/// Each window is optimized by the ordinary search, under `config`'s budget *per window*
/// rather than for the whole circuit, and its result is spliced back only if it is
/// cheaper by `model`'s reckoning. A window that does not improve is left exactly as it
/// was, so a windowed run can never make a circuit worse.
#[allow(clippy::too_many_arguments)]
pub fn optimize_windowed(
    dag: &mut Dag,
    windows: &WindowConfig,
    config: &SearchConfig,
    registry: &GateRegistry,
    transformations: &TransformationSet,
    resynth: &(dyn Resynthesizer + Sync),
    limits: SymbolicLimits,
    model: &CostModel,
    progress: WindowProgressFn<'_>,
) -> WindowReport {
    let mut report = WindowReport {
        before: dag.gate_count(),
        ..Default::default()
    };
    // Once for the run, not once per window. Building it scans every rule's pattern --
    // about 12 ms over the Nam set -- and a million-gate circuit is nearly two thousand
    // windows a round, so rebuilding it per window cost several times the whole budget.
    let eligibility =
        std::sync::Arc::new(crate::eligibility::EligibilityIndex::build(transformations));
    if windows.gates == 0 {
        report.after = report.before;
        return report;
    }

    for round in 0..windows.rounds.max(1) {
        if out_of_time(windows) {
            break;
        }
        // Half a window per round, so a gate at a seam is in the interior next time.
        let offset = (round * windows.gates / 2) % windows.gates.max(1);
        let order = dag.topological_gates();
        if order.len() <= 1 {
            break;
        }

        let mut start = 0usize;
        let mut bounds: Vec<(usize, usize)> = Vec::new();
        if offset > 0 && offset < order.len() {
            bounds.push((0, offset.min(order.len())));
            start = offset;
        }
        while start < order.len() {
            let end = (start + windows.gates).min(order.len());
            bounds.push((start, end));
            start = end;
        }

        // Cut every window of the round out first. The slabs are disjoint and each
        // sub-circuit is detached, so optimizing them is embarrassingly parallel; only
        // the splicing has to touch the shared circuit.
        let jobs: Vec<(usize, usize, Vec<NodeIndex>, Dag)> = bounds
            .into_iter()
            // A one-gate window has nothing to rearrange.
            .filter(|&(lo, hi)| hi - lo >= 2)
            .map(|(lo, hi)| {
                let slab: Vec<NodeIndex> = order[lo..hi].to_vec();
                let sub = dag.subcircuit(&slab);
                (lo, hi, slab, sub)
            })
            .collect();

        let threads = if windows.parallel {
            rayon::current_num_threads()
        } else {
            1
        };
        let total_jobs = jobs.len();
        let done_jobs = std::sync::atomic::AtomicUsize::new(0);

        let optimize = |(lo, _hi, slab, sub): (usize, usize, Vec<NodeIndex>, Dag)| {
            if out_of_time(windows) {
                return None;
            }
            let started = done_jobs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let before_key = model.key(&sub);
            // The stream is keyed by the window's position, not by how many windows came
            // before it, so a window's result is the same whether rounds run in sequence
            // or in parallel.
            let window_config = SearchConfig {
                seed: window_seed(config.seed, round, lo),
                time_limit: per_window_budget(
                    windows,
                    config,
                    round,
                    total_jobs.saturating_sub(started),
                    threads,
                ),
                ..config.clone()
            };
            let mut search = Search::with_eligibility(
                &window_config,
                registry,
                transformations,
                resynth,
                SymbolicLimits {
                    deadline: windows.deadline,
                    ..limits
                },
                std::sync::Arc::clone(&eligibility),
            );
            let mut noop = |_: &Candidate, _: f64| {};
            let result = search.run(sub, &mut noop);
            Some((slab, result.best.dag, result.best.key < before_key))
        };

        // Splice sequentially and in position order, so the circuit that comes out does
        // not depend on which window finished first — but *as results arrive*, through a
        // reorder buffer, rather than after the whole round. The splice sequence is
        // byte-identical to the collect-then-splice it replaced; what streaming buys is a
        // circuit that is consistent mid-round, so progress can be reported against it
        // and a best-so-far can be written to disk while the run is still going. Workers
        // only read their pre-cut sub-circuits, never the shared one, which is what makes
        // mutating it mid-round sound.
        struct Splicer<'a> {
            dag: &'a mut Dag,
            /// Results that arrived ahead of their turn, by job index.
            pending: rustc_hash::FxHashMap<usize, Option<WindowOutcome>>,
            /// The job index the next splice is waiting for.
            next: usize,
            windows: usize,
            improved: usize,
            last_report: Instant,
        }
        let splicer = std::sync::Mutex::new(Splicer {
            dag: &mut *dag,
            pending: rustc_hash::FxHashMap::default(),
            next: 0,
            windows: 0,
            improved: 0,
            last_report: Instant::now(),
        });
        let prior = (report.windows, report.improved);
        let rounds = windows.rounds.max(1);

        let finish = |seq: usize, outcome: Option<WindowOutcome>| {
            let mut s = splicer.lock().expect("splice lock");
            s.pending.insert(seq, outcome);
            loop {
                let turn = s.next;
                let Some(outcome) = s.pending.remove(&turn) else {
                    break;
                };
                s.next += 1;
                if let Some((slab, best, improved)) = outcome {
                    s.windows += 1;
                    if improved && s.dag.replace_span(&slab, &best) {
                        s.improved += 1;
                    }
                }
            }
            if s.last_report.elapsed() >= PROGRESS_INTERVAL {
                s.last_report = Instant::now();
                let p = WindowProgress {
                    round: round + 1,
                    rounds,
                    done: s.next,
                    total: total_jobs,
                    improved: prior.1 + s.improved,
                    gates_start: report.before,
                    gates_now: s.dag.gate_count(),
                };
                progress(s.dag, &p);
            }
        };

        if windows.parallel {
            jobs.into_par_iter()
                .enumerate()
                .for_each(|(seq, job)| finish(seq, optimize(job)));
        } else {
            jobs.into_iter()
                .enumerate()
                .for_each(|(seq, job)| finish(seq, optimize(job)));
        }

        let s = splicer.into_inner().expect("splice lock");
        report.windows = prior.0 + s.windows;
        report.improved = prior.1 + s.improved;
        let done = s.next;
        drop(s);

        // One unconditional report per round, so short runs are not silent and the last
        // line a round leaves behind is its true final state.
        progress(
            dag,
            &WindowProgress {
                round: round + 1,
                rounds,
                done,
                total: total_jobs,
                improved: report.improved,
                gates_start: report.before,
                gates_now: dag.gate_count(),
            },
        );
    }

    report.after = dag.gate_count();
    report
}

fn out_of_time(windows: &WindowConfig) -> bool {
    windows.deadline.is_some_and(|d| Instant::now() >= d)
}

/// A window's share of the run's remaining time.
///
/// Dividing matters more than it looks. Handing each window whatever is left, capped at
/// some constant, means the first few windows spend the entire budget and the rest never
/// run at all: on a hundred-thousand-gate circuit at a twenty-second budget that was 36
/// windows out of 784, so seven eighths of the circuit was never examined. The budget is
/// therefore split evenly across the rounds, and a round's slice across the *batches* of
/// windows it will run — batches rather than windows, because `threads` of them run at
/// once.
///
/// A window still finishes early when its search converges, and the next call sees the
/// time it gave back, so the split self-corrects rather than being fixed up front.
fn per_window_budget(
    windows: &WindowConfig,
    config: &SearchConfig,
    round: usize,
    remaining_in_round: usize,
    threads: usize,
) -> Option<std::time::Duration> {
    let Some(deadline) = windows.deadline else {
        return config.time_limit;
    };
    let rounds_left = windows.rounds.max(1).saturating_sub(round).max(1);
    let left = deadline.saturating_duration_since(Instant::now());
    let this_round = left / rounds_left as u32;
    let batches = remaining_in_round.div_ceil(threads.max(1)).max(1);
    // A floor, so that a circuit with more windows than milliseconds still gives each one
    // enough time to attempt something rather than timing out on arrival.
    Some((this_round / batches as u32).max(std::time::Duration::from_millis(2)))
}

/// A per-window seed derived from the run's seed and the window's identity.
///
/// Keyed by position rather than by visit order so that the result of a window does not
/// depend on how many windows preceded it, which is what makes running them concurrently
/// reproducible.
fn window_seed(seed: u64, round: usize, lo: usize) -> u64 {
    let mut z = seed
        .wrapping_add((round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add((lo as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::Objective;
    use crate::resynth::NullResynthesizer;
    use qcircuit::qasm;
    use qrules::Rule;

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

    fn config() -> SearchConfig {
        SearchConfig {
            objective: Objective::Total,
            queue_size: 1,
            max_iters: Some(400),
            ..SearchConfig::default()
        }
    }

    /// Windows find cancellations spread across a circuit far larger than any one window.
    ///
    /// Six rounds are needed to reach the optimum, not two: a pair straddling a seam is
    /// invisible until a later round's shifted boundary puts it in one window's interior.
    /// That is the cost of windowing, and it is why `rounds` exists.
    #[test]
    fn windows_optimize_a_circuit_larger_than_one_window() {
        let mut src = String::new();
        for i in 0..60 {
            src.push_str(&format!("h q{}; h q{}; t q{};\n", i % 4, i % 4, i % 4));
        }
        let mut dag = qasm::parse(&src).unwrap();
        let registry = GateRegistry::with_builtins();
        let set = cancel_rules();
        let null = NullResynthesizer;
        let cfg = config();
        let model = CostModel::new(Objective::Total, 1, registry.clone());

        let before = dag.gate_count();
        let report = optimize_windowed(
            &mut dag,
            &WindowConfig {
                gates: 16,
                rounds: 6,
                deadline: None,
                parallel: false,
            },
            &cfg,
            &registry,
            &set,
            &null,
            SymbolicLimits::default(),
            &model,
            &|_, _| {},
        );

        assert!(
            report.windows > 5,
            "expected many windows, got {}",
            report.windows
        );
        assert!(
            dag.gate_count() < before,
            "windowing did not improve {before} gates"
        );
        // Only the `t` gates should survive: every `h` pair cancels.
        assert_eq!(dag.gate_count(), 60, "expected the 60 t gates to remain");
        assert!(dag.is_acyclic());
    }

    /// The progress callback sees consistent, monotonically advancing state, ends each
    /// round on its true final count — and observing a run must not change its result.
    #[test]
    fn progress_reports_without_changing_the_result() {
        let mut src = String::new();
        for i in 0..40 {
            src.push_str(&format!("h q{}; h q{}; x q{};\n", i % 3, i % 3, i % 3));
        }
        let registry = GateRegistry::with_builtins();
        let null = NullResynthesizer;
        let model = CostModel::new(Objective::Total, 1, registry.clone());
        let windows = WindowConfig {
            gates: 12,
            rounds: 2,
            deadline: None,
            parallel: true,
        };

        let mut silent = qasm::parse(&src).unwrap();
        optimize_windowed(
            &mut silent,
            &windows,
            &config(),
            &registry,
            &cancel_rules(),
            &null,
            SymbolicLimits::default(),
            &model,
            &|_, _| {},
        );

        let seen: std::sync::Mutex<Vec<WindowProgress>> = std::sync::Mutex::new(Vec::new());
        let mut watched = qasm::parse(&src).unwrap();
        let report = optimize_windowed(
            &mut watched,
            &windows,
            &config(),
            &registry,
            &cancel_rules(),
            &null,
            SymbolicLimits::default(),
            &model,
            &|dag, p| {
                assert!(dag.is_acyclic(), "progress saw an inconsistent circuit");
                assert_eq!(dag.gate_count(), p.gates_now);
                seen.lock().unwrap().push(*p);
            },
        );

        assert_eq!(
            qasm::to_qasm(&silent),
            qasm::to_qasm(&watched),
            "watching the run changed its result"
        );
        let seen = seen.into_inner().unwrap();
        assert!(
            seen.len() >= windows.rounds,
            "one report per round at least"
        );
        for pair in seen.windows(2) {
            assert!(
                (pair[0].round, pair[0].done) <= (pair[1].round, pair[1].done),
                "progress went backwards"
            );
        }
        let last = seen.last().unwrap();
        assert_eq!(last.done, last.total, "the last report is the round's end");
        assert_eq!(last.improved, report.improved);
        assert_eq!(last.gates_now, report.after);
    }

    /// A windowed run never makes a circuit worse: a window whose search finds nothing is
    /// left exactly as it was.
    #[test]
    fn windows_never_worsen_a_circuit() {
        let mut src = String::new();
        for i in 0..40 {
            src.push_str(&format!("t q{};\n", i % 3));
        }
        let mut dag = qasm::parse(&src).unwrap();
        let before = dag.structural_hash();
        let registry = GateRegistry::with_builtins();
        let model = CostModel::new(Objective::Total, 1, registry.clone());
        let null = NullResynthesizer;

        let report = optimize_windowed(
            &mut dag,
            &WindowConfig {
                gates: 8,
                rounds: 2,
                deadline: None,
                parallel: false,
            },
            &config(),
            &registry,
            &cancel_rules(),
            &null,
            SymbolicLimits::default(),
            &model,
            &|_, _| {},
        );

        assert!(report.windows > 0);
        assert_eq!(report.improved, 0);
        assert_eq!(dag.structural_hash(), before);
    }

    /// The same seed gives the same result, and a window's stream does not depend on how
    /// many windows preceded it.
    #[test]
    fn windowing_is_deterministic() {
        let mut src = String::new();
        for i in 0..40 {
            src.push_str(&format!(
                "h q{}; h q{}; x q{}; x q{};\n",
                i % 3,
                i % 3,
                i % 3,
                i % 3
            ));
        }
        let registry = GateRegistry::with_builtins();
        let model = CostModel::new(Objective::Total, 1, registry.clone());
        let null = NullResynthesizer;
        let windows = WindowConfig {
            gates: 12,
            rounds: 2,
            deadline: None,
            parallel: false,
        };

        let run = || {
            let mut dag = qasm::parse(&src).unwrap();
            optimize_windowed(
                &mut dag,
                &windows,
                &config(),
                &registry,
                &cancel_rules(),
                &null,
                SymbolicLimits::default(),
                &model,
                &|_, _| {},
            );
            qasm::to_qasm(&dag)
        };
        assert_eq!(run(), run());

        // A window's seed depends on where it is, not on what came before it.
        assert_eq!(window_seed(7, 1, 512), window_seed(7, 1, 512));
        assert_ne!(window_seed(7, 1, 512), window_seed(7, 1, 1024));
        assert_ne!(window_seed(7, 1, 512), window_seed(7, 2, 512));
    }

    /// Running the windows of a round concurrently gives the same circuit as running them
    /// one at a time.
    ///
    /// This is the property that makes the parallelism safe to turn on by default, and it
    /// is not automatic: it holds because a window's RNG stream comes from its position
    /// rather than from when it is visited, and because splicing stays sequential and in
    /// position order however the threads interleave.
    #[test]
    fn parallel_windows_match_serial_windows() {
        let mut src = String::new();
        for i in 0..120 {
            src.push_str(&format!(
                "h q{}; h q{}; t q{}; x q{}; x q{};\n",
                i % 5,
                i % 5,
                i % 5,
                (i + 1) % 5,
                (i + 1) % 5
            ));
        }
        let registry = GateRegistry::with_builtins();
        let model = CostModel::new(Objective::Total, 1, registry.clone());
        let null = NullResynthesizer;

        let run = |parallel: bool| {
            let mut dag = qasm::parse(&src).unwrap();
            let report = optimize_windowed(
                &mut dag,
                &WindowConfig {
                    gates: 24,
                    rounds: 3,
                    deadline: None,
                    parallel,
                },
                &config(),
                &registry,
                &cancel_rules(),
                &null,
                SymbolicLimits::default(),
                &model,
                &|_, _| {},
            );
            (qasm::to_qasm(&dag), report.windows, report.improved)
        };

        let (serial, sw, si) = run(false);
        let (parallel, pw, pi) = run(true);
        assert!(sw > 8, "expected many windows, got {sw}");
        assert_eq!((sw, si), (pw, pi));
        assert_eq!(serial, parallel, "parallel windowing changed the circuit");
    }
}
