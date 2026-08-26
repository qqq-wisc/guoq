//! Progress logging.
//!
//! One JSON object per line, with the keys `evaluation/run_guoq.py` scrapes. Keeping the
//! shape identical is what lets the existing harness read this binary's output.

use qcircuit::{Dag, GateRegistry};
use qopt::{Candidate, CostModel, GateCounts, SearchResult};
use serde_json::{json, Map, Value};

/// One iteration's worth of state, for [`Logger::iteration`].
pub struct IterationInfo<'a> {
    pub queue_size: usize,
    pub seen: usize,
    pub best: &'a Candidate,
    pub current: &'a Candidate,
    pub seconds_elapsed: f64,
    pub time_to_best: f64,
    pub num_rules: usize,
    pub num_symb_rules: usize,
}

/// Emits the reference's log lines.
pub struct Logger {
    pub verbosity: u8,
    pub filename: String,
}

impl Logger {
    pub fn new(verbosity: u8, filename: impl Into<String>) -> Self {
        Self {
            verbosity,
            filename: filename.into(),
        }
    }

    fn emit(&self, value: Value) {
        println!("{value}");
    }

    /// Counts of the input circuit, before any optimization.
    pub fn initial(&self, dag: &Dag, registry: &GateRegistry) {
        if self.verbosity < 1 {
            return;
        }
        let c = GateCounts::of(dag, registry);
        self.emit(json!({
            "original_total": c.total.to_string(),
            "original_2q": c.multi_qubit.to_string(),
            "original_t": c.t.to_string(),
        }));
    }

    /// The configuration, echoed at verbosity 2 and above.
    pub fn config(&self, config: &Value) {
        if self.verbosity < 2 {
            return;
        }
        self.emit(config.clone());
    }

    /// How resynthesis went, when a backend was configured. Without this line a backend
    /// that declines or misses the budget on every call is indistinguishable from one
    /// that was never invoked.
    pub fn resynth_stats(&self, stats: &qopt::ResynthStats) {
        if self.verbosity < 1 {
            return;
        }
        self.emit(json!({
            "resynth_attempts": stats.attempts.to_string(),
            "resynth_accepted": stats.accepted.to_string(),
            "resynth_no_partition": stats.no_partition.to_string(),
            "resynth_declined": stats.backend_declined.to_string(),
            "resynth_unusable": stats.unusable.to_string(),
            "resynth_over_budget": stats.over_budget.to_string(),
            "resynth_errored": stats.errored.to_string(),
            "resynth_total_error": stats.total_error.to_string(),
        }));
    }

    /// Mid-run progress of a windowed optimization, at verbosity 1 and above.
    ///
    /// New keys rather than the final summary's, so nothing scraping the summary line
    /// mistakes a snapshot for the result.
    pub fn window_progress(&self, p: &qopt::WindowProgress, seconds_elapsed: f64) {
        if self.verbosity < 1 {
            return;
        }
        self.emit(json!({
            "windows_done": p.done,
            "windows_total": p.total,
            "round": p.round,
            "rounds": p.rounds,
            "windows_improved_so_far": p.improved,
            "gates_start": p.gates_start,
            "gates_now": p.gates_now,
            "seconds_elapsed": format!("{seconds_elapsed:.1}"),
        }));
    }

    /// One line summarizing a windowed run.
    pub fn windows(&self, report: &qopt::WindowReport) {
        if self.verbosity == 0 {
            return;
        }
        self.emit(serde_json::json!({
            "windows": report.windows,
            "windows_improved": report.improved,
            "gates_before": report.before,
            "gates_after": report.after,
        }));
    }

    pub fn improvement(
        &self,
        candidate: &Candidate,
        model: &CostModel,
        seconds_elapsed: f64,
        time_to_best: f64,
    ) {
        if self.verbosity < 1 {
            return;
        }
        let mut map = self.common(candidate, model, seconds_elapsed, time_to_best);
        let rules = if self.verbosity >= 3 {
            json!(candidate
                .history
                .iter()
                .map(|s| json!([s.transformation, s.gate_count]))
                .collect::<Vec<_>>())
        } else {
            json!({})
        };
        map.insert("rules_applied_to_best".into(), json!(rules.to_string()));
        map.insert("all_rules_applied".into(), json!("{}"));
        self.emit(Value::Object(map));
    }

    /// One search iteration, at verbosity 2 and above.
    pub fn iteration(&self, info: &IterationInfo<'_>, model: &CostModel) {
        if self.verbosity < 2 {
            return;
        }
        let mut map = self.common(info.best, model, info.seconds_elapsed, info.time_to_best);
        map.insert("filename".into(), json!(self.filename));
        map.insert("queue_size".into(), json!(info.queue_size.to_string()));
        map.insert("seen_set_size".into(), json!(info.seen.to_string()));
        map.insert("num_rules_using".into(), json!(info.num_rules.to_string()));
        map.insert(
            "num_symb_rules_using".into(),
            json!(info.num_symb_rules.to_string()),
        );
        map.insert(
            "current_circuit_size".into(),
            json!(info.current.dag.gate_count().to_string()),
        );
        self.emit(Value::Object(map));
    }

    /// The final result.
    pub fn final_result(&self, result: &SearchResult, model: &CostModel) {
        if self.verbosity < 1 {
            return;
        }
        let total = result.elapsed.as_secs_f64();
        let mut map = self.common(&result.best, model, total, result.time_to_best);
        map.insert("total_time".into(), json!((total as u64).to_string()));
        map.insert("iterations".into(), json!(result.iterations.to_string()));
        map.insert(
            "accumulated_error".into(),
            json!(result.best.accumulated_error.to_string()),
        );
        let rules = json!(result
            .best
            .history
            .iter()
            .map(|s| json!([s.transformation, s.gate_count]))
            .collect::<Vec<_>>());
        map.insert("rules_applied_to_best".into(), json!(rules.to_string()));
        let all: Map<String, Value> = result
            .applications
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect();
        map.insert(
            "all_rules_applied".into(),
            json!(Value::Object(all).to_string()),
        );
        self.emit(Value::Object(map));
    }

    fn common(
        &self,
        candidate: &Candidate,
        model: &CostModel,
        seconds_elapsed: f64,
        time_to_best: f64,
    ) -> Map<String, Value> {
        let c = model.counts(&candidate.dag);
        let mut map = Map::new();
        map.insert("best_circuit_size".into(), json!(c.total.to_string()));
        map.insert("best_size_2q".into(), json!(c.multi_qubit.to_string()));
        map.insert("best_size_t".into(), json!(c.t.to_string()));
        map.insert(
            "time_to_best".into(),
            json!((time_to_best as u64).to_string()),
        );
        map.insert(
            "seconds_elapsed".into(),
            json!(format!("{seconds_elapsed}")),
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;
    use qopt::Objective;

    fn model() -> CostModel {
        CostModel::new(Objective::Total, 1, GateRegistry::with_builtins())
    }

    fn candidate(src: &str) -> Candidate {
        Candidate::new(qasm::parse(src).unwrap(), &model(), 0)
    }

    #[test]
    fn common_fields_match_the_reference_keys() {
        let logger = Logger::new(3, "tof_3.qasm");
        let map = logger.common(&candidate("h a; cx a, b; t b;"), &model(), 1.5, 1.0);
        for key in [
            "best_circuit_size",
            "best_size_2q",
            "best_size_t",
            "time_to_best",
            "seconds_elapsed",
        ] {
            assert!(map.contains_key(key), "missing {key}");
        }
        // The reference emitted every value as a string, which the harness parses.
        for (k, v) in &map {
            assert!(v.is_string(), "{k} should be a string, got {v}");
        }
        assert_eq!(map["best_circuit_size"], json!("3"));
        assert_eq!(map["best_size_2q"], json!("1"));
        assert_eq!(map["best_size_t"], json!("1"));
    }

    #[test]
    fn verbosity_gates_output() {
        // Nothing should panic at any verbosity; the guard is on emission.
        for v in 0..=3u8 {
            let logger = Logger::new(v, "c.qasm");
            let m = model();
            let c = candidate("h a;");
            logger.initial(&c.dag, &GateRegistry::with_builtins());
            logger.improvement(&c, &m, 1.0, 0.5);
            logger.iteration(
                &IterationInfo {
                    queue_size: 1,
                    seen: 1,
                    best: &c,
                    current: &c,
                    seconds_elapsed: 1.0,
                    time_to_best: 0.5,
                    num_rules: 2,
                    num_symb_rules: 3,
                },
                &m,
            );
            logger.config(&json!({"k": "v"}));
        }
    }

    #[test]
    fn counts_track_the_circuit() {
        let logger = Logger::new(1, "c.qasm");
        let m = model();
        let a = logger.common(&candidate(""), &m, 0.0, 0.0);
        assert_eq!(a["best_circuit_size"], json!("0"));
        let b = logger.common(&candidate("ccz a, b, c; ccz a, b, c;"), &m, 0.0, 0.0);
        assert_eq!(b["best_circuit_size"], json!("2"));
        assert_eq!(b["best_size_2q"], json!("2"), "ccz counts as multi-qubit");
    }
}
