//! A circuit under consideration, with its provenance.

use std::cmp::Ordering;
use std::sync::Arc;

use qcircuit::Dag;
use qrules::MatchIndex;

use crate::cost::{CostKey, CostModel};

/// One step in a candidate's history.
#[derive(Debug, Clone)]
pub struct Step {
    /// The transformation's id.
    pub transformation: String,
    /// Gate count immediately afterwards.
    pub gate_count: usize,
    /// Error this step added, for resynthesis steps.
    pub error: f64,
}

/// A circuit the search is considering.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub dag: Arc<Dag>,
    /// The match index for exactly this circuit, carried so the search does not rebuild
    /// it every iteration.
    ///
    /// It is keyed by node index, so it is valid only for *this* `dag` -- not for a
    /// structurally identical one, whose nodes may be numbered differently. The two
    /// travel together for that reason.
    pub index: Arc<MatchIndex>,
    pub key: CostKey,
    /// Transformations applied to reach this circuit, oldest first.
    pub history: Arc<Vec<Step>>,
    /// Total resynthesis error accumulated along `history`.
    ///
    /// The reference had no equivalent: it divided the user's budget by
    /// `MAX_RESYNTH_ALLOWED` and never measured anything, so the error of a finished
    /// circuit was unknown. Carrying the real total on the candidate is what lets a
    /// resynthesis result be rejected for being unaffordable.
    pub accumulated_error: f64,
    /// Monotonic sequence number, used to break ties in favour of older candidates.
    pub sequence: u64,
}

impl Candidate {
    pub fn new(dag: Dag, model: &CostModel, sequence: u64) -> Self {
        let key = model.key(&dag);
        let index = Arc::new(MatchIndex::build(&dag));
        Self {
            dag: Arc::new(dag),
            index,
            key,
            history: Arc::new(Vec::new()),
            accumulated_error: 0.0,
            sequence,
        }
    }

    /// A successor produced by applying `transformation`.
    /// A successor whose index has to be rebuilt, for transformations that do not report
    /// what they changed.
    pub fn derive(
        &self,
        dag: Dag,
        model: &CostModel,
        transformation: &str,
        error: f64,
        sequence: u64,
    ) -> Self {
        let index = MatchIndex::build(&dag);
        self.derive_with_index(dag, index, model, transformation, error, sequence)
    }

    /// A successor carrying an index already brought up to date with `dag`.
    #[allow(clippy::too_many_arguments)]
    pub fn derive_with_index(
        &self,
        dag: Dag,
        index: MatchIndex,
        model: &CostModel,
        transformation: &str,
        error: f64,
        sequence: u64,
    ) -> Self {
        let key = model.key(&dag);
        let mut history = Vec::with_capacity(self.history.len() + 1);
        history.extend(self.history.iter().cloned());
        history.push(Step {
            transformation: transformation.to_string(),
            gate_count: dag.gate_count(),
            error,
        });
        Self {
            dag: Arc::new(dag),
            index: Arc::new(index),
            key,
            history: Arc::new(history),
            accumulated_error: self.accumulated_error + error,
            sequence,
        }
    }

    /// Mutable access to the circuit and its index together.
    ///
    /// Separate field borrows, so both can be held at once; `Arc::make_mut` copies only
    /// if something else still holds them, which after `detached` is used for `best` is
    /// the exceptional case rather than the rule.
    pub fn parts_mut(&mut self) -> (&mut Dag, &mut MatchIndex) {
        let Self { dag, index, .. } = self;
        (Arc::make_mut(dag), Arc::make_mut(index))
    }

    /// Re-score this candidate after its circuit was changed in place, recording `step`.
    pub fn rescore_in_place(
        &mut self,
        model: &CostModel,
        transformation: &str,
        error: f64,
        sequence: u64,
    ) {
        self.key = model.key(&self.dag);
        let mut history = Vec::with_capacity(self.history.len() + 1);
        history.extend(self.history.iter().cloned());
        history.push(Step {
            transformation: transformation.to_string(),
            gate_count: self.dag.gate_count(),
            error,
        });
        self.history = Arc::new(history);
        self.accumulated_error += error;
        self.sequence = sequence;
    }

    /// A copy sharing nothing with this candidate.
    ///
    /// The search mutates the circuit it is expanding in place, which requires that
    /// nothing else holds it. `best` is the one long-lived alias, so it takes a detached
    /// copy at the moment it is set -- an O(circuit) cost paid only when the search
    /// improves, in exchange for every ordinary iteration avoiding one.
    pub fn detached(&self) -> Self {
        Self {
            dag: Arc::new((*self.dag).clone()),
            index: Arc::new((*self.index).clone()),
            key: self.key,
            history: self.history.clone(),
            accumulated_error: self.accumulated_error,
            sequence: self.sequence,
        }
    }

    /// How many times `transformation` appears in this candidate's history.
    pub fn count_applications(&self, transformation: &str) -> usize {
        self.history
            .iter()
            .filter(|s| s.transformation == transformation)
            .count()
    }

    /// How many resynthesis calls this candidate carries.
    pub fn count_resynth(&self) -> usize {
        self.count_applications("resynth")
    }
}

/// Orders candidates best-first: lower cost wins, then the older candidate.
///
/// Total for every objective, because [`CostKey`] is. The reference's
/// `OptCircuitComparator` threw for two of the six.
#[derive(Debug, Clone, Copy)]
pub struct BestFirst;

impl BestFirst {
    pub fn compare(a: &Candidate, b: &Candidate) -> Ordering {
        a.key.cmp(&b.key).then(a.sequence.cmp(&b.sequence))
    }
}

/// Wrapper giving a max-heap the *worst* candidate at its top, so pruning is cheap.
#[derive(Debug, Clone)]
pub struct WorstFirst(pub Candidate);

impl PartialEq for WorstFirst {
    fn eq(&self, other: &Self) -> bool {
        BestFirst::compare(&self.0, &other.0) == Ordering::Equal
    }
}

impl Eq for WorstFirst {}

impl PartialOrd for WorstFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorstFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        BestFirst::compare(&self.0, &other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::Objective;
    use qcircuit::{qasm, GateRegistry};

    fn model() -> CostModel {
        CostModel::new(Objective::Total, 1, GateRegistry::with_builtins())
    }

    fn cand(src: &str, seq: u64) -> Candidate {
        Candidate::new(qasm::parse(src).unwrap(), &model(), seq)
    }

    #[test]
    fn a_fresh_candidate_has_no_history() {
        let c = cand("h a;", 0);
        assert!(c.history.is_empty());
        assert_eq!(c.accumulated_error, 0.0);
        assert_eq!(c.count_resynth(), 0);
    }

    #[test]
    fn deriving_records_the_step() {
        let m = model();
        let a = cand("h a; h a;", 0);
        let b = a.derive(qasm::parse("").unwrap(), &m, "h q0; h q0; | ", 0.0, 1);
        assert_eq!(b.history.len(), 1);
        assert_eq!(b.history[0].gate_count, 0);
        assert_eq!(b.count_applications("h q0; h q0; | "), 1);
        // The parent is untouched.
        assert!(a.history.is_empty());
    }

    #[test]
    fn accumulated_error_adds_along_the_chain() {
        let m = model();
        let mut c = cand("h a;", 0);
        for i in 0..4 {
            c = c.derive(qasm::parse("h a;").unwrap(), &m, "resynth", 1e-9, i + 1);
        }
        assert!((c.accumulated_error - 4e-9).abs() < 1e-18);
        assert_eq!(c.count_resynth(), 4);
    }

    #[test]
    fn ordering_prefers_lower_cost() {
        let small = cand("h a;", 0);
        let big = cand("h a; h a;", 1);
        assert_eq!(BestFirst::compare(&small, &big), Ordering::Less);
        assert_eq!(BestFirst::compare(&big, &small), Ordering::Greater);
    }

    #[test]
    fn ties_go_to_the_older_candidate() {
        let older = cand("h a;", 1);
        let newer = cand("h b;", 2);
        assert_eq!(older.key, newer.key);
        assert_eq!(BestFirst::compare(&older, &newer), Ordering::Less);
    }

    /// The queue accepts candidates under every objective.
    #[test]
    fn queue_accepts_every_objective() {
        use std::collections::BinaryHeap;
        for o in Objective::ALL {
            let m = CostModel::new(o, 38, GateRegistry::with_builtins());
            let mut heap: BinaryHeap<WorstFirst> = BinaryHeap::new();
            for (i, src) in ["h a;", "cx a, b;", "t a; cx a, b;", "ccz a, b, c;"]
                .iter()
                .enumerate()
            {
                // The reference threw here on the second insert for FIDELITY and FT.
                heap.push(WorstFirst(Candidate::new(
                    qasm::parse(src).unwrap(),
                    &m,
                    i as u64,
                )));
            }
            assert_eq!(heap.len(), 4, "{o}");
            // The heap's top is the worst candidate, so pruning pops the right ones.
            let worst = heap.peek().unwrap().0.key;
            for c in heap.iter() {
                assert!(c.0.key <= worst, "{o}: heap top is not the worst");
            }
        }
    }

    #[test]
    fn worst_first_pops_the_most_expensive() {
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<WorstFirst> = BinaryHeap::new();
        heap.push(WorstFirst(cand("h a;", 0)));
        heap.push(WorstFirst(cand("h a; h a; h a;", 1)));
        heap.push(WorstFirst(cand("h a; h a;", 2)));
        assert_eq!(heap.pop().unwrap().0.dag.gate_count(), 3);
        assert_eq!(heap.pop().unwrap().0.dag.gate_count(), 2);
        assert_eq!(heap.pop().unwrap().0.dag.gate_count(), 1);
    }

    #[test]
    fn history_is_shared_not_copied_per_clone() {
        let c = cand("h a;", 0);
        let d = c.clone();
        assert!(Arc::ptr_eq(&c.history, &d.history));
        assert!(Arc::ptr_eq(&c.dag, &d.dag));
    }
}
