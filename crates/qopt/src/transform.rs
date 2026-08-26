//! The set of moves the search can make.
//!
//! # Why this is a type rather than an index range
//!
//! Plain rules, symbolic rules, and resynthesis share one index space, and the offset
//! arithmetic that partitions it is easy to get subtly wrong when repeated at every use
//! site — a misattributed index surfaces as a score credited to the wrong rule.
//! [`TransformationSet::get`] makes the arithmetic impossible to get wrong, because there
//! is only one place that does it.

use qrules::{Rule, SymbolicRule};

/// One move available to the search.
#[derive(Debug, Clone, Copy)]
pub enum Transformation<'a> {
    Plain(&'a Rule),
    Symbolic(&'a SymbolicRule),
    /// Hand a partition to a resynthesis backend.
    Resynth,
}

impl<'a> Transformation<'a> {
    /// A stable identity for scoring and logging.
    ///
    /// Borrows from the rule set, not from the `Transformation` value, so the id outlives
    /// the (copyable) handle it came from.
    pub fn id(&self) -> &'a str {
        match self {
            Transformation::Plain(r) => &r.id,
            Transformation::Symbolic(r) => &r.id,
            Transformation::Resynth => "resynth",
        }
    }

    /// Net change in gate count if this fires, where it is known ahead of time.
    pub fn size_delta(&self) -> Option<isize> {
        match self {
            Transformation::Plain(r) => Some(r.size_delta()),
            Transformation::Symbolic(r) => Some(r.size_delta()),
            Transformation::Resynth => None,
        }
    }
}

/// Everything the search may try, in one index space.
#[derive(Debug, Default)]
pub struct TransformationSet {
    plain: Vec<Rule>,
    symbolic: Vec<SymbolicRule>,
    /// How many index slots resynthesis occupies, so that a uniform draw picks it with a
    /// tunable probability.
    resynth_weight: usize,
}

impl TransformationSet {
    pub fn new(plain: Vec<Rule>, symbolic: Vec<SymbolicRule>, resynth_weight: usize) -> Self {
        Self {
            plain,
            symbolic,
            resynth_weight,
        }
    }

    pub fn plain(&self) -> &[Rule] {
        &self.plain
    }

    pub fn symbolic(&self) -> &[SymbolicRule] {
        &self.symbolic
    }

    pub fn num_plain(&self) -> usize {
        self.plain.len()
    }

    pub fn num_symbolic(&self) -> usize {
        self.symbolic.len()
    }

    pub fn resynth_weight(&self) -> usize {
        self.resynth_weight
    }

    pub fn set_resynth_weight(&mut self, w: usize) {
        self.resynth_weight = w;
    }

    /// Total number of index slots.
    pub fn len(&self) -> usize {
        self.plain.len() + self.symbolic.len() + self.resynth_weight
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The transformation at `index`, or `None` if out of range.
    ///
    /// The single place index arithmetic happens.
    pub fn get(&self, index: usize) -> Option<Transformation<'_>> {
        let n_plain = self.plain.len();
        let n_symb = self.symbolic.len();
        if index < n_plain {
            Some(Transformation::Plain(&self.plain[index]))
        } else if index < n_plain + n_symb {
            Some(Transformation::Symbolic(&self.symbolic[index - n_plain]))
        } else if index < n_plain + n_symb + self.resynth_weight {
            Some(Transformation::Resynth)
        } else {
            None
        }
    }

    /// Every transformation, in index order.
    pub fn iter(&self) -> impl Iterator<Item = Transformation<'_>> {
        (0..self.len()).filter_map(|i| self.get(i))
    }

    /// Indices of every rule, excluding resynthesis.
    pub fn rule_indices(&self) -> std::ops::Range<usize> {
        0..(self.plain.len() + self.symbolic.len())
    }

    /// Derive the resynthesis weight from the rule count.
    ///
    /// Resynthesis is one move among many thousands of rules; without a weight a uniform
    /// draw would almost never pick it. The reference used
    /// `RESYNTH_PERCENTAGE * totalRules` and applied it only when the weight was still at
    /// its default of 1 (`Params.setResynthWeight`).
    pub fn weight_from_fraction(&mut self, fraction: f64) {
        let n = self.plain.len() + self.symbolic.len();
        if n == 0 {
            self.resynth_weight = self.resynth_weight.max(1);
            return;
        }
        self.resynth_weight = ((fraction * n as f64) as usize).max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qrules::Rule;

    fn plain(n: usize) -> Vec<Rule> {
        (0..n)
            .map(|i| Rule::new(&format!("h q{i}; h q{i};"), "").unwrap())
            .collect()
    }

    fn symbolic(n: usize) -> Vec<SymbolicRule> {
        let c = "[{[false, false]=[false, false], [true, false]=[true, false], \
                 [false, true]=[false, true], [true, true]=[true, true]}]";
        (0..n)
            .map(|i| {
                SymbolicRule::parse_legacy_with_builtins(&format!(
                    "h q0; rz({i}.5) q1; symb q; | h q0; symb q; rz({i}.5) q1; | {c}"
                ))
                .unwrap()
            })
            .collect()
    }

    /// Plain, symbolic, and resynthesis each own exactly the index range they should.
    #[test]
    fn index_space_is_partitioned_correctly() {
        let set = TransformationSet::new(plain(5), symbolic(3), 2);
        assert_eq!(set.len(), 10);

        for i in 0..5 {
            assert!(matches!(set.get(i), Some(Transformation::Plain(_))), "{i}");
        }
        for i in 5..8 {
            assert!(
                matches!(set.get(i), Some(Transformation::Symbolic(_))),
                "{i}"
            );
        }
        for i in 8..10 {
            assert!(matches!(set.get(i), Some(Transformation::Resynth)), "{i}");
        }
        assert!(set.get(10).is_none());

        // Each symbolic rule is reachable exactly once, at the right offset.
        let ids: Vec<&str> = (5..8).map(|i| set.get(i).unwrap().id()).collect();
        let expected: Vec<&str> = set.symbolic().iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, expected);

        // The reference's arithmetic, reproduced, disagrees.
        let (n_plain, n_symb) = (5usize, 3usize);
        let reference_bound = n_symb + n_symb; // should be n_plain + n_symb
        assert_ne!(reference_bound, n_plain + n_symb);
    }

    #[test]
    fn every_index_maps_to_something() {
        let set = TransformationSet::new(plain(4), symbolic(2), 3);
        for i in 0..set.len() {
            assert!(set.get(i).is_some(), "index {i} is unmapped");
        }
        assert_eq!(set.iter().count(), set.len());
    }

    #[test]
    fn rule_indices_exclude_resynthesis() {
        let set = TransformationSet::new(plain(4), symbolic(2), 7);
        assert_eq!(set.rule_indices(), 0..6);
        for i in set.rule_indices() {
            assert!(!matches!(set.get(i), Some(Transformation::Resynth)));
        }
    }

    #[test]
    fn ids_are_stable_and_distinct() {
        let set = TransformationSet::new(plain(3), symbolic(2), 1);
        let ids: Vec<&str> = set.iter().map(|t| t.id()).collect();
        assert_eq!(ids.len(), 6);
        assert_eq!(ids[5], "resynth");
        let unique: std::collections::HashSet<&&str> = ids[..5].iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn size_deltas_are_known_for_rules_only() {
        let set = TransformationSet::new(plain(1), symbolic(1), 1);
        assert_eq!(set.get(0).unwrap().size_delta(), Some(-2));
        assert!(set.get(1).unwrap().size_delta().is_some());
        assert_eq!(set.get(2).unwrap().size_delta(), None);
    }

    #[test]
    fn resynth_weight_from_fraction() {
        let mut set = TransformationSet::new(plain(100), symbolic(100), 1);
        set.weight_from_fraction(0.015);
        assert_eq!(set.resynth_weight(), 3);
        assert_eq!(set.len(), 203);

        // Never zero, or resynthesis could never be chosen at all.
        let mut small = TransformationSet::new(plain(1), symbolic(0), 1);
        small.weight_from_fraction(0.015);
        assert_eq!(small.resynth_weight(), 1);

        let mut none = TransformationSet::default();
        none.weight_from_fraction(0.015);
        assert_eq!(none.resynth_weight(), 1);
    }

    /// A set with no plain rules is an ordinary set, not a special case.
    #[test]
    fn a_set_with_no_plain_rules_is_usable() {
        let set = TransformationSet::new(Vec::new(), symbolic(3), 1);
        assert_eq!(set.num_plain(), 0);
        assert_eq!(set.len(), 4);
        assert!(matches!(set.get(0), Some(Transformation::Symbolic(_))));
        assert!(matches!(set.get(3), Some(Transformation::Resynth)));
        assert_eq!(set.iter().count(), 4);
    }

    #[test]
    fn an_empty_set_is_usable() {
        let set = TransformationSet::default();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(set.get(0).is_none());
        assert_eq!(set.iter().count(), 0);
    }
}
