//! Narrowing the transformation set to what could actually fire.
//!
//! # The problem
//!
//! The search samples one transformation per iteration from the whole set. Most of them
//! cannot possibly match: a rule whose pattern contains an `sx` will never fire on a
//! circuit with no `sx` in it, but it still occupies a slot in the draw and wastes the
//! iteration.
//!
//! That makes rule-set *size* work against rule-set *quality*. This port loads about
//! 22,954 plain Nam rules where the reference loads 12,558 — the difference is the rules
//! the reference's `+`/`-` pattern filter sets aside, which are perfectly usable (see
//! docs/PORTING-NOTES.md on compound pattern angles). Loading them is right, but at a
//! fixed sampling rate it halves the chance of drawing a rule that fires, and a
//! differential run showed the cost: filtering them back out recovered about a gate per
//! circuit.
//!
//! # The fix
//!
//! A rule can only fire if the circuit contains every gate its pattern names. Gate names
//! are few — a gate set has a handful — so the requirement fits in a bitmask, and the
//! eligible set depends only on which gates the circuit currently contains. There are
//! few distinct such masks over a run, so the eligible lists are built once each and
//! reused.
//!
//! This discards nothing: a transformation excluded from a draw is one that provably could
//! not have matched.

use rustc_hash::FxHashMap;
use std::sync::Arc;

use qcircuit::Dag;

use crate::transform::{Transformation, TransformationSet};

/// Strictly-reducing classifications, memoized per (objective, breakeven).
type ReducingMemo = FxHashMap<(crate::cost::Objective, i64), Arc<Vec<bool>>>;

/// Gate names beyond this many are all folded into one bit, which is conservative: a rule
/// needing a folded gate stays eligible whenever any folded gate is present.
const MAX_TRACKED_GATES: u32 = 31;

/// Which transformations can fire, given which gates a circuit contains.
#[derive(Debug, Default)]
pub struct EligibilityIndex {
    gate_bits: FxHashMap<String, u32>,
    /// Bit set of gates each transformation's pattern requires.
    required: Vec<u32>,
    /// Memoized eligible lists, behind a lock so one index can serve every window of a
    /// parallel run.
    ///
    /// Building the index means scanning every rule's pattern, which takes ~12 ms over the
    /// 22,954-rule Nam set. That is once-per-run work, but the windowed search creates a
    /// `Search` per window: a million-gate circuit is 1,954 windows a round, so rebuilding
    /// it per window cost more than the entire time budget. The immutable part is
    /// therefore shared, and the only mutable part is this memo.
    ///
    /// Contention is negligible: there are a handful of distinct masks over a whole run,
    /// so after the first few windows every access is a read hit.
    cache: std::sync::RwLock<FxHashMap<u32, Arc<Vec<usize>>>>,
    /// Which transformations are strictly cost-reducing plain rules, memoized per
    /// objective for the same reason as `cache`: classifying means two cost keys per
    /// rule, ~130 ms over a 63k-rule set, which the reduction strategy must not pay per
    /// window — at a tight budget that is more than a window's whole slice.
    reducing: std::sync::RwLock<ReducingMemo>,
    /// Transformations that no gate requirement applies to, such as resynthesis.
    always: Vec<usize>,
}

impl EligibilityIndex {
    /// Build an index over a transformation set.
    pub fn build(set: &TransformationSet) -> Self {
        let mut gate_bits: FxHashMap<String, u32> = FxHashMap::default();
        let mut next_bit = 0u32;
        let bit_of = |name: &str, bits: &mut FxHashMap<String, u32>, next: &mut u32| -> u32 {
            if let Some(b) = bits.get(name) {
                return *b;
            }
            let b = if *next < MAX_TRACKED_GATES {
                let b = 1u32 << *next;
                *next += 1;
                b
            } else {
                1u32 << MAX_TRACKED_GATES
            };
            bits.insert(name.to_string(), b);
            b
        };

        let mut required = Vec::with_capacity(set.len());
        let mut always = Vec::new();
        for i in 0..set.len() {
            match set.get(i) {
                Some(Transformation::Plain(rule)) => {
                    let mut mask = 0u32;
                    for idx in rule.find.dag().gate_indices() {
                        mask |= bit_of(
                            &rule.find.dag().gate(idx).gate,
                            &mut gate_bits,
                            &mut next_bit,
                        );
                    }
                    required.push(mask);
                }
                Some(Transformation::Symbolic(rule)) => {
                    // Both halves must be present for a symbolic rule to fire.
                    let mut mask = 0u32;
                    for half in [&rule.find_before, &rule.find_after] {
                        for idx in half.dag().gate_indices() {
                            mask |=
                                bit_of(&half.dag().gate(idx).gate, &mut gate_bits, &mut next_bit);
                        }
                    }
                    required.push(mask);
                }
                // Resynthesis needs no particular gate, only a circuit to partition.
                Some(Transformation::Resynth) => {
                    required.push(0);
                    always.push(i);
                }
                None => required.push(u32::MAX),
            }
        }

        Self {
            gate_bits,
            required,
            cache: std::sync::RwLock::new(FxHashMap::default()),
            reducing: std::sync::RwLock::new(FxHashMap::default()),
            always,
        }
    }

    /// The bit set of gates `dag` contains.
    pub fn circuit_mask(&self, dag: &Dag) -> u32 {
        let mut mask = 0u32;
        for idx in dag.gate_indices() {
            if let Some(b) = self.gate_bits.get(&*dag.gate(idx).gate) {
                mask |= b;
            }
        }
        mask
    }

    /// Transformation indices that could fire on a circuit with gate set `mask`.
    pub fn eligible(&self, mask: u32) -> Arc<Vec<usize>> {
        if let Ok(cache) = self.cache.read() {
            if let Some(list) = cache.get(&mask) {
                return Arc::clone(list);
            }
        }
        let list = Arc::new(
            self.required
                .iter()
                .enumerate()
                .filter(|(_, &req)| req != u32::MAX && req & !mask == 0)
                .map(|(i, _)| i)
                .collect::<Vec<usize>>(),
        );
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(mask, Arc::clone(&list));
        }
        list
    }

    /// Which transformations are strictly cost-reducing plain rules under `model`.
    ///
    /// Strictly reducing under the *model*, not by gate count: a rule trading one gate
    /// for two cheaper ones shrinks a T-count objective while growing a total-count one.
    /// The `(objective, breakeven)` pair is the part of the model the keys depend on, so
    /// it is the memo key.
    pub fn reducing_plain(
        &self,
        set: &TransformationSet,
        model: &crate::cost::CostModel,
        objective: crate::cost::Objective,
        breakeven: i64,
    ) -> Arc<Vec<bool>> {
        let key = (objective, breakeven);
        if let Ok(cache) = self.reducing.read() {
            if let Some(list) = cache.get(&key) {
                return Arc::clone(list);
            }
        }
        let list = Arc::new(
            (0..set.len())
                .map(|i| match set.get(i) {
                    Some(Transformation::Plain(r)) => {
                        model.key(r.find.dag()) > model.key(r.replace.dag())
                    }
                    _ => false,
                })
                .collect::<Vec<bool>>(),
        );
        if let Ok(mut cache) = self.reducing.write() {
            cache.insert(key, Arc::clone(&list));
        }
        list
    }

    /// Transformations with no gate requirement.
    pub fn always(&self) -> &[usize] {
        &self.always
    }

    /// How many distinct circuit gate sets have been seen.
    pub fn cached_masks(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::{qasm, GateRegistry};
    use qrules::{Rule, SymbolicRule};

    fn set_of(rules: &[(&str, &str)], resynth_weight: usize) -> TransformationSet {
        let plain: Vec<Rule> = rules
            .iter()
            .map(|(f, r)| Rule::new(f, r).unwrap())
            .collect();
        TransformationSet::new(plain, Vec::new(), resynth_weight)
    }

    #[test]
    fn a_rule_needing_an_absent_gate_is_excluded() {
        let set = set_of(
            &[
                ("h q0; h q0;", ""),
                ("sx q0; sx q0;", ""),
                ("cx q0, q1; cx q0, q1;", ""),
            ],
            0,
        );
        let idx = EligibilityIndex::build(&set);

        let dag = qasm::parse("h a; h a; cx a, b;").unwrap();
        let eligible = idx.eligible(idx.circuit_mask(&dag));
        // The h rule and the cx rule can fire; the sx rule cannot.
        assert_eq!(*eligible, vec![0, 2]);
    }

    #[test]
    fn a_rule_needing_several_gates_needs_all_of_them() {
        let set = set_of(&[("h q0; cx q0, q1;", "cx q0, q1; h q0;")], 0);
        let idx = EligibilityIndex::build(&set);

        let both = qasm::parse("h a; cx a, b;").unwrap();
        assert_eq!(*idx.eligible(idx.circuit_mask(&both)), vec![0]);

        let only_h = qasm::parse("h a; h a;").unwrap();
        assert!(idx.eligible(idx.circuit_mask(&only_h)).is_empty());

        let only_cx = qasm::parse("cx a, b;").unwrap();
        assert!(idx.eligible(idx.circuit_mask(&only_cx)).is_empty());
    }

    #[test]
    fn resynthesis_is_always_eligible() {
        let set = set_of(&[("sx q0; sx q0;", "")], 3);
        let idx = EligibilityIndex::build(&set);
        let dag = qasm::parse("h a; h a;").unwrap();
        let eligible = idx.eligible(idx.circuit_mask(&dag));
        // The sx rule is excluded; the three resynthesis slots are not.
        assert_eq!(*eligible, vec![1, 2, 3]);
        assert_eq!(idx.always(), &[1, 2, 3]);
    }

    #[test]
    fn an_empty_circuit_admits_only_gate_free_transformations() {
        let set = set_of(&[("h q0; h q0;", "")], 1);
        let idx = EligibilityIndex::build(&set);
        let dag = qasm::parse("qreg q[2];").unwrap();
        assert_eq!(*idx.eligible(idx.circuit_mask(&dag)), vec![1]);
    }

    /// Nothing that could fire may be excluded: the index is an optimization, not a
    /// filter on which rules exist.
    #[test]
    fn eligibility_never_excludes_a_rule_that_matches() {
        use qrules::{matches_anywhere, MatchContext};

        let reg = GateRegistry::with_builtins();
        let rules: Vec<(&str, &str)> = vec![
            ("h q0; h q0;", ""),
            ("x q0; x q0;", ""),
            ("cx q0, q1; cx q0, q1;", ""),
            ("h q0; cx q0, q1;", "cx q0, q1; h q0;"),
            ("sx q0; sx q0;", ""),
            ("t q0; t q0;", "s q0;"),
            ("cx q0, q1; h q1;", "h q1; cx q0, q1;"),
        ];
        let set = set_of(&rules, 0);
        let idx = EligibilityIndex::build(&set);

        for src in [
            "h a; h a; cx a, b;",
            "x a; x a;",
            "t a; t a; cx a, b;",
            "sx a; sx a; h b;",
            "cx a, b; h b; h b;",
            "h a; x a; cx a, b; t b;",
        ] {
            let dag = qasm::parse(src).unwrap();
            let eligible = idx.eligible(idx.circuit_mask(&dag));
            let ctx = MatchContext::new(&dag);
            for (i, _) in rules.iter().enumerate() {
                let Some(Transformation::Plain(rule)) = set.get(i) else {
                    continue;
                };
                if matches_anywhere(&ctx, rule) {
                    assert!(
                        eligible.contains(&i),
                        "`{src}`: rule {i} matches but was excluded"
                    );
                }
            }
        }
        let _ = reg;
    }

    #[test]
    fn results_are_cached_per_gate_set() {
        let set = set_of(&[("h q0; h q0;", ""), ("x q0; x q0;", "")], 0);
        let idx = EligibilityIndex::build(&set);
        let a = qasm::parse("h a; h a;").unwrap();
        let b = qasm::parse("h b; h b;").unwrap();
        let c = qasm::parse("x a; x a;").unwrap();

        assert_eq!(idx.cached_masks(), 0);
        idx.eligible(idx.circuit_mask(&a));
        assert_eq!(idx.cached_masks(), 1);
        // Same gate set, different circuit: no new entry.
        idx.eligible(idx.circuit_mask(&b));
        assert_eq!(idx.cached_masks(), 1);
        idx.eligible(idx.circuit_mask(&c));
        assert_eq!(idx.cached_masks(), 2);
    }

    #[test]
    fn symbolic_rules_need_both_halves_gates() {
        let c = "[{[false, false]=[false, false], [true, false]=[true, false], \
                 [false, true]=[false, true], [true, true]=[true, true]}]";
        let rule = SymbolicRule::parse_legacy_with_builtins(&format!(
            "h q0; rz(theta1) q1; symb q; | h q0; symb q; rz(theta1) q1; | {c}"
        ))
        .unwrap();
        let set = TransformationSet::new(Vec::new(), vec![rule], 0);
        let idx = EligibilityIndex::build(&set);

        let both = qasm::parse("h a; rz(0.3) b;").unwrap();
        assert_eq!(*idx.eligible(idx.circuit_mask(&both)), vec![0]);
        let only_h = qasm::parse("h a;").unwrap();
        assert!(idx.eligible(idx.circuit_mask(&only_h)).is_empty());
    }

    /// The index must shrink the draw substantially on a real rule set, or it is not
    /// worth its cost.
    #[test]
    fn the_index_narrows_a_real_rule_set() {
        use qrules::legacy::{self, LoadOptions};

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let text = std::fs::read_to_string(root.join("rules/rules_q3_s6_nam.txt")).unwrap();
        let head: String = text.lines().take(20_000).collect::<Vec<_>>().join("\n");
        let report = legacy::load_str(
            &head,
            &LoadOptions::default(),
            &GateRegistry::with_builtins(),
        );
        let set = TransformationSet::new(report.rules, Vec::new(), 1);
        let idx = EligibilityIndex::build(&set);

        let src = std::fs::read_to_string(root.join("benchmarks/nam_rz/tof_3.qasm")).unwrap();
        let dag = qasm::parse(&src).unwrap();
        let eligible = idx.eligible(idx.circuit_mask(&dag));
        println!(
            "{} of {} transformations eligible",
            eligible.len(),
            set.len()
        );
        assert!(!eligible.is_empty());
        assert!(
            eligible.len() < set.len(),
            "the index excluded nothing: {} of {}",
            eligible.len(),
            set.len()
        );
    }
}
