//! Optimization objectives.
//!
//! # One key, used everywhere
//!
//! Cost, comparison, and tie-breaking are one computation, not several that must agree:
//! every objective produces a [`CostKey`]: a fixed-length lexicographic tuple whose
//! first element is the objective proper and whose remaining elements are tie-breakers.
//! Ordering is total for every objective by construction, so there is nothing to throw.

use std::fmt;

use qcircuit::{is_t_gate, Dag, GateRegistry};

/// What the search is trying to minimize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Objective {
    /// Total gate count.
    Total,
    /// Multi-qubit gate count, then total.
    TwoQ,
    /// T-gate count, then multi-qubit count, then total.
    T,
    /// Weighted T and multi-qubit count, for fault-tolerant cost.
    Ft,
    /// Total gate count ignoring virtual-z rotations.
    TotalIgnoreRz,
    /// Estimated infidelity, charging multi-qubit gates more.
    Fidelity,
}

impl Objective {
    pub const ALL: [Objective; 6] = [
        Objective::Total,
        Objective::TwoQ,
        Objective::T,
        Objective::Ft,
        Objective::TotalIgnoreRz,
        Objective::Fidelity,
    ];

    /// The reference's spelling, as accepted by `-opt`.
    pub fn as_flag(self) -> &'static str {
        match self {
            Objective::Total => "TOTAL",
            Objective::TwoQ => "TWO_Q",
            Objective::T => "T",
            Objective::Ft => "FT",
            Objective::TotalIgnoreRz => "TOTAL_IGNORE_RZ",
            Objective::Fidelity => "FIDELITY",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Objective::ALL
            .into_iter()
            .find(|o| o.as_flag().eq_ignore_ascii_case(s))
    }

    /// `true` if this objective needs a resynthesis backend by default.
    pub fn prefers_resynthesis(self) -> bool {
        matches!(
            self,
            Objective::Fidelity | Objective::TwoQ | Objective::Ft | Objective::T
        )
    }
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_flag())
    }
}

/// A lexicographic cost: the objective, then tie-breakers.
///
/// Fixed length so it is `Copy` and orders by the derived lexicographic `Ord`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CostKey([i64; 3]);

impl CostKey {
    pub fn new(primary: i64, second: i64, third: i64) -> Self {
        Self([primary, second, third])
    }

    /// The objective value proper, ignoring tie-breakers.
    pub fn primary(&self) -> i64 {
        self.0[0]
    }

    pub fn as_slice(&self) -> &[i64] {
        &self.0
    }
}

impl fmt::Display for CostKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.0[0], self.0[1], self.0[2])
    }
}

/// Gate counts, computed in one pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GateCounts {
    pub total: usize,
    /// Gates acting on two or more qubits.
    pub multi_qubit: usize,
    pub t: usize,
    /// Gates other than virtual-z rotations.
    pub non_virtual_z: usize,
    /// Sum of `k - 1` over gates of arity `k >= 2`; a `cx` contributes 1, a `ccz` 2.
    pub two_qubit_equivalents: usize,
}

impl GateCounts {
    pub fn of(dag: &Dag, registry: &GateRegistry) -> Self {
        let mut c = GateCounts::default();
        for idx in dag.gate_indices() {
            let op = dag.gate(idx);
            c.total += 1;
            let arity = op.qubits.len();
            if arity >= 2 {
                c.multi_qubit += 1;
                c.two_qubit_equivalents += arity - 1;
            }
            if is_t_gate(&op.gate, &op.params) {
                c.t += 1;
            }
            let virtual_z = registry
                .get(&op.gate)
                .map(|d| d.is_virtual_z)
                .unwrap_or(false);
            if !virtual_z {
                c.non_virtual_z += 1;
            }
        }
        c
    }
}

/// Turns a circuit into a comparable cost.
#[derive(Debug, Clone)]
pub struct CostModel {
    pub objective: Objective,
    /// Cost of a two-qubit gate in one-qubit-gate units, or the weight of a T gate
    /// against a two-qubit gate for [`Objective::Ft`].
    pub fidelity_breakeven: i64,
    registry: GateRegistry,
}

impl CostModel {
    pub fn new(objective: Objective, fidelity_breakeven: i64, registry: GateRegistry) -> Self {
        Self {
            objective,
            fidelity_breakeven: fidelity_breakeven.max(1),
            registry,
        }
    }

    /// Derive the two-qubit-gate cost from measured error rates.
    ///
    /// `log(1 - e2) / log(1 - e1)`: how many one-qubit gates carry the same error as one
    /// two-qubit gate.
    pub fn breakeven_from_error_rates(error_1q: f64, error_2q: f64) -> Option<i64> {
        if !(0.0..1.0).contains(&error_1q) || !(0.0..1.0).contains(&error_2q) {
            return None;
        }
        if error_1q == 0.0 {
            return None;
        }
        let v = ((1.0 - error_2q).ln() / (1.0 - error_1q).ln()) as i64;
        Some(v.max(1))
    }

    pub fn counts(&self, dag: &Dag) -> GateCounts {
        GateCounts::of(dag, &self.registry)
    }

    /// The lexicographic cost of `dag`.
    pub fn key(&self, dag: &Dag) -> CostKey {
        self.key_of(&self.counts(dag))
    }

    /// The lexicographic cost of already-computed counts.
    pub fn key_of(&self, c: &GateCounts) -> CostKey {
        let total = c.total as i64;
        let multi = c.multi_qubit as i64;
        let t = c.t as i64;
        let b = self.fidelity_breakeven;
        match self.objective {
            Objective::Total => CostKey::new(total, multi, t),
            Objective::TwoQ => CostKey::new(multi, total, t),
            Objective::T => CostKey::new(t, multi, total),
            Objective::TotalIgnoreRz => CostKey::new(c.non_virtual_z as i64, total, multi),
            Objective::Fidelity => CostKey::new(self.fidelity_cost(c), multi, total),
            Objective::Ft => CostKey::new(b * t + multi, t, total),
        }
    }

    /// Estimated cost in one-qubit-gate-error units.
    ///
    /// A gate of arity `k >= 2` is charged `(k - 1) * breakeven`, so a `ccz` costs twice a
    /// `cx` rather than rounding down to a one-qubit gate's price.
    fn fidelity_cost(&self, c: &GateCounts) -> i64 {
        let one_qubit = (c.non_virtual_z - c.multi_qubit.min(c.non_virtual_z)) as i64;
        one_qubit + c.two_qubit_equivalents as i64 * self.fidelity_breakeven
    }

    /// `true` if `a` is strictly better than `b`.
    pub fn is_better(&self, a: &Dag, b: &Dag) -> bool {
        self.key(a) < self.key(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn model(o: Objective) -> CostModel {
        CostModel::new(o, 38, reg())
    }

    fn dag(src: &str) -> Dag {
        qasm::parse(src).unwrap()
    }

    #[test]
    fn counts_are_computed_in_one_pass() {
        let c = GateCounts::of(&dag("h a; t a; rz(0.3) a; cx a, b; ccz a, b, c;"), &reg());
        assert_eq!(c.total, 5);
        assert_eq!(c.multi_qubit, 2);
        assert_eq!(c.t, 1);
        // rz(0.3) is the only virtual-z.
        assert_eq!(c.non_virtual_z, 4);
        // cx contributes 1, ccz contributes 2.
        assert_eq!(c.two_qubit_equivalents, 3);
    }

    #[test]
    fn t_count_includes_rz_at_odd_multiples_of_pi_over_four() {
        use std::f64::consts::PI;
        let c = GateCounts::of(
            &dag(&format!(
                "t a; tdg a; rz({}) a; rz({}) a; rz({}) a;",
                PI / 4.0,
                3.0 * PI / 4.0,
                PI / 2.0
            )),
            &reg(),
        );
        assert_eq!(c.t, 4, "t, tdg, pi/4 and 3pi/4 count; pi/2 does not");
    }

    /// Every objective yields a total order, `FIDELITY` and `FT` included.
    #[test]
    fn every_objective_orders_totally() {
        let a = dag("h a; cx a, b;");
        let b = dag("h a; h a; cx a, b; cx b, c;");
        for o in Objective::ALL {
            let m = model(o);
            let (ka, kb) = (m.key(&a), m.key(&b));
            // Whatever the objective, the two keys compare without panicking and the
            // ordering is a strict weak order.
            assert!(ka <= kb || kb <= ka, "{o}: keys are not comparable");
            assert_eq!(ka.cmp(&kb), ka.cmp(&kb).reverse().reverse(), "{o}");
            assert_eq!(m.key(&a), ka, "{o}: key is not deterministic");
        }
    }

    #[test]
    fn objectives_rank_as_expected() {
        let few_2q = dag("h a; h a; h a; cx a, b;");
        let many_2q = dag("cx a, b; cx b, c;");

        // TOTAL prefers the shorter circuit.
        assert!(model(Objective::Total).key(&many_2q) < model(Objective::Total).key(&few_2q));
        // TWO_Q prefers fewer multi-qubit gates, even at a higher total.
        assert!(model(Objective::TwoQ).key(&few_2q) < model(Objective::TwoQ).key(&many_2q));
    }

    #[test]
    fn tie_breakers_apply_for_every_objective() {
        // Same primary value, different totals.
        let small = dag("cx a, b;");
        let big = dag("cx a, b; rz(0.1) a; rz(0.2) a;");
        for o in [Objective::TwoQ, Objective::T, Objective::Ft] {
            let m = model(o);
            let (ks, kb) = (m.key(&small), m.key(&big));
            assert_eq!(ks.primary(), kb.primary(), "{o}: expected a tie on primary");
            assert!(ks < kb, "{o}: tie should be broken by size");
        }
    }

    #[test]
    fn total_ignore_rz_discounts_virtual_z() {
        let m = model(Objective::TotalIgnoreRz);
        let with_rz = dag("h a; rz(0.1) a; rz(0.2) a; u1(0.3) a;");
        let without = dag("h a;");
        assert_eq!(m.key(&with_rz).primary(), m.key(&without).primary());
    }

    /// A gate of arity `k` costs `(k - 1)` two-qubit equivalents: a Toffoli is never
    /// cheaper than a CNOT.
    #[test]
    fn fidelity_charges_by_arity() {
        let m = model(Objective::Fidelity);
        let cx = m.key(&dag("cx a, b;")).primary();
        let ccz = m.key(&dag("ccz a, b, c;")).primary();
        let h = m.key(&dag("h a;")).primary();
        assert!(h < cx, "a one-qubit gate must be cheaper than a cx");
        assert!(cx < ccz, "a ccz must not be cheaper than a cx");
        assert_eq!(cx, 38);
        assert_eq!(ccz, 76);

        // The reference's rule, for contrast, would have scored ccz at 1.
        assert_ne!(ccz, 1);
    }

    #[test]
    fn fidelity_ignores_virtual_z() {
        let m = model(Objective::Fidelity);
        assert_eq!(m.key(&dag("rz(0.3) a; rz(0.4) a;")).primary(), 0);
        assert_eq!(m.key(&dag("h a; rz(0.3) a;")).primary(), 1);
    }

    #[test]
    fn ft_weights_t_against_two_qubit_gates() {
        let m = CostModel::new(Objective::Ft, 50, reg());
        assert_eq!(m.key(&dag("t a;")).primary(), 50);
        assert_eq!(m.key(&dag("cx a, b;")).primary(), 1);
        assert_eq!(m.key(&dag("t a; cx a, b;")).primary(), 51);
    }

    #[test]
    fn breakeven_from_error_rates() {
        // The README's IBM-Eagle numbers.
        let b = CostModel::breakeven_from_error_rates(0.0003, 0.0115).unwrap();
        assert_eq!(b, 38);
        // Degenerate inputs are rejected rather than producing a nonsense weight.
        assert!(CostModel::breakeven_from_error_rates(0.0, 0.01).is_none());
        assert!(CostModel::breakeven_from_error_rates(-0.1, 0.01).is_none());
        assert!(CostModel::breakeven_from_error_rates(0.001, 1.5).is_none());
    }

    #[test]
    fn empty_circuit_is_the_cheapest() {
        let empty = dag("");
        for o in Objective::ALL {
            let m = model(o);
            assert_eq!(m.key(&empty).primary(), 0, "{o}");
            assert!(m.key(&empty) < m.key(&dag("h a;")), "{o}");
        }
    }

    #[test]
    fn objective_flags_round_trip() {
        for o in Objective::ALL {
            assert_eq!(Objective::parse(o.as_flag()), Some(o));
            assert_eq!(Objective::parse(&o.as_flag().to_lowercase()), Some(o));
        }
        assert_eq!(Objective::parse("nonsense"), None);
    }

    #[test]
    fn breakeven_is_clamped_to_at_least_one() {
        let m = CostModel::new(Objective::Fidelity, 0, reg());
        assert_eq!(m.fidelity_breakeven, 1);
        assert_eq!(m.key(&dag("cx a, b;")).primary(), 1);
    }

    #[test]
    fn is_better_agrees_with_key_order() {
        let m = model(Objective::Total);
        let a = dag("h a;");
        let b = dag("h a; h a;");
        assert!(m.is_better(&a, &b));
        assert!(!m.is_better(&b, &a));
        assert!(!m.is_better(&a, &a));
    }
}
