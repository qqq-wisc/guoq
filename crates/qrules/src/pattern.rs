//! Rewrite-rule patterns.

use qcircuit::{qasm, AngleExpr, CircuitError, Dag, GateRegistry, NodeIndex, QubitId};
use rustc_hash::FxHashSet;

use crate::error::{Result, RuleError};

/// One side of a rewrite rule: a small circuit over symbolic qubits, whose gate
/// parameters may contain free angle variables such as `theta1`.
#[derive(Debug, Clone)]
pub struct Pattern {
    dag: Dag,
    /// Gate nodes grouped by connected component, each in BFS order over shared wires.
    ///
    /// Matching walks each component in order, so every gate after a component's first is
    /// adjacent to an already-mapped gate and its position in the circuit is forced. Only
    /// the first gate of each component needs searching.
    ///
    /// A plain rewrite rule must have exactly one component — a disconnected replacement
    /// has no single insertion point. The halves of a *symbolic* rule may have several,
    /// because the hole determines where they go: the real rule
    /// `symb q; h q1; rz((theta1+theta2)) q0;` has an after-half of two gates on
    /// unconnected wires.
    components: Vec<Vec<NodeIndex>>,
    order: Vec<NodeIndex>,
    /// For each gate of the *first* component, a BFS order of that component starting
    /// there.
    ///
    /// The search for a match is linear in the pattern but scans every circuit gate
    /// whose name matches the anchor's, so which gate anchors decides the scan's size:
    /// `rz(theta1) q0; t q0;` anchored on `rz` walks every `rz` in the circuit even when
    /// `t` is four times rarer. Precomputing an order per possible anchor lets the
    /// matcher pick the rarest name *per circuit* at zero per-attempt cost. Any BFS
    /// order from any start preserves the invariant matching relies on — every gate
    /// after the first is adjacent to an earlier one — so the anchor is a free choice.
    ///
    /// Only the first component gets this: later components of a symbolic half are
    /// anchored under qubit bindings that already prune their scans.
    anchor_orders: Vec<(NodeIndex, Vec<NodeIndex>)>,
    /// The original text, kept for diagnostics and rule identity.
    text: String,
}

impl Pattern {
    /// Parse a connected pattern from QASM text such as `cx q0, q1; h q1;`.
    pub fn parse(text: &str) -> Result<Self> {
        let p = Self::parse_multi(text)?;
        if p.components.len() > 1 {
            return Err(RuleError::Disconnected(text.trim().to_string()));
        }
        Ok(p)
    }

    /// Parse a pattern that may have several connected components.
    ///
    /// Used for the halves of a symbolic rule, where the hole fixes the insertion point.
    pub fn parse_multi(text: &str) -> Result<Self> {
        let dag = qasm::parse(text).map_err(RuleError::Parse)?;
        Ok(Self::from_dag(dag, text.trim().to_string()))
    }

    /// Build a pattern from an already-parsed circuit.
    pub fn from_dag(dag: Dag, text: String) -> Self {
        let components = components(&dag);
        let order: Vec<NodeIndex> = components.iter().flatten().copied().collect();
        debug_assert_eq!(order.len(), dag.gate_count());
        let anchor_orders = match components.first() {
            Some(first) => first
                .iter()
                .map(|&start| (start, bfs_from(&dag, start, first)))
                .collect(),
            None => Vec::new(),
        };
        Self {
            dag,
            components,
            order,
            anchor_orders,
            text,
        }
    }

    /// An empty pattern, which matches nothing and inserts nothing.
    pub fn empty() -> Self {
        Self {
            dag: Dag::new(Vec::<String>::new()),
            components: Vec::new(),
            order: Vec::new(),
            anchor_orders: Vec::new(),
            text: String::new(),
        }
    }

    /// The first component's possible anchors, each with a BFS order starting there.
    pub fn anchor_orders(&self) -> &[(NodeIndex, Vec<NodeIndex>)] {
        &self.anchor_orders
    }

    /// The pattern's connected components, each in BFS order.
    pub fn components(&self) -> &[Vec<NodeIndex>] {
        &self.components
    }

    /// `true` if the pattern is a single connected fragment.
    pub fn is_connected(&self) -> bool {
        self.components.len() <= 1
    }

    pub fn dag(&self) -> &Dag {
        &self.dag
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn order(&self) -> &[NodeIndex] {
        &self.order
    }

    pub fn is_empty(&self) -> bool {
        self.dag.gate_count() == 0
    }

    pub fn gate_count(&self) -> usize {
        self.dag.gate_count()
    }

    pub fn qubits(&self) -> &[QubitId] {
        self.dag.qubits()
    }

    /// Every distinct gate-parameter expression appearing in the pattern that contains a
    /// free variable, keyed by its canonical text.
    ///
    /// A whole expression is the unit of binding, not just its variables. The reference
    /// implementation did the same thing (`matchAngles` keyed on `angle.toString()`), and
    /// the shipped rule files depend on it: a rule may write `rz((theta1+theta2))` on one
    /// side and `rz(theta1); rz(theta2)` on the other, and only the latter direction
    /// binds the two variables individually.
    pub fn symbolic_param_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for idx in self.dag.topological_gates() {
            for p in &self.dag.gate(idx).params {
                if !p.free_vars().is_empty() {
                    let k = p.to_string();
                    if !keys.contains(&k) {
                        keys.push(k);
                    }
                }
            }
        }
        keys
    }

    /// Variables the pattern binds *individually*, i.e. those appearing as a parameter
    /// on their own rather than inside a larger expression.
    ///
    /// The distinction decides whether a rule can be reversed. A pattern writing
    /// `rz((theta1+theta2))` constrains only the sum, so a replacement needing `theta1`
    /// alone has nothing to draw on, even though `theta1` does occur in the pattern text.
    pub fn bound_vars(&self) -> Vec<String> {
        let mut vars: Vec<String> = Vec::new();
        for idx in self.dag.topological_gates() {
            for p in &self.dag.gate(idx).params {
                if let AngleExpr::Var(name) = p {
                    if !vars.contains(name) {
                        vars.push(name.clone());
                    }
                }
            }
        }
        vars
    }

    /// Every free angle variable appearing anywhere in the pattern.
    pub fn free_vars(&self) -> Vec<String> {
        let mut vars: Vec<String> = Vec::new();
        for idx in self.dag.topological_gates() {
            for p in &self.dag.gate(idx).params {
                for v in p.free_vars() {
                    if !vars.iter().any(|x| x == v) {
                        vars.push(v.to_string());
                    }
                }
            }
        }
        vars
    }

    /// Check that every gate in the pattern is known to `registry` and well formed.
    pub fn validate(&self, registry: &GateRegistry) -> Result<()> {
        for idx in self.dag.gate_indices() {
            self.dag
                .gate(idx)
                .validate(registry)
                .map_err(|e: CircuitError| RuleError::Parse(e))?;
        }
        Ok(())
    }
}

/// BFS order of `members` starting at `start`, over shared wires.
fn bfs_from(dag: &Dag, start: NodeIndex, members: &[NodeIndex]) -> Vec<NodeIndex> {
    let member_set: FxHashSet<NodeIndex> = members.iter().copied().collect();
    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut queue = std::collections::VecDeque::new();
    let mut out = Vec::with_capacity(members.len());
    queue.push_back(start);
    seen.insert(start);
    while let Some(n) = queue.pop_front() {
        out.push(n);
        for q in dag.gate(n).qubits.clone() {
            for neighbour in [dag.predecessor_on(n, &q), dag.successor_on(n, &q)]
                .into_iter()
                .flatten()
            {
                if member_set.contains(&neighbour) && seen.insert(neighbour) {
                    queue.push_back(neighbour);
                }
            }
        }
    }
    debug_assert_eq!(out.len(), members.len());
    out
}

/// Group gate nodes into connected components, each ordered so that every gate after the
/// component's first shares a wire with an earlier one in that component.
fn components(dag: &Dag) -> Vec<Vec<NodeIndex>> {
    let gates = dag.topological_gates();
    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut out: Vec<Vec<NodeIndex>> = Vec::new();

    for &start in &gates {
        if seen.contains(&start) {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        seen.insert(start);
        while let Some(n) = queue.pop_front() {
            component.push(n);
            for q in dag.gate(n).qubits.clone() {
                for neighbour in [dag.predecessor_on(n, &q), dag.successor_on(n, &q)]
                    .into_iter()
                    .flatten()
                {
                    if seen.insert(neighbour) {
                        queue.push_back(neighbour);
                    }
                }
            }
        }
        out.push(component);
    }
    out
}

/// How a pattern gate's parameter relates to a circuit gate's.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamShape {
    /// Fully determined: compare numerically.
    Concrete(f64),
    /// A symbolic expression, bound as a whole under its canonical text.
    Symbolic(String, AngleExpr),
}

impl ParamShape {
    pub fn of(expr: &AngleExpr) -> Self {
        match expr.eval() {
            Some(v) => ParamShape::Concrete(v),
            None => ParamShape::Symbolic(expr.to_string(), expr.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_connected_pattern() {
        let p = Pattern::parse("cx q0, q1; h q1;").unwrap();
        assert_eq!(p.gate_count(), 2);
        assert_eq!(p.order().len(), 2);
        assert_eq!(p.qubits().len(), 2);
    }

    #[test]
    fn rejects_a_disconnected_pattern() {
        // Two gates on unrelated wires: no single point to splice a replacement in at.
        let err = Pattern::parse("h q0; h q1;").unwrap_err();
        assert!(matches!(err, RuleError::Disconnected(_)));
    }

    /// A symbolic rule's half may be disconnected; the hole fixes where it goes.
    #[test]
    fn multi_component_patterns_parse() {
        let p = Pattern::parse_multi("h q1; rz((theta1+theta2)) q0;").unwrap();
        assert_eq!(p.gate_count(), 2);
        assert_eq!(p.components().len(), 2);
        assert!(!p.is_connected());
        assert_eq!(p.order().len(), 2);

        let c = Pattern::parse_multi("cx q0, q1; h q1;").unwrap();
        assert_eq!(c.components().len(), 1);
        assert!(c.is_connected());
    }

    #[test]
    fn components_partition_the_gates() {
        let p = Pattern::parse_multi("h q0; x q0; cx q1, q2; t q3;").unwrap();
        assert_eq!(p.components().len(), 3);
        let total: usize = p.components().iter().map(Vec::len).sum();
        assert_eq!(total, p.gate_count());
        let mut sizes: Vec<usize> = p.components().iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 1, 2]);
    }

    #[test]
    fn accepts_a_pattern_connected_through_a_shared_wire() {
        let p = Pattern::parse("h q0; cx q0, q1; h q1;").unwrap();
        assert_eq!(p.order().len(), 3);
    }

    #[test]
    fn component_order_starts_adjacent() {
        let p = Pattern::parse("h q0; cx q0, q1; t q1;").unwrap();
        let dag = p.dag();
        let comp = &p.components()[0];
        for (i, &n) in comp.iter().enumerate().skip(1) {
            let earlier = &comp[..i];
            let adjacent = dag.gate(n).qubits.iter().any(|q| {
                [dag.predecessor_on(n, q), dag.successor_on(n, q)]
                    .into_iter()
                    .flatten()
                    .any(|m| earlier.contains(&m))
            });
            assert!(adjacent, "gate {i} is not adjacent to any earlier gate");
        }
    }

    #[test]
    fn empty_pattern() {
        let p = Pattern::empty();
        assert!(p.is_empty());
        assert_eq!(p.gate_count(), 0);
        assert_eq!(p.order().len(), 0);
        // An empty rule side parses to the same thing.
        let q = Pattern::parse("").unwrap();
        assert!(q.is_empty());
    }

    #[test]
    fn symbolic_keys_are_whole_expressions() {
        let p = Pattern::parse("rz(theta1) q0; rz((theta1+theta2)) q0;").unwrap();
        let keys = p.symbolic_param_keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"theta1".to_string()));
        assert!(keys.contains(&"(theta1+theta2)".to_string()));
        assert_eq!(p.free_vars(), vec!["theta1", "theta2"]);
        // Only `theta1` stands alone as a parameter; `theta2` occurs solely inside a sum.
        assert_eq!(p.bound_vars(), vec!["theta1"]);
    }

    #[test]
    fn concrete_params_are_not_symbolic_keys() {
        let p = Pattern::parse("rz(0.5) q0; h q0;").unwrap();
        assert!(p.symbolic_param_keys().is_empty());
        assert!(p.free_vars().is_empty());
    }

    #[test]
    fn param_shape_classifies() {
        assert!(matches!(
            ParamShape::of(&AngleExpr::Num(0.5)),
            ParamShape::Concrete(_)
        ));
        assert!(matches!(
            ParamShape::of(&AngleExpr::var("theta1")),
            ParamShape::Symbolic(_, _)
        ));
        // `pi/4` is concrete even though it is written symbolically.
        let e = AngleExpr::div(AngleExpr::Pi, AngleExpr::Num(4.0));
        assert!(matches!(ParamShape::of(&e), ParamShape::Concrete(_)));
    }

    #[test]
    fn validate_catches_unknown_gates() {
        let p = Pattern::parse("nosuchgate q0; h q0;").unwrap();
        assert!(p.validate(&GateRegistry::with_builtins()).is_err());
        let q = Pattern::parse("cx q0, q1; h q1;").unwrap();
        assert!(q.validate(&GateRegistry::with_builtins()).is_ok());
    }

    #[test]
    fn arbitrary_arity_patterns_are_fine() {
        let p = Pattern::parse("ccz q0, q1, q2; h q2;").unwrap();
        assert_eq!(p.gate_count(), 2);
        assert_eq!(p.qubits().len(), 3);
        assert!(p.validate(&GateRegistry::with_builtins()).is_ok());
    }
}
