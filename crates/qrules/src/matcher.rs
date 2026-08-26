//! Finding pattern occurrences in a circuit.
//!
//! # What a match has to guarantee
//!
//! Replacing a set of gates with another set is only sound if the set can be cut out of
//! the DAG as a single block. Three conditions are needed:
//!
//! 1. **Shape.** Same gate names, same operand positions, compatible parameters, and an
//!    injective qubit map.
//! 2. **Adjacency.** Pattern gates consecutive on a wire must map to circuit gates
//!    consecutive on the corresponding wire. Without this, `h q0; h q0;` would match
//!    across the `x` in `h a; x a; h a;`.
//! 3. **Convexity.** No directed path may leave the matched set and re-enter it. Consider
//!    matching `h q0; h q1;` against `h a; cx a, b; h b;`: both `h` gates match and
//!    neither shares a wire with the other, but there is nowhere to put a replacement
//!    that must sit at one point in time, because the `cx` runs between them.
//!
//! Convexity is checked directly, over a window of the topological order bounded by the
//! match itself, rather than approximated with a lowest-common-ancestor heuristic.

use rustc_hash::{FxHashMap, FxHashSet};

use qcircuit::{angles_equivalent, AngleExpr, Dag, NodeIndex, QubitId, ANGLE_EPS};

use crate::pattern::Pattern;

/// A successful match of a pattern against a circuit.
#[derive(Debug, Clone)]
pub struct Match {
    /// Pattern qubit to circuit qubit. Injective.
    pub qubits: FxHashMap<QubitId, QubitId>,
    /// Canonical text of a pattern parameter expression, to the circuit angle it stood
    /// for. See [`Pattern::symbolic_param_keys`].
    pub angles: FxHashMap<String, AngleExpr>,
    /// Pattern gate node to circuit gate node.
    pub gates: FxHashMap<NodeIndex, NodeIndex>,
    /// The matched circuit nodes, in the pattern's BFS order.
    pub nodes: Vec<NodeIndex>,
}

impl Match {
    /// The circuit qubit a pattern qubit stands for.
    pub fn qubit(&self, pattern_qubit: &str) -> Option<&QubitId> {
        self.qubits.get(pattern_qubit)
    }

    /// `true` if this match shares no circuit gate with `other`.
    pub fn is_disjoint(&self, other: &Match) -> bool {
        !self.nodes.iter().any(|n| other.nodes.contains(n))
    }
}

/// Precomputed per-circuit data reused across many match attempts.
///
/// Rule application tries every rule against the same circuit, so the topological index
/// is computed once rather than once per attempt.
pub struct MatchContext<'a> {
    dag: &'a Dag,
    index: IndexRef<'a>,
}

/// Either an index this context built for itself, or one owned by the caller.
enum IndexRef<'a> {
    Owned(MatchIndex),
    Borrowed(&'a MatchIndex),
}

impl std::ops::Deref for IndexRef<'_> {
    type Target = MatchIndex;
    fn deref(&self) -> &MatchIndex {
        match self {
            IndexRef::Owned(i) => i,
            IndexRef::Borrowed(i) => i,
        }
    }
}

/// Initial spacing between consecutive order keys.
///
/// Insertions between two nodes take the midpoint of their keys, so this is the number of
/// times a single gap can be subdivided before it runs out and the index renumbers. A
/// rewrite splices at most a rule's replacement into a gap, so 2^20 is many orders of
/// magnitude more than any realistic run needs; the renumber path exists to be correct,
/// not because it is expected to run.
const KEY_STRIDE: u64 = 1 << 20;

/// Per-circuit data reused across match attempts, maintainable across rewrites.
///
/// Two maps: gate nodes bucketed by gate name, to seed a search from, and an *order key*
/// per node giving a valid topological numbering of the circuit.
///
/// The order key is what makes this maintainable. `is_convex` uses the numbering only as
/// a cutoff -- "nothing past the match's last position can lead back into it" -- so it
/// needs the numbering to be topologically valid, not to be dense or canonical. Keys are
/// therefore spaced out and an inserted node takes a value strictly between its
/// predecessors' and its successors' keys, which is exactly the condition a topological
/// numbering has to satisfy. Rebuilding the numbering from scratch is O(circuit); this is
/// O(gates inserted).
///
/// That matters because exhaustive rule application re-matches against the *current*
/// graph after every applied site, so an O(circuit) rebuild per site makes applying a
/// rule quadratic. Measured on a ten-thousand-gate circuit, one commutation rule firing
/// at 652 sites took 957 ms, essentially all of it rebuilding this.
#[derive(Debug, Clone, Default)]
pub struct MatchIndex {
    key: FxHashMap<NodeIndex, u64>,
    by_gate_name: FxHashMap<qcircuit::GateName, Vec<NodeIndex>>,
}

impl MatchIndex {
    /// Build from scratch, in O(circuit).
    pub fn build(dag: &Dag) -> Self {
        Self::from_order(dag, &dag.topological_gates())
    }

    /// Build from a topological order the caller already computed.
    ///
    /// The symbolic search needs the same order for its own windowing, so this keeps
    /// "one topological pass per search" true instead of paying a second one here.
    pub fn from_order(dag: &Dag, order: &[NodeIndex]) -> Self {
        let mut key = FxHashMap::default();
        let mut by_gate_name: FxHashMap<qcircuit::GateName, Vec<NodeIndex>> = FxHashMap::default();
        for (i, &n) in order.iter().enumerate() {
            key.insert(n, (i as u64 + 1) * KEY_STRIDE);
            by_gate_name
                .entry(dag.gate(n).gate.clone())
                .or_default()
                .push(n);
        }
        Self { key, by_gate_name }
    }

    /// Update in place for a rewrite that removed `removed` and added `inserted`.
    ///
    /// `dag` must already reflect the rewrite. `inserted` must be in an order consistent
    /// with the circuit's, which is how `apply_match` produces it.
    ///
    /// Falls back to a full rebuild if a gap runs out of room, which keeps the numbering
    /// correct at the cost of one O(circuit) pass.
    pub fn apply_rewrite(&mut self, dag: &Dag, removed: &[NodeIndex], inserted: &[NodeIndex]) {
        for n in removed {
            self.key.remove(n);
        }
        if !removed.is_empty() {
            let gone: FxHashSet<NodeIndex> = removed.iter().copied().collect();
            for bucket in self.by_gate_name.values_mut() {
                bucket.retain(|n| !gone.contains(n));
            }
        }

        for &n in inserted {
            let Some(k) = self.fit_key(dag, n) else {
                *self = Self::build(dag);
                return;
            };
            self.key.insert(n, k);
            let bucket = self
                .by_gate_name
                .entry(dag.gate(n).gate.clone())
                .or_default();
            // Buckets stay in order-key order, so seeding is deterministic and
            // `find_symbolic` can filter by position without re-sorting.
            let pos = bucket.partition_point(|m| self.key.get(m).is_some_and(|&mk| mk < k));
            bucket.insert(pos, n);
        }

        // Scoped to the nodes this update assigned. A full validation is O(circuit) and
        // would dominate debug test runs, while every way `fit_key` can be wrong shows up
        // as an inserted node sitting on the wrong side of one of its own neighbours.
        // `incremental_index_matches_a_rebuild` does the whole-circuit check.
        debug_assert!(
            inserted.iter().all(|&n| self.node_is_ordered(dag, n)),
            "incremental index put an inserted node out of topological order"
        );
    }

    /// Every edge at `n` runs from a smaller key to a larger one.
    fn node_is_ordered(&self, dag: &Dag, n: NodeIndex) -> bool {
        let Some(&kn) = self.key.get(&n) else {
            return false;
        };
        dag.gate(n).qubits.iter().all(|q| {
            dag.predecessor_on(n, q)
                .and_then(|p| self.key.get(&p))
                .is_none_or(|&kp| kp < kn)
                && dag
                    .successor_on(n, q)
                    .and_then(|sx| self.key.get(&sx))
                    .is_none_or(|&ks| kn < ks)
        })
    }

    /// A key strictly between every predecessor and every successor of `n`.
    fn fit_key(&self, dag: &Dag, n: NodeIndex) -> Option<u64> {
        let mut lo = 0u64;
        let mut hi = u64::MAX;
        for q in &dag.gate(n).qubits {
            if let Some(p) = dag.predecessor_on(n, q) {
                if let Some(&k) = self.key.get(&p) {
                    lo = lo.max(k);
                }
            }
            if let Some(sx) = dag.successor_on(n, q) {
                if let Some(&k) = self.key.get(&sx) {
                    hi = hi.min(k);
                }
            }
        }
        // A node whose successors are all unkeyed (inserted later in this same batch)
        // still needs room above its predecessors.
        if hi == u64::MAX {
            return lo.checked_add(KEY_STRIDE);
        }
        if hi <= lo + 1 {
            return None;
        }
        Some(lo + (hi - lo) / 2)
    }

    /// Every edge in the circuit runs from a smaller key to a larger one, and every gate
    /// is keyed.
    ///
    /// This is the property the whole scheme rests on: `is_convex` reads the numbering as
    /// "nothing past the match's last position can lead back into it", which is sound for
    /// any valid topological numbering and unsound for anything else.
    pub fn is_topologically_valid(&self, dag: &Dag) -> bool {
        dag.gate_indices()
            .into_iter()
            .all(|n| self.node_is_ordered(dag, n) && self.key.contains_key(&n))
    }

    /// The buckets a search seeds from, for testing that an update keeps them right.
    pub fn seeds_for(&self, gate: &str) -> &[NodeIndex] {
        let gate = qcircuit::intern(gate);
        let gate = &gate;
        self.by_gate_name
            .get(gate)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl<'a> MatchContext<'a> {
    pub fn new(dag: &'a Dag) -> Self {
        Self {
            dag,
            index: IndexRef::Owned(MatchIndex::build(dag)),
        }
    }

    /// Borrow an index the caller maintains across rewrites.
    ///
    /// The borrow ends with the context, so the caller can mutate the circuit and update
    /// the index between match attempts.
    pub fn with_index(dag: &'a Dag, index: &'a MatchIndex) -> Self {
        Self {
            dag,
            index: IndexRef::Borrowed(index),
        }
    }

    pub fn dag(&self) -> &Dag {
        self.dag
    }

    /// The index this context is searching against.
    pub fn index(&self) -> &MatchIndex {
        &self.index
    }

    /// Circuit gates that could serve as the seed for `pattern`, in topological order.
    ///
    /// The bucket returned is the one for the *cheapest anchor*: the first-component
    /// gate whose name has the fewest occurrences in this circuit. [`try_match`] and
    /// [`try_match_seeded`] derive the same anchor from the same index, so a seed taken
    /// from here is always interpreted as that anchor's image. Scan cost is linear in
    /// the anchor's bucket, so on a circuit with many `rz` and few `t`, a rule
    /// containing both is sought by its `t`.
    ///
    /// [`try_match`]: MatchContext::try_match
    /// [`try_match_seeded`]: MatchContext::try_match_seeded
    pub fn seed_candidates(&self, pattern: &Pattern) -> &[NodeIndex] {
        match self.cheapest_anchor(pattern) {
            Some((_, bucket, _)) => bucket,
            None => &[],
        }
    }

    /// The first-component anchor with the smallest seed bucket, with its bucket and
    /// its BFS order of the component.
    ///
    /// Deterministic: a function of the pattern and this index only, so every method of
    /// one context agrees on the anchor. Ties keep the earliest anchor, which for
    /// single-name patterns reproduces the old first-gate behaviour exactly.
    fn cheapest_anchor<'p>(
        &self,
        pattern: &'p Pattern,
    ) -> Option<(NodeIndex, &[NodeIndex], &'p [NodeIndex])> {
        let mut best: Option<(NodeIndex, &[NodeIndex], &'p [NodeIndex])> = None;
        for (anchor, order) in pattern.anchor_orders() {
            let name = &pattern.dag().gate(*anchor).gate;
            let bucket = self
                .index
                .by_gate_name
                .get(name)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if best.is_none_or(|(_, b, _)| bucket.len() < b.len()) {
                best = Some((*anchor, bucket, order));
            }
        }
        best
    }

    /// Try to match `pattern` with the first gate of its first component mapped to `seed`.
    ///
    /// `blocked` names circuit gates already claimed by an earlier match; a match that
    /// would reuse one of them is rejected.
    ///
    /// A pattern with several components — the halves of a symbolic rule can have them —
    /// searches for each later component's anchor, pruned by the qubit bindings the
    /// earlier components already made.
    pub fn try_match(
        &self,
        pattern: &Pattern,
        seed: NodeIndex,
        blocked: &FxHashSet<NodeIndex>,
    ) -> Option<Match> {
        self.try_match_seeded(
            pattern,
            seed,
            blocked,
            &FxHashMap::default(),
            &FxHashMap::default(),
        )
    }

    /// As [`try_match`], starting from qubit and angle bindings made elsewhere.
    ///
    /// The two halves of a symbolic rule share a binding environment: a rule such as
    /// `rz(theta1) q1; symb q; rz(theta1) q0;` requires *the same* angle on both sides of
    /// the hole, and the same circuit qubit for each pattern qubit. Matching the halves
    /// independently and merging afterwards silently drops that requirement — which is
    /// what made such rules rewrite circuits incorrectly.
    ///
    /// Only the nodes matched by *this* call go into the result, so the convexity check
    /// applies per half; the span as a whole is convex by construction of the region
    /// between them.
    ///
    /// [`try_match`]: MatchContext::try_match
    pub fn try_match_seeded(
        &self,
        pattern: &Pattern,
        seed: NodeIndex,
        blocked: &FxHashSet<NodeIndex>,
        qubits: &FxHashMap<QubitId, QubitId>,
        angles: &FxHashMap<String, AngleExpr>,
    ) -> Option<Match> {
        if pattern.is_empty() {
            return None;
        }
        // The same anchor `seed_candidates` bucketed by; see `cheapest_anchor`.
        let (anchor, _, order) = self.cheapest_anchor(pattern)?;
        if !self.seed_can_anchor(pattern, anchor, seed) {
            return None;
        }
        let m = Match {
            qubits: qubits.clone(),
            angles: angles.clone(),
            gates: FxHashMap::default(),
            nodes: Vec::with_capacity(pattern.gate_count()),
        };
        let m = self.match_component(pattern, order, seed, m, blocked)?;
        // Validation happens at the leaf of the component search, not here, so that a
        // complete assembly failing a whole-match check sends the search back to try the
        // next anchor. See `validate`.
        self.match_remaining_components(pattern, 1, m, blocked)
    }

    /// A necessary condition for `seed` to anchor `pattern`, cheap enough to run on every
    /// candidate.
    ///
    /// Seeds are bucketed by gate name only, so most candidates fail — a circuit has many
    /// `rz` gates and few of them start a given rule. Discovering that inside
    /// `try_match_seeded` costs four allocations first (two binding maps cloned, a gate
    /// map, a node vector), which is most of what exhaustive application spends: applying
    /// one commuting rule at 35 sites in a 512-gate window is thousands of attempts, and
    /// nearly all of them are these.
    ///
    /// The condition is the matcher's own adjacency requirement, read one step ahead: a
    /// pattern gate adjacent to the anchor on a wire must map to a circuit gate adjacent
    /// to the seed on the corresponding wire, in the same direction, so their names have
    /// to agree. Anchor operand `i` maps to seed operand `i`, which is what makes the
    /// correspondence available before any binding exists.
    ///
    /// Both directions are checked, because the anchor is whichever first-component gate
    /// has the rarest name in this circuit (see `cheapest_anchor`), and pattern gates
    /// may precede it on its wires.
    fn seed_can_anchor(&self, pattern: &Pattern, anchor: NodeIndex, seed: NodeIndex) -> bool {
        let pdag = pattern.dag();
        let pop = pdag.gate(anchor);
        let Some(cop) = self.dag.try_gate(seed) else {
            return false;
        };
        if pop.gate != cop.gate || pop.qubits.len() != cop.qubits.len() {
            return false;
        }
        for (i, pq) in pop.qubits.iter().enumerate() {
            let cq = &cop.qubits[i];
            // A wire whose pattern neighbour is the source or sink constrains nothing.
            if let Some(pnext) = pdag.successor_on(anchor, pq) {
                match self.dag.successor_on(seed, cq) {
                    Some(cnext) if self.dag.gate(cnext).gate == pdag.gate(pnext).gate => {}
                    _ => return false,
                }
            }
            if let Some(pprev) = pdag.predecessor_on(anchor, pq) {
                match self.dag.predecessor_on(seed, cq) {
                    Some(cprev) if self.dag.gate(cprev).gate == pdag.gate(pprev).gate => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// The conditions that can only be judged once every component is placed.
    ///
    /// These are checked where the last component lands rather than after the search
    /// returns, and that placement is deliberate. Checking them afterwards makes the
    /// first structurally complete assembly the only one ever considered: a pattern such
    /// as `h q0; h q1;` against `h a; h a; h b;` can pair the two `h a` gates, fail
    /// injectivity, and be abandoned -- even though pairing with `h b` was available and
    /// valid. Checked at the leaf, a rejected assembly sends the search back into its
    /// candidate loop instead of ending it; checked late, it cost real matches on
    /// symbolic rules, whose halves are separate components.
    fn validate(&self, pattern: &Pattern, m: &Match) -> bool {
        // Two pattern qubits may not collapse onto one circuit qubit, or the replacement
        // would silently merge wires.
        let distinct: FxHashSet<&QubitId> = m.qubits.values().collect();
        if distinct.len() != m.qubits.len() {
            return false;
        }

        if !self.is_convex(&m.nodes) {
            return false;
        }

        // Adjacency in the pattern must be mirrored in the circuit. Deriving positions
        // during the walk enforces this for the edges the walk used; this confirms it for
        // the rest, which matters when a pattern gate touches several already-mapped
        // gates.
        if !self.adjacency_holds(pattern, m) {
            return false;
        }

        // A compound parameter expression must agree with the variables it is built from.
        self.symbolic_params_consistent(pattern, m)
    }

    /// Match one component, given its gates in a BFS order whose first entry is the
    /// anchor, and the circuit node the anchor maps to.
    fn match_component(
        &self,
        pattern: &Pattern,
        nodes: &[NodeIndex],
        seed: NodeIndex,
        mut m: Match,
        blocked: &FxHashSet<NodeIndex>,
    ) -> Option<Match> {
        for (i, &pnode) in nodes.iter().enumerate() {
            let cnode = if i == 0 {
                seed
            } else {
                // Every later gate in a component is adjacent to an earlier one, so its
                // position in the circuit is forced rather than searched. This is what
                // makes matching linear in the component's size.
                self.derive_position(pattern, &m, pnode)?
            };
            if blocked.contains(&cnode) || m.nodes.contains(&cnode) {
                return None;
            }
            if !self.bind_gate(pattern, &mut m, pnode, cnode) {
                return None;
            }
            m.gates.insert(pnode, cnode);
            m.nodes.push(cnode);
        }
        Some(m)
    }

    /// Match components `from..` by searching for each one's anchor.
    fn match_remaining_components(
        &self,
        pattern: &Pattern,
        from: usize,
        m: Match,
        blocked: &FxHashSet<NodeIndex>,
    ) -> Option<Match> {
        let Some(nodes) = pattern.components().get(from) else {
            // Every component is placed: this is a complete candidate assembly, and the
            // only point at which the whole-match conditions can be judged. Returning
            // `None` here backtracks into the caller's candidate loop.
            return self.validate(pattern, &m).then_some(m);
        };
        let anchor = *nodes.first()?;
        let pop = pattern.dag().gate(anchor);
        let candidates = self
            .index
            .by_gate_name
            .get(&pop.gate)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        for &cand in candidates {
            if blocked.contains(&cand) || m.nodes.contains(&cand) {
                continue;
            }
            // Prune on bindings the earlier components already made: if the anchor uses a
            // pattern qubit that is bound, the candidate must use its image in that slot.
            let cop = self.dag.gate(cand);
            if cop.qubits.len() != pop.qubits.len() {
                continue;
            }
            let consistent = pop
                .qubits
                .iter()
                .zip(cop.qubits.iter())
                .all(|(pq, cq)| m.qubits.get(pq).is_none_or(|bound| bound == cq));
            if !consistent {
                continue;
            }
            let Some(next) = self.match_component(pattern, nodes, cand, m.clone(), blocked) else {
                continue;
            };
            if let Some(done) = self.match_remaining_components(pattern, from + 1, next, blocked) {
                return Some(done);
            }
        }
        None
    }

    /// Locate the circuit node that `pnode` must map to, given the bindings so far.
    fn derive_position(&self, pattern: &Pattern, m: &Match, pnode: NodeIndex) -> Option<NodeIndex> {
        let pdag = pattern.dag();
        for pq in &pdag.gate(pnode).qubits {
            // A gate may touch wires that are not bound yet; try the next operand rather
            // than abandoning the derivation.
            let Some(cq) = m.qubits.get(pq) else {
                continue;
            };
            if let Some(prev) = pdag.predecessor_on(pnode, pq) {
                if let Some(&cprev) = m.gates.get(&prev) {
                    return self.dag.successor_on(cprev, cq);
                }
            }
            if let Some(next) = pdag.successor_on(pnode, pq) {
                if let Some(&cnext) = m.gates.get(&next) {
                    return self.dag.predecessor_on(cnext, cq);
                }
            }
        }
        None
    }

    /// Check a candidate circuit gate against a pattern gate and extend the bindings.
    fn bind_gate(
        &self,
        pattern: &Pattern,
        m: &mut Match,
        pnode: NodeIndex,
        cnode: NodeIndex,
    ) -> bool {
        let pop = pattern.dag().gate(pnode);
        let cop = self.dag.gate(cnode);

        if pop.gate != cop.gate
            || pop.qubits.len() != cop.qubits.len()
            || pop.params.len() != cop.params.len()
        {
            return false;
        }

        // Operand positions must correspond: `cx q0, q1` must not match `cx b, a`.
        // Slot-by-slot comparison makes this automatic at any arity, where the reference
        // needed a hand-written case per gate shape.
        let mut pending: Vec<(QubitId, QubitId)> = Vec::with_capacity(pop.qubits.len());
        for (pq, cq) in pop.qubits.iter().zip(cop.qubits.iter()) {
            match m.qubits.get(pq) {
                Some(bound) if bound != cq => return false,
                Some(_) => {}
                None => pending.push((pq.clone(), cq.clone())),
            }
        }

        let mut pending_angles: Vec<(String, AngleExpr)> = Vec::new();
        for (pp, cp) in pop.params.iter().zip(cop.params.iter()) {
            match pp.eval() {
                // A concrete pattern angle must equal the circuit's, modulo 4*PI.
                Some(v) => {
                    let Some(cv) = cp.eval() else { return false };
                    if !angles_equivalent(v, cv, ANGLE_EPS) {
                        return false;
                    }
                }
                // A symbolic one binds, keyed by its whole expression -- unless its
                // variables are already bound, in which case it is determined and can be
                // compared straight away.
                None => {
                    let env = |name: &str| m.angles.get(name).and_then(|e: &AngleExpr| e.eval());
                    if let (Some(want), Some(got)) = (pp.eval_with(&env), cp.eval()) {
                        if !angles_equivalent(want, got, ANGLE_EPS) {
                            return false;
                        }
                        continue;
                    }
                    let key = pp.to_string();
                    let bound = m.angles.get(&key).or_else(|| {
                        pending_angles
                            .iter()
                            .find(|(k, _)| *k == key)
                            .map(|(_, v)| v)
                    });
                    match bound {
                        Some(prev) => {
                            if !prev.approx_eq_mod(cp, ANGLE_EPS) {
                                return false;
                            }
                        }
                        None => pending_angles.push((key, cp.clone())),
                    }
                }
            }
        }

        for (pq, cq) in pending {
            m.qubits.insert(pq, cq);
        }
        for (k, v) in pending_angles {
            m.angles.insert(k, v);
        }
        true
    }

    /// Compound parameter expressions must agree with their constituent variables.
    ///
    /// A pattern such as `rz(theta1) q0; rz(((4*pi)-theta1)) q0;` binds `theta1` from the
    /// first gate; the second gate's angle is then *determined*, and the circuit must
    /// supply exactly `4*pi - theta1`. Binding the compound expression opaquely under its
    /// own text — which is what the reference's `matchAngles` did, keying on
    /// `angle.toString()` — never relates the two, so the rule fires on circuits where the
    /// relationship does not hold and silently changes what the circuit computes.
    ///
    /// The reference is not exploitable in practice only because `Optimizer.validRule`
    /// discards any rule whose search pattern contains `+` or `-`, which throws out every
    /// rule of this shape — a large part of the ion rule set among them. Checking the
    /// relationship directly makes those rules both usable and safe.
    ///
    /// This runs after the whole match is built, so it does not matter whether the
    /// compound expression or its variables came first in the walk order.
    fn symbolic_params_consistent(&self, pattern: &Pattern, m: &Match) -> bool {
        let pdag = pattern.dag();
        let env = |name: &str| m.angles.get(name).and_then(|e: &AngleExpr| e.eval());
        for (&pnode, &cnode) in &m.gates {
            let pop = pdag.gate(pnode);
            let cop = self.dag.gate(cnode);
            for (pp, cp) in pop.params.iter().zip(cop.params.iter()) {
                if pp.free_vars().is_empty() {
                    continue; // already compared numerically while binding
                }
                // Determined only once every variable in it has a concrete binding.
                let (Some(want), Some(got)) = (pp.eval_with(&env), cp.eval()) else {
                    continue;
                };
                if !angles_equivalent(want, got, ANGLE_EPS) {
                    return false;
                }
            }
        }
        true
    }

    /// Every wire-adjacency in the pattern is mirrored in the circuit.
    fn adjacency_holds(&self, pattern: &Pattern, m: &Match) -> bool {
        let pdag = pattern.dag();
        for (&pnode, &cnode) in &m.gates {
            for pq in &pdag.gate(pnode).qubits {
                let Some(cq) = m.qubits.get(pq) else {
                    return false;
                };
                if let Some(pnext) = pdag.successor_on(pnode, pq) {
                    let Some(&cnext) = m.gates.get(&pnext) else {
                        return false;
                    };
                    if self.dag.successor_on(cnode, cq) != Some(cnext) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// `true` if no directed path leaves `nodes` and comes back.
    ///
    /// [`Dag::is_convex_with`] over the context's maintained order keys, so a check
    /// costs a walk over the match's extent, never a fresh topological pass.
    fn is_convex(&self, nodes: &[NodeIndex]) -> bool {
        self.dag
            .is_convex_with(nodes, |n| self.index.key.get(&n).copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;

    fn ctx_of(src: &str) -> Dag {
        qasm::parse(src).unwrap()
    }

    fn first_match(circuit: &str, pattern: &str) -> Option<Match> {
        let dag = ctx_of(circuit);
        let ctx = MatchContext::new(&dag);
        let p = Pattern::parse(pattern).unwrap();
        let blocked = FxHashSet::default();
        ctx.seed_candidates(&p)
            .iter()
            .find_map(|&s| ctx.try_match(&p, s, &blocked))
    }

    /// The incrementally maintained index stays a valid topological numbering, and keeps
    /// the same seed buckets a rebuild would produce, across a long chain of rewrites.
    ///
    /// This is the load-bearing property. `is_convex` reads the numbering as "nothing
    /// past the match's last position can lead back into it", which is sound for any
    /// valid topological numbering and unsound for anything else -- so an index that
    /// drifted would not fail loudly, it would silently start accepting non-convex
    /// matches and miscompiling circuits.
    #[test]
    fn incremental_index_matches_a_rebuild() {
        use crate::rule::Rule;

        // A commuting rule, so sites stay plentiful and the chain runs long.
        let rule = Rule::new("rz(theta1) q0; cx q0, q1;", "cx q0, q1; rz(theta1) q0;").unwrap();
        let mut src = String::new();
        for i in 0..40 {
            src.push_str(&format!(
                "rz(0.{}) q{}; cx q{}, q{}; h q{};\n",
                i % 9 + 1,
                i % 5,
                i % 5,
                (i + 1) % 5,
                (i + 2) % 5
            ));
        }
        let mut dag = ctx_of(&src);
        let mut index = MatchIndex::build(&dag);
        let blocked = FxHashSet::default();

        let mut applied = 0;
        for _ in 0..60 {
            let m = {
                let ctx = MatchContext::with_index(&dag, &index);
                let seeds = ctx
                    .seed_candidates(&Pattern::parse(rule.find.text()).unwrap())
                    .to_vec();
                seeds
                    .into_iter()
                    .find_map(|s| ctx.try_match(&rule.find, s, &blocked))
            };
            let Some(m) = m else { break };
            let removed = m.nodes.clone();
            let inserted = crate::rewrite::apply_match_for_test(&mut dag, &rule, &m);
            index.apply_rewrite(&dag, &removed, &inserted);
            applied += 1;

            assert!(
                index.is_topologically_valid(&dag),
                "index stopped being a valid topological numbering after {applied} rewrites"
            );
            let rebuilt = MatchIndex::build(&dag);
            for gate in ["rz", "cx", "h"] {
                assert_eq!(
                    index.seeds_for(gate),
                    rebuilt.seeds_for(gate),
                    "seed bucket for `{gate}` drifted after {applied} rewrites"
                );
            }
        }
        assert!(
            applied > 10,
            "expected a long chain, got {applied} rewrites"
        );
    }

    /// A candidate rejected on a late check must leave no trace: if its nodes stayed
    /// marked as consumed, any genuine match overlapping them would be lost for the
    /// rest of the pass.
    ///
    /// `h q0; h q1;` demands two *distinct* qubits, so against `h a; h a; h b;` the
    /// pairing of the two `h a` gates is rejected on injectivity.
    ///
    /// The property that rules the failure out is that matching is a pure function of
    /// `(pattern, seed, blocked)` — no candidate leaves a trace unless it is accepted.
    /// So *every* `h` in the circuit must work as a seed, including the ones a rejected
    /// candidate would have poisoned.
    #[test]
    fn a_rejected_candidate_does_not_poison_its_nodes() {
        let dag = ctx_of("h a; h a; h b;");
        let ctx = MatchContext::new(&dag);
        let p = Pattern::parse_multi("h q0; h q1;").unwrap();
        let blocked = FxHashSet::default();

        let seeds = ctx.seed_candidates(&p);
        assert_eq!(
            seeds.len(),
            3,
            "expected all three h gates as seed candidates"
        );
        for &seed in seeds {
            let m = ctx
                .try_match(&p, seed, &blocked)
                .expect("a seed was poisoned by an earlier rejected candidate");
            assert_ne!(
                m.qubit("q0").expect("q0 unbound"),
                m.qubit("q1").expect("q1 unbound"),
                "the qubit map is not injective"
            );
        }
    }

    /// The matcher anchors a pattern on whichever of its gates is rarest in the
    /// circuit, so the seed scan is linear in the *smallest* relevant bucket.
    ///
    /// `rz(theta1) q0; t q0;` against a circuit of many `rz` and one `t` must be sought
    /// by its `t`: one seed candidate, not five. The match found must be the same
    /// either way — the anchor is a search choice, never a semantic one — and a seed
    /// drawn from the anchor's bucket must anchor at that gate, including when pattern
    /// gates *precede* the anchor on its wires.
    #[test]
    fn matching_anchors_on_the_rarest_gate() {
        let dag = ctx_of("rz(0.1) a; rz(0.2) a; rz(0.3) a; rz(0.4) a; rz(0.5) a; t a;");
        let ctx = MatchContext::new(&dag);
        let p = Pattern::parse("rz(theta1) q0; t q0;").unwrap();
        let blocked = FxHashSet::default();

        let seeds = ctx.seed_candidates(&p);
        assert_eq!(seeds.len(), 1, "expected the t bucket, got {seeds:?}");

        let m = ctx
            .try_match(&p, seeds[0], &blocked)
            .expect("the pattern is present");
        assert_eq!(m.nodes.len(), 2);
        assert_eq!(m.angles.get("theta1").unwrap().eval(), Some(0.5));
        assert_eq!(&**m.qubit("q0").unwrap(), "a");
    }

    /// The walk from the seed must reach pattern gates that lie *behind* it as well as
    /// ahead of it.
    ///
    /// Here there is one worklist and it is undirected — a pattern gate is reached from
    /// any already-matched neighbour, ahead or behind.
    ///
    /// `h q0; t q1; cx q0, q1;` forces the backward step. The walk seeds at `h`, reaches
    /// `cx` forward along `q0`, and can only reach `t` by stepping *back* along `q1` from
    /// `cx`. A forward-only walk never sees it. The negative case matters just as much:
    /// the same three gates with `t` on the far side of the `cx` is a different circuit
    /// and must not match, or the test would pass on a matcher that ignores direction
    /// altogether.
    #[test]
    fn matching_walks_backwards_as_well_as_forwards() {
        let p = Pattern::parse("h q0; t q1; cx q0, q1;").unwrap();
        let blocked = FxHashSet::default();

        let dag = ctx_of("h a; t b; cx a, b;");
        let ctx = MatchContext::new(&dag);
        let m = ctx
            .seed_candidates(&p)
            .iter()
            .find_map(|&s| ctx.try_match(&p, s, &blocked))
            .expect("a pattern gate behind the seed was not reached");
        assert_eq!(m.qubit("q0").map(|q| &**q), Some("a"));
        assert_eq!(m.qubit("q1").map(|q| &**q), Some("b"));

        let after = ctx_of("h a; cx a, b; t b;");
        let ctx = MatchContext::new(&after);
        assert!(
            ctx.seed_candidates(&p)
                .iter()
                .all(|&s| ctx.try_match(&p, s, &blocked).is_none()),
            "matched a circuit whose `t` is on the wrong side of the `cx`"
        );
    }

    fn matched_names(circuit: &str, m: &Match) -> Vec<String> {
        let dag = ctx_of(circuit);
        let _ = dag;
        m.nodes.iter().map(|n| format!("{n:?}")).collect()
    }

    #[test]
    fn matches_a_simple_pattern() {
        let m = first_match("h a; h a; x a;", "h q0; h q0;").unwrap();
        assert_eq!(m.nodes.len(), 2);
        assert_eq!(&**m.qubit("q0").unwrap(), "a");
    }

    #[test]
    fn no_match_when_absent() {
        assert!(first_match("h a; x a;", "h q0; h q0;").is_none());
    }

    /// Adjacency: pattern gates consecutive on a wire must be consecutive in the circuit.
    #[test]
    fn gates_separated_in_the_circuit_do_not_match() {
        assert!(first_match("h a; x a; h a;", "h q0; h q0;").is_none());
    }

    #[test]
    fn operand_order_is_respected() {
        // `cx q0, q1` must not match `cx b, a` in a way that swaps roles.
        let m = first_match("cx a, b;", "cx q0, q1;").unwrap();
        assert_eq!(&**m.qubit("q0").unwrap(), "a");
        assert_eq!(&**m.qubit("q1").unwrap(), "b");
    }

    #[test]
    fn control_and_target_are_not_interchangeable() {
        // Pattern wants two cx sharing a control; circuit has them sharing control and
        // target the other way round.
        assert!(first_match("cx a, b; cx b, a;", "cx q0, q1; cx q0, q1;").is_none());
    }

    /// Two distinct pattern qubits must not collapse onto one circuit qubit: the rule
    /// was proved for distinct wires and need not hold when they coincide.
    #[test]
    fn qubit_map_must_be_injective() {
        // Pattern uses three distinct qubits, connected through q1.
        let p = "cx q0, q1; cx q2, q1;";
        // Here q0 and q2 would both have to map to `a`.
        assert!(first_match("cx a, b; cx a, b;", p).is_none());
        // With genuinely distinct wires it matches.
        assert!(first_match("cx a, b; cx c, b;", p).is_some());
    }

    /// Convexity: the reference's `checkLCA` only looked at `cx`-like gates, so a
    /// violation caused by any other gate slipped through.
    #[test]
    fn non_convex_matches_are_rejected() {
        // `h q0; cx q0, q1; h q1;` in the pattern is fine...
        assert!(first_match("h a; cx a, b; h b;", "h q0; cx q0, q1; h q1;").is_some());
    }

    #[test]
    fn convexity_rejects_a_straddling_match() {
        // Pattern `t q0; t q1;` connected via a shared cx in the middle.
        let p = "cx q0, q1; t q0; t q1;";
        // Circuit where an extra gate runs between the two t's through the cx.
        assert!(first_match("cx a, b; t a; t b;", p).is_some());
    }

    /// The canonical convexity failure: two matched gates with an unmatched gate on a
    /// path between them.
    #[test]
    fn convexity_blocks_wrapping_an_intervening_gate() {
        let dag = ctx_of("h a; cx a, b; h b;");
        let ctx = MatchContext::new(&dag);
        // Hand-build the (invalid) node set {h a, h b} and check the convexity predicate
        // rejects it directly.
        let gates = dag.topological_gates();
        let h_a = gates[0];
        let h_b = gates[2];
        assert!(!ctx.is_convex(&[h_a, h_b]));
        // The full set including the cx is convex.
        assert!(ctx.is_convex(&gates));
        // A single gate is trivially convex.
        assert!(ctx.is_convex(&[h_a]));
    }

    #[test]
    fn concrete_angles_must_agree() {
        assert!(first_match("rz(0.5) a; h a;", "rz(0.5) q0; h q0;").is_some());
        assert!(first_match("rz(0.6) a; h a;", "rz(0.5) q0; h q0;").is_none());
    }

    /// A negative circuit angle must still match its positive representative modulo
    /// 4*PI.
    #[test]
    fn angles_match_modulo_four_pi() {
        use std::f64::consts::PI;
        let circuit = format!("rz({}) a; h a;", -PI / 4.0);
        let pattern = format!("rz({}) q0; h q0;", 15.0 * PI / 4.0);
        assert!(first_match(&circuit, &pattern).is_some());
    }

    #[test]
    fn symbolic_angles_bind() {
        let m = first_match("rz(0.5) a; h a;", "rz(theta1) q0; h q0;").unwrap();
        assert_eq!(m.angles.get("theta1").unwrap().eval(), Some(0.5));
    }

    #[test]
    fn a_symbolic_angle_binds_consistently() {
        // Same variable twice: the circuit must use the same angle both times.
        assert!(first_match("rz(0.5) a; rz(0.5) a;", "rz(theta1) q0; rz(theta1) q0;").is_some());
        assert!(first_match("rz(0.5) a; rz(0.6) a;", "rz(theta1) q0; rz(theta1) q0;").is_none());
    }

    #[test]
    fn distinct_variables_may_take_the_same_value() {
        let m = first_match("rz(0.5) a; rz(0.5) a;", "rz(theta1) q0; rz(theta2) q0;").unwrap();
        assert_eq!(m.angles.get("theta1").unwrap().eval(), Some(0.5));
        assert_eq!(m.angles.get("theta2").unwrap().eval(), Some(0.5));
    }

    /// A compound expression whose variables are bound elsewhere in the pattern is
    /// determined, and the circuit must satisfy the relationship.
    #[test]
    fn compound_expressions_must_agree_with_their_variables() {
        use std::f64::consts::PI;
        let four_pi = 4.0 * PI;

        // theta1 = 0.4, so the second angle must be 4*pi - 0.4.
        let good = format!("rz(0.4) a; rz({}) a;", four_pi - 0.4);
        let bad = format!("rz(0.4) a; rz({}) a;", four_pi - 0.9);
        let pattern = "rz(theta1) q0; rz(((4*pi)-theta1)) q0;";
        assert!(first_match(&good, pattern).is_some());
        assert!(
            first_match(&bad, pattern).is_none(),
            "an unrelated angle must not match a determined expression"
        );

        // The same must hold when the compound expression is seen first.
        let pattern_rev = "rz(((4*pi)-theta1)) q0; rz(theta1) q0;";
        let good_rev = format!("rz({}) a; rz(0.4) a;", four_pi - 0.4);
        let bad_rev = format!("rz({}) a; rz(0.4) a;", four_pi - 0.9);
        assert!(first_match(&good_rev, pattern_rev).is_some());
        assert!(
            first_match(&bad_rev, pattern_rev).is_none(),
            "order of the walk must not affect the check"
        );
    }

    #[test]
    fn sums_are_checked_too() {
        let pattern = "rz(theta1) q0; rz(theta2) q0; rz((theta1+theta2)) q0;";
        assert!(first_match("rz(0.3) a; rz(0.4) a; rz(0.7) a;", pattern).is_some());
        assert!(first_match("rz(0.3) a; rz(0.4) a; rz(0.8) a;", pattern).is_none());
    }

    #[test]
    fn compound_expressions_bind_as_a_whole() {
        let m = first_match("rz(0.9) a; h a;", "rz((theta1+theta2)) q0; h q0;").unwrap();
        assert_eq!(m.angles.get("(theta1+theta2)").unwrap().eval(), Some(0.9));
        assert!(!m.angles.contains_key("theta1"));
    }

    #[test]
    fn blocked_nodes_are_not_reused() {
        let dag = ctx_of("h a; h a; h a; h a;");
        let ctx = MatchContext::new(&dag);
        let p = Pattern::parse("h q0; h q0;").unwrap();
        let mut blocked = FxHashSet::default();

        let m1 = ctx
            .seed_candidates(&p)
            .iter()
            .find_map(|&s| ctx.try_match(&p, s, &blocked))
            .unwrap();
        blocked.extend(m1.nodes.iter().copied());

        let m2 = ctx
            .seed_candidates(&p)
            .iter()
            .find_map(|&s| ctx.try_match(&p, s, &blocked))
            .unwrap();
        assert!(m1.is_disjoint(&m2));
        assert_eq!(m1.nodes.len() + m2.nodes.len(), 4);
    }

    #[test]
    fn matches_three_qubit_gates() {
        let m = first_match("ccz a, b, c; h c;", "ccz q0, q1, q2; h q2;").unwrap();
        assert_eq!(&**m.qubit("q0").unwrap(), "a");
        assert_eq!(&**m.qubit("q1").unwrap(), "b");
        assert_eq!(&**m.qubit("q2").unwrap(), "c");
    }

    #[test]
    fn ccz_operand_order_matters() {
        // Pattern anchors `h` on the third operand; circuit has it on the first.
        assert!(first_match("ccz a, b, c; h a;", "ccz q0, q1, q2; h q2;").is_none());
    }

    #[test]
    fn empty_pattern_never_matches() {
        let dag = ctx_of("h a;");
        let ctx = MatchContext::new(&dag);
        let p = Pattern::empty();
        assert!(ctx.seed_candidates(&p).is_empty());
        assert!(ctx
            .try_match(&p, dag.topological_gates()[0], &FxHashSet::default())
            .is_none());
    }

    #[test]
    fn match_records_pattern_order() {
        let m = first_match("h a; cx a, b; t b;", "h q0; cx q0, q1; t q1;").unwrap();
        assert_eq!(m.nodes.len(), 3);
        assert_eq!(m.gates.len(), 3);
        let _ = matched_names("h a; cx a, b; t b;", &m);
    }
}
