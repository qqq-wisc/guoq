//! The circuit DAG.
//!
//! Every qubit has a source node and a sink node, and the gates on that qubit form a
//! path between them. That invariant holds at all times: there is no
//! partially built state, no second construction phase, and no side structures for a
//! clone to forget.

use petgraph::stable_graph::{EdgeIndex, StableDiGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::Direction;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::{Hash, Hasher};

use crate::angle::AngleExpr;
use crate::error::{CircuitError, Result};
use crate::gate::GateRegistry;

/// An opaque handle to a node in a [`Dag`].
///
/// Re-exported from `qcircuit` rather than taken from `petgraph` at each use site, so
/// that the graph library is an implementation detail of this module. Callers treat it
/// as an opaque, copyable, hashable token; nothing outside `dag.rs` depends on how a
/// node is addressed, which is what makes the representation replaceable.
pub type NodeIndex = petgraph::stable_graph::NodeIndex;

/// A qubit name, e.g. `q[3]` in a circuit or `q0` in a rewrite rule.
///
/// Shared rather than owned, because names are copied far more often than they are
/// created. Cloning a circuit clones every gate's operand list, and at ten thousand gates
/// that was tens of thousands of string allocations — 57% of the cost of `Dag::clone`,
/// which the search pays on every rule that fires. A shared pointer makes the copy a
/// refcount bump.
pub type QubitId = std::sync::Arc<str>;

/// A gate name. Shared for the same reason as [`QubitId`].
pub type GateName = std::sync::Arc<str>;

/// Return the shared name for `s`, allocating only the first time it is seen.
///
/// A circuit has a handful of distinct qubit names and gate names but millions of
/// *references* to them, so interning keeps one allocation per distinct name for the
/// whole process. The table is thread-local, so two threads may hold separate copies of
/// the same name; that costs one extra allocation and nothing else, since equality is by
/// value.
pub fn intern(s: &str) -> std::sync::Arc<str> {
    use std::sync::Arc;
    thread_local! {
        static TABLE: std::cell::RefCell<FxHashMap<Arc<str>, Arc<str>>> =
            std::cell::RefCell::new(FxHashMap::default());
    }
    TABLE.with(|t| {
        if let Some(hit) = t.borrow().get(s) {
            return hit.clone();
        }
        let owned: Arc<str> = Arc::from(s);
        t.borrow_mut().insert(owned.clone(), owned.clone());
        owned
    })
}

/// A gate applied to specific qubits with specific parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct GateOp {
    pub gate: GateName,
    pub qubits: Vec<QubitId>,
    pub params: Vec<AngleExpr>,
}

impl GateOp {
    pub fn new(
        gate: impl AsRef<str>,
        qubits: impl IntoIterator<Item = impl AsRef<str>>,
        params: impl IntoIterator<Item = AngleExpr>,
    ) -> Self {
        Self {
            gate: intern(gate.as_ref()),
            qubits: qubits.into_iter().map(|q| intern(q.as_ref())).collect(),
            params: params.into_iter().collect(),
        }
    }

    /// The operand index of `qubit` in this gate, or `None` if it is not an operand.
    pub fn slot_of(&self, qubit: &str) -> Option<usize> {
        self.qubits.iter().position(|q| &**q == qubit)
    }

    /// Validate the op against a registry: known gate, right arity, right parameter
    /// count, no repeated qubit.
    pub fn validate(&self, registry: &GateRegistry) -> Result<()> {
        let def = registry.get(&self.gate)?;
        if self.qubits.len() != def.arity {
            return Err(CircuitError::Arity {
                gate: self.gate.to_string(),
                expected: def.arity,
                got: self.qubits.len(),
            });
        }
        if self.params.len() != def.num_params {
            return Err(CircuitError::ParamCount {
                gate: self.gate.to_string(),
                expected: def.num_params,
                got: self.params.len(),
            });
        }
        for (i, q) in self.qubits.iter().enumerate() {
            if self.qubits[..i].contains(q) {
                return Err(CircuitError::DuplicateQubit {
                    gate: self.gate.to_string(),
                    qubit: q.to_string(),
                });
            }
        }
        Ok(())
    }
}

/// A DAG vertex.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Gate(GateOp),
    /// The input end of a qubit wire.
    Source(QubitId),
    /// The output end of a qubit wire.
    Sink(QubitId),
}

impl Node {
    pub fn as_gate(&self) -> Option<&GateOp> {
        match self {
            Node::Gate(op) => Some(op),
            _ => None,
        }
    }

    pub fn is_gate(&self) -> bool {
        matches!(self, Node::Gate(_))
    }

    /// The operand index of `qubit` in this node. Source and sink nodes carry exactly one
    /// wire, at slot 0.
    fn slot_of(&self, qubit: &str) -> Option<usize> {
        match self {
            Node::Gate(op) => op.slot_of(qubit),
            Node::Source(q) | Node::Sink(q) => (&**q == qubit).then_some(0),
        }
    }
}

/// A DAG edge: one qubit wire running between two nodes.
///
/// `src_slot` and `dst_slot` are the wire's **operand indices** in the source and target
/// gates. This replaces the reference implementation's
/// `Edge.Label::{CONTROL, CONTROL2, TARGET, NONE}` enum, which could only describe gates
/// of arity at most three and needed a new variant per additional qubit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    pub qubit: QubitId,
    pub src_slot: usize,
    pub dst_slot: usize,
}

/// Everything preceding the gate list in a QASM file, preserved for round-tripping.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct QasmHeader {
    pub version: Option<String>,
    pub includes: Vec<String>,
    /// Quantum registers, as `(name, size)`, in declaration order.
    pub qregs: Vec<(String, usize)>,
    /// Classical registers, preserved verbatim so measurement-bearing files round-trip.
    pub cregs: Vec<(String, usize)>,
}

/// A quantum circuit as a directed acyclic graph over gate applications.
#[derive(Debug, Clone)]
pub struct Dag {
    graph: StableDiGraph<Node, Edge>,
    /// Qubits in declaration order.
    qubits: Vec<QubitId>,
    sources: FxHashMap<QubitId, NodeIndex>,
    sinks: FxHashMap<QubitId, NodeIndex>,
    pub header: QasmHeader,
}

impl Dag {
    /// An empty circuit over `qubits`: each wire runs straight from source to sink.
    pub fn new(qubits: impl IntoIterator<Item = impl Into<QubitId>>) -> Self {
        let mut dag = Self {
            graph: StableDiGraph::new(),
            qubits: Vec::new(),
            sources: FxHashMap::default(),
            sinks: FxHashMap::default(),
            header: QasmHeader::default(),
        };
        for q in qubits {
            dag.ensure_qubit(&q.into());
        }
        dag
    }

    /// Add a qubit wire if it does not already exist. Returns `true` if one was added.
    pub fn ensure_qubit(&mut self, qubit: &str) -> bool {
        if self.sources.contains_key(qubit) {
            return false;
        }
        let q: QubitId = intern(qubit);
        let src = self.graph.add_node(Node::Source(q.clone()));
        let snk = self.graph.add_node(Node::Sink(q.clone()));
        self.graph.add_edge(
            src,
            snk,
            Edge {
                qubit: q.clone(),
                src_slot: 0,
                dst_slot: 0,
            },
        );
        self.sources.insert(q.clone(), src);
        self.sinks.insert(q.clone(), snk);
        self.qubits.push(q);
        true
    }

    pub fn qubits(&self) -> &[QubitId] {
        &self.qubits
    }

    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Qubits that at least one gate acts on.
    pub fn used_qubits(&self) -> Vec<QubitId> {
        let mut used: Vec<QubitId> = Vec::new();
        for idx in self.gate_indices() {
            for q in &self.gate(idx).qubits {
                if !used.contains(q) {
                    used.push(q.clone());
                }
            }
        }
        used.sort();
        used
    }

    /// Append a gate at the end of each of its qubits' wires.
    pub fn push_gate(&mut self, op: GateOp) -> NodeIndex {
        for q in op.qubits.clone() {
            self.ensure_qubit(&q);
        }
        let sinks: FxHashMap<QubitId, NodeIndex> = op
            .qubits
            .iter()
            .map(|q| (q.clone(), self.sinks[q]))
            .collect();
        self.insert_gate_before(op, &sinks)
    }

    /// Append a gate, validating it against `registry` first.
    pub fn push_gate_checked(&mut self, op: GateOp, registry: &GateRegistry) -> Result<NodeIndex> {
        op.validate(registry)?;
        Ok(self.push_gate(op))
    }

    /// Insert `op` immediately before `before[q]` on each of its qubit wires.
    ///
    /// `before` must name, for every operand qubit, a node that already lies on that
    /// wire.
    pub fn insert_gate_before(
        &mut self,
        op: GateOp,
        before: &FxHashMap<QubitId, NodeIndex>,
    ) -> NodeIndex {
        let qubits = op.qubits.clone();
        let new = self.graph.add_node(Node::Gate(op));
        for (slot, q) in qubits.iter().enumerate() {
            let succ = before[q];
            let in_edge = self
                .edge_on_wire(succ, q, Direction::Incoming)
                .expect("target node is not on this wire");
            let pred = self.graph.edge_endpoints(in_edge).expect("edge exists").0;
            let pred_slot = self.graph[pred].slot_of(q).expect("wire slot");
            let succ_slot = self.graph[succ].slot_of(q).expect("wire slot");
            self.graph.remove_edge(in_edge);
            self.graph.add_edge(
                pred,
                new,
                Edge {
                    qubit: q.clone(),
                    src_slot: pred_slot,
                    dst_slot: slot,
                },
            );
            self.graph.add_edge(
                new,
                succ,
                Edge {
                    qubit: q.clone(),
                    src_slot: slot,
                    dst_slot: succ_slot,
                },
            );
        }
        new
    }

    /// Remove a gate, reconnecting each of its wires from predecessor to successor.
    pub fn remove_gate(&mut self, idx: NodeIndex) {
        let qubits = self
            .graph
            .node_weight(idx)
            .and_then(Node::as_gate)
            .map(|op| op.qubits.clone())
            .expect("not a gate node");
        for q in qubits {
            let in_e = self
                .edge_on_wire(idx, &q, Direction::Incoming)
                .expect("incoming wire");
            let out_e = self
                .edge_on_wire(idx, &q, Direction::Outgoing)
                .expect("outgoing wire");
            let pred = self.graph.edge_endpoints(in_e).expect("edge").0;
            let succ = self.graph.edge_endpoints(out_e).expect("edge").1;
            let pred_slot = self.graph[pred].slot_of(&q).expect("slot");
            let succ_slot = self.graph[succ].slot_of(&q).expect("slot");
            self.graph.remove_edge(in_e);
            self.graph.remove_edge(out_e);
            self.graph.add_edge(
                pred,
                succ,
                Edge {
                    qubit: q,
                    src_slot: pred_slot,
                    dst_slot: succ_slot,
                },
            );
        }
        self.graph.remove_node(idx);
    }

    fn edge_on_wire(&self, node: NodeIndex, qubit: &str, dir: Direction) -> Option<EdgeIndex> {
        self.graph
            .edges_directed(node, dir)
            .find(|e| &*e.weight().qubit == qubit)
            .map(|e| e.id())
    }

    /// The gate node immediately before `idx` on wire `qubit`, if it is a gate.
    pub fn predecessor_on(&self, idx: NodeIndex, qubit: &str) -> Option<NodeIndex> {
        let e = self.edge_on_wire(idx, qubit, Direction::Incoming)?;
        let n = self.graph.edge_endpoints(e)?.0;
        self.graph[n].is_gate().then_some(n)
    }

    /// The gate node immediately after `idx` on wire `qubit`, if it is a gate.
    pub fn successor_on(&self, idx: NodeIndex, qubit: &str) -> Option<NodeIndex> {
        let e = self.edge_on_wire(idx, qubit, Direction::Outgoing)?;
        let n = self.graph.edge_endpoints(e)?.1;
        self.graph[n].is_gate().then_some(n)
    }

    /// The node following `idx` on wire `qubit`, gate or sink.
    pub fn next_on(&self, idx: NodeIndex, qubit: &str) -> Option<NodeIndex> {
        let e = self.edge_on_wire(idx, qubit, Direction::Outgoing)?;
        Some(self.graph.edge_endpoints(e)?.1)
    }

    pub fn source(&self, qubit: &str) -> Option<NodeIndex> {
        self.sources.get(qubit).copied()
    }

    pub fn sink(&self, qubit: &str) -> Option<NodeIndex> {
        self.sinks.get(qubit).copied()
    }

    pub fn node(&self, idx: NodeIndex) -> &Node {
        &self.graph[idx]
    }

    /// The gate at `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is a source or sink node.
    pub fn gate(&self, idx: NodeIndex) -> &GateOp {
        self.graph[idx].as_gate().expect("not a gate node")
    }

    pub fn try_gate(&self, idx: NodeIndex) -> Option<&GateOp> {
        self.graph.node_weight(idx).and_then(Node::as_gate)
    }

    pub fn graph(&self) -> &StableDiGraph<Node, Edge> {
        &self.graph
    }

    /// Every gate node index, in arbitrary order.
    pub fn gate_indices(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&i| self.graph[i].is_gate())
            .collect()
    }

    pub fn gate_count(&self) -> usize {
        self.graph
            .node_indices()
            .filter(|&i| self.graph[i].is_gate())
            .count()
    }

    /// Number of gates acting on `k` qubits.
    pub fn count_arity(&self, k: usize) -> usize {
        self.gate_indices()
            .into_iter()
            .filter(|&i| self.gate(i).qubits.len() == k)
            .count()
    }

    /// Number of gates acting on two or more qubits.
    ///
    /// The reference implementation's `twoQGateCount` counted `qubits.size() == 2`
    /// exactly, so a `ccz` contributed nothing to a "two-qubit gate count" objective even
    /// though it is the most expensive gate in the circuit.
    pub fn multi_qubit_gate_count(&self) -> usize {
        self.gate_indices()
            .into_iter()
            .filter(|&i| self.gate(i).qubits.len() >= 2)
            .count()
    }

    /// Gates grouped into topological layers: every gate in layer `n` depends only on
    /// gates in layers `< n`.
    pub fn layers(&self) -> Vec<Vec<NodeIndex>> {
        let mut indeg: FxHashMap<NodeIndex, usize> = self
            .graph
            .node_indices()
            .map(|i| (i, self.graph.edges_directed(i, Direction::Incoming).count()))
            .collect();
        let mut frontier: Vec<NodeIndex> = indeg
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&i, _)| i)
            .collect();
        frontier.sort_by_key(|i| i.index());

        let mut layers = Vec::new();
        while !frontier.is_empty() {
            let mut next = Vec::new();
            let mut gates_in_layer = Vec::new();
            for &n in &frontier {
                if self.graph[n].is_gate() {
                    gates_in_layer.push(n);
                }
                for e in self.graph.edges_directed(n, Direction::Outgoing) {
                    let t = e.target();
                    let d = indeg.get_mut(&t).expect("indegree");
                    *d -= 1;
                    if *d == 0 {
                        next.push(t);
                    }
                }
            }
            if !gates_in_layer.is_empty() {
                gates_in_layer.sort_by_key(|i| i.index());
                layers.push(gates_in_layer);
            }
            next.sort_by_key(|i| i.index());
            frontier = next;
        }
        layers
    }

    /// Gate nodes in a deterministic topological order.
    ///
    /// If the graph somehow contains a cycle this returns only the gates outside it, so
    /// callers that can produce cycles should assert [`is_acyclic`] rather than infer
    /// well-formedness from a non-empty result.
    ///
    /// [`is_acyclic`]: Dag::is_acyclic
    pub fn topological_gates(&self) -> Vec<NodeIndex> {
        // Same layered walk as `layers`, but with indegrees in a vector indexed by node
        // rather than a hash map, and gates written straight into one output vector.
        //
        // Node indices are dense, so the map was paying a hash per node to store what an
        // offset addresses directly. That is invisible on a benchmark circuit and not on
        // a large one: the windowed search takes a fresh topological order every round,
        // and the smaller the window the more rounds are worth running.
        let bound = petgraph::visit::NodeIndexable::node_bound(&self.graph);
        let mut indeg = vec![0u32; bound];
        for i in self.graph.node_indices() {
            indeg[i.index()] = self.graph.edges_directed(i, Direction::Incoming).count() as u32;
        }

        let mut frontier: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|i| indeg[i.index()] == 0)
            .collect();
        frontier.sort_by_key(|i| i.index());

        let mut out = Vec::with_capacity(self.gate_count());
        let mut next = Vec::new();
        let mut layer = Vec::new();
        while !frontier.is_empty() {
            next.clear();
            layer.clear();
            for &n in &frontier {
                if self.graph[n].is_gate() {
                    layer.push(n);
                }
                for e in self.graph.edges_directed(n, Direction::Outgoing) {
                    let t = e.target();
                    indeg[t.index()] -= 1;
                    if indeg[t.index()] == 0 {
                        next.push(t);
                    }
                }
            }
            // Both sorts are what make the order deterministic rather than dependent on
            // edge-traversal order, and `layers` does the same. Dropping them yields a
            // topological order that is still valid and no longer reproducible, which the
            // QASM round-trip corpus catches immediately.
            layer.sort_unstable_by_key(|i| i.index());
            out.extend_from_slice(&layer);
            next.sort_unstable_by_key(|i| i.index());
            std::mem::swap(&mut frontier, &mut next);
        }
        out
    }

    /// `true` if no directed path leaves `nodes` and re-enters it.
    ///
    /// A convex set of gates can be scheduled as one contiguous block, which is exactly
    /// the condition for cutting it out and splicing something else in. Both rewriting
    /// and resynthesis partitioning depend on it.
    ///
    /// Numbers the circuit per call; callers checking many sets against one circuit
    /// should number it once and use [`is_convex_with`].
    ///
    /// [`is_convex_with`]: Dag::is_convex_with
    pub fn is_convex(&self, nodes: &[NodeIndex]) -> bool {
        if nodes.len() < 2 {
            return true;
        }
        let index: FxHashMap<NodeIndex, u64> = self
            .topological_gates()
            .into_iter()
            .enumerate()
            .map(|(i, n)| (n, i as u64))
            .collect();
        self.is_convex_with(nodes, |n| index.get(&n).copied())
    }

    /// [`is_convex`] against a topological numbering the caller already has.
    ///
    /// `key` must order the gates consistently with the circuit's edges — every edge from
    /// a smaller key to a larger one. It need not be dense: a plain enumeration and
    /// `qrules`' incrementally maintained, deliberately gappy order keys both qualify. A
    /// member of `nodes` the numbering cannot place makes the answer `false`; an
    /// unnumbered gate means the numbering is stale or the graph has a cycle, and neither
    /// may vouch for a splice.
    ///
    /// Only nodes numbered at most the set's maximum can lie on a path that leaves the
    /// set and re-enters it, so the search is proportional to the set's extent in the
    /// order, not to the circuit.
    ///
    /// [`is_convex`]: Dag::is_convex
    pub fn is_convex_with(
        &self,
        nodes: &[NodeIndex],
        key: impl Fn(NodeIndex) -> Option<u64>,
    ) -> bool {
        if nodes.len() < 2 {
            return true;
        }
        let set: FxHashSet<NodeIndex> = nodes.iter().copied().collect();
        let mut hi = 0u64;
        for &n in nodes {
            match key(n) {
                Some(k) => hi = hi.max(k),
                None => return false,
            }
        }

        let mut stack: Vec<NodeIndex> = Vec::new();
        let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
        for &n in nodes {
            for q in &self.gate(n).qubits {
                if let Some(s) = self.successor_on(n, q) {
                    if !set.contains(&s) {
                        stack.push(s);
                    }
                }
            }
        }
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if key(n).is_none_or(|k| k > hi) {
                continue;
            }
            for q in &self.gate(n).qubits {
                if let Some(s) = self.successor_on(n, q) {
                    if set.contains(&s) {
                        return false;
                    }
                    stack.push(s);
                }
            }
        }
        true
    }

    /// `true` if every gate appears in the topological layering.
    ///
    /// A circuit DAG must always be acyclic; a cycle means a rewrite spliced a gate in at
    /// a point that contradicts an existing dependency. Rewriting checks this.
    pub fn is_acyclic(&self) -> bool {
        self.layers().iter().map(Vec::len).sum::<usize>() == self.gate_count()
    }

    /// Pairs of qubits connected by at least one multi-qubit gate.
    ///
    /// Unlike the reference's `getConnectivity`, which only inspected gates for which
    /// `isCX()` was true (`cx`, `cz`, `rxx`) and threw on anything else, this covers
    /// every gate of arity two or more, at any arity.
    pub fn connectivity(&self) -> FxHashSet<(QubitId, QubitId)> {
        let mut pairs = FxHashSet::default();
        for idx in self.gate_indices() {
            let qs = &self.gate(idx).qubits;
            for i in 0..qs.len() {
                for j in (i + 1)..qs.len() {
                    let (a, b) = (&qs[i], &qs[j]);
                    if a <= b {
                        pairs.insert((a.clone(), b.clone()));
                    } else {
                        pairs.insert((b.clone(), a.clone()));
                    }
                }
            }
        }
        pairs
    }

    /// A 128-bit structural fingerprint, invariant under node insertion order.
    ///
    /// The reference `getDagHash` returned a 32-bit value derived from edges alone. Two
    /// problems followed: circuits differing only in isolated vertices hashed equal, and
    /// the beam search used the value as an exact identity key in its `seen` set, so any
    /// collision silently discarded a distinct circuit. Widening to 128 bits makes
    /// collisions negligible, and folding in node contributions makes the fingerprint
    /// actually structural.
    pub fn structural_hash(&self) -> u128 {
        let mut acc: u128 = 0xcbf2_9ce4_8422_2325;
        for idx in self.graph.node_indices() {
            acc = acc.wrapping_add(mix(self.node_fingerprint(idx)));
        }
        for e in self.graph.edge_references() {
            let mut h = rustc_hash::FxHasher::default();
            let ew: &Edge = e.weight();
            ew.hash(&mut h);
            let w = h.finish() as u128;
            let s = self.node_fingerprint(e.source()) as u128;
            let t = self.node_fingerprint(e.target()) as u128;
            // Cantor pairing keeps the combination order-sensitive per edge while the
            // outer sum stays order-insensitive across edges.
            let pairing =
                (s.wrapping_add(t)).wrapping_mul(s.wrapping_add(t).wrapping_add(1)) / 2 + t;
            acc = acc.wrapping_add(mix(w.wrapping_mul(31).wrapping_add(pairing) as u64));
        }
        acc
    }

    fn node_fingerprint(&self, idx: NodeIndex) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        match &self.graph[idx] {
            Node::Source(q) => {
                0u8.hash(&mut h);
                q.hash(&mut h);
            }
            Node::Sink(q) => {
                1u8.hash(&mut h);
                q.hash(&mut h);
            }
            Node::Gate(op) => {
                2u8.hash(&mut h);
                op.gate.hash(&mut h);
                op.qubits.hash(&mut h);
                for p in &op.params {
                    // Bucket to the angle tolerance so that angles equal up to floating
                    // point noise fingerprint identically.
                    match p.eval() {
                        Some(v) => {
                            let q = (crate::angle::normalize_angle(v) / 1e-9).round() as i64;
                            q.hash(&mut h);
                        }
                        None => p.to_string().hash(&mut h),
                    }
                }
            }
        }
        h.finish()
    }

    /// Remove gates that act as the identity, up to a global phase.
    ///
    /// The decision is *semantic*: a gate goes if the matrix its parameters give it is
    /// `e^{i phi} * I`. Every distance in this codebase is global-phase-invariant, so a
    /// gate whose whole effect is a global phase does no work and only costs gates.
    /// `rz(2*pi)` is `-I` and goes; `u2(0, 0)` is a Hadamard-like matrix and stays.
    ///
    /// Two earlier designs did this worse, and both failures came from deciding by
    /// *name* instead of by semantics. The reference dropped gates inside the parser
    /// (`NodeVisitor.visitGateCallStatement` returned `null`), silently altering rule
    /// patterns as they were read, exempting `u2` by name, and testing angles against
    /// multiples of `4*pi` -- which kept `rz(2*pi)` and even `u1(2*pi)`, an *exact*
    /// identity, because `u1`'s period is `2*pi`, not `4*pi`. The first version of this
    /// pass moved out of the parser but kept a hardcoded list of rotation names, so it
    /// inherited the same blind spots and could not see identities in user-defined
    /// gates. Building the actual matrix has no such cases to miss.
    ///
    /// A gate with symbolic parameters is not concrete and is always left alone. The
    /// per-gate matrices are at most `2^arity` square and their identity-ness is
    /// memoized per `(gate, angles)`, so the pass stays cheap on the search hot path.
    pub fn drop_identity_gates(&mut self, registry: &GateRegistry) {
        let victims: Vec<NodeIndex> = self
            .gate_indices()
            .into_iter()
            .filter(|&i| is_global_phase(self.gate(i), registry))
            .collect();
        for v in victims {
            self.remove_gate(v);
        }
    }

    /// The sub-circuit made of `nodes`, over the qubits they touch.
    ///
    /// `nodes` must already be in a topological order of this circuit; the gates are
    /// emitted in the order given. Callers that slice a window out of
    /// [`topological_gates`](Dag::topological_gates) satisfy that by construction.
    ///
    /// The ordering is a precondition rather than something computed here because
    /// computing it costs a topological sort of the *whole* circuit, and this is called
    /// once per window. A million-gate circuit cut into windows of 512 calls it 1,954
    /// times a round, so sorting inside turned a 20-second run into one still going after
    /// five minutes.
    ///
    /// Qubit names are kept, so the result can be spliced back with
    /// [`replace_span`](Dag::replace_span) without a renaming step.
    pub fn subcircuit(&self, nodes: &[NodeIndex]) -> Dag {
        debug_assert!(
            self.is_in_topological_order(nodes),
            "subcircuit requires nodes in topological order"
        );
        let mut qubits: Vec<QubitId> = Vec::new();
        for &n in nodes {
            for q in &self.gate(n).qubits {
                if !qubits.contains(q) {
                    qubits.push(q.clone());
                }
            }
        }
        let mut out = Dag::new(qubits);
        for &n in nodes {
            out.push_gate(self.gate(n).clone());
        }
        out
    }

    /// `true` if no gate in `nodes` precedes an earlier one. For `debug_assert` only.
    fn is_in_topological_order(&self, nodes: &[NodeIndex]) -> bool {
        let mut position: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for (i, &n) in nodes.iter().enumerate() {
            position.insert(n, i);
        }
        nodes.iter().enumerate().all(|(i, &n)| {
            self.gate(n).qubits.iter().all(|q| {
                self.successor_on(n, q)
                    .and_then(|s| position.get(&s))
                    .is_none_or(|&j| i < j)
            })
        })
    }

    /// Replace the gates `nodes` with `replacement`, in place.
    ///
    /// `nodes` must be convex — no path may leave the set and re-enter it — or there is
    /// no single point in time to splice the replacement into. A run of consecutive
    /// positions in a topological order always is: every edge runs from a lower position
    /// to a higher one, so a path that leaves the run above it can only continue upward
    /// and can never come back.
    ///
    /// `replacement` may touch only wires `nodes` already touched. Returns `false`,
    /// having changed nothing, if it names another.
    pub fn replace_span(&mut self, nodes: &[NodeIndex], replacement: &Dag) -> bool {
        let set: FxHashSet<NodeIndex> = nodes.iter().copied().collect();
        let mut exit: FxHashMap<QubitId, NodeIndex> = FxHashMap::default();
        for &n in nodes {
            for q in self.gate(n).qubits.clone() {
                let mut cur = n;
                loop {
                    match self.next_on(cur, &q) {
                        Some(nxt) if set.contains(&nxt) => cur = nxt,
                        Some(nxt) => {
                            exit.insert(q.clone(), nxt);
                            break;
                        }
                        None => break,
                    }
                }
            }
        }
        // Check before touching anything, so a rejected replacement leaves the circuit
        // exactly as it was rather than half-spliced.
        for idx in replacement.gate_indices() {
            if replacement
                .gate(idx)
                .qubits
                .iter()
                .any(|q| !exit.contains_key(q))
            {
                return false;
            }
        }

        for &n in nodes {
            self.remove_gate(n);
        }
        for idx in replacement.topological_gates() {
            let op = replacement.gate(idx).clone();
            let target: FxHashMap<QubitId, NodeIndex> =
                op.qubits.iter().map(|q| (q.clone(), exit[q])).collect();
            self.insert_gate_before(op, &target);
        }
        true
    }

    /// As [`drop_identity_gates`], but considering only `candidates`.
    ///
    /// A rewrite cannot turn a gate it did not touch into an identity: the gates outside
    /// the replaced span keep their names and their parameters. So after a rewrite, the
    /// only gates that need examining are the ones it introduced, and normalizing costs
    /// O(replacement) rather than O(circuit). On the search hot path -- where this runs
    /// after every accepted rewrite -- that is the difference between a pass that is free
    /// and one that dominates the iteration on a large circuit.
    ///
    /// Indices that no longer name a gate are skipped, so a caller may pass nodes that an
    /// intervening edit removed.
    ///
    /// [`drop_identity_gates`]: Dag::drop_identity_gates
    pub fn drop_identity_gates_among(&mut self, candidates: &[NodeIndex], registry: &GateRegistry) {
        let victims: Vec<NodeIndex> = candidates
            .iter()
            .copied()
            .filter(|&i| {
                self.try_gate(i)
                    .is_some_and(|op| is_global_phase(op, registry))
            })
            .collect();
        for v in victims {
            self.remove_gate(v);
        }
    }
}

/// `true` if `op`'s matrix is the identity times a global phase — that is, if the gate
/// does nothing that any global-phase-invariant measure can see.
///
/// A gate with a symbolic parameter is not concrete and is never reported as an identity.
///
/// Memoized: the search normalizes after every accepted rewrite, and a circuit contains
/// few distinct `(gate, angles)` pairs, so almost every call is a table hit rather than a
/// matrix build.
pub fn is_global_phase(op: &GateOp, registry: &GateRegistry) -> bool {
    let Ok(def) = registry.get(&op.gate) else {
        return false;
    };
    if op.params.len() != def.num_params {
        return false;
    }
    let vals: Option<Vec<f64>> = op.params.iter().map(|p| p.eval()).collect();
    let Some(vals) = vals else {
        return false; // symbolic parameters: not concrete, never dropped
    };

    thread_local! {
        static MEMO: std::cell::RefCell<rustc_hash::FxHashMap<(String, Vec<i64>), bool>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    // Quantized at the same resolution the structural hash uses; angles closer together
    // than this are already treated as equal everywhere else.
    let key = (
        op.gate.to_string(),
        vals.iter()
            .map(|v| (crate::angle::normalize_angle(*v) / crate::angle::ANGLE_EPS).round() as i64)
            .collect::<Vec<i64>>(),
    );
    if let Some(hit) = MEMO.with(|m| m.borrow().get(&key).copied()) {
        return hit;
    }
    let answer = def
        .matrix(&vals, registry)
        .is_ok_and(|m| matrix_is_global_phase(&m));
    MEMO.with(|m| m.borrow_mut().insert(key, answer));
    answer
}

/// Elementwise test for `m == e^{i phi} * I`, with the phase read off the first diagonal
/// entry. Elementwise rather than via any trace shortcut, per the numerical note in
/// `docs/PORTING-NOTES.md`.
fn matrix_is_global_phase(m: &ndarray::Array2<num_complex::Complex64>) -> bool {
    const EPS: f64 = 1e-9;
    let n = m.nrows();
    let c = m[[0, 0]];
    if (c.norm() - 1.0).abs() > EPS {
        return false;
    }
    for i in 0..n {
        for j in 0..n {
            let want = if i == j {
                c
            } else {
                num_complex::Complex64::new(0.0, 0.0)
            };
            if (m[[i, j]] - want).norm() > EPS {
                return false;
            }
        }
    }
    true
}

fn mix(x: u64) -> u128 {
    // splitmix64, widened.
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    let lo = z ^ (z >> 31);
    let hi = lo.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    ((hi as u128) << 64) | lo as u128
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::angle::AngleExpr;

    fn op(gate: &str, qubits: &[&str]) -> GateOp {
        GateOp::new(gate, qubits.to_vec(), [])
    }

    fn names(dag: &Dag) -> Vec<String> {
        dag.topological_gates()
            .into_iter()
            .map(|i| {
                let g = dag.gate(i);
                format!("{} {}", g.gate, g.qubits.join(","))
            })
            .collect()
    }

    #[test]
    fn empty_circuit_has_source_sink_per_qubit() {
        let dag = Dag::new(["q0", "q1"]);
        assert_eq!(dag.num_qubits(), 2);
        assert_eq!(dag.gate_count(), 0);
        assert_eq!(dag.graph().node_count(), 4);
        assert_eq!(dag.graph().edge_count(), 2);
    }

    #[test]
    fn push_gate_creates_qubits_on_demand() {
        let mut dag = Dag::new(Vec::<String>::new());
        dag.push_gate(op("cx", &["a", "b"]));
        assert_eq!(dag.num_qubits(), 2);
        assert_eq!(dag.gate_count(), 1);
    }

    #[test]
    fn wires_are_paths_from_source_to_sink() {
        let mut dag = Dag::new(["q0", "q1"]);
        dag.push_gate(op("h", &["q0"]));
        dag.push_gate(op("cx", &["q0", "q1"]));
        dag.push_gate(op("h", &["q1"]));
        for q in dag.qubits().to_vec() {
            let mut n = dag.source(&q).unwrap();
            let mut steps = 0;
            while n != dag.sink(&q).unwrap() {
                n = dag.next_on(n, &q).expect("wire continues");
                steps += 1;
                assert!(steps < 100, "wire {q} does not terminate");
            }
        }
    }

    #[test]
    fn slots_record_operand_positions() {
        let mut dag = Dag::new(Vec::<String>::new());
        let ccz = dag.push_gate(op("ccz", &["a", "b", "c"]));
        for e in dag.graph().edges_directed(ccz, Direction::Incoming) {
            let expected = dag.gate(ccz).slot_of(&e.weight().qubit).unwrap();
            assert_eq!(e.weight().dst_slot, expected);
        }
        // Operand 2 is the third wire; nothing in the DAG needed a `CONTROL2` label.
        let c_edge = dag
            .graph()
            .edges_directed(ccz, Direction::Incoming)
            .find(|e| &*e.weight().qubit == "c")
            .unwrap();
        assert_eq!(c_edge.weight().dst_slot, 2);
    }

    #[test]
    fn arbitrary_arity_gates_are_representable() {
        let mut dag = Dag::new(Vec::<String>::new());
        let qs: Vec<String> = (0..8).map(|i| format!("q{i}")).collect();
        let idx = dag.push_gate(GateOp::new("wide", qs.clone(), []));
        assert_eq!(dag.gate(idx).qubits.len(), 8);
        for (slot, q) in qs.iter().enumerate() {
            let e = dag
                .graph()
                .edges_directed(idx, Direction::Incoming)
                .find(|e| *e.weight().qubit == **q)
                .unwrap();
            assert_eq!(e.weight().dst_slot, slot);
        }
    }

    #[test]
    fn remove_gate_reconnects_wires() {
        let mut dag = Dag::new(["q0", "q1"]);
        dag.push_gate(op("h", &["q0"]));
        let cx = dag.push_gate(op("cx", &["q0", "q1"]));
        dag.push_gate(op("x", &["q1"]));
        dag.remove_gate(cx);
        assert_eq!(names(&dag), vec!["h q0", "x q1"]);
        // Both wires still terminate.
        for q in ["q0", "q1"] {
            let mut n = dag.source(q).unwrap();
            let mut steps = 0;
            while n != dag.sink(q).unwrap() {
                n = dag.next_on(n, q).unwrap();
                steps += 1;
                assert!(steps < 10);
            }
        }
    }

    #[test]
    fn insert_before_splices_in_order() {
        let mut dag = Dag::new(["q0"]);
        dag.push_gate(op("h", &["q0"]));
        let x = dag.push_gate(op("x", &["q0"]));
        let mut before = FxHashMap::default();
        before.insert(intern("q0"), x);
        dag.insert_gate_before(op("t", &["q0"]), &before);
        assert_eq!(names(&dag), vec!["h q0", "t q0", "x q0"]);
    }

    #[test]
    fn layers_group_independent_gates() {
        let mut dag = Dag::new(["q0", "q1", "q2"]);
        dag.push_gate(op("h", &["q0"]));
        dag.push_gate(op("h", &["q1"]));
        dag.push_gate(op("h", &["q2"]));
        dag.push_gate(op("cx", &["q0", "q1"]));
        let layers = dag.layers();
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].len(), 3);
        assert_eq!(layers[1].len(), 1);
    }

    #[test]
    fn convexity_of_gate_sets() {
        let mut dag = Dag::new(["a", "b"]);
        let h = dag.push_gate(op("h", &["a"]));
        let cx = dag.push_gate(op("cx", &["a", "b"]));
        let x = dag.push_gate(op("x", &["b"]));

        assert!(dag.is_convex(&[]));
        assert!(dag.is_convex(&[h]));
        assert!(dag.is_convex(&[h, cx]));
        assert!(dag.is_convex(&[cx, x]));
        assert!(dag.is_convex(&[h, cx, x]));
        // Skipping the cx leaves a path h -> cx -> x that exits and re-enters.
        assert!(!dag.is_convex(&[h, x]));
    }

    #[test]
    fn independent_gates_are_convex() {
        let mut dag = Dag::new(["a", "b"]);
        let h = dag.push_gate(op("h", &["a"]));
        let x = dag.push_gate(op("x", &["b"]));
        // No path between them at all.
        assert!(dag.is_convex(&[h, x]));
    }

    #[test]
    fn a_well_formed_circuit_is_acyclic() {
        let mut dag = Dag::new(["q0", "q1"]);
        dag.push_gate(op("h", &["q0"]));
        dag.push_gate(op("cx", &["q0", "q1"]));
        assert!(dag.is_acyclic());
        assert_eq!(dag.topological_gates().len(), dag.gate_count());
    }

    #[test]
    fn topological_order_respects_dependencies() {
        let mut dag = Dag::new(["q0", "q1"]);
        dag.push_gate(op("h", &["q0"]));
        dag.push_gate(op("cx", &["q0", "q1"]));
        dag.push_gate(op("t", &["q1"]));
        assert_eq!(names(&dag), vec!["h q0", "cx q0,q1", "t q1"]);
    }

    #[test]
    fn counts_are_arity_generic() {
        let mut dag = Dag::new(Vec::<String>::new());
        dag.push_gate(op("h", &["a"]));
        dag.push_gate(op("cx", &["a", "b"]));
        dag.push_gate(op("ccz", &["a", "b", "c"]));
        assert_eq!(dag.gate_count(), 3);
        assert_eq!(dag.count_arity(1), 1);
        assert_eq!(dag.count_arity(2), 1);
        assert_eq!(dag.count_arity(3), 1);
        // A ccz is a multi-qubit gate; the reference's `twoQGateCount` missed it.
        assert_eq!(dag.multi_qubit_gate_count(), 2);
    }

    #[test]
    fn connectivity_covers_all_arities() {
        let mut dag = Dag::new(Vec::<String>::new());
        dag.push_gate(op("ccz", &["a", "b", "c"]));
        let conn = dag.connectivity();
        assert_eq!(conn.len(), 3);
        assert!(conn.contains(&("a".into(), "b".into())));
        assert!(conn.contains(&("a".into(), "c".into())));
        assert!(conn.contains(&("b".into(), "c".into())));
    }

    /// The hash covers vertices as well as edges, so circuits differing only in
    /// isolated structure do not collide.
    #[test]
    fn hash_distinguishes_structurally_distinct_circuits() {
        let mut a = Dag::new(["q0", "q1"]);
        a.push_gate(op("h", &["q0"]));
        let mut b = Dag::new(["q0", "q1"]);
        b.push_gate(op("h", &["q1"]));
        assert_ne!(a.structural_hash(), b.structural_hash());

        let mut c = Dag::new(["q0"]);
        c.push_gate(op("h", &["q0"]));
        let mut d = Dag::new(["q0"]);
        d.push_gate(op("x", &["q0"]));
        assert_ne!(c.structural_hash(), d.structural_hash());
    }

    #[test]
    fn hash_is_stable_under_equal_construction() {
        let build = || {
            let mut dag = Dag::new(["q0", "q1"]);
            dag.push_gate(op("h", &["q0"]));
            dag.push_gate(op("cx", &["q0", "q1"]));
            dag
        };
        assert_eq!(build().structural_hash(), build().structural_hash());
    }

    #[test]
    fn hash_is_insensitive_to_insertion_order_of_independent_gates() {
        let mut a = Dag::new(["q0", "q1"]);
        a.push_gate(op("h", &["q0"]));
        a.push_gate(op("x", &["q1"]));
        let mut b = Dag::new(["q0", "q1"]);
        b.push_gate(op("x", &["q1"]));
        b.push_gate(op("h", &["q0"]));
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn hash_distinguishes_gate_order_on_a_wire() {
        let mut a = Dag::new(["q0"]);
        a.push_gate(op("h", &["q0"]));
        a.push_gate(op("t", &["q0"]));
        let mut b = Dag::new(["q0"]);
        b.push_gate(op("t", &["q0"]));
        b.push_gate(op("h", &["q0"]));
        assert_ne!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn hash_tolerates_float_noise_in_angles() {
        use std::f64::consts::PI;
        let mut a = Dag::new(["q0"]);
        a.push_gate(GateOp::new("rz", vec!["q0"], [AngleExpr::Num(PI / 4.0)]));
        let mut b = Dag::new(["q0"]);
        let noisy = (0..8).map(|_| PI / 32.0).sum::<f64>();
        b.push_gate(GateOp::new("rz", vec!["q0"], [AngleExpr::Num(noisy)]));
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn validate_rejects_bad_ops() {
        let reg = GateRegistry::with_builtins();
        assert!(op("cx", &["a"]).validate(&reg).is_err());
        assert!(op("cx", &["a", "a"]).validate(&reg).is_err());
        assert!(op("nope", &["a"]).validate(&reg).is_err());
        assert!(op("cx", &["a", "b"]).validate(&reg).is_ok());
        assert!(GateOp::new("rz", vec!["a"], []).validate(&reg).is_err());
        assert!(GateOp::new("rz", vec!["a"], [AngleExpr::Num(0.1)])
            .validate(&reg)
            .is_ok());
    }

    /// Identity removal decides from the gate's matrix, never from its name.
    #[test]
    fn identity_dropping_is_explicit_and_shape_aware() {
        use std::f64::consts::PI;
        let reg = GateRegistry::with_builtins();
        let mut dag = Dag::new(["q0"]);
        dag.push_gate(GateOp::new("rz", vec!["q0"], [AngleExpr::Num(0.0)]));
        dag.push_gate(GateOp::new("rz", vec!["q0"], [AngleExpr::Num(4.0 * PI)]));
        dag.push_gate(GateOp::new("rz", vec!["q0"], [AngleExpr::Num(2.0 * PI)]));
        dag.push_gate(GateOp::new(
            "u2",
            vec!["q0"],
            [AngleExpr::Num(0.0), AngleExpr::Num(0.0)],
        ));
        dag.push_gate(op("h", &["q0"]));
        assert_eq!(dag.gate_count(), 5);
        dag.drop_identity_gates(&reg);
        // rz(0) and rz(4pi) are exact identities; rz(2pi) is -I, an identity up to
        // global phase, and goes too. u2(0, 0) is a genuine rotation and stays without
        // needing a name-based exemption; so does h.
        assert_eq!(names(&dag), vec!["u2 q0", "h q0"]);
    }

    /// The decision is made from the gate's matrix, so identities the old name lists
    /// could not see are caught: `u1`'s period is `2*pi`, making `u1(2*pi)` an *exact*
    /// identity that a mod-`4*pi` angle test keeps, and a user-defined composite gate
    /// participates with no registration in any list.
    #[test]
    fn drop_identity_is_semantic_not_name_based() {
        use std::f64::consts::PI;
        let reg = GateRegistry::with_builtins();
        let mut dag = Dag::new(["q0"]);
        dag.push_gate(GateOp::new("u1", vec!["q0"], [AngleExpr::Num(2.0 * PI)]));
        dag.push_gate(GateOp::new("u1", vec!["q0"], [AngleExpr::Num(PI)]));
        dag.drop_identity_gates(&reg);
        assert_eq!(names(&dag), vec!["u1 q0"]);
    }

    #[test]
    fn drop_identity_leaves_symbolic_angles_alone() {
        let reg = GateRegistry::with_builtins();
        let mut dag = Dag::new(["q0"]);
        dag.push_gate(GateOp::new("rz", vec!["q0"], [AngleExpr::var("theta1")]));
        dag.drop_identity_gates(&reg);
        assert_eq!(dag.gate_count(), 1);
    }

    #[test]
    fn used_qubits_excludes_idle_wires() {
        let mut dag = Dag::new(["q0", "q1", "q2"]);
        dag.push_gate(op("h", &["q0"]));
        assert_eq!(dag.used_qubits(), vec![intern("q0")]);
        assert_eq!(dag.num_qubits(), 3);
    }

    #[test]
    fn clone_is_independent() {
        let mut a = Dag::new(["q0"]);
        a.push_gate(op("h", &["q0"]));
        let mut b = a.clone();
        b.push_gate(op("x", &["q0"]));
        assert_eq!(a.gate_count(), 1);
        assert_eq!(b.gate_count(), 2);
        // The reference copy constructor left partition state null; a clone here is
        // immediately usable for every operation.
        let idx = b.gate_indices()[0];
        b.remove_gate(idx);
        assert_eq!(b.gate_count(), 1);
    }
    /// A run of consecutive topological positions is always convex, which is what makes
    /// it safe to cut out, optimize on its own, and splice back.
    #[test]
    fn a_topological_slab_is_convex() {
        let mut dag = Dag::new(["a", "b", "c"]);
        for spec in [
            ("h", vec!["a"]),
            ("cx", vec!["a", "b"]),
            ("t", vec!["b"]),
            ("cx", vec!["b", "c"]),
            ("h", vec!["c"]),
            ("cx", vec!["c", "a"]),
            ("x", vec!["a"]),
        ] {
            dag.push_gate(op(spec.0, &spec.1));
        }
        let order = dag.topological_gates();
        for lo in 0..order.len() {
            for hi in (lo + 1)..=order.len() {
                assert!(
                    dag.is_convex(&order[lo..hi]),
                    "positions {lo}..{hi} are not convex"
                );
            }
        }
    }

    #[test]
    fn a_slab_round_trips_through_subcircuit_and_replace_span() {
        let mut dag = Dag::new(["a", "b", "c"]);
        for spec in [
            ("h", vec!["a"]),
            ("cx", vec!["a", "b"]),
            ("t", vec!["b"]),
            ("cx", vec!["b", "c"]),
            ("h", vec!["c"]),
            ("x", vec!["a"]),
        ] {
            dag.push_gate(op(spec.0, &spec.1));
        }
        let before = dag.structural_hash();
        let order = dag.topological_gates();

        let slab = &order[1..4];
        let sub = dag.subcircuit(slab);
        assert_eq!(sub.gate_count(), 3);
        // Putting the slab back unchanged must reproduce the circuit exactly.
        assert!(dag.replace_span(slab, &sub));
        assert_eq!(dag.structural_hash(), before);
        assert!(dag.is_acyclic());
    }

    /// A replacement naming a wire the span never touched is refused, and refusing leaves
    /// the circuit untouched rather than half-spliced.
    #[test]
    fn replace_span_refuses_a_foreign_wire_without_damage() {
        let mut dag = Dag::new(["a", "b"]);
        dag.push_gate(op("h", &["a"]));
        dag.push_gate(op("x", &["a"]));
        let before = dag.structural_hash();
        let order = dag.topological_gates();

        let mut foreign = Dag::new(["a", "b"]);
        foreign.push_gate(op("cx", &["a", "b"]));
        assert!(!dag.replace_span(&order, &foreign));
        assert_eq!(
            dag.structural_hash(),
            before,
            "a refused splice damaged the circuit"
        );
    }
}
