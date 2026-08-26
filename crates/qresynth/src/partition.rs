//! Choosing a piece of a circuit to hand to a resynthesis backend.
//!
//! A partition must be **convex**: no directed path may leave it and come back, or there
//! is no point in the circuit where the replacement can be spliced in. It must also stay
//! within a qubit budget, since a backend has to build a dense unitary.
//!
//! Growth is arity-generic — a gate contributes exactly the wires it declares, however
//! many — and convexity is checked directly rather than approximated with lowest common
//! ancestors.

use rand::seq::SliceRandom;
use rand::Rng;
use rustc_hash::FxHashSet;

use qcircuit::{Dag, GateOp, NodeIndex, QubitId};

/// How large a partition may get.
#[derive(Debug, Clone, Copy)]
pub struct PartitionLimits {
    /// Largest number of distinct qubits the partition may touch.
    pub max_qubits: usize,
    /// Largest number of gates, as a guard against pathological growth.
    pub max_gates: usize,
}

impl Default for PartitionLimits {
    fn default() -> Self {
        // `IPartitionPicker.MAX_PARTITION_QUBITS = 3` in the reference, where it was a
        // compile-time constant on an interface.
        Self {
            max_qubits: 3,
            max_gates: 64,
        }
    }
}

/// A convex block of gates, and the qubits it spans.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Gate nodes, in circuit order.
    pub nodes: Vec<NodeIndex>,
    /// Qubits the block touches, in a stable order.
    pub qubits: Vec<QubitId>,
}

impl Partition {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The block as a standalone circuit over `q[0]..q[k-1]`.
    ///
    /// Backends work in their own register naming, so the partition is renamed on the way
    /// out and mapped back on the way in. [`Partition::qubits`] is the mapping: index `i`
    /// of that list is the backend's `q[i]`.
    pub fn to_circuit(&self, dag: &Dag) -> Dag {
        let names: Vec<qcircuit::QubitId> = (0..self.qubits.len())
            .map(|i| qcircuit::intern(&format!("q[{i}]")))
            .collect();
        let mut out = Dag::new(names.clone());
        out.header.version = Some("2.0".into());
        out.header.includes = vec!["qelib1.inc".into()];
        out.header.qregs = vec![("q".into(), self.qubits.len())];
        for &n in &self.nodes {
            let op = dag.gate(n);
            let qubits: Vec<QubitId> = op
                .qubits
                .iter()
                .map(|q| {
                    let i = self
                        .qubits
                        .iter()
                        .position(|x| x == q)
                        .expect("qubit in partition");
                    names[i].clone()
                })
                .collect();
            out.push_gate(GateOp {
                gate: op.gate.clone(),
                qubits,
                params: op.params.clone(),
            });
        }
        out
    }

    /// Map a backend's `q[i]` back to the circuit's own qubit name.
    pub fn rename_back(&self, backend_qubit: &str) -> Option<QubitId> {
        let idx = backend_qubit
            .strip_prefix("q[")
            .and_then(|s| s.strip_suffix(']'))
            .and_then(|s| s.parse::<usize>().ok())?;
        self.qubits.get(idx).cloned()
    }
}

/// Picks partitions.
pub trait PartitionPicker {
    fn pick<R: Rng + ?Sized>(
        &self,
        dag: &Dag,
        limits: &PartitionLimits,
        rng: &mut R,
    ) -> Option<Partition>;
}

/// Grows a partition outward from a randomly chosen gate.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomPartition;

impl PartitionPicker for RandomPartition {
    fn pick<R: Rng + ?Sized>(
        &self,
        dag: &Dag,
        limits: &PartitionLimits,
        rng: &mut R,
    ) -> Option<Partition> {
        let gates = dag.topological_gates();
        // A circuit with no gates has nothing to partition; report that rather than
        // hunt for one.
        if gates.is_empty() || limits.max_qubits == 0 {
            return None;
        }
        let order: Vec<usize> = {
            let mut v: Vec<usize> = (0..gates.len()).collect();
            v.shuffle(rng);
            v
        };
        let index: rustc_hash::FxHashMap<NodeIndex, usize> =
            gates.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        for &seed_pos in &order {
            let seed = gates[seed_pos];
            if dag.gate(seed).qubits.len() > limits.max_qubits {
                continue; // a single gate already too wide for the budget
            }
            let p = grow(dag, seed, limits, &index, rng);
            if !p.is_empty() {
                return Some(p);
            }
        }
        None
    }
}

/// Grow a convex block outward from `seed`.
fn grow<R: Rng + ?Sized>(
    dag: &Dag,
    seed: NodeIndex,
    limits: &PartitionLimits,
    index: &rustc_hash::FxHashMap<NodeIndex, usize>,
    rng: &mut R,
) -> Partition {
    let mut nodes: Vec<NodeIndex> = vec![seed];
    let mut set: FxHashSet<NodeIndex> = [seed].into_iter().collect();
    let mut qubits: Vec<QubitId> = dag.gate(seed).qubits.clone();

    loop {
        if nodes.len() >= limits.max_gates {
            break;
        }
        // Everything adjacent to the block, in a shuffled order so growth is varied.
        let mut frontier: Vec<NodeIndex> = Vec::new();
        for &n in &nodes {
            for q in &dag.gate(n).qubits {
                for cand in [dag.predecessor_on(n, q), dag.successor_on(n, q)]
                    .into_iter()
                    .flatten()
                {
                    if !set.contains(&cand) && !frontier.contains(&cand) {
                        frontier.push(cand);
                    }
                }
            }
        }
        frontier.shuffle(rng);

        let mut grew = false;
        for cand in frontier {
            let op = dag.gate(cand);
            let new_qubits: Vec<&QubitId> =
                op.qubits.iter().filter(|q| !qubits.contains(q)).collect();
            if qubits.len() + new_qubits.len() > limits.max_qubits {
                continue;
            }
            // Convexity is the invariant that makes the block replaceable; check it
            // rather than approximating it with lowest common ancestors.
            let mut trial: Vec<NodeIndex> = nodes.clone();
            trial.push(cand);
            if !dag.is_convex_with(&trial, |n| index.get(&n).map(|&i| i as u64)) {
                continue;
            }
            for q in op.qubits.clone() {
                if !qubits.contains(&q) {
                    qubits.push(q);
                }
            }
            nodes.push(cand);
            set.insert(cand);
            grew = true;
            break;
        }
        if !grew {
            break;
        }
    }

    nodes.sort_by_key(|n| index.get(n).copied().unwrap_or(usize::MAX));
    qubits.sort();
    Partition { nodes, qubits }
}

/// Replace a partition's gates with `replacement`, which must act on the same qubits.
///
/// The partition is convex, so the replacement can be spliced in at a single point.
pub fn replace_partition(dag: &Dag, partition: &Partition, replacement: &Dag) -> Option<Dag> {
    let mut out = dag.clone();

    // Where each wire continues after the block.
    let set: FxHashSet<NodeIndex> = partition.nodes.iter().copied().collect();
    let mut exit: rustc_hash::FxHashMap<QubitId, NodeIndex> = rustc_hash::FxHashMap::default();
    for &n in &partition.nodes {
        for q in out.gate(n).qubits.clone() {
            let mut cur = n;
            loop {
                match out.next_on(cur, &q) {
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

    for &n in &partition.nodes {
        out.remove_gate(n);
    }

    for idx in replacement.topological_gates() {
        let op = replacement.gate(idx);
        let target: rustc_hash::FxHashMap<QubitId, NodeIndex> = op
            .qubits
            .iter()
            .filter_map(|q| exit.get(q).map(|&n| (q.clone(), n)))
            .collect();
        if target.len() != op.qubits.len() {
            // The replacement names a wire the partition never touched.
            return None;
        }
        out.insert_gate_before(op.clone(), &target);
    }

    out.is_acyclic().then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    fn dag(src: &str) -> Dag {
        qasm::parse(src).unwrap()
    }

    /// There is no two-phase construction here: a `Dag` is fully built the moment it
    /// exists and its wire structure is an invariant, so a clone is not a second-class
    /// citizen. This pins that a clone partitions identically to its source.
    #[test]
    fn a_cloned_dag_partitions_like_its_source() {
        let original = dag("h a; cx a, b; t b; cx b, c; h c;");
        let copy = original.clone();
        let limits = PartitionLimits::default();

        let from_original = RandomPartition.pick(&original, &limits, &mut rng(7));
        let from_copy = RandomPartition.pick(&copy, &limits, &mut rng(7));

        let a = from_original.expect("original yielded no partition");
        let b = from_copy.expect("clone yielded no partition");
        assert_eq!(a.len(), b.len());
        assert_eq!(
            qcircuit::qasm::to_qasm(&a.to_circuit(&original)),
            qcircuit::qasm::to_qasm(&b.to_circuit(&copy)),
        );
    }

    /// Qubit names are opaque strings throughout this port; nothing parses an index back
    /// out of one. A circuit over an indexed register partitions like any other.
    #[test]
    fn indexed_register_names_are_not_parsed_as_integers() {
        let d = dag("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\n\
             h q[0];\ncx q[0], q[1];\nt q[1];\ncx q[1], q[2];\n");
        assert_eq!(d.gate_count(), 4);
        let p = RandomPartition
            .pick(&d, &PartitionLimits::default(), &mut rng(3))
            .expect("indexed register names yielded no partition");
        assert!(!p.is_empty());
        // Every qubit the partition names round-trips back to a real circuit qubit.
        let circuit = p.to_circuit(&d);
        for q in circuit.qubits() {
            assert!(
                p.rename_back(q).is_some(),
                "backend qubit {q} does not map back"
            );
        }
    }

    #[test]
    fn picks_a_partition_from_a_simple_circuit() {
        let d = dag("h a; cx a, b; t b; cx b, c; h c;");
        let p = RandomPartition
            .pick(&d, &PartitionLimits::default(), &mut rng(1))
            .unwrap();
        assert!(!p.is_empty());
        assert!(p.qubits.len() <= 3);
        assert!(d.is_convex(&p.nodes));
    }

    /// A circuit with no gates yields `None`, not a hunt for one.
    #[test]
    fn a_gate_free_circuit_returns_none() {
        let d = dag("qreg q[4];");
        assert_eq!(d.gate_count(), 0);
        // Must return, not spin.
        assert!(RandomPartition
            .pick(&d, &PartitionLimits::default(), &mut rng(0))
            .is_none());
        assert!(RandomPartition
            .pick(&dag(""), &PartitionLimits::default(), &mut rng(0))
            .is_none());
    }

    #[test]
    fn a_zero_qubit_budget_returns_none() {
        let d = dag("h a;");
        let limits = PartitionLimits {
            max_qubits: 0,
            max_gates: 8,
        };
        assert!(RandomPartition.pick(&d, &limits, &mut rng(0)).is_none());
    }

    /// A gate wider than the budget can never be partitioned around.
    #[test]
    fn gates_wider_than_the_budget_are_skipped() {
        let d = dag("ccz a, b, c;");
        let limits = PartitionLimits {
            max_qubits: 2,
            max_gates: 8,
        };
        assert!(RandomPartition.pick(&d, &limits, &mut rng(0)).is_none());

        // With room for three qubits it is pickable.
        let limits = PartitionLimits {
            max_qubits: 3,
            max_gates: 8,
        };
        let p = RandomPartition.pick(&d, &limits, &mut rng(0)).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p.qubits.len(), 3);
    }

    /// Partitions are always convex and always within budget, across many circuits and
    /// seeds. This is the property that makes splicing sound.
    #[test]
    fn partitions_are_always_convex_and_within_budget() {
        let circuits = [
            "h a; cx a, b; t b; cx b, c; h c; cx a, c; t a;",
            "ccz a, b, c; h c; cx a, b; ccz a, b, c;",
            "cx a, b; cx c, d; cx b, c; cx a, d; h a; h d;",
            "rz(0.3) a; cx a, b; rz(0.4) b; cx b, a; rz(0.5) a;",
        ];
        for max_qubits in 1..=4 {
            let limits = PartitionLimits {
                max_qubits,
                max_gates: 32,
            };
            for src in circuits {
                let d = dag(src);
                for seed in 0..25 {
                    let Some(p) = RandomPartition.pick(&d, &limits, &mut rng(seed)) else {
                        continue;
                    };
                    assert!(
                        p.qubits.len() <= max_qubits,
                        "budget {max_qubits} exceeded: {} qubits",
                        p.qubits.len()
                    );
                    assert!(
                        d.is_convex(&p.nodes),
                        "seed {seed} on `{src}` gave a non-convex partition"
                    );
                    // Every gate's qubits are within the partition's qubit set.
                    for &n in &p.nodes {
                        for q in &d.gate(n).qubits {
                            assert!(p.qubits.contains(q));
                        }
                    }
                }
            }
        }
    }

    /// Three-qubit gates are grown correctly, where the reference treated them as
    /// one-qubit gates because `isCX()` was false.
    #[test]
    fn three_qubit_gates_contribute_all_their_qubits() {
        let d = dag("ccz a, b, c; h a;");
        let limits = PartitionLimits {
            max_qubits: 3,
            max_gates: 8,
        };
        for seed in 0..20 {
            let p = RandomPartition.pick(&d, &limits, &mut rng(seed)).unwrap();
            if p.nodes.iter().any(|&n| &*d.gate(n).gate == "ccz") {
                assert_eq!(
                    p.qubits.len(),
                    3,
                    "a ccz must contribute three qubits to the budget"
                );
            }
        }
    }

    #[test]
    fn partition_renders_to_a_standalone_circuit() {
        let d = dag("h x; cx x, y; t y; cx y, z;");
        let limits = PartitionLimits {
            max_qubits: 2,
            max_gates: 8,
        };
        let p = RandomPartition.pick(&d, &limits, &mut rng(4)).unwrap();
        let circuit = p.to_circuit(&d);
        assert_eq!(circuit.gate_count(), p.len());
        assert_eq!(circuit.num_qubits(), p.qubits.len());
        // Renamed into the backend's register.
        for q in circuit.qubits() {
            assert!(q.starts_with("q["), "unexpected qubit name {q}");
        }
        // ...and the text is valid QASM with a header.
        let text = qasm::to_qasm(&circuit);
        assert!(text.starts_with("OPENQASM 2.0;"));
        assert!(qasm::parse(&text).is_ok());
    }

    #[test]
    fn renaming_round_trips() {
        let d = dag("h x; cx x, y;");
        let limits = PartitionLimits {
            max_qubits: 2,
            max_gates: 8,
        };
        let p = RandomPartition.pick(&d, &limits, &mut rng(2)).unwrap();
        for (i, original) in p.qubits.iter().enumerate() {
            assert_eq!(p.rename_back(&format!("q[{i}]")).as_ref(), Some(original));
        }
        assert!(p.rename_back("q[99]").is_none());
        assert!(p.rename_back("nonsense").is_none());
    }

    #[test]
    fn replacing_a_partition_preserves_the_rest() {
        let d = dag("h a; cx a, b; t b; x a;");
        let limits = PartitionLimits {
            max_qubits: 2,
            max_gates: 2,
        };
        let p = RandomPartition.pick(&d, &limits, &mut rng(5)).unwrap();

        // Replace the block with itself: the circuit must come back identical.
        let same: Dag = {
            let mut r = Dag::new(p.qubits.clone());
            for &n in &p.nodes {
                r.push_gate(d.gate(n).clone());
            }
            r
        };
        let out = replace_partition(&d, &p, &same).unwrap();
        assert_eq!(out.gate_count(), d.gate_count());
        assert_eq!(out.structural_hash(), d.structural_hash());
    }

    #[test]
    fn replacing_with_fewer_gates_shrinks_the_circuit() {
        let d = dag("h a; h a; t b;");
        let gates = d.topological_gates();
        let p = Partition {
            nodes: vec![gates[0], gates[1]],
            qubits: vec![qcircuit::intern("a")],
        };
        assert!(d.is_convex(&p.nodes));
        let empty = Dag::new(vec!["a".to_string()]);
        let out = replace_partition(&d, &p, &empty).unwrap();
        assert_eq!(out.gate_count(), 1);
        assert!(out.is_acyclic());
    }

    #[test]
    fn replacement_naming_an_unknown_wire_is_refused() {
        let d = dag("h a; t b;");
        let gates = d.topological_gates();
        let p = Partition {
            nodes: vec![gates[0]],
            qubits: vec![qcircuit::intern("a")],
        };
        let mut bad = Dag::new(vec!["zzz".to_string()]);
        bad.push_gate(GateOp::new("h", vec!["zzz"], []));
        assert!(replace_partition(&d, &p, &bad).is_none());
    }

    #[test]
    fn picking_is_deterministic_for_a_seed() {
        let d = dag("h a; cx a, b; t b; cx b, c; h c; x a;");
        let limits = PartitionLimits::default();
        let a = RandomPartition.pick(&d, &limits, &mut rng(9)).unwrap();
        let b = RandomPartition.pick(&d, &limits, &mut rng(9)).unwrap();
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.qubits, b.qubits);
    }

    /// Different seeds must reach different partitions, or resynthesis would keep
    /// handing the backend the same block. The circuit needs more qubits than the budget
    /// for this to mean anything: with three qubits and a three-qubit budget, growth
    /// saturates to the whole circuit whatever the seed.
    #[test]
    fn different_seeds_explore_different_partitions() {
        let d = dag("h a; cx a, b; t b; cx b, c; h c; x d; cx d, e; t e; \
             cx c, d; h f; cx e, f; t a; cx a, c; x f;");
        assert!(d.num_qubits() > PartitionLimits::default().max_qubits);
        let limits = PartitionLimits::default();
        let mut distinct = std::collections::HashSet::new();
        for seed in 0..40 {
            if let Some(p) = RandomPartition.pick(&d, &limits, &mut rng(seed)) {
                assert!(d.is_convex(&p.nodes));
                distinct.insert(p.nodes.clone());
            }
        }
        assert!(
            distinct.len() > 3,
            "only {} distinct partitions",
            distinct.len()
        );
    }

    /// Growth saturates: with a budget at least as wide as the circuit, the partition is
    /// the whole circuit however it was seeded.
    #[test]
    fn a_budget_wider_than_the_circuit_takes_everything() {
        let d = dag("h a; cx a, b; t b;");
        let limits = PartitionLimits {
            max_qubits: 8,
            max_gates: 64,
        };
        for seed in 0..10 {
            let p = RandomPartition.pick(&d, &limits, &mut rng(seed)).unwrap();
            assert_eq!(p.len(), d.gate_count(), "seed {seed}");
        }
    }

    #[test]
    fn partition_gate_limit_is_respected() {
        let d = dag("h a; h a; h a; h a; h a; h a;");
        let limits = PartitionLimits {
            max_qubits: 1,
            max_gates: 3,
        };
        for seed in 0..10 {
            let p = RandomPartition.pick(&d, &limits, &mut rng(seed)).unwrap();
            assert!(p.len() <= 3, "got {} gates", p.len());
        }
    }
}
