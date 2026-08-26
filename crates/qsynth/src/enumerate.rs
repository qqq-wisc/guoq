//! Enumerating small circuits and grouping the equivalent ones.
//!
//! # The pruning that makes this tractable
//!
//! Naively there are `(gates * qubits * angles)^size` circuits, which for the Nam gate
//! set at size 6 is around 85 million. The saving is that a circuit only needs
//! enumerating if every one of its suffixes is already the *canonical representative* of
//! its equivalence class: if a suffix can be rewritten smaller, so can the whole circuit,
//! and the smaller form will be enumerated instead.
//!
//! The reference implements this as
//! `previousReps.containsKey(caa.getQasmStringDropFirst())` — extend a representative of
//! size `i` by one gate, and keep the result only if dropping its first gate leaves a
//! representative of size `i`. That is reproduced here.

use rustc_hash::{FxHashMap, FxHashSet};

use qcircuit::{AngleExpr, Dag, GateOp, GateRegistry, GateSet};

use crate::fingerprint::{fingerprint, AngleSample, Equivalence, Fingerprint};

/// A circuit under enumeration.
#[derive(Debug, Clone, PartialEq)]
pub struct Enumerated {
    pub ops: Vec<GateOp>,
    /// Canonical text, used as an identity for the suffix-pruning table.
    pub key: String,
}

impl Enumerated {
    pub fn empty() -> Self {
        Self {
            ops: Vec::new(),
            key: String::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Extend by one gate.
    pub fn extend(&self, op: GateOp) -> Self {
        let mut ops = self.ops.clone();
        ops.push(op);
        let key = render(&ops);
        Self { ops, key }
    }

    /// This circuit with its first gate removed.
    pub fn drop_first(&self) -> String {
        if self.ops.is_empty() {
            String::new()
        } else {
            render(&self.ops[1..])
        }
    }

    pub fn to_dag(&self, qubits: &[qcircuit::QubitId]) -> Dag {
        let mut dag = Dag::new(qubits.to_vec());
        for op in &self.ops {
            dag.push_gate(op.clone());
        }
        dag
    }

    /// The rule-file spelling.
    pub fn to_rule_text(&self) -> String {
        if self.ops.is_empty() {
            ";".to_string()
        } else {
            render(&self.ops)
        }
    }
}

fn render(ops: &[GateOp]) -> String {
    ops.iter()
        .map(|op| {
            let params = if op.params.is_empty() {
                String::new()
            } else {
                format!(
                    "({})",
                    op.params
                        .iter()
                        .map(qcircuit::qasm::format_angle)
                        .collect::<Vec<_>>()
                        .join(",")
                )
            };
            format!("{}{} {}", op.gate, params, op.qubits.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
        + ";"
}

/// What to enumerate.
#[derive(Debug, Clone)]
pub struct EnumerationConfig {
    /// Gates available, and the angle basis for parameterized ones.
    pub gates: Vec<String>,
    pub angles: Vec<AngleExpr>,
    /// Fixed-angle instantiations, for hardware exposing a rotation only at set values.
    pub fixed: Vec<(String, Vec<AngleExpr>)>,
    pub max_qubits: usize,
    pub max_size: usize,
    pub equivalence: Equivalence,
    pub seed: u64,
    /// Independent random rational points exact verification tries; see
    /// [`crate::exact`]. One already leaves only a probability-zero failure mode.
    pub verify_rounds: usize,
    /// Gates whose angles must not repeat an expression already in the circuit; see
    /// [`qcircuit::GateSet::single_use_angles`].
    pub single_use_angles: Vec<String>,
}

impl EnumerationConfig {
    /// Build from a gate set definition.
    pub fn from_gate_set(set: &GateSet, max_qubits: usize, max_size: usize) -> Self {
        Self {
            gates: set.gates.clone(),
            angles: set.synth_angles.clone(),
            fixed: set
                .fixed_params
                .iter()
                .map(|f| (f.gate.clone(), f.params.clone()))
                .collect(),
            max_qubits,
            max_size,
            equivalence: Equivalence::Exact,
            seed: 0x5EED,
            verify_rounds: 2,
            single_use_angles: set.single_use_angles.clone(),
        }
    }

    pub fn qubit_names(&self) -> Vec<qcircuit::QubitId> {
        (0..self.max_qubits)
            .map(|i| qcircuit::intern(&format!("q{i}")))
            .collect()
    }
}

/// A group of circuits that all compute the same thing.
#[derive(Debug, Clone)]
pub struct EquivalenceClass {
    pub representative: Enumerated,
    pub members: Vec<Enumerated>,
}

impl EquivalenceClass {
    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// Every gate application available at each enumeration step.
pub fn gate_applications(config: &EnumerationConfig, registry: &GateRegistry) -> Vec<GateOp> {
    let qubits = config.qubit_names();
    let mut out = Vec::new();

    for name in &config.gates {
        let Ok(def) = registry.get(name) else {
            continue;
        };
        // Fixed-angle instantiations replace the free angle basis for that gate.
        let fixed: Vec<&Vec<AngleExpr>> = config
            .fixed
            .iter()
            .filter(|(g, _)| g == name)
            .map(|(_, p)| p)
            .collect();

        for operands in operand_tuples(&qubits, def.arity) {
            if def.num_params == 0 {
                out.push(GateOp {
                    gate: qcircuit::intern(name),
                    qubits: operands,
                    params: Vec::new(),
                });
            } else if !fixed.is_empty() {
                for params in &fixed {
                    out.push(GateOp {
                        gate: qcircuit::intern(name),
                        qubits: operands.clone(),
                        params: (*params).clone(),
                    });
                }
            } else {
                for params in param_tuples(&config.angles, def.num_params) {
                    out.push(GateOp {
                        gate: qcircuit::intern(name),
                        qubits: operands.clone(),
                        params,
                    });
                }
            }
        }
    }
    out
}

/// All ordered tuples of `k` distinct qubits.
fn operand_tuples(qubits: &[qcircuit::QubitId], k: usize) -> Vec<Vec<qcircuit::QubitId>> {
    if k == 0 || k > qubits.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current = Vec::new();
    fn go(
        qubits: &[qcircuit::QubitId],
        k: usize,
        current: &mut Vec<qcircuit::QubitId>,
        out: &mut Vec<Vec<qcircuit::QubitId>>,
    ) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for q in qubits {
            if current.contains(q) {
                continue;
            }
            current.push(q.clone());
            go(qubits, k, current, out);
            current.pop();
        }
    }
    go(qubits, k, &mut current, &mut out);
    out
}

/// All tuples of `k` angles, with no angle repeated.
///
/// Repeating a symbol within one gate would express a constraint the rule cannot use,
/// which is why the reference skipped those too (`containsAngle` in `Synthesizer`).
fn param_tuples(angles: &[AngleExpr], k: usize) -> Vec<Vec<AngleExpr>> {
    if k == 0 || angles.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current: Vec<AngleExpr> = Vec::new();
    fn go(
        angles: &[AngleExpr],
        k: usize,
        current: &mut Vec<AngleExpr>,
        out: &mut Vec<Vec<AngleExpr>>,
    ) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for a in angles {
            if current.iter().any(|x| x.to_string() == a.to_string()) {
                continue;
            }
            current.push(a.clone());
            go(angles, k, current, out);
            current.pop();
        }
    }
    go(angles, k, &mut current, &mut out);
    out
}

/// Enumerate circuits and group them by what they compute.
pub struct Enumerator<'a> {
    config: EnumerationConfig,
    registry: &'a GateRegistry,
    sample: AngleSample,
    /// Fingerprint to the circuits carrying it.
    buckets: FxHashMap<Fingerprint, Vec<Enumerated>>,
    /// Circuits already seen, so a circuit is fingerprinted once.
    seen: FxHashSet<String>,
    /// How many circuits were enumerated, for reporting.
    pub enumerated: usize,
}

impl<'a> Enumerator<'a> {
    pub fn new(config: EnumerationConfig, registry: &'a GateRegistry) -> Self {
        let seed = config.seed;
        Self {
            config,
            registry,
            sample: AngleSample::new(seed),
            buckets: FxHashMap::default(),
            seen: FxHashSet::default(),
            enumerated: 0,
        }
    }

    pub fn config(&self) -> &EnumerationConfig {
        &self.config
    }

    pub fn sample(&self) -> &AngleSample {
        &self.sample
    }

    /// Run the enumeration, returning the equivalence classes.
    pub fn run(&mut self) -> Vec<EquivalenceClass> {
        let qubits = self.config.qubit_names();
        let applications = gate_applications(&self.config, self.registry);
        // Which applications carry the single-use-angle constraint, precomputed so the
        // hot loop tests a bool instead of scanning a name list.
        let single_use: Vec<bool> = applications
            .iter()
            .map(|op| {
                self.config
                    .single_use_angles
                    .iter()
                    .any(|g| *op.gate == **g)
            })
            .collect();

        let empty = Enumerated::empty();
        self.record(&empty, &qubits);
        // Size 0: the empty circuit is the only representative to extend.
        let mut previous: Vec<Enumerated> = vec![empty];
        let mut previous_keys: FxHashSet<String> = [String::new()].into_iter().collect();

        for size in 1..=self.config.max_size {
            for base in &previous.clone() {
                for (op, &restricted) in applications.iter().zip(&single_use) {
                    // The reference's `containsAngle` pruning: a restricted gate may
                    // not reuse an angle expression the circuit already contains, from
                    // any gate. For the u family this is what keeps the space finite
                    // enough to enumerate at all.
                    if restricted
                        && op
                            .params
                            .iter()
                            .any(|p| base.ops.iter().any(|prior| prior.params.contains(p)))
                    {
                        continue;
                    }
                    let candidate = base.extend(op.clone());
                    // Suffix pruning: only circuits all of whose suffixes are canonical
                    // can themselves be canonical.
                    if size > 1 && !previous_keys.contains(&candidate.drop_first()) {
                        continue;
                    }
                    self.record(&candidate, &qubits);
                }
            }

            let classes = self.classes();
            previous = classes
                .iter()
                .map(|c| c.representative.clone())
                .filter(|r| r.len() == size)
                .collect();
            previous_keys = previous.iter().map(|r| r.key.clone()).collect();
            if previous.is_empty() {
                break;
            }
        }

        self.classes()
    }

    fn record(&mut self, circuit: &Enumerated, qubits: &[qcircuit::QubitId]) {
        if !self.seen.insert(circuit.key.clone()) {
            return;
        }
        self.enumerated += 1;
        let dag = circuit.to_dag(qubits);
        let Some(fp) = fingerprint(&dag, qubits, self.registry, &mut self.sample) else {
            return;
        };
        self.buckets.entry(fp).or_default().push(circuit.clone());
    }

    /// The current buckets, each with its canonical representative chosen.
    pub fn classes(&self) -> Vec<EquivalenceClass> {
        let mut out: Vec<EquivalenceClass> = self
            .buckets
            .values()
            .filter_map(|members| {
                let representative = pick_representative(members)?;
                Some(EquivalenceClass {
                    representative,
                    members: members.clone(),
                })
            })
            .collect();
        out.sort_by(|a, b| a.representative.key.cmp(&b.representative.key));
        out
    }
}

/// The canonical member of a class: fewest gates, then lexicographically first.
///
/// Matching the reference's `pickSmallest`. Ties must be broken deterministically or the
/// suffix-pruning table depends on hash iteration order.
pub fn pick_representative(members: &[Enumerated]) -> Option<Enumerated> {
    members
        .iter()
        .min_by(|a, b| a.len().cmp(&b.len()).then(a.key.cmp(&b.key)))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::GateSetLibrary;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn nam_config(max_qubits: usize, max_size: usize) -> EnumerationConfig {
        let lib = GateSetLibrary::builtin();
        EnumerationConfig::from_gate_set(lib.get("nam").unwrap(), max_qubits, max_size)
    }

    /// The reference's `containsAngle` pruning, matched: a single-use gate (ion's
    /// `rxx` here) never reuses an angle expression already present in the circuit,
    /// while unrestricted gates (`rz`) still may. Without this pruning the ibmo gate
    /// set cannot be enumerated at any size on real hardware.
    #[test]
    fn single_use_angle_gates_never_reuse_an_expression() {
        let lib = GateSetLibrary::builtin();
        let cfg = EnumerationConfig::from_gate_set(lib.get("ion").unwrap(), 2, 2);
        assert!(cfg.single_use_angles.contains(&"rxx".to_string()));
        let registry = reg();
        let mut e = Enumerator::new(cfg, &registry);
        let classes = e.run();

        let mut rz_reused = false;
        for class in &classes {
            for m in std::iter::once(&class.representative).chain(class.members.iter()) {
                for (i, op) in m.ops.iter().enumerate() {
                    let repeats_prior = op
                        .params
                        .iter()
                        .any(|p| m.ops[..i].iter().any(|prior| prior.params.contains(p)));
                    if &*op.gate == "rxx" {
                        assert!(
                            !repeats_prior,
                            "rxx reused an angle the circuit already had: {}",
                            m.key
                        );
                    } else if &*op.gate == "rz" && repeats_prior {
                        rz_reused = true;
                    }
                }
            }
        }
        assert!(
            rz_reused,
            "unrestricted gates should still be free to repeat an angle"
        );
    }

    #[test]
    fn operand_tuples_are_ordered_and_distinct() {
        let q: Vec<qcircuit::QubitId> = ["q0", "q1", "q2"]
            .iter()
            .map(|s| qcircuit::intern(s))
            .collect();
        assert_eq!(operand_tuples(&q, 1).len(), 3);
        // Ordered pairs of distinct qubits: 3 * 2.
        assert_eq!(operand_tuples(&q, 2).len(), 6);
        assert_eq!(operand_tuples(&q, 3).len(), 6);
        // A gate wider than the register has no placements.
        assert!(operand_tuples(&q, 4).is_empty());
        for t in operand_tuples(&q, 2) {
            assert_ne!(t[0], t[1]);
        }
    }

    #[test]
    fn param_tuples_do_not_repeat_an_angle() {
        let angles = vec![
            AngleExpr::var("theta1"),
            AngleExpr::var("theta2"),
            AngleExpr::var("theta3"),
        ];
        assert_eq!(param_tuples(&angles, 1).len(), 3);
        assert_eq!(param_tuples(&angles, 2).len(), 6);
        for t in param_tuples(&angles, 2) {
            assert_ne!(t[0].to_string(), t[1].to_string());
        }
        assert!(param_tuples(&[], 1).is_empty());
    }

    #[test]
    fn gate_applications_cover_the_gate_set() {
        let cfg = nam_config(2, 1);
        let apps = gate_applications(&cfg, &reg());
        // h and x on 2 qubits, rz on 2 qubits with 3 angles, cx on 2 ordered pairs.
        let count = |g: &str| apps.iter().filter(|a| &*a.gate == g).count();
        assert_eq!(count("h"), 2);
        assert_eq!(count("x"), 2);
        assert_eq!(count("rz"), 2 * 3);
        assert_eq!(count("cx"), 2);
        assert_eq!(apps.len(), 12);
    }

    #[test]
    fn fixed_angle_gates_replace_the_free_basis() {
        let lib = GateSetLibrary::builtin();
        let cfg = EnumerationConfig::from_gate_set(lib.get("rigetti").unwrap(), 2, 1);
        let apps = gate_applications(&cfg, &reg());
        let rx: Vec<&GateOp> = apps.iter().filter(|a| &*a.gate == "rx").collect();
        // Three fixed angles on each of two qubits, and no symbolic rx.
        assert_eq!(rx.len(), 6);
        for a in rx {
            assert!(
                a.params[0].eval().is_some(),
                "rx should be at a fixed angle"
            );
        }
        // rz still uses the symbolic basis.
        let rz: Vec<&GateOp> = apps.iter().filter(|a| &*a.gate == "rz").collect();
        assert!(rz.iter().any(|a| a.params[0].eval().is_none()));
    }

    #[test]
    fn rendering_matches_the_rule_file_spelling() {
        let c = Enumerated::empty()
            .extend(GateOp::new("cx", vec!["q0", "q1"], []))
            .extend(GateOp::new("h", vec!["q1"], []));
        assert_eq!(c.key, "cx q0, q1; h q1;");
        assert_eq!(c.to_rule_text(), "cx q0, q1; h q1;");
        assert_eq!(Enumerated::empty().to_rule_text(), ";");
    }

    #[test]
    fn dropping_the_first_gate() {
        let c = Enumerated::empty()
            .extend(GateOp::new("h", vec!["q0"], []))
            .extend(GateOp::new("x", vec!["q1"], []))
            .extend(GateOp::new("h", vec!["q0"], []));
        assert_eq!(c.drop_first(), "x q1; h q0;");
        assert_eq!(Enumerated::empty().drop_first(), "");
    }

    #[test]
    fn representative_is_smallest_then_lexicographic() {
        let a = Enumerated::empty().extend(GateOp::new("x", vec!["q0"], []));
        let b = Enumerated::empty().extend(GateOp::new("h", vec!["q0"], []));
        let long = a.extend(GateOp::new("h", vec!["q0"], []));
        let rep = pick_representative(&[long.clone(), a.clone(), b.clone()]).unwrap();
        assert_eq!(rep.key, b.key, "h sorts before x at equal size");
        assert!(pick_representative(&[]).is_none());
    }

    #[test]
    fn enumeration_finds_the_identity_class() {
        let registry = reg();
        let cfg = nam_config(1, 2);
        let mut e = Enumerator::new(cfg, &registry);
        let classes = e.run();
        // The empty circuit's class must contain h;h and x;x.
        let identity = classes
            .iter()
            .find(|c| c.representative.is_empty())
            .expect("no identity class");
        let keys: Vec<&str> = identity.members.iter().map(|m| m.key.as_str()).collect();
        assert!(keys.contains(&"h q0; h q0;"), "{keys:?}");
        assert!(keys.contains(&"x q0; x q0;"), "{keys:?}");
    }

    #[test]
    fn enumeration_groups_only_genuine_equivalences() {
        let registry = reg();
        let cfg = nam_config(2, 2);
        let mut e = Enumerator::new(cfg.clone(), &registry);
        let classes = e.run();
        let qubits = cfg.qubit_names();
        let base = e.sample().clone();

        for class in &classes {
            for member in &class.members {
                assert!(
                    matches!(
                        crate::exact::verify(
                            &crate::exact::Side::plain(&class.representative.ops),
                            &crate::exact::Side::plain(&member.ops),
                            None,
                            &qubits,
                            &reg(),
                            Equivalence::Exact,
                            2,
                            base.seed(),
                        ),
                        Ok(true)
                    ),
                    "`{}` was grouped with `{}` but is not equal to it",
                    member.key,
                    class.representative.key
                );
            }
        }
    }

    #[test]
    fn suffix_pruning_reduces_the_search() {
        // Without pruning, size 3 over the nam set on 2 qubits would be 12^3 = 1728
        // circuits plus smaller ones. Pruning must cut that substantially.
        let registry = reg();
        let mut e = Enumerator::new(nam_config(2, 3), &registry);
        e.run();
        assert!(
            e.enumerated < 1728,
            "pruning did nothing: {} circuits",
            e.enumerated
        );
        assert!(
            e.enumerated > 50,
            "pruning was too aggressive: {}",
            e.enumerated
        );
    }

    #[test]
    fn enumeration_is_deterministic() {
        let registry = reg();
        let a = {
            let mut e = Enumerator::new(nam_config(2, 3), &registry);
            let c = e.run();
            c.iter()
                .map(|x| x.representative.key.clone())
                .collect::<Vec<_>>()
        };
        let b = {
            let mut e = Enumerator::new(nam_config(2, 3), &registry);
            let c = e.run();
            c.iter()
                .map(|x| x.representative.key.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn a_zero_size_enumeration_yields_only_the_empty_circuit() {
        let registry = reg();
        let mut e = Enumerator::new(nam_config(2, 0), &registry);
        let classes = e.run();
        assert_eq!(classes.len(), 1);
        assert!(classes[0].representative.is_empty());
    }
}
