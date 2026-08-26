//! Synthesizing symbolic rules: those with a hole for an arbitrary sub-circuit.
//!
//! A symbolic rule says two circuits agree *provided* whatever sits in the hole acts on
//! the boundary qubits like one of a listed set of basis permutations. Synthesizing one
//! means enumerating circuits that contain a hole, and for each pair working out the set
//! of permutations under which they agree.
//!
//! # Width and cost
//!
//! The reference fixed the boundary at two qubits (`MAX_QUBITS_SYMB = 2`), and the reason
//! is combinatorial: there are `(2^k)!` permutations of the basis of `k` qubits — 24 at
//! `k = 2`, but 40,320 at `k = 3` and over 20 trillion at `k = 4`. Testing every one of
//! them against every pair of circuits does not scale.
//!
//! The width is a parameter here rather than a constant, and the cost is managed by
//! doing the expensive part late: per permutation, circuits are bucketed by a cheap
//! fingerprint of their evaluation, and only same-bucket pairs reach exact
//! verification. `k = 3` is usable on a small enumeration; `k >= 4` is refused rather
//! than attempted.

use ndarray::Array2;
use num_complex::Complex64;
use rustc_hash::FxHashMap;

use qcircuit::{Dag, GateOp, GateRegistry};
use qrules::BasisPermutation;
use qsemantics::Unitary;

use crate::enumerate::{Enumerated, EnumerationConfig};
use crate::fingerprint::{fingerprint_matrix_up_to_phase, AngleSample, Fingerprint};

/// The gate name standing for the hole.
pub const HOLE_GATE: &str = "__symb";

/// Above this width, enumerating permutations is not attempted.
pub const MAX_SUPPORTED_WIDTH: usize = 3;

/// Every permutation of the `2^width` basis states.
///
/// `(2^width)!` of them: 2, 24, 40320 for widths 1, 2, 3.
pub fn all_permutations(width: usize) -> Vec<BasisPermutation> {
    let n = 1usize << width;
    let mut items: Vec<u32> = (0..n as u32).collect();
    let mut out = Vec::new();
    permute(&mut items, 0, &mut out, width);
    out
}

fn permute(items: &mut Vec<u32>, k: usize, out: &mut Vec<BasisPermutation>, width: usize) {
    if k == items.len() {
        if let Ok(p) = BasisPermutation::new(width, items.clone()) {
            out.push(p);
        }
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, out, width);
        items.swap(k, i);
    }
}

/// The permutation as a matrix over `width` qubits.
pub fn permutation_matrix(p: &BasisPermutation) -> Array2<Complex64> {
    let n = p.len();
    let mut m = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        m[[p.apply(i), i]] = Complex64::new(1.0, 0.0);
    }
    m
}

/// A circuit containing exactly one hole.
#[derive(Debug, Clone)]
pub struct SymbolicCircuit {
    /// Gates before the hole.
    pub before: Vec<GateOp>,
    /// Gates after the hole.
    pub after: Vec<GateOp>,
}

impl SymbolicCircuit {
    /// Split an enumerated circuit at its hole, or `None` if it has none or several.
    pub fn split(circuit: &Enumerated) -> Option<Self> {
        let positions: Vec<usize> = circuit
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| &*op.gate == HOLE_GATE)
            .map(|(i, _)| i)
            .collect();
        if positions.len() != 1 {
            return None;
        }
        let at = positions[0];
        Some(Self {
            before: circuit.ops[..at].to_vec(),
            after: circuit.ops[at + 1..].to_vec(),
        })
    }

    pub fn gate_count(&self) -> usize {
        self.before.len() + self.after.len()
    }

    /// The unitary with `permutation` substituted for the hole.
    pub fn evaluate(
        &self,
        permutation: &BasisPermutation,
        qubits: &[qcircuit::QubitId],
        boundary: &[qcircuit::QubitId],
        registry: &GateRegistry,
        sample: &mut AngleSample,
    ) -> Option<Unitary> {
        let bind = |ops: &[GateOp], sample: &mut AngleSample| {
            for op in ops {
                for p in &op.params {
                    for v in p.free_vars() {
                        sample.get(v);
                    }
                }
            }
        };
        bind(&self.before, sample);
        bind(&self.after, sample);
        let bindings = sample.bindings().clone();
        let env = move |name: &str| bindings.get(name).copied();

        let mut u = Unitary::identity(qubits.to_vec());
        let apply = |u: &mut Unitary, ops: &[GateOp]| -> Option<()> {
            let mut dag = Dag::new(qubits.to_vec());
            for op in ops {
                dag.push_gate(op.clone());
            }
            let part =
                Unitary::from_dag_over_with_env(&dag, qubits.to_vec(), registry, &env).ok()?;
            *u = compose(part.matrix(), u.matrix(), qubits.to_vec());
            Some(())
        };

        apply(&mut u, &self.before)?;
        // The hole acts on the boundary qubits, in order.
        let operands: Vec<usize> = boundary
            .iter()
            .map(|q| qubits.iter().position(|x| x == q))
            .collect::<Option<_>>()?;
        u.apply(&permutation_matrix(permutation), &operands);
        apply(&mut u, &self.after)?;
        Some(u)
    }

    /// The rule-file spelling: gates, `symb q`, gates.
    pub fn to_rule_text(&self) -> String {
        let render = |ops: &[GateOp]| {
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
        };
        let before = render(&self.before);
        let after = render(&self.after);
        let mut parts: Vec<String> = Vec::new();
        if !before.is_empty() {
            parts.push(format!("{before};"));
        }
        parts.push("symb q;".to_string());
        if !after.is_empty() {
            parts.push(format!("{after};"));
        }
        parts.join(" ")
    }
}

fn compose(
    later: &Array2<Complex64>,
    earlier: &Array2<Complex64>,
    qubits: Vec<qcircuit::QubitId>,
) -> Unitary {
    let product = later.dot(earlier);
    let mut u = Unitary::identity(qubits);
    let n = product.shape()[0];
    let identity_operands: Vec<usize> = (0..(n.trailing_zeros() as usize)).collect();
    u.apply(&product, &identity_operands);
    u
}

/// An opt-in restriction on which gates may appear on either side of the hole.
///
/// Evaluation-based agreement (see [`synthesize_symbolic`]) needs no such restriction —
/// any gate the registry can build a matrix for is admissible — so the default is
/// [`HalfRestrictions::none`]. [`HalfRestrictions::reference`] reproduces the reference
/// synthesizer's rule, for regenerating corpora shaped like the shipped ones.
#[derive(Debug, Clone, Default)]
pub struct HalfRestrictions {
    /// Gates that may not appear before the hole.
    pub not_before: Vec<String>,
    /// Gates that may not appear after the hole.
    pub not_after: Vec<String>,
}

impl HalfRestrictions {
    /// No restriction: any gate on either side of the hole.
    pub fn none() -> Self {
        Self::default()
    }

    /// The reference synthesizer's restriction, exactly: it refused to put a hole into
    /// any circuit containing `cx`, `h`, `cz`, `rx`, `ry`, `rxx`, or `sx` — so none of
    /// those could precede a hole — and refused to apply `cx` or `cz` after one. Note
    /// the asymmetry: `h` *after* the hole was allowed.
    ///
    /// Gates absent from the gate set being enumerated simply never come up; the lists
    /// need no intersecting with the set.
    pub fn reference() -> Self {
        Self {
            not_before: ["cx", "h", "cz", "rx", "ry", "rxx", "sx"]
                .map(String::from)
                .to_vec(),
            not_after: ["cx", "cz"].map(String::from).to_vec(),
        }
    }

    pub fn banned_before(&self, gate: &str) -> bool {
        self.not_before.iter().any(|g| g == gate)
    }

    pub fn banned_after(&self, gate: &str) -> bool {
        self.not_after.iter().any(|g| g == gate)
    }

    pub fn is_none(&self) -> bool {
        self.not_before.is_empty() && self.not_after.is_empty()
    }
}

/// A symbolic rule as synthesized.
#[derive(Debug, Clone)]
pub struct SynthesizedSymbolicRule {
    pub smaller: String,
    pub larger: String,
    pub constraints: Vec<BasisPermutation>,
}

impl SynthesizedSymbolicRule {
    /// The rule-file line, `smaller | larger | constraints`.
    pub fn to_line(&self) -> String {
        format!(
            "{} | {} | {}",
            self.smaller,
            self.larger,
            qrules::format_constraints(&self.constraints)
        )
    }
}

/// The shared context a symbolic comparison needs.
pub struct SymbolicContext<'a> {
    pub config: &'a EnumerationConfig,
    /// Boundary qubits, in the order the constraint's bits refer to them.
    ///
    /// Must be a prefix of the enumeration's qubit list: exact verification addresses
    /// the hole by qubit index.
    pub boundary: &'a [qcircuit::QubitId],
    pub registry: &'a GateRegistry,
    /// Independent random rational points an agreement must survive.
    pub rounds: usize,
}

/// Constraints that survive floating-point confirmation at several fresh samples.
///
/// The cheap middle tier of the pipeline: bucketing proposes candidates from one
/// sample, this confirms them at a few more in f64 — microseconds per permutation
/// against milliseconds of exact rational arithmetic — and only what survives reaches
/// [`constraints_for`]'s exact check at rule gathering. It can only over-approve
/// (every true rule passes a float evaluation of itself), so nothing exact would
/// accept is lost here.
fn float_confirmed_constraints(
    a: &SymbolicCircuit,
    b: &SymbolicCircuit,
    permutations: &[BasisPermutation],
    ctx: &SymbolicContext<'_>,
    base: &AngleSample,
) -> Vec<BasisPermutation> {
    let qubits = ctx.config.qubit_names();
    permutations
        .iter()
        .filter(|p| {
            (0..ctx.rounds.max(1)).all(|round| {
                let mut s = base.resample(round as u64 + 1);
                match (
                    a.evaluate(p, &qubits, ctx.boundary, ctx.registry, &mut s),
                    b.evaluate(p, &qubits, ctx.boundary, ctx.registry, &mut s),
                ) {
                    (Some(ua), Some(ub)) => qsemantics::phase_invariant_distance(&ua, &ub) <= 1e-9,
                    _ => false,
                }
            })
        })
        .cloned()
        .collect()
}

/// Find every permutation under which two symbolic circuits agree, **exactly**.
///
/// Each candidate permutation is checked by [`crate::exact::verify`] under the strong
/// contract: the hole is the permutation times an arbitrary phase per boundary basis
/// state, so a surviving constraint licenses *every* region whose boundary support is
/// the permutation — no per-application soundness check needed. Agreement is up to one
/// global phase, like every consumer of an emitted rule. A pair outside the exact
/// fragment yields no constraints rather than approximately-checked ones.
pub fn constraints_for(
    a: &SymbolicCircuit,
    b: &SymbolicCircuit,
    permutations: &[BasisPermutation],
    ctx: &SymbolicContext<'_>,
) -> Vec<BasisPermutation> {
    let qubits = ctx.config.qubit_names();
    debug_assert!(
        qubits.starts_with(ctx.boundary),
        "boundary must be a prefix of the qubit list"
    );
    let width = ctx.boundary.len();
    let side_a = crate::exact::Side {
        before: &a.before,
        after: &a.after,
    };
    let side_b = crate::exact::Side {
        before: &b.before,
        after: &b.after,
    };
    permutations
        .iter()
        .filter(|p| {
            matches!(
                crate::exact::verify(
                    &side_a,
                    &side_b,
                    Some((p, width)),
                    &qubits,
                    ctx.registry,
                    crate::fingerprint::Equivalence::UpToPhase,
                    ctx.rounds,
                    ctx.config.seed,
                ),
                Ok(true)
            )
        })
        .cloned()
        .collect()
}

/// Synthesize symbolic rules for a gate set: enumerate hole circuits, group, pair.
///
/// # No gate restriction
///
/// The reference refused to put a hole into any circuit containing `cx`, `h`, `cz`,
/// `rx`, `ry`, `rxx`, or `sx`, and refused to apply `cx` or `cz` after one — so its
/// symbolic rules could never carry a `cx` in their halves. The restriction guarded its
/// symbolic path-sum verifier, which could not push those gates through the hole's
/// boolean function. Here agreement is decided by *evaluating* both circuits with each
/// permutation substituted for the hole, so how a gate's state change propagates through
/// the hole's function is handled by the arithmetic itself, and any gate the registry
/// can build a matrix for is admissible on either side of the hole.
///
/// (Deciding by evaluation rather than by gate list is also what keeps unverifiable
/// shapes out: nothing is emitted that the evaluator has not confirmed.)
///
/// The reference's restriction remains available via `restrictions` —
/// [`HalfRestrictions::reference`] for its exact rule, or any other gate lists — and it
/// is what the `queso` CLI applies by default (`--symb-all-gates` lifts it). It narrows
/// what is emitted, never what is soundly emittable.
///
/// Halves are enumerated over the `width` boundary qubits only: a half gate on a
/// non-boundary wire would commute with the hole trivially and add nothing a plain rule
/// does not already say.
pub fn synthesize_symbolic(
    config: &EnumerationConfig,
    width: usize,
    registry: &GateRegistry,
    restrictions: &HalfRestrictions,
) -> Vec<SynthesizedSymbolicRule> {
    assert!(
        width <= MAX_SUPPORTED_WIDTH,
        "boundary width {width} needs {}! permutations",
        1usize << width
    );
    // Halves live on the boundary qubits, so enumerate over exactly those.
    let mut half_config = config.clone();
    half_config.max_qubits = width;
    // A gate banned on both sides of the hole can never appear in a half at all, so
    // sequences containing it are dead weight; drop it from the enumeration pool rather
    // than filtering its every cut away below.
    let banned_everywhere = |g: &str| restrictions.banned_before(g) && restrictions.banned_after(g);
    half_config.gates.retain(|g| !banned_everywhere(g));
    half_config.fixed.retain(|(g, _)| !banned_everywhere(g));

    // The pool of "atoms" halves are cut from is the *suffix-pruned, deduplicated*
    // enumeration `crate::enumerate::Enumerator` already does for plain rules -- not a
    // fresh brute force. That pruning is not an optimization of an otherwise-equivalent
    // computation, it is the difference between tractable and not: a gate set with a
    // wide angle basis (`ibmo`'s `u3` alone contributes 5^3 = 125 combinations per
    // position) makes the raw `apps^size` sequence count run into the tens of millions
    // at size 3, which is what the first version of this function did and it ran an
    // `ibm` generation out of several gigabytes of RAM before finishing. The enumerator
    // instead keeps, at each size, only sequences whose every suffix is already a
    // *canonical representative* of its own equivalence class -- a bound on genuinely
    // distinct behaviors, not on raw gate-string count -- which is exactly why the
    // plain-rule pass over the same gate set finishes in a few thousand circuits rather
    // than tens of millions.
    let mut enumerator = crate::enumerate::Enumerator::new(half_config.clone(), registry);
    let classes = enumerator.run();
    // Every recorded circuit, not just each class's chosen representative. Suffix
    // pruning already bounds *which* sequences get recorded at all -- that is what
    // fixes the blowup -- but collapsing further to one representative per behavior
    // loses literal sequences that matter here specifically because of their literal
    // form. `rz(theta1+theta2) q0; h q1;` and `h q1; rz(theta1+theta2) q0;` are the two
    // cuts of one sequence; if a class's representative happens to be some other,
    // shorter member with the same unitary behavior, that literal compound-angle
    // sequence is gone before a single cut point is ever tried. Measured: taking
    // representatives only reproduced 69 of nam's 127 valid shipped rules; every member
    // reproduces all 127.
    let sequences: Vec<Vec<GateOp>> = classes
        .into_iter()
        .flat_map(|c| c.members)
        .map(|e| e.ops)
        .collect();

    let mut circuits: Vec<SymbolicCircuit> = Vec::new();
    let mut seen: FxHashMap<String, ()> = FxHashMap::default();
    for seq in &sequences {
        // The cut is where the hole goes: `seq[..cut]` before it, `seq[cut..]` after.
        // A gate banned after the hole at position `j` forces `cut > j`; one banned
        // before the hole at position `i` forces `cut <= i`. An empty range means this
        // sequence admits no hole at all under the restriction.
        let min_cut = seq
            .iter()
            .rposition(|op| restrictions.banned_after(&op.gate))
            .map_or(0, |j| j + 1);
        let max_cut = seq
            .iter()
            .position(|op| restrictions.banned_before(&op.gate))
            .unwrap_or(seq.len());
        for cut in min_cut..=max_cut {
            let c = SymbolicCircuit {
                before: seq[..cut].to_vec(),
                after: seq[cut..].to_vec(),
            };
            let key = c.to_rule_text();
            if seen.insert(key, ()).is_none() {
                circuits.push(c);
            }
        }
    }

    let permutations = all_permutations(width);
    let boundary = half_config.qubit_names();
    let mut sample = AngleSample::new(config.seed);

    // Candidate discovery runs per permutation, because that is what a symbolic rule
    // *is*: a pair agreeing under some permutations and not others. Grouping by joint
    // behaviour across several permutations — the first version of this — silently finds
    // only pairs whose constraint sets contain every probe, which excludes almost every
    // real rule. Instead, for each permutation, circuits are bucketed by the fingerprint
    // of their evaluation with that permutation in the hole; every same-bucket pair
    // agrees there at the sampled angles, and the union over permutations gives each
    // pair its candidate constraint set.
    let qubit_names = half_config.qubit_names();
    let mut pair_constraints: FxHashMap<(usize, usize), Vec<usize>> = FxHashMap::default();
    for (pi, p) in permutations.iter().enumerate() {
        let mut buckets: FxHashMap<Fingerprint, Vec<usize>> = FxHashMap::default();
        for (i, c) in circuits.iter().enumerate() {
            if let Some(u) = c.evaluate(p, &qubit_names, &boundary, registry, &mut sample) {
                buckets
                    .entry(fingerprint_matrix_up_to_phase(u.matrix()))
                    .or_default()
                    .push(i);
            }
        }
        // All pairs within a bucket, not just each member against a chosen
        // representative. Agreement at fixed pi is an equivalence, so pairing everyone
        // through one representative says nothing *semantically* different -- but the
        // rules this emits are literal-pattern matches, not algebra the optimizer
        // composes at runtime, so a rule between two non-representative members is not
        // recoverable from the representative pairs and is a real loss. Concretely:
        // `tdg q0` and `tdg q1` both agree with `s q1; ...; t q1` shapes under some
        // permutation, but neither is the bucket's chosen representative, so
        // representative-only pairing produces two rules involving that third circuit
        // and never the direct `tdg q0 <-> tdg q1` rule the shipped corpus has.
        //
        // This was quadratic against the *raw* brute-forced circuit pool (168,324 lines
        // for nam) before `synthesize_symbolic` switched to enumerating from
        // `Enumerator`'s suffix-pruned, deduplicated members instead of every gate
        // sequence directly -- against that much smaller pool, all-pairs is the
        // corpus's actual size, not a blowup to guard against.
        for members in buckets.values() {
            for (mi, &a) in members.iter().enumerate() {
                for &b in members.iter().skip(mi + 1) {
                    let key = if a < b { (a, b) } else { (b, a) };
                    pair_constraints.entry(key).or_default().push(pi);
                }
            }
        }
    }

    let ctx = SymbolicContext {
        config: &half_config,
        boundary: &boundary,
        registry,
        rounds: config.verify_rounds.max(1),
    };
    let float_base = sample.clone();

    let mut out = Vec::new();
    for ((a_idx, b_idx), candidate) in &pair_constraints {
        // Agreement under *every* permutation is a plain rule wearing a costume; the
        // plain synthesizer already covers it.
        if candidate.len() == permutations.len() {
            continue;
        }
        let (a, b) = (&circuits[*a_idx], &circuits[*b_idx]);
        if a.to_rule_text() == b.to_rule_text() {
            continue;
        }
        // The fingerprints matched at one angle sample; confirm cheaply across several
        // more before paying for exact arithmetic on what remains.
        let selected: Vec<BasisPermutation> =
            candidate.iter().map(|&i| permutations[i].clone()).collect();
        let selected = float_confirmed_constraints(a, b, &selected, &ctx, &float_base);
        if selected.is_empty() {
            continue;
        }
        let constraints = constraints_for(a, b, &selected, &ctx);
        if constraints.is_empty() {
            continue;
        }
        let (smaller, larger) = if a.gate_count() <= b.gate_count() {
            (a, b)
        } else {
            (b, a)
        };
        out.push(SynthesizedSymbolicRule {
            smaller: smaller.to_rule_text(),
            larger: larger.to_rule_text(),
            constraints,
        });
    }
    out.sort_by(|x, y| x.smaller.cmp(&y.smaller).then(x.larger.cmp(&y.larger)));
    out.dedup_by(|x, y| x.smaller == y.smaller && x.larger == y.larger);
    out
}

#[cfg(test)]
mod synthesize_tests {
    use super::*;
    use crate::enumerate::EnumerationConfig;
    use qcircuit::gateset::GateSetLibrary;

    fn nam_config(size: usize) -> EnumerationConfig {
        let lib = GateSetLibrary::builtin();
        let set = lib.get("nam").unwrap();
        let mut c = EnumerationConfig::from_gate_set(set, 3, size);
        c.seed = 7;
        c
    }

    /// Symbolic synthesis emits rules with `cx` in their halves -- the shape the
    /// reference's gate restrictions made impossible -- and every emitted line survives
    /// the loader's independent identity check.
    #[test]
    fn synthesis_emits_cx_rules_and_every_line_is_an_identity() {
        let registry = GateRegistry::with_builtins();
        let rules = synthesize_symbolic(&nam_config(2), 2, &registry, &HalfRestrictions::none());
        assert!(
            rules.len() > 100,
            "expected a real corpus, got {}",
            rules.len()
        );

        let with_cx = rules
            .iter()
            .filter(|r| r.smaller.contains("cx") || r.larger.contains("cx"))
            .count();
        assert!(with_cx > 0, "no rule carries a cx in its halves");

        for r in &rules {
            let line = r.to_line();
            qrules::SymbolicRule::parse_legacy(&line, &registry)
                .unwrap_or_else(|e| panic!("emitted a rule the loader refuses: {e}\n  {line}"));
        }
    }

    /// Gate names in a rule half's spelling, hole excluded.
    fn half_gates(text: &str) -> impl Iterator<Item = &str> {
        text.split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|stmt| stmt.split(['(', ' ']).next().unwrap_or(stmt))
            .filter(|g| *g != "symb")
    }

    /// Under [`HalfRestrictions::reference`], no emitted rule carries a banned gate on
    /// the banned side: nothing from the reference's refusal list before the hole, no
    /// `cx` after it — the shapes its synthesizer could produce, and nothing else.
    #[test]
    fn reference_restrictions_reproduce_the_reference_shapes() {
        let registry = GateRegistry::with_builtins();
        let restricted =
            synthesize_symbolic(&nam_config(2), 2, &registry, &HalfRestrictions::reference());
        assert!(
            !restricted.is_empty(),
            "restriction must not empty the corpus"
        );

        let banned_before = ["cx", "h", "cz", "rx", "ry", "rxx", "sx"];
        for r in &restricted {
            for text in [&r.smaller, &r.larger] {
                let (before, after) = text.split_once("symb q;").unwrap();
                for g in half_gates(before) {
                    assert!(
                        !banned_before.contains(&g),
                        "`{g}` before the hole in `{text}`"
                    );
                }
                for g in half_gates(after) {
                    assert!(g != "cx" && g != "cz", "`{g}` after the hole in `{text}`");
                }
            }
        }
    }

    /// Regenerate a gate set's symbolic corpus and assert every *valid* shipped rule is
    /// reproduced, textually and in either field order. `skip` names the shipped lines
    /// that fail the load-time identity check and so must not be reproduced.
    fn assert_covers_shipped(gate_set: &str, shipped_file: &str, skip: fn(&str) -> bool) {
        let lib = GateSetLibrary::builtin();
        let set = lib.get(gate_set).unwrap();
        let mut cfg = EnumerationConfig::from_gate_set(set, 3, 3);
        cfg.seed = 7;
        let registry = GateRegistry::with_builtins();
        let rules = synthesize_symbolic(&cfg, 2, &registry, &HalfRestrictions::none());
        let mut gen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
        for r in &rules {
            gen.insert((r.smaller.clone(), r.larger.clone()));
            gen.insert((r.larger.clone(), r.smaller.clone()));
        }

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .unwrap()
            .join(shipped_file);
        let text = std::fs::read_to_string(root).unwrap();
        let mut shipped = 0;
        let mut covered = 0;
        let mut weak = 0;
        let mut missing: Vec<&str> = Vec::new();
        for line in text.lines() {
            let parts: Vec<&str> = line.split('|').map(str::trim).collect();
            if parts.len() < 3 || skip(parts[1]) {
                continue;
            }
            // The shipped corpora were float-verified under a single shared hole
            // phase, so they contain rules valid only for regions that are *exactly*
            // their permutation. Exact synthesis verifies the strong contract — free
            // phases per boundary state — and deliberately does not emit those; the
            // loader's `phase_safe` classification identifies them.
            match qrules::SymbolicRule::parse_legacy(line, &registry) {
                Ok(rule) if !rule.phase_safe => {
                    weak += 1;
                    continue;
                }
                Ok(_) => {}
                Err(_) => continue,
            }
            shipped += 1;
            if gen.contains(&(parts[0].to_string(), parts[1].to_string())) {
                covered += 1;
            } else if missing.len() < 5 {
                missing.push(line);
            }
        }
        eprintln!("{gate_set}: {weak} shipped weak-contract rules excluded from coverage");
        assert_eq!(
            covered, shipped,
            "{gate_set}: generated set covers {covered} of {shipped} valid phase-safe \
             shipped rules; first missing: {missing:#?}"
        );
    }

    /// Regenerating the nam symbolic rules reproduces every valid *phase-safe* shipped
    /// rule.
    ///
    /// The nine shipped rules that fail the identity check are refused outright, and
    /// the weak-contract rules — sound only for regions that are exactly their
    /// permutation — are deliberately not emitted by exact synthesis; everything else
    /// is reproduced.
    #[test]
    #[ignore = "regenerates the full nam symbolic corpus; run in release"]
    fn regenerating_nam_covers_the_shipped_valid_rules() {
        // The x-family is the invalid set, refused by the loader too.
        assert_covers_shipped("nam", "rules/rules_q3_s3_nam_symb.txt", |searched| {
            searched.starts_with("x q")
        });
    }

    /// As for nam: everything valid reproduced, the x-family exactly what is not.
    #[test]
    #[ignore = "regenerates the full ibmnew symbolic corpus; run in release"]
    fn regenerating_ibmnew_covers_the_shipped_valid_rules() {
        assert_covers_shipped("ibmn", "rules/rules_q3_s3_ibmnew_symb.txt", |searched| {
            searched.starts_with("x q")
        });
    }

    /// Nearly half the shipped cliffordt rules are identities only up to a global phase;
    /// exact-phase pair discovery missed every one of them, and coverage sat at 46%
    /// until discovery matched the phase-invariance of everything downstream of it.
    #[test]
    #[ignore = "regenerates the full cliffordt symbolic corpus; run in release"]
    fn regenerating_cliffordt_covers_the_shipped_rules() {
        assert_covers_shipped("cliffordt", "rules/rules_q3_s3_cliffordt_symb.txt", |_| {
            false
        });
    }

    /// Rigetti's fixed `rx` angles must print symbolically (`(pi/2)`, not
    /// `1.5707963267948966`) for the textual comparison to see the reproduction; the
    /// decimal spelling hid a fifth of the corpus.
    #[test]
    #[ignore = "regenerates the full rigetti symbolic corpus; run in release"]
    fn regenerating_rigetti_covers_the_shipped_rules() {
        assert_covers_shipped("rigetti", "rules/rules_q3_s3_rigetti_symb.txt", |_| false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::GateSetLibrary;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn config(qubits: usize, size: usize) -> EnumerationConfig {
        let lib = GateSetLibrary::builtin();
        EnumerationConfig::from_gate_set(lib.get("nam").unwrap(), qubits, size)
    }

    fn boundary(k: usize) -> Vec<qcircuit::QubitId> {
        (0..k).map(|i| qcircuit::intern(&format!("q{i}"))).collect()
    }

    #[test]
    fn permutation_counts_are_factorial() {
        assert_eq!(all_permutations(1).len(), 2);
        assert_eq!(all_permutations(2).len(), 24);
        // 8! = 40320; enumerating it is the reason the reference stopped at width 2.
        assert_eq!(all_permutations(3).len(), 40_320);
    }

    #[test]
    fn permutations_are_distinct_and_valid() {
        let ps = all_permutations(2);
        let mut seen = std::collections::HashSet::new();
        for p in &ps {
            assert_eq!(p.width(), 2);
            assert!(seen.insert(p.image().to_vec()));
        }
        assert_eq!(seen.len(), 24);
    }

    #[test]
    fn permutation_matrices_are_unitary_and_correct() {
        for p in all_permutations(2) {
            let m = permutation_matrix(&p);
            assert_eq!(m.shape(), &[4, 4]);
            // Exactly one 1 per column, at the image row.
            for i in 0..4 {
                let ones: Vec<usize> = (0..4).filter(|&r| m[[r, i]].norm() > 0.5).collect();
                assert_eq!(ones, vec![p.apply(i)]);
            }
        }
        // The identity permutation gives the identity matrix.
        let m = permutation_matrix(&BasisPermutation::identity(2));
        for i in 0..4 {
            assert!((m[[i, i]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn splitting_at_the_hole() {
        let c = Enumerated::empty()
            .extend(GateOp::new("h", vec!["q0"], []))
            .extend(GateOp::new(HOLE_GATE, vec!["q0", "q1"], []))
            .extend(GateOp::new("x", vec!["q1"], []));
        let s = SymbolicCircuit::split(&c).unwrap();
        assert_eq!(s.before.len(), 1);
        assert_eq!(s.after.len(), 1);
        assert_eq!(s.gate_count(), 2);

        // No hole, or several, is not a symbolic circuit.
        let none = Enumerated::empty().extend(GateOp::new("h", vec!["q0"], []));
        assert!(SymbolicCircuit::split(&none).is_none());
        let two = c.extend(GateOp::new(HOLE_GATE, vec!["q0", "q1"], []));
        assert!(SymbolicCircuit::split(&two).is_none());
    }

    #[test]
    fn rule_text_matches_the_reference_spelling() {
        let c = Enumerated::empty()
            .extend(GateOp::new(
                "rz",
                vec!["q0"],
                [qcircuit::AngleExpr::var("theta1")],
            ))
            .extend(GateOp::new(HOLE_GATE, vec!["q0", "q1"], []))
            .extend(GateOp::new("h", vec!["q1"], []));
        let s = SymbolicCircuit::split(&c).unwrap();
        assert_eq!(s.to_rule_text(), "rz(theta1) q0; symb q; h q1;");

        // Halves may be empty on either side.
        let only_after = Enumerated::empty()
            .extend(GateOp::new(HOLE_GATE, vec!["q0", "q1"], []))
            .extend(GateOp::new("h", vec!["q1"], []));
        assert_eq!(
            SymbolicCircuit::split(&only_after).unwrap().to_rule_text(),
            "symb q; h q1;"
        );
    }

    /// Constraints are found under the strong contract: the hole is the permutation
    /// times an arbitrary phase per boundary basis state.
    ///
    /// A diagonal gate commutes with those phases, so `rz(theta1) q0; symb` versus
    /// `symb; rz(theta1) q0` agrees under exactly the permutations that preserve
    /// `q0`'s bit — a proper, nonempty subset containing the identity. An `h`-hole
    /// commutation, by contrast, holds only when the hole's phases are trivial — the
    /// weak contract this synthesizer deliberately does not emit — so it must yield
    /// *no* constraints, where the float verifier's single shared phase accepted it.
    #[test]
    fn constraints_capture_when_two_circuits_agree() {
        let registry = reg();
        let cfg = config(2, 3);
        let b = boundary(2);
        let perms = all_permutations(2);

        let rz = |side: bool| SymbolicCircuit {
            before: if side {
                vec![GateOp::new(
                    "rz",
                    vec!["q0"],
                    [qcircuit::AngleExpr::var("theta1")],
                )]
            } else {
                vec![]
            },
            after: if side {
                vec![]
            } else {
                vec![GateOp::new(
                    "rz",
                    vec!["q0"],
                    [qcircuit::AngleExpr::var("theta1")],
                )]
            },
        };
        let ctx = SymbolicContext {
            config: &cfg,
            boundary: &b,
            registry: &registry,
            rounds: 2,
        };
        let found = constraints_for(&rz(true), &rz(false), &perms, &ctx);
        assert!(
            !found.is_empty(),
            "the identity permutation at least should agree"
        );
        assert!(found.len() < perms.len(), "not every permutation can agree");
        assert!(found.iter().any(BasisPermutation::is_identity));
        // Exactly the permutations that fix q0's bit: 2 boundary states with q0 = 0
        // permute among themselves, likewise q0 = 1, so 2! * 2! = 4 of the 24.
        assert_eq!(found.len(), 4);

        // A circuit against itself agrees under every permutation.
        let all = constraints_for(&rz(true), &rz(true), &perms, &ctx);
        assert_eq!(all.len(), perms.len());

        // The h-hole commutation is weak-contract only: no constraints survive.
        let h = |side: bool| SymbolicCircuit {
            before: if side {
                vec![GateOp::new("h", vec!["q0"], [])]
            } else {
                vec![]
            },
            after: if side {
                vec![]
            } else {
                vec![GateOp::new("h", vec!["q0"], [])]
            },
        };
        assert!(constraints_for(&h(true), &h(false), &perms, &ctx).is_empty());
    }

    #[test]
    fn circuits_that_never_agree_have_no_constraints() {
        let registry = reg();
        let cfg = config(2, 3);
        let b = boundary(2);
        let perms = all_permutations(2);

        // `x q0` before the hole versus `h q0` before it: never the same.
        let a = SymbolicCircuit {
            before: vec![GateOp::new("x", vec!["q0"], [])],
            after: vec![],
        };
        let c = SymbolicCircuit {
            before: vec![GateOp::new("h", vec!["q0"], [])],
            after: vec![],
        };
        let ctx = SymbolicContext {
            config: &cfg,
            boundary: &b,
            registry: &registry,
            rounds: 2,
        };
        assert!(constraints_for(&a, &c, &perms, &ctx).is_empty());
    }

    /// A hole that is the identity permutation leaves a circuit unchanged.
    #[test]
    fn the_identity_permutation_is_a_no_op() {
        let registry = reg();
        let cfg = config(2, 2);
        let qubits = cfg.qubit_names();
        let b = boundary(2);
        let mut sample = AngleSample::new(1);

        let with_hole = SymbolicCircuit {
            before: vec![GateOp::new("h", vec!["q0"], [])],
            after: vec![GateOp::new("x", vec!["q1"], [])],
        };
        let u = with_hole
            .evaluate(
                &BasisPermutation::identity(2),
                &qubits,
                &b,
                &registry,
                &mut sample,
            )
            .unwrap();

        // The same circuit with no hole at all.
        let mut plain = Dag::new(qubits.clone());
        plain.push_gate(GateOp::new("h", vec!["q0"], []));
        plain.push_gate(GateOp::new("x", vec!["q1"], []));
        let v = Unitary::from_dag_over(&plain, qubits, &registry).unwrap();
        assert!(qsemantics::equivalent_exact(&u, &v, 1e-9));
    }

    /// Width generality, demonstrated: a three-qubit boundary works.
    ///
    /// This is what the reference could not express at all — its constraints were
    /// `boolean[2]` and its matcher compared against the literals `"q0"` and `"q1"`.
    #[test]
    fn a_three_qubit_boundary_can_be_constrained() {
        let registry = reg();
        let cfg = config(3, 3);
        let b = boundary(3);

        // A small, hand-picked set of width-3 permutations rather than all 40,320.
        let candidates: Vec<BasisPermutation> = vec![
            BasisPermutation::identity(3),
            // Swap the low two bits.
            BasisPermutation::new(
                3,
                (0..8)
                    .map(|i| {
                        let (a, bb, c) = ((i >> 2) & 1, (i >> 1) & 1, i & 1);
                        ((a << 2) | (c << 1) | bb) as u32
                    })
                    .collect(),
            )
            .unwrap(),
            // Flip the top bit.
            BasisPermutation::new(3, (0..8).map(|i| (i ^ 0b100) as u32).collect()).unwrap(),
        ];

        // `rz(theta1) q2; symb` versus `symb; rz(theta1) q2`: agrees exactly when the
        // hole leaves qubit 2 alone.
        let a = SymbolicCircuit {
            before: vec![GateOp::new(
                "rz",
                vec!["q2"],
                [qcircuit::AngleExpr::var("theta1")],
            )],
            after: vec![],
        };
        let c = SymbolicCircuit {
            before: vec![],
            after: vec![GateOp::new(
                "rz",
                vec!["q2"],
                [qcircuit::AngleExpr::var("theta1")],
            )],
        };
        let ctx = SymbolicContext {
            config: &cfg,
            boundary: &b,
            registry: &registry,
            rounds: 3,
        };
        let found = constraints_for(&a, &c, &candidates, &ctx);
        assert!(!found.is_empty(), "at least the identity should agree");
        for p in &found {
            assert_eq!(p.width(), 3, "the constraint must carry its width");
        }
        // The identity leaves q2 alone, so it agrees.
        assert!(found.iter().any(BasisPermutation::is_identity));
        // Flipping the top bit is qubit 0 in big-endian order, which also leaves q2 alone.
        assert!(found.len() >= 2, "found {} constraints", found.len());
    }

    #[test]
    fn rule_lines_round_trip_through_the_reader() {
        let rule = SynthesizedSymbolicRule {
            smaller: "rz(theta1) q0; symb q;".into(),
            larger: "symb q; rz(theta1) q0;".into(),
            constraints: vec![BasisPermutation::identity(2)],
        };
        let line = rule.to_line();
        let parsed = qrules::SymbolicRule::parse_legacy_with_builtins(&line)
            .unwrap_or_else(|e| panic!("`{line}` did not parse: {e}"));
        assert_eq!(parsed.constraints.len(), 1);
        assert!(parsed.constraints[0].is_identity());
        assert_eq!(
            parsed.boundary,
            vec![qcircuit::intern("q0"), qcircuit::intern("q1")]
        );
    }
}
