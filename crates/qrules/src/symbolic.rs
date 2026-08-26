//! Symbolic rewrite rules.
//!
//! A symbolic rule has a *hole*: it says that some gates before an arbitrary sub-circuit
//! and some gates after it can be rewritten, provided the sub-circuit acts on the rule's
//! boundary qubits like one of a listed set of basis permutations. The shipped files
//! spell the hole `symb q`:
//!
//! ```text
//! rz((theta1+theta2)) q0; symb q; h q1;  |  symb q; h q1; rz((theta1+theta2)) q0;  |  [{...}, ...]
//! ```
//!
//! # What changed from the reference
//!
//! The reference fixed the boundary at two qubits and matched by walking a fixed number
//! of topological layers.
//!
//! Here the boundary is whatever qubits the rule's two halves actually touch, of any
//! number, and the region between the halves is derived from the DAG's causal structure
//! rather than from layer arithmetic.

use rand::seq::SliceRandom;
use rand::Rng;
use rustc_hash::{FxHashMap, FxHashSet};

use qcircuit::{
    angles_equivalent, AngleExpr, Dag, GateOp, GateRegistry, NodeIndex, QubitId, ANGLE_EPS,
};

use crate::constraint::{self, BasisPermutation};
use crate::error::{Result, RuleError};
use crate::matcher::{Match, MatchContext, MatchIndex};
use crate::pattern::Pattern;
use crate::rewrite::substitute_angle;

/// The marker the synthesizer writes for the hole.
pub const HOLE: &str = "symb q";

/// Limits on the sub-circuit that may sit in the hole.
#[derive(Debug, Clone)]
pub struct SymbolicLimits {
    /// Largest number of qubits the sub-circuit may touch.
    pub max_qubits: usize,
    /// Largest number of gates the sub-circuit may contain.
    pub max_gates: usize,
    /// How far past the before-half, in topological positions, the after-half may be
    /// sought.
    ///
    /// The search is a nested loop -- every before-seed against every after-seed -- and
    /// each pair runs a forward and a backward reachability walk to find the region
    /// between them. Left unbounded that is cubic in the circuit, which on a
    /// ten-thousand-gate circuit meant a *single* symbolic rule taking 55 seconds and
    /// running clean past the search's whole wall-clock budget.
    ///
    /// This is a search limit in the same sense as `max_gates` and `max_qubits`: it
    /// restricts which matches are *looked for*, never which are valid, so no rewrite it
    /// admits could have been wrong without it. The reference had an equivalent bound
    /// implicitly -- `findSymb` looked a fixed number of layers ahead -- and matching by
    /// causal structure removed it; this puts it back deliberately, as a number.
    ///
    /// The default is far larger than any distance a rule with at most `max_gates` gates
    /// between its halves plausibly spans, so small-circuit behaviour is unchanged.
    /// `None` searches the whole circuit.
    pub max_span: Option<usize>,
    /// Stop searching once this instant has passed. `None` searches without a clock.
    pub deadline: Option<std::time::Instant>,
    /// Re-verify each candidate rewrite's whole span by simulation before accepting it.
    ///
    /// Off by default: a rule's constraints are verified at load (and, for corpora from
    /// this port's synthesizer, exactly at synthesis), and the region check establishes
    /// per match that the hole's content satisfies one — so the span check re-proves an
    /// implication whose premises are already established. It remains available as
    /// defense in depth against implementation bugs in matching itself; the QCEC
    /// equivalence tests are the independent backstop for that class of bug.
    pub verify_rewrites: bool,
}

impl Default for SymbolicLimits {
    fn default() -> Self {
        // The reference's defaults, from `Params.MAX_SYMB_QUBITS` and `MAX_SYMB_SIZE`.
        Self {
            max_qubits: 7,
            max_gates: 10,
            max_span: Some(512),
            deadline: None,
            verify_rewrites: false,
        }
    }
}

/// A rewrite rule with a hole.
#[derive(Debug, Clone)]
pub struct SymbolicRule {
    /// Gates before the hole, on the search side.
    pub find_before: Pattern,
    /// Gates after the hole, on the search side.
    pub find_after: Pattern,
    /// Gates before the hole, on the replacement side.
    pub replace_before: Pattern,
    /// Gates after the hole, on the replacement side.
    pub replace_after: Pattern,
    /// Boundary qubits, in the order the constraint's bits refer to them.
    ///
    /// Derived from the rule's own halves rather than assumed to be `["q0", "q1"]`.
    pub boundary: Vec<QubitId>,
    pub constraints: Vec<BasisPermutation>,
    /// Whether the rule tolerates regions that carry basis-state-dependent phases.
    ///
    /// A rule is *verified*, at synthesis time, with the hole as the literal permutation
    /// matrix. A matched region is only required to have the permutation's **support**:
    /// its unitary factors as `V = (Pi (x) I) W` with `W` block-diagonal over boundary
    /// basis states — arbitrary phases per state, arbitrary action on non-boundary
    /// qubits conditioned on the boundary. `cx a, b; t a;` is such a region, and it is
    /// not the phase-free permutation the verification used.
    ///
    /// Whether that gap matters reduces to one operator. Writing the rule as
    /// `C1; hole; C2 | D1; hole; D2` (halves as unitaries over the boundary), validity
    /// for a region `V = Pi W` works out to `[B, W] = 0` for `B = D1 C1†`. The
    /// commutant of the block-diagonal algebra is the diagonal, so:
    ///
    /// - **`B` diagonal** — every region with the right support is safe, phases and
    ///   all. The before-halves differ by a diagonal, whose action depends only on
    ///   basis-state values, which is exactly the information the permutation preserves.
    /// - **`B` not diagonal** — some supported region breaks the rule. Concretely,
    ///   `cx q0,q1; symb | symb; cx q0,q1` with the cx permutation as constraint
    ///   rewrites `cx; cz; cx` into `cz`, and those differ. Such rules stay usable, but
    ///   only against regions that *are* the bare permutation with one global phase and
    ///   no boundary-conditioned action ([`satisfies_exactly`]).
    ///
    /// Computed at parse time by sampling the halves' angles at three independent
    /// points. `rz`-style gates are diagonal at every angle, and a gate that is only
    /// accidentally diagonal has off-diagonal entries that are analytic in its angles,
    /// so vanishing at three random points without vanishing identically is a
    /// measure-zero event. `SymbolicLimits::verify_rewrites` re-enables the whole-span
    /// re-check as defense in depth on top of this.
    pub phase_safe: bool,
    pub id: String,
}

impl SymbolicRule {
    /// Parse a symbolic rule from a legacy rule-file line.
    ///
    /// Fields are `smaller | larger | constraints`; as with plain rules the *larger* side
    /// is what gets searched for.
    /// Parse and verify against a registry the caller already has.
    ///
    /// Building a registry is not free, and this runs once per rule in a file: a
    /// generated file can have tens of thousands of lines, and a version that built its
    /// own `GateRegistry::with_builtins()` here paid for one per rule while every caller
    /// already had one in scope. `parse_legacy_with_builtins` below is a convenience for
    /// callers -- mostly tests -- with no registry handy.
    pub fn parse_legacy(line: &str, registry: &GateRegistry) -> Result<Self> {
        let fields: Vec<&str> = line.split(" | ").collect();
        if fields.len() < 3 {
            return Err(RuleError::Malformed(line.to_string()));
        }
        let (replace_src, find_src, constraint_src) = (fields[0], fields[1], fields[2]);
        let (find_before, find_after) = split_hole(find_src)?;
        let (replace_before, replace_after) = split_hole(replace_src)?;

        // A half may be disconnected -- `symb q; h q1; rz(theta) q0;` has an after-half
        // of two gates on unconnected wires -- because the hole fixes where it goes.
        let find_before = Pattern::parse_multi(&find_before)?;
        let find_after = Pattern::parse_multi(&find_after)?;
        let replace_before = Pattern::parse_multi(&replace_before)?;
        let replace_after = Pattern::parse_multi(&replace_after)?;
        let constraints = constraint::parse_constraints(constraint_src)?;

        let width = constraints[0].width();
        if constraints.iter().any(|c| c.width() != width) {
            return Err(RuleError::Malformed(format!(
                "constraints of mixed width in `{line}`"
            )));
        }

        // In the legacy format, constraint bit `j` refers to the pattern qubit named
        // `qj`; the reference encoded this by comparing qubit names against the literals
        // "q0" and "q1". Naming the convention once here keeps it out of the matcher, and
        // a future format can set `boundary` to anything.
        let boundary: Vec<QubitId> = (0..width)
            .map(|j| qcircuit::intern(&format!("q{j}")))
            .collect();

        // Every qubit any half touches must be a boundary qubit. Note this spans all four
        // halves, not just the two searched for: a rule may move a gate from `q1` before
        // the hole to `q0` after it, so the replacement side can introduce a boundary
        // qubit the search side never mentions.
        for half in [&find_before, &find_after, &replace_before, &replace_after] {
            for q in half.qubits() {
                if !boundary.contains(q) {
                    return Err(RuleError::Malformed(format!(
                        "qubit `{q}` is outside the width-{width} boundary: `{line}`"
                    )));
                }
            }
        }

        let phase_safe =
            before_halves_differ_by_a_diagonal(&find_before, &replace_before, &boundary, registry);

        let mut rule = Self {
            find_before,
            find_after,
            replace_before,
            replace_after,
            boundary,
            constraints,
            phase_safe,
            id: line.to_string(),
        };

        // Keep only constraints under which the rule is actually an identity.
        //
        // A constraint is a claim: with the hole acting as this permutation, the two
        // sides are equal. The claim is checkable in microseconds -- build both sides
        // over the boundary with the permutation as the hole and compare -- so it is
        // checked, and a false claim is dropped rather than trusted.
        //
        // This is not paranoia: several shipped corpora contain rules that fail this
        // check for *every* listed permutation -- 540 of the 1,152 `ibm` rules among
        // them -- and this port applies rules the reference's `+`/`-` pattern filter
        // would have set aside, so it must do the checking that filter happened to be
        // standing in for.
        rule.constraints = rule.verified_constraints(registry);
        if rule.constraints.is_empty() {
            return Err(RuleError::Malformed(format!(
                "no constraint makes the rule an identity: `{line}`"
            )));
        }
        Ok(rule)
    }

    /// As [`parse_legacy`](Self::parse_legacy), building a throwaway registry.
    ///
    /// For one-off calls -- tests, a single ad hoc rule -- where constructing a registry
    /// just to pass it in would be more code than it saves. Anything parsing more than a
    /// handful of rules should build one registry and call `parse_legacy` directly.
    pub fn parse_legacy_with_builtins(line: &str) -> Result<Self> {
        Self::parse_legacy(line, &GateRegistry::with_builtins())
    }

    /// The subset of this rule's constraints under which it is an identity.
    ///
    /// Checked numerically over the boundary at sampled angles, up to a global phase --
    /// the same equivalence the optimizer's distances use. A permutation that fails at
    /// any sample is out.
    fn verified_constraints(&self, registry: &GateRegistry) -> Vec<BasisPermutation> {
        const SAMPLES: [f64; 2] = [0.734_121_9, 1.912_837_3];
        let mut vars: Vec<String> = Vec::new();
        for half in [
            &self.find_before,
            &self.find_after,
            &self.replace_before,
            &self.replace_after,
        ] {
            for idx in half.dag().gate_indices() {
                for p in &half.dag().gate(idx).params {
                    for v in p.free_vars() {
                        if !vars.iter().any(|x| x == v) {
                            vars.push(v.to_string());
                        }
                    }
                }
            }
        }

        let half_u = |half: &Pattern, env: &dyn Fn(&str) -> Option<f64>| {
            qsemantics::Unitary::from_dag_over_with_env(
                half.dag(),
                self.boundary.to_vec(),
                registry,
                env,
            )
        };

        // Per round, the four half-unitaries are the same for every constraint -- only
        // the permutation matrix `pi` varies. The first version of this rebuilt all four
        // inside the per-constraint filter, so a rule with 24 constraints paid for 24
        // redundant unitary builds per round instead of one; over a 31,573-rule
        // generated file that was most of a 4.5-second load. Building each half once per
        // round and checking every constraint against it is the same math, done once.
        let mut alive: Vec<bool> = vec![true; self.constraints.len()];
        for round in 0..SAMPLES.len() {
            let assignment: FxHashMap<String, f64> = vars
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        v.clone(),
                        SAMPLES[(i + round) % SAMPLES.len()] + round as f64,
                    )
                })
                .collect();
            let env = |name: &str| assignment.get(name).copied();
            let (Ok(c1), Ok(c2), Ok(d1), Ok(d2)) = (
                half_u(&self.find_before, &env),
                half_u(&self.find_after, &env),
                half_u(&self.replace_before, &env),
                half_u(&self.replace_after, &env),
            ) else {
                return Vec::new(); // no gate registered / bad env: nothing verifiable
            };
            let dim = 1usize << self.boundary.len();

            for (perm, keep) in self.constraints.iter().zip(alive.iter_mut()) {
                if !*keep {
                    continue;
                }
                let mut pi: ndarray::Array2<num_complex::Complex64> =
                    ndarray::Array2::zeros((dim, dim));
                for x in 0..dim {
                    pi[[perm.apply(x), x]] = num_complex::Complex64::new(1.0, 0.0);
                }
                let lhs = c2.matrix().dot(&pi).dot(c1.matrix());
                let rhs = d2.matrix().dot(&pi).dot(d1.matrix());
                // Up to one global phase, aligned by the overlap trace.
                let tr: num_complex::Complex64 =
                    lhs.iter().zip(rhs.iter()).map(|(x, y)| x.conj() * y).sum();
                if tr.norm() < 1e-9 {
                    *keep = false;
                    continue;
                }
                let c = tr / tr.norm();
                let dist = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(x, y)| (x - c.conj() * y).norm())
                    .fold(0.0, f64::max);
                if dist > 1e-9 {
                    *keep = false;
                }
            }
            if alive.iter().all(|&k| !k) {
                return Vec::new(); // nothing survived; later rounds cannot resurrect it
            }
        }

        self.constraints
            .iter()
            .zip(alive.iter())
            .filter(|(_, &keep)| keep)
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Net change in gate count when this rule fires.
    pub fn size_delta(&self) -> isize {
        (self.replace_before.gate_count() + self.replace_after.gate_count()) as isize
            - (self.find_before.gate_count() + self.find_after.gate_count()) as isize
    }

    pub fn validate(&self, registry: &GateRegistry) -> Result<()> {
        for p in [
            &self.find_before,
            &self.find_after,
            &self.replace_before,
            &self.replace_after,
        ] {
            p.validate(registry)?;
        }
        Ok(())
    }
}

/// Split a rule side at the hole marker.
fn split_hole(src: &str) -> Result<(String, String)> {
    let idx = src
        .find("symb")
        .ok_or_else(|| RuleError::Malformed(format!("no hole in `{src}`")))?;
    let before = src[..idx].trim().trim_end_matches(';').trim().to_string();
    let rest = &src[idx..];
    let after = match rest.find(';') {
        Some(semi) => rest[semi + 1..].trim().to_string(),
        None => String::new(),
    };
    Ok((
        if before.is_empty() {
            String::new()
        } else {
            format!("{before};")
        },
        after,
    ))
}

/// Is `D1 * C1^dagger` diagonal, sampled over the halves' free angles?
///
/// This is the operator the [`SymbolicRule::phase_safe`] doc derives: diagonal means the
/// two before-halves differ only by something whose action depends on basis-state values
/// alone, so any region with the constraint's support commutes with the difference.
fn before_halves_differ_by_a_diagonal(
    find_before: &Pattern,
    replace_before: &Pattern,
    boundary: &[QubitId],
    registry: &GateRegistry,
) -> bool {
    // Distinct, irrational-looking samples: an angle at which a parameterized gate is
    // accidentally diagonal (rx(0), ry(2*pi)...) must not decide the classification.
    const SAMPLES: [f64; 3] = [0.734_121_9, 1.912_837_3, 2.541_113_7];
    let mut vars: Vec<String> = Vec::new();
    for half in [find_before, replace_before] {
        for idx in half.dag().gate_indices() {
            for p in &half.dag().gate(idx).params {
                for v in p.free_vars() {
                    if !vars.iter().any(|x| x == v) {
                        vars.push(v.to_string());
                    }
                }
            }
        }
    }
    for round in 0..SAMPLES.len() {
        let assignment: FxHashMap<String, f64> = vars
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    v.clone(),
                    SAMPLES[(i + round) % SAMPLES.len()] + round as f64,
                )
            })
            .collect();
        let env = |name: &str| assignment.get(name).copied();
        let Ok(c1) = qsemantics::Unitary::from_dag_over_with_env(
            find_before.dag(),
            boundary.to_vec(),
            registry,
            &env,
        ) else {
            return false;
        };
        let Ok(d1) = qsemantics::Unitary::from_dag_over_with_env(
            replace_before.dag(),
            boundary.to_vec(),
            registry,
            &env,
        ) else {
            return false;
        };
        let b = d1.matrix().dot(c1.adjoint().matrix());
        let n = b.nrows();
        for i in 0..n {
            for j in 0..n {
                if i != j && b[[i, j]].norm() > 1e-9 {
                    return false;
                }
            }
        }
    }
    true
}

/// A region's unitary, arranged for constraint checks: boundary qubits first, so
/// boundary bits are the high bits.
///
/// A candidate region is tested against *every* constraint of a rule — up to 24 at
/// width 2 — and the unitary depends only on the region, never on the permutation.
/// Building it once here and checking each permutation against the matrix keeps the
/// expensive part (`Unitary::from_dag_over`, exponential in the region's width) out of
/// the per-constraint loop, where the first version paid it per permutation.
struct RegionAction {
    m: ndarray::Array2<num_complex::Complex64>,
    /// Boundary width in qubits.
    k: usize,
    /// Non-boundary qubits of the region.
    rest: usize,
}

impl RegionAction {
    /// `None` if the region is too wide to decide or a gate has no matrix — both are
    /// treated as "no constraint can be satisfied" rather than guessed at.
    fn new(sub: &Dag, boundary: &[QubitId], registry: &GateRegistry) -> Option<Self> {
        let mut order: Vec<QubitId> = boundary.to_vec();
        for q in sub.qubits() {
            if !order.contains(q) {
                order.push(q.clone());
            }
        }
        let k = boundary.len();
        let n = order.len();
        if n > 12 {
            return None;
        }
        let u = qsemantics::Unitary::from_dag_over(sub, order, registry).ok()?;
        Some(Self {
            m: u.matrix().clone(),
            k,
            rest: n - k,
        })
    }

    /// Does the region act on the boundary like `perm`?
    ///
    /// For every classical input on the boundary qubits — and every assignment of the
    /// region's other qubits — the output must be supported only on states whose
    /// boundary bits are `perm`'s image of the input. That is the property the rule's
    /// path-sum derivation assumes, and it admits regions that are not themselves
    /// classical, as long as their action *on the boundary* is.
    fn supports(&self, perm: &BasisPermutation, tol: f64) -> bool {
        debug_assert_eq!(self.k, perm.width());
        let dim = self.m.nrows();
        for input in 0..(1usize << self.k) {
            let want = perm.apply(input);
            for other in 0..(1usize << self.rest) {
                let col = (input << self.rest) | other;
                for row in 0..dim {
                    if self.m[[row, col]].norm() <= tol {
                        continue;
                    }
                    if (row >> self.rest) != want {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Does the region act on the boundary *exactly* as `perm`, up to one global phase?
    ///
    /// The stricter tier of region acceptance, for rules that are not
    /// [`phase_safe`](SymbolicRule::phase_safe): the region's unitary must factor as
    /// `Pi (x) U`, the literal permutation on the boundary times a single unitary on
    /// the remaining qubits — no basis-state-dependent phases, no boundary-conditioned
    /// action on the rest. Every classical circuit (`x`, `cx`, `ccx`) qualifies;
    /// `cx a,b; t a;` does not, and for these rules must not.
    fn is_exactly(&self, perm: &BasisPermutation, tol: f64) -> bool {
        debug_assert_eq!(self.k, perm.width());
        let env_dim = 1usize << self.rest;
        // The block at input x sits at rows `perm(x)`; all blocks must equal the block
        // at x = 0. Extracting the reference block from the unitary itself absorbs the
        // one global phase the factorization allows.
        for input in 0..(1usize << self.k) {
            let want = perm.apply(input);
            for other in 0..env_dim {
                let col = (input << self.rest) | other;
                for row in 0..self.m.nrows() {
                    let expected = if (row >> self.rest) == want {
                        self.m[[(perm.apply(0) << self.rest) | (row & (env_dim - 1)), other]]
                    } else {
                        num_complex::Complex64::new(0.0, 0.0)
                    };
                    if (self.m[[row, col]] - expected).norm() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Does `sub` act on `boundary` like `perm`? See [`RegionAction::supports`].
pub fn satisfies(
    sub: &Dag,
    boundary: &[QubitId],
    perm: &BasisPermutation,
    registry: &GateRegistry,
    tol: f64,
) -> bool {
    RegionAction::new(sub, boundary, registry).is_some_and(|r| r.supports(perm, tol))
}

/// Does `sub` act on `boundary` *exactly* as `perm`, up to one global phase? See
/// [`RegionAction::is_exactly`].
pub fn satisfies_exactly(
    sub: &Dag,
    boundary: &[QubitId],
    perm: &BasisPermutation,
    registry: &GateRegistry,
    tol: f64,
) -> bool {
    RegionAction::new(sub, boundary, registry).is_some_and(|r| r.is_exactly(perm, tol))
}

/// A located symbolic match: the two halves and everything between them.
#[derive(Debug, Clone)]
pub struct SymbolicMatch {
    pub before: Match,
    pub after: Match,
    /// Gates strictly between the two halves.
    pub between: Vec<NodeIndex>,
    /// Which constraint the in-between region satisfied.
    pub constraint: usize,
    /// Every rule boundary qubit to the circuit qubit it stands for.
    ///
    /// The halves' own matches cover the qubits they touch; this also covers boundary
    /// qubits bound through the region, which the replacement side is allowed to name.
    pub boundary_map: FxHashMap<QubitId, QubitId>,
    /// The rewrite's whole span — both halves and everything between — in circuit
    /// order.
    ///
    /// Ordered by the search from the topological numbering it already holds, so that
    /// applying the match never pays a fresh topological pass over the whole circuit
    /// just to order a handful of nodes.
    pub ordered_span: Vec<NodeIndex>,
}

impl SymbolicMatch {
    /// The angle-binding keys the two halves established: variables and whole
    /// expressions, by their canonical text.
    pub fn bound_angle_keys(&self) -> std::collections::BTreeSet<String> {
        self.before
            .angles
            .keys()
            .chain(self.after.angles.keys())
            .cloned()
            .collect()
    }

    /// Every node in the rewrite's span: both halves and everything between them.
    ///
    /// The halves' node lists are in the pattern's own BFS order; for circuit order use
    /// [`ordered_span`](SymbolicMatch::ordered_span).
    pub fn span(&self) -> Vec<NodeIndex> {
        let mut v = self.before.nodes.clone();
        v.extend(self.between.iter().copied());
        v.extend(self.after.nodes.iter().copied());
        v
    }
}

/// The per-search invariants every candidate check needs.
///
/// One symbolic search touches these from several levels — seed loops, region growth,
/// boundary completion, constraint checks, the final soundness check — and passing them
/// individually had every helper carrying eight parameters. The topological numbering is
/// the load-bearing member: computed once per search, it serves seed windowing, span
/// ordering, and every convexity check.
#[derive(Clone, Copy)]
struct SearchEnv<'a> {
    dag: &'a Dag,
    rule: &'a SymbolicRule,
    limits: &'a SymbolicLimits,
    registry: &'a GateRegistry,
    topo_index: &'a FxHashMap<NodeIndex, usize>,
}

impl SearchEnv<'_> {
    /// `true` once the search's deadline has passed.
    fn deadline_hit(&self) -> bool {
        self.limits
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
    }

    /// The search's topological numbering, in the form [`Dag::is_convex_with`] takes.
    fn key_of(&self, n: NodeIndex) -> Option<u64> {
        self.topo_index.get(&n).map(|&i| i as u64)
    }

    /// Complete the boundary injectively and try every completion's constraints.
    ///
    /// `pinned` holds the boundary qubits the halves bound, in the rule's boundary
    /// order; a boundary qubit neither half touches is `None` and is completed from the
    /// region's own qubits. Two boundary qubits pinned to one circuit wire cannot be
    /// completed injectively, so that is refused here rather than at each caller.
    fn try_boundary_completions(
        &self,
        sub: &Dag,
        pinned: &[Option<QubitId>],
        before: &Match,
        after: &Match,
        between: &[NodeIndex],
    ) -> Option<SymbolicMatch> {
        let bound: Vec<&QubitId> = pinned.iter().flatten().collect();
        if bound.iter().collect::<FxHashSet<_>>().len() != bound.len() {
            return None;
        }
        let taken: FxHashSet<QubitId> = pinned.iter().flatten().cloned().collect();
        let free_candidates: Vec<QubitId> = sub
            .qubits()
            .iter()
            .filter(|q| !taken.contains(*q))
            .cloned()
            .collect();
        for boundary in boundary_assignments(pinned, &free_candidates) {
            if let Some(m) = self.try_constraints(sub, &boundary, before, after, between) {
                return Some(m);
            }
        }
        None
    }

    /// Try every constraint against a fully bound boundary; on a hit, verify and return.
    fn try_constraints(
        &self,
        sub: &Dag,
        boundary: &[QubitId],
        before: &Match,
        after: &Match,
        between: &[NodeIndex],
    ) -> Option<SymbolicMatch> {
        // The region's unitary is the same for every constraint; build it once. So is
        // the span's circuit order, from the numbering the search already holds.
        let action = RegionAction::new(sub, boundary, self.registry)?;
        let mut ordered_span: Vec<NodeIndex> = before
            .nodes
            .iter()
            .chain(between.iter())
            .chain(after.nodes.iter())
            .copied()
            .collect();
        ordered_span
            .sort_unstable_by_key(|n| self.topo_index.get(n).copied().unwrap_or(usize::MAX));
        for (i, c) in self.rule.constraints.iter().enumerate() {
            // The tier is the load-time classification: support alone licenses a
            // phase-safe rule, and only the exact permutation licenses the rest.
            // See `SymbolicRule::phase_safe` for the argument.
            let region_ok = if self.rule.phase_safe {
                action.supports(c, 1e-9)
            } else {
                action.is_exactly(c, 1e-9)
            };
            if !region_ok {
                continue;
            }
            let boundary_map: FxHashMap<QubitId, QubitId> = self
                .rule
                .boundary
                .iter()
                .cloned()
                .zip(boundary.iter().cloned())
                .collect();
            let m = SymbolicMatch {
                before: before.clone(),
                after: after.clone(),
                between: between.to_vec(),
                constraint: i,
                boundary_map,
                ordered_span: ordered_span.clone(),
            };
            // Opt-in defense in depth (`SymbolicLimits::verify_rewrites`): build the
            // span before and after and compare unitaries. A convex span is a
            // contiguous factor of the circuit, so local equivalence implies global
            // equivalence. Failing it abandons this candidate, not the whole search.
            if self.limits.verify_rewrites && !self.rewrite_is_sound(&m) {
                continue;
            }
            return Some(m);
        }
        None
    }

    /// Check the rewrite on the span alone.
    ///
    /// Returns `false` if the span is too wide to simulate, which is treated as "do not
    /// apply" rather than "assume fine".
    fn rewrite_is_sound(&self, m: &SymbolicMatch) -> bool {
        let original = extract(self.dag, &m.ordered_span);
        if original.num_qubits() > self.limits.max_qubits.max(4) + 2 {
            return false;
        }

        // The rewritten span: replacement-before, the untouched region, replacement-after.
        let mut rewritten = Dag::new(Vec::<String>::new());
        append_half(&mut rewritten, &self.rule.replace_before, m);
        for &n in &m.between {
            rewritten.push_gate(self.dag.gate(n).clone());
        }
        append_half(&mut rewritten, &self.rule.replace_after, m);

        let mut order: Vec<QubitId> = original.qubits().to_vec();
        for q in rewritten.qubits() {
            if !order.contains(q) {
                order.push(q.clone());
            }
        }
        order.sort();
        if order.len() > 12 {
            return false;
        }

        let (Ok(a), Ok(b)) = (
            qsemantics::Unitary::from_dag_over(&original, order.clone(), self.registry),
            qsemantics::Unitary::from_dag_over(&rewritten, order, self.registry),
        ) else {
            return false;
        };
        let d = qsemantics::phase_invariant_distance(&a, &b);
        d <= 1e-9
    }
}

/// Compound angle expressions must agree with their variables *across* the halves.
///
/// Each half's own match already enforces this within itself
/// (`MatchContext::validate`), but a rule can bind `(theta1+theta2)` opaquely in one
/// half and `theta1`, `theta2` individually in the other — and then neither half's
/// check ever relates them, so the pair matched sites where the compound was not the
/// sum, and the rewrite changed what the circuit computes. The span simulation used
/// to mask this; the check belongs here, where the two halves' bindings first meet.
/// `merged` must carry both halves' bindings (the after-half inherits the before's).
fn cross_half_angles_consistent(rule: &SymbolicRule, merged: &Match) -> bool {
    let env = |name: &str| merged.angles.get(name).and_then(|e: &AngleExpr| e.eval());
    for half in [&rule.find_before, &rule.find_after] {
        for idx in half.dag().gate_indices() {
            for p in &half.dag().gate(idx).params {
                if p.free_vars().is_empty() {
                    continue;
                }
                // Determined only once every variable in it has an individual binding.
                let (Some(want), Some(bound)) =
                    (p.eval_with(&env), merged.angles.get(&p.to_string()))
                else {
                    continue;
                };
                let Some(got) = bound.eval() else {
                    continue;
                };
                if !angles_equivalent(want, got, ANGLE_EPS) {
                    return false;
                }
            }
        }
    }
    true
}

/// Find one applicable occurrence of `rule` in `dag`.
pub fn find_symbolic<R: Rng + ?Sized>(
    dag: &Dag,
    rule: &SymbolicRule,
    limits: &SymbolicLimits,
    registry: &GateRegistry,
    rng: &mut R,
) -> Option<SymbolicMatch> {
    if rule.find_before.is_empty() && rule.find_after.is_empty() {
        return None; // a bare hole matches everything and rewrites nothing coherent
    }
    // One topological pass serves the whole search: the match index is built from it,
    // the seed windowing reads positions from it, and every convexity check keys off
    // it. Building the context with `MatchContext::new` paid the same pass twice.
    let topo = dag.topological_gates();
    let index = MatchIndex::from_order(dag, &topo);
    let ctx = MatchContext::with_index(dag, &index);
    let topo_index: FxHashMap<NodeIndex, usize> =
        topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let env = SearchEnv {
        dag,
        rule,
        limits,
        registry,
        topo_index: &topo_index,
    };

    // A rule whose find side starts or ends at the hole has only one half to anchor on;
    // the region is then grown causally from that half instead of being derived as the
    // in-between. This shape is most of what ships: 77 of nam's 136 symbolic rules and
    // 852 of ion's 1,126 begin `symb q; ...`, and none of them could match before.
    if rule.find_before.is_empty() {
        return find_symbolic_one_sided(&env, &ctx, &topo, rng, Direction::Backward);
    }
    if rule.find_after.is_empty() {
        return find_symbolic_one_sided(&env, &ctx, &topo, rng, Direction::Forward);
    }
    let no_block: FxHashSet<NodeIndex> = FxHashSet::default();

    let mut before_seeds: Vec<NodeIndex> = ctx.seed_candidates(&rule.find_before).to_vec();
    before_seeds.shuffle(rng);

    for bseed in before_seeds {
        if env.deadline_hit() {
            return None;
        }
        let Some(before) = ctx.try_match(&rule.find_before, bseed, &no_block) else {
            continue;
        };
        let before_set: FxHashSet<NodeIndex> = before.nodes.iter().copied().collect();
        let before_max = before
            .nodes
            .iter()
            .filter_map(|n| topo_index.get(n))
            .max()
            .copied()
            .unwrap_or(0);

        // The after-half must lie past the before-half, and within the search window.
        let horizon = limits.max_span.map(|w| before_max.saturating_add(w));
        let mut after_seeds: Vec<NodeIndex> = ctx
            .seed_candidates(&rule.find_after)
            .iter()
            .copied()
            .filter(|n| {
                topo_index
                    .get(n)
                    .is_some_and(|&i| i > before_max && horizon.is_none_or(|h| i <= h))
            })
            .collect();
        after_seeds.shuffle(rng);

        for aseed in after_seeds {
            if env.deadline_hit() {
                return None;
            }
            // The after-half inherits the before-half's bindings, so a rule naming the
            // same pattern qubit or the same angle variable on both sides of the hole
            // really does require them to agree.
            let Some(after) = ctx.try_match_seeded(
                &rule.find_after,
                aseed,
                &before_set,
                &before.qubits,
                &before.angles,
            ) else {
                continue;
            };
            // The after-half's bindings include the before-half's, so this sees both.
            if !cross_half_angles_consistent(rule, &after) {
                continue;
            }
            let Some(between) = region_between(dag, &before, &after, &topo_index, limits.max_gates)
            else {
                continue;
            };

            // Build the in-between region as a standalone circuit to test.
            let sub = extract(dag, &between);
            if sub.num_qubits() > limits.max_qubits {
                continue;
            }

            // Boundary qubits pinned by the halves. A boundary qubit neither half
            // touches is *free*: it exists only in the constraints, standing for a wire
            // the hole acts on. `u3(..) q1; symb q; u1(..) q1;` is the shipped shape --
            // every gate sits on `q1`, and `q0` is whichever other wire the region
            // permutes. Free qubits are therefore bound by trying the region's own
            // qubits, injectively; before this, any rule with such a qubit simply never
            // matched, which silently killed 540 of the 1152 shipped `ibm` rules.
            let pinned: Vec<Option<QubitId>> = rule
                .boundary
                .iter()
                .map(|pq| {
                    before
                        .qubits
                        .get(pq)
                        .or_else(|| after.qubits.get(pq))
                        .cloned()
                })
                .collect();
            if let Some(m) = env.try_boundary_completions(&sub, &pinned, &before, &after, &between)
            {
                return Some(m);
            }
        }
    }
    None
}

/// Match a rule with exactly one non-empty find half, growing the region from it.
///
/// `direction` is where the region lies relative to the matched half: `Backward` for
/// rules shaped `symb q; gates;` (the region precedes the gates), `Forward` for
/// `gates; symb q;`.
///
/// The region grows one gate at a time along *tracked* wires — the wires the anchor
/// half touches, plus every wire a region gate drags in — and each extent is offered to
/// the constraints. Gates on untracked wires are skipped, which preserves convexity of
/// the span: a skipped gate's wires carry no span gates on the far side of it, so no
/// path can leave the span through it and re-enter. The span is still checked with
/// [`Dag::is_convex_with`] before use, because that argument is subtle enough to want a
/// guard; keyed off the search's own numbering, the guard costs a walk over the span,
/// not a topological pass per extent.
fn find_symbolic_one_sided<R: Rng + ?Sized>(
    env: &SearchEnv,
    ctx: &MatchContext,
    topo: &[NodeIndex],
    rng: &mut R,
    direction: Direction,
) -> Option<SymbolicMatch> {
    let SearchEnv {
        dag, rule, limits, ..
    } = *env;
    let anchor_pattern = match direction {
        Direction::Backward => &rule.find_after,
        Direction::Forward => &rule.find_before,
    };
    let no_block: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut seeds: Vec<NodeIndex> = ctx.seed_candidates(anchor_pattern).to_vec();
    seeds.shuffle(rng);

    let empty_match = Match {
        qubits: FxHashMap::default(),
        angles: FxHashMap::default(),
        gates: FxHashMap::default(),
        nodes: Vec::new(),
    };

    for seed in seeds {
        if env.deadline_hit() {
            return None;
        }
        let Some(anchor) = ctx.try_match(anchor_pattern, seed, &no_block) else {
            continue;
        };
        let anchor_set: FxHashSet<NodeIndex> = anchor.nodes.iter().copied().collect();

        // Walk the topological order away from the anchor.
        let positions: Vec<usize> = anchor
            .nodes
            .iter()
            .filter_map(|n| env.topo_index.get(n).copied())
            .collect();
        let (start, step): (isize, isize) = match direction {
            Direction::Backward => (*positions.iter().min()? as isize - 1, -1),
            Direction::Forward => (*positions.iter().max()? as isize + 1, 1),
        };

        let mut tracked: FxHashSet<QubitId> = FxHashSet::default();
        for &n in &anchor.nodes {
            tracked.extend(dag.gate(n).qubits.iter().cloned());
        }
        let mut region: Vec<NodeIndex> = Vec::new();
        let mut pos = start;
        while pos >= 0 && (pos as usize) < topo.len() && region.len() < limits.max_gates {
            let n = topo[pos as usize];
            pos += step;
            if anchor_set.contains(&n) {
                continue;
            }
            let touches = dag.gate(n).qubits.iter().any(|q| tracked.contains(q));
            if !touches {
                continue;
            }
            tracked.extend(dag.gate(n).qubits.iter().cloned());
            if tracked.len() > limits.max_qubits {
                break;
            }
            region.push(n);

            // The walk visits positions monotonically, so the region is already in
            // circuit order for a forward walk and in reverse circuit order for a
            // backward one; reversing replaces a per-extent sort.
            let mut between = region.clone();
            if matches!(direction, Direction::Backward) {
                between.reverse();
            }

            let (before, after) = match direction {
                Direction::Backward => (&empty_match, &anchor),
                Direction::Forward => (&anchor, &empty_match),
            };
            let mut span: Vec<NodeIndex> = between.clone();
            span.extend(anchor.nodes.iter().copied());
            if !dag.is_convex_with(&span, |n| env.key_of(n)) {
                continue;
            }

            let sub = extract(dag, &between);
            let pinned: Vec<Option<QubitId>> = rule
                .boundary
                .iter()
                .map(|pq| anchor.qubits.get(pq).cloned())
                .collect();
            if let Some(m) = env.try_boundary_completions(&sub, &pinned, before, after, &between) {
                return Some(m);
            }
        }
    }
    None
}

/// Every way to complete a partially pinned boundary from `candidates`, injectively.
///
/// The pinned entries stay fixed; each `None` slot takes a distinct candidate. With no
/// free slots this is exactly one assignment, so the common case costs nothing.
fn boundary_assignments(pinned: &[Option<QubitId>], candidates: &[QubitId]) -> Vec<Vec<QubitId>> {
    let mut out = Vec::new();
    let mut current: Vec<QubitId> = Vec::with_capacity(pinned.len());
    fn go(
        pinned: &[Option<QubitId>],
        candidates: &[QubitId],
        current: &mut Vec<QubitId>,
        out: &mut Vec<Vec<QubitId>>,
    ) {
        let i = current.len();
        if i == pinned.len() {
            out.push(current.clone());
            return;
        }
        match &pinned[i] {
            Some(q) => {
                current.push(q.clone());
                go(pinned, candidates, current, out);
                current.pop();
            }
            None => {
                for c in candidates {
                    if current.contains(c) {
                        continue;
                    }
                    current.push(c.clone());
                    go(pinned, candidates, current, out);
                    current.pop();
                }
            }
        }
    }
    go(pinned, candidates, &mut current, &mut out);
    out
}

/// Append a replacement half's gates, with the match's bindings applied.
fn append_half(dag: &mut Dag, half: &Pattern, m: &SymbolicMatch) {
    for pidx in half.dag().topological_gates() {
        dag.push_gate(bind_op(half.dag().gate(pidx), m));
    }
}

/// Apply a match's qubit and angle bindings to a pattern gate.
fn bind_op(pop: &GateOp, m: &SymbolicMatch) -> GateOp {
    let qubits: Vec<QubitId> = pop
        .qubits
        .iter()
        .map(|pq| {
            m.before
                .qubits
                .get(pq)
                .or_else(|| m.after.qubits.get(pq))
                // A replacement gate may sit on a boundary qubit that only the region
                // bound, so the full boundary map is the fallback before giving up.
                .or_else(|| m.boundary_map.get(pq))
                .cloned()
                .unwrap_or_else(|| pq.clone())
        })
        .collect();
    let mut angles: FxHashMap<String, AngleExpr> = m.before.angles.clone();
    for (k, v) in &m.after.angles {
        angles.entry(k.clone()).or_insert_with(|| v.clone());
    }
    GateOp {
        gate: pop.gate.clone(),
        qubits,
        params: pop
            .params
            .iter()
            .map(|p| substitute_angle(p, &angles))
            .collect(),
    }
}

/// Gates strictly causally between `before` and `after`, or `None` if there are more
/// than `max_gates` of them.
///
/// A "between" set is convex by construction: any node on a path joining two of its
/// members is itself a descendant of `before` and an ancestor of `after`. That is what
/// makes the whole span safe to cut out, and it replaces the reference's `checkLCA`
/// approximation with a property rather than a heuristic.
///
/// Returns `None` if the two halves are not causally ordered, or if any gate of `after`
/// precedes any gate of `before`.
///
/// This runs once per seed *pair*, so its two walks are kept off the whole circuit:
/// unbounded, each was a cone reaching to the circuit's edge, paid thousands of times to
/// almost always find a region larger than `max_gates` and discard it.
fn region_between(
    dag: &Dag,
    before: &Match,
    after: &Match,
    topo_index: &FxHashMap<NodeIndex, usize>,
    max_gates: usize,
) -> Option<Vec<NodeIndex>> {
    let before_set: FxHashSet<NodeIndex> = before.nodes.iter().copied().collect();
    let after_set: FxHashSet<NodeIndex> = after.nodes.iter().copied().collect();
    if before_set.intersection(&after_set).next().is_some() {
        return None;
    }
    let lo = before
        .nodes
        .iter()
        .filter_map(|n| topo_index.get(n))
        .max()?;
    let hi = after.nodes.iter().filter_map(|n| topo_index.get(n)).min()?;
    if hi <= lo {
        return None;
    }

    // Forward from the before half, pruned to the window: keys grow along every edge, so
    // nothing positioned past the last after gate can lie on a before -> after path.
    let hi_outer = *after.nodes.iter().filter_map(|n| topo_index.get(n)).max()?;
    let desc = descendants_within(dag, &before.nodes, &after_set, |n| {
        topo_index.get(&n).is_some_and(|&i| i <= hi_outer)
    });

    // Every gate of `after` must be reachable from `before`, otherwise the two halves are
    // causally independent and there is no single region joining them.
    if !after.nodes.iter().any(|n| desc.contains(n)) {
        return None;
    }

    // Backward from the after half, expanding only inside `desc`. This is complete: a
    // node on a backward path toward a member of the intersection is a descendant of
    // that member, hence of the before half, so it is in `desc` itself -- unless the
    // forward walk stopped at it as an after gate, and every after gate's predecessors
    // are already seeds here. Restricting the walk this way prices it at the region's
    // size rather than the ancestor cone's, and lets it stop the moment the region is
    // too large to accept anyway.
    let mut between: Vec<NodeIndex> = Vec::new();
    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut stack: Vec<NodeIndex> = Vec::new();
    for &s in &after.nodes {
        for q in &dag.gate(s).qubits {
            if let Some(p) = dag.predecessor_on(s, q) {
                stack.push(p);
            }
        }
    }
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if before_set.contains(&n) || after_set.contains(&n) || !desc.contains(&n) {
            continue;
        }
        between.push(n);
        if between.len() > max_gates {
            return None;
        }
        for q in &dag.gate(n).qubits {
            if let Some(p) = dag.predecessor_on(n, q) {
                stack.push(p);
            }
        }
    }
    between.sort_unstable_by_key(|n| topo_index.get(n).copied().unwrap_or(usize::MAX));
    Some(between)
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Backward,
}

/// Nodes forward-reachable from `seeds`, not passing *through* `stop` (but including
/// it), and not traversing nodes outside `within`.
fn descendants_within(
    dag: &Dag,
    seeds: &[NodeIndex],
    stop: &FxHashSet<NodeIndex>,
    within: impl Fn(NodeIndex) -> bool,
) -> FxHashSet<NodeIndex> {
    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut stack: Vec<NodeIndex> = Vec::new();
    for &s in seeds {
        for q in &dag.gate(s).qubits {
            if let Some(n) = dag.successor_on(s, q) {
                stack.push(n);
            }
        }
    }
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if stop.contains(&n) {
            continue; // reached the far half; do not traverse past it
        }
        if !within(n) {
            continue;
        }
        for q in &dag.gate(n).qubits {
            if let Some(m) = dag.successor_on(n, q) {
                stack.push(m);
            }
        }
    }
    seen
}

/// Build a standalone circuit from a set of nodes, preserving their relative order.
fn extract(dag: &Dag, nodes: &[NodeIndex]) -> Dag {
    let mut out = Dag::new(Vec::<String>::new());
    for &n in nodes {
        out.push_gate(dag.gate(n).clone());
    }
    out
}

/// Apply a symbolic rule to `dag`, returning `None` if it does not apply.
pub fn apply_symbolic<R: Rng + ?Sized>(
    dag: &Dag,
    rule: &SymbolicRule,
    limits: &SymbolicLimits,
    registry: &GateRegistry,
    rng: &mut R,
) -> Option<Dag> {
    let m = find_symbolic(dag, rule, limits, registry, rng)?;
    // Node indices survive a clone, so the match — its ordered span included — stays
    // valid against the copy.
    let mut out = dag.clone();
    let span = m.ordered_span.clone();

    // Everything between the halves stays put; the halves are removed and the replacement
    // halves are spliced in around the surviving region. Positions are computed against
    // the whole span, not just one half: a replacement gate can land on a boundary qubit
    // the half it belongs to never touched -- a rule moving `rz(theta) q0` from after the
    // hole to before it is exactly that shape -- and taking positions from one half alone
    // left such gates with nowhere to go.
    //
    // `exit` is the first node past the span on each wire; `entry` is the first surviving
    // (in-between) node on each wire. The before-half goes at `entry` where there is one
    // and `exit` otherwise; the after-half always goes at `exit`, which puts it after the
    // region and after anything already inserted there.
    let exit = exit_map(&out, &span);
    let mut entry: FxHashMap<QubitId, NodeIndex> = FxHashMap::default();
    for &n in &m.between {
        for q in out.gate(n).qubits.clone() {
            entry.entry(q).or_insert(n);
        }
    }
    let before_positions: FxHashMap<QubitId, NodeIndex> = exit
        .iter()
        .map(|(q, &n)| (q.clone(), entry.get(q).copied().unwrap_or(n)))
        .collect();

    // Every wire the replacement needs must have a position, and every replacement
    // parameter must be determined by the match's bindings — either the variable
    // itself or a whole expression containing it. A rule can bind `(4*pi-theta1)` as
    // an opaque unit without ever learning `theta1`; applying a replacement that needs
    // the bare variable would splice in a gate with an unresolved angle. Several
    // shipped ion rules have exactly this shape, and the span simulation used to be
    // what stopped them — by failing to build, not by design. Refusing the match here
    // is the designed check, and it does not depend on `verify_rewrites`.
    let bound_keys = m.bound_angle_keys();
    for half in [&rule.replace_before, &rule.replace_after] {
        for pidx in half.dag().gate_indices() {
            let pop = half.dag().gate(pidx);
            for p in &pop.params {
                let whole_bound = bound_keys.contains(&p.to_string());
                if !whole_bound && p.free_vars().iter().any(|v| !bound_keys.contains(*v)) {
                    return None;
                }
            }
            for q in &bind_op(pop, &m).qubits {
                if !exit.contains_key(q) {
                    return None;
                }
            }
        }
    }

    for &n in m.before.nodes.iter().chain(m.after.nodes.iter()) {
        out.remove_gate(n);
    }

    insert_half(&mut out, &rule.replace_before, &m, &before_positions);
    insert_half(&mut out, &rule.replace_after, &m, &exit);

    if !out.is_acyclic() {
        return None;
    }
    Some(out)
}

/// For each wire a node set touches, the node just after its last member.
fn exit_map(dag: &Dag, nodes: &[NodeIndex]) -> FxHashMap<QubitId, NodeIndex> {
    let set: FxHashSet<NodeIndex> = nodes.iter().copied().collect();
    let mut out = FxHashMap::default();
    for &n in nodes {
        for q in dag.gate(n).qubits.clone() {
            let mut cur = n;
            loop {
                match dag.next_on(cur, &q) {
                    Some(nxt) if set.contains(&nxt) => cur = nxt,
                    Some(nxt) => {
                        out.insert(q.clone(), nxt);
                        break;
                    }
                    None => break,
                }
            }
        }
    }
    out
}

fn insert_half(
    dag: &mut Dag,
    half: &Pattern,
    m: &SymbolicMatch,
    positions: &FxHashMap<QubitId, NodeIndex>,
) {
    for pidx in half.dag().topological_gates() {
        let op = bind_op(half.dag().gate(pidx), m);
        let target: FxHashMap<QubitId, NodeIndex> = op
            .qubits
            .iter()
            .filter_map(|q| positions.get(q).map(|&n| (q.clone(), n)))
            .collect();
        if target.len() == op.qubits.len() {
            dag.insert_gate_before(op, &target);
        } else {
            dag.push_gate(op);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(11)
    }

    const REAL_RULE: &str = "rz((theta1+theta2)) q0; symb q; h q1; | symb q; h q1; rz((theta1+theta2)) q0; | [{[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, true], [true, true]=[true, false]}, {[false, false]=[false, true], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, false]}, {[false, false]=[false, false], [true, false]=[true, false], [false, true]=[false, true], [true, true]=[true, true]}, {[false, false]=[false, true], [true, false]=[true, false], [false, true]=[false, false], [true, true]=[true, true]}]";

    /// Applying a rule must not mutate the rule: patterns are shared across the whole
    /// run, so any leak of interior state would corrupt every later use of them.
    ///
    /// `Dag::qubits` returns `&[QubitId]`; there is no live set to hand out and no
    /// caller that could mutate one. This pins the observable consequence: a rule's
    /// pattern halves report the same qubits and gate counts after being applied as
    /// before.
    #[test]
    fn applying_a_rule_does_not_mutate_its_pattern() {
        let rule = SymbolicRule::parse_legacy_with_builtins(REAL_RULE).unwrap();
        let before = (
            rule.find_before.gate_count(),
            rule.find_after.gate_count(),
            rule.find_before.dag().qubits().to_vec(),
            rule.find_after.dag().qubits().to_vec(),
        );

        let dag = qasm::parse("rz(0.3) a; cx a, b; h b;").unwrap();
        let _ = apply_symbolic(&dag, &rule, &SymbolicLimits::default(), &reg(), &mut rng());

        assert_eq!(
            (
                rule.find_before.gate_count(),
                rule.find_after.gate_count(),
                rule.find_before.dag().qubits().to_vec(),
                rule.find_after.dag().qubits().to_vec(),
            ),
            before,
            "applying the rule changed the rule"
        );
    }

    #[test]
    fn splits_a_rule_side_at_the_hole() {
        let (b, a) = split_hole("rz(theta1) q0; symb q; h q1;").unwrap();
        assert_eq!(b, "rz(theta1) q0;");
        assert_eq!(a, "h q1;");

        let (b, a) = split_hole("symb q; h q1;").unwrap();
        assert_eq!(b, "");
        assert_eq!(a, "h q1;");

        let (b, a) = split_hole("rz(theta1) q0; symb q;").unwrap();
        assert_eq!(b, "rz(theta1) q0;");
        assert_eq!(a, "");

        assert!(split_hole("h q0;").is_err());
    }

    #[test]
    fn parses_a_real_symbolic_rule() {
        let r = SymbolicRule::parse_legacy_with_builtins(REAL_RULE).unwrap();
        // The larger side is searched for.
        assert_eq!(r.find_before.gate_count(), 0);
        assert_eq!(r.find_after.gate_count(), 2);
        assert_eq!(r.replace_before.gate_count(), 1);
        assert_eq!(r.replace_after.gate_count(), 1);
        assert_eq!(r.constraints.len(), 4);
        assert_eq!(
            r.boundary,
            vec![qcircuit::intern("q0"), qcircuit::intern("q1")]
        );
        assert!(r.validate(&reg()).is_ok());
    }

    #[test]
    fn rejects_rules_whose_boundary_does_not_match_the_constraint_width() {
        // Two boundary qubits but a width-1 constraint.
        let line = "h q0; symb q; h q1; | symb q; h q1; h q0; | [{[false]=[false], [true]=[true]}]";
        assert!(SymbolicRule::parse_legacy_with_builtins(line).is_err());
    }

    #[test]
    fn rejects_malformed_lines() {
        assert!(SymbolicRule::parse_legacy_with_builtins("a | b").is_err());
        assert!(SymbolicRule::parse_legacy_with_builtins("h q0; | h q0; | [{}]").is_err());
    }

    /// The classical-action check, on circuits whose behaviour is known by hand.
    #[test]
    fn identity_region_satisfies_the_identity_constraint() {
        let sub = qasm::parse("rz(0.3) a;").unwrap();
        let boundary = vec![qcircuit::intern("a"), qcircuit::intern("b")];
        let id = BasisPermutation::identity(2);
        assert!(satisfies(&sub, &boundary, &id, &reg(), 1e-9));
    }

    #[test]
    fn a_cnot_region_satisfies_the_matching_permutation() {
        let sub = qasm::parse("cx a, b;").unwrap();
        let boundary = vec![qcircuit::intern("a"), qcircuit::intern("b")];
        // cx maps |ab> to |a, a xor b>: 00->00, 01->01, 10->11, 11->10.
        let perm = BasisPermutation::new(2, vec![0b00, 0b01, 0b11, 0b10]).unwrap();
        assert!(satisfies(&sub, &boundary, &perm, &reg(), 1e-9));
        assert!(!satisfies(
            &sub,
            &boundary,
            &BasisPermutation::identity(2),
            &reg(),
            1e-9
        ));
    }

    #[test]
    fn an_x_region_satisfies_the_bit_flip() {
        let sub = qasm::parse("x a;").unwrap();
        let boundary = vec![qcircuit::intern("a"), qcircuit::intern("b")];
        let perm = BasisPermutation::new(2, vec![0b10, 0b11, 0b00, 0b01]).unwrap();
        assert!(satisfies(&sub, &boundary, &perm, &reg(), 1e-9));
    }

    /// A Hadamard puts the boundary into superposition, so no permutation describes it.
    #[test]
    fn a_hadamard_on_the_boundary_satisfies_nothing() {
        let sub = qasm::parse("h a;").unwrap();
        let boundary = vec![qcircuit::intern("a"), qcircuit::intern("b")];
        for i in 0..24u32 {
            // Every permutation of four elements.
            let mut items = [0u32, 1, 2, 3];
            let mut n = i;
            for k in (1..4).rev() {
                let j = (n % (k as u32 + 1)) as usize;
                items.swap(k, j);
                n /= k as u32 + 1;
            }
            if let Ok(p) = BasisPermutation::new(2, items.to_vec()) {
                assert!(
                    !satisfies(&sub, &boundary, &p, &reg(), 1e-9),
                    "h should satisfy no permutation, but matched {p}"
                );
            }
        }
    }

    /// A Hadamard *off* the boundary is fine: the boundary bits stay determined.
    #[test]
    fn a_hadamard_off_the_boundary_is_allowed() {
        let sub = qasm::parse("h c; cx a, b;").unwrap();
        let boundary = vec![qcircuit::intern("a"), qcircuit::intern("b")];
        let perm = BasisPermutation::new(2, vec![0b00, 0b01, 0b11, 0b10]).unwrap();
        assert!(satisfies(&sub, &boundary, &perm, &reg(), 1e-9));
    }

    /// The check is width-generic, which is the whole point of the module.
    #[test]
    fn three_qubit_boundaries_work() {
        let sub = qasm::parse("cx a, b; cx b, c;").unwrap();
        let boundary = vec![
            qcircuit::intern("a"),
            qcircuit::intern("b"),
            qcircuit::intern("c"),
        ];
        // |abc> -> |a, a^b, a^b^c>
        let image: Vec<u32> = (0..8)
            .map(|i| {
                let a = (i >> 2) & 1;
                let b = (i >> 1) & 1;
                let c = i & 1;
                let nb = a ^ b;
                let nc = nb ^ c;
                ((a << 2) | (nb << 1) | nc) as u32
            })
            .collect();
        let perm = BasisPermutation::new(3, image).unwrap();
        assert!(satisfies(&sub, &boundary, &perm, &reg(), 1e-9));
        assert!(!satisfies(
            &sub,
            &boundary,
            &BasisPermutation::identity(3),
            &reg(),
            1e-9
        ));
    }

    #[test]
    fn an_empty_region_is_the_identity() {
        let sub = Dag::new(["a", "b"]);
        let boundary = vec![qcircuit::intern("a"), qcircuit::intern("b")];
        assert!(satisfies(
            &sub,
            &boundary,
            &BasisPermutation::identity(2),
            &reg(),
            1e-9
        ));
    }

    /// Matching at the very end of a circuit must simply find nothing; nothing here
    /// indexes by topological layer.
    #[test]
    fn matching_near_the_end_of_a_circuit_does_not_panic() {
        let rule = SymbolicRule::parse_legacy_with_builtins(REAL_RULE).unwrap();
        for src in [
            "h q1;",
            "rz(0.5) q0;",
            "rz(0.5) q0; h q1;",
            "rz(0.5) q0; cx q0, q1; h q1;",
            "h q1; rz(0.5) q0;",
            "",
        ] {
            let dag = qasm::parse(src).unwrap();
            // Must return an answer, not panic, whatever the circuit's shape.
            let _ = find_symbolic(&dag, &rule, &SymbolicLimits::default(), &reg(), &mut rng());
        }
    }

    #[test]
    fn region_between_is_empty_for_adjacent_halves() {
        let dag = qasm::parse("h a; x a;").unwrap();
        let ctx = MatchContext::new(&dag);
        let topo: Vec<NodeIndex> = dag.topological_gates();
        let ti: FxHashMap<NodeIndex, usize> =
            topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let no_block = FxHashSet::default();
        let pb = Pattern::parse("h q0;").unwrap();
        let pa = Pattern::parse("x q0;").unwrap();
        let b = ctx.try_match(&pb, topo[0], &no_block).unwrap();
        let a = ctx.try_match(&pa, topo[1], &no_block).unwrap();
        let between = region_between(&dag, &b, &a, &ti, usize::MAX).unwrap();
        assert!(between.is_empty());
    }

    #[test]
    fn region_between_collects_intervening_gates() {
        let dag = qasm::parse("h a; cx a, b; t b; x a;").unwrap();
        let ctx = MatchContext::new(&dag);
        let topo: Vec<NodeIndex> = dag.topological_gates();
        let ti: FxHashMap<NodeIndex, usize> =
            topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let no_block = FxHashSet::default();
        let b = ctx
            .try_match(&Pattern::parse("h q0;").unwrap(), topo[0], &no_block)
            .unwrap();
        let last = *topo.last().unwrap();
        let a = ctx
            .try_match(&Pattern::parse("x q0;").unwrap(), last, &no_block)
            .unwrap();
        let between = region_between(&dag, &b, &a, &ti, usize::MAX).unwrap();
        // The cx lies between; the t on b does not reach `x a`.
        assert_eq!(between.len(), 1);
        assert_eq!(&*dag.gate(between[0]).gate, "cx");
    }

    #[test]
    fn region_between_rejects_reversed_halves() {
        let dag = qasm::parse("x a; h a;").unwrap();
        let ctx = MatchContext::new(&dag);
        let topo: Vec<NodeIndex> = dag.topological_gates();
        let ti: FxHashMap<NodeIndex, usize> =
            topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let no_block = FxHashSet::default();
        let b = ctx
            .try_match(&Pattern::parse("h q0;").unwrap(), topo[1], &no_block)
            .unwrap();
        let a = ctx
            .try_match(&Pattern::parse("x q0;").unwrap(), topo[0], &no_block)
            .unwrap();
        assert!(region_between(&dag, &b, &a, &ti, usize::MAX).is_none());
    }

    #[test]
    fn limits_are_respected() {
        let rule = SymbolicRule::parse_legacy_with_builtins(REAL_RULE).unwrap();
        let dag = qasm::parse("rz(0.5) q0; cx q0,q1; cx q1,q2; cx q2,q3; h q1;").unwrap();
        let tight = SymbolicLimits {
            max_qubits: 2,
            max_gates: 1,
            ..Default::default()
        };
        assert!(find_symbolic(&dag, &rule, &tight, &reg(), &mut rng()).is_none());
    }

    /// A rule whose *find* side (the second field) has gates on both sides of the hole,
    /// with constraints licensing a `cx` between them.
    ///
    /// `REAL_RULE` cannot serve here: its find-half before the hole is empty, so it never
    /// enters the seed loop at all. The replacement is deliberately identical to the
    /// pattern, so the soundness check on the rewrite is not what these tests measure --
    /// they are about which matches the search *looks for*, and a rule that rewrites
    /// nothing isolates that from whether the rewrite would have been valid.
    const SPANNING_RULE: &str = concat!(
        "rz(theta1) q0; symb q; h q1; | rz(theta1) q0; symb q; h q1; | ",
        "[{[false, false]=[false, false], [true, false]=[true, true], ",
        "[false, true]=[false, true], [true, true]=[true, false]}]"
    );

    /// `max_span` bounds how far apart the two halves may sit.
    ///
    /// Without it the seed loop is quadratic in the circuit and every pair pays a
    /// reachability walk, which is how one symbolic rule came to take 55 seconds on a
    /// ten-thousand-gate circuit. The bound is a search limit, not a validity condition:
    /// the same match is found when the window is wide enough to reach it.
    #[test]
    fn max_span_bounds_how_far_the_halves_may_sit_apart() {
        let rule = SymbolicRule::parse_legacy_with_builtins(SPANNING_RULE).unwrap();
        // The halves are `rz(0.5) a` and `h b`, with a long run between them. The pairs
        // of `x` cancel as an operator, so the region still acts as the licensed `cx`.
        let mut src = String::from("rz(0.5) a; cx a, b;");
        for _ in 0..40 {
            src.push_str(" x b; x b;");
        }
        src.push_str(" h b;");
        let dag = qasm::parse(&src).unwrap();

        let wide = SymbolicLimits {
            max_gates: 200,
            max_span: None,
            ..Default::default()
        };
        assert!(
            find_symbolic(&dag, &rule, &wide, &reg(), &mut rng()).is_some(),
            "the match this test narrows should exist when unbounded"
        );

        let narrow = SymbolicLimits {
            max_span: Some(2),
            ..wide
        };
        assert!(
            find_symbolic(&dag, &rule, &narrow, &reg(), &mut rng()).is_none(),
            "a window of 2 should not reach a half 80 gates away"
        );
    }

    /// A compound angle bound opaquely in one half must agree with its variables
    /// bound individually in the other; a site where it does not is not a match.
    ///
    /// The rule below is a true identity: with the hole flipping `q0`,
    /// `rz(a) . X = X . rz(-a)`, so the search side collapses to the replacement
    /// exactly when the first angle is `theta1 + theta2`. Each half's own consistency
    /// check cannot see that — the compound lives in one half, the variables in the
    /// other — and before the cross-half check this matched circuits where the sum
    /// did not hold and miscompiled them. Found by the QCEC benchmark sweep.
    #[test]
    fn compound_angles_bound_across_the_halves_must_agree() {
        let line = "rz(theta2) q0; rz(theta2) q1; symb q; | \
                    rz((theta1+theta2)) q0; symb q; rz(theta1) q0; rz(theta2) q1; | \
                    [{[false, false]=[true, false], [false, true]=[true, true], \
                    [true, false]=[false, false], [true, true]=[false, true]}]";
        let rule = SymbolicRule::parse_legacy_with_builtins(line).unwrap();
        let registry = reg();

        // 0.8 = 0.5 + 0.3: the relationship holds, so the rule applies — and the
        // rewrite preserves the circuit's unitary.
        let consistent = qasm::parse("rz(0.8) a; x a; rz(0.5) a; rz(0.3) b;").unwrap();
        let out = apply_symbolic(
            &consistent,
            &rule,
            &SymbolicLimits::default(),
            &registry,
            &mut rng(),
        )
        .expect("the consistent site must match");
        let qs: Vec<qcircuit::QubitId> = vec![qcircuit::intern("a"), qcircuit::intern("b")];
        let ua = qsemantics::Unitary::from_dag_over(&consistent, qs.clone(), &registry).unwrap();
        let ub = qsemantics::Unitary::from_dag_over(&out, qs, &registry).unwrap();
        assert!(qsemantics::phase_invariant_distance(&ua, &ub) < 1e-9);

        // 0.9 != 0.5 + 0.3: the compound disagrees with its variables, so there is no
        // match at all — this is the site that used to be rewritten wrongly.
        let inconsistent = qasm::parse("rz(0.9) a; x a; rz(0.5) a; rz(0.3) b;").unwrap();
        assert!(
            find_symbolic(
                &inconsistent,
                &rule,
                &SymbolicLimits::default(),
                &registry,
                &mut rng()
            )
            .is_none(),
            "a site where the compound is not the sum must not match"
        );
    }

    /// A match that binds a compound expression as an opaque unit never learns its
    /// constituent variables, so a replacement needing a bare variable is
    /// underdetermined and must not apply — with or without `verify_rewrites`, which
    /// used to be what caught it, by failing to simulate rather than by design.
    /// Several shipped ion rules have this shape.
    #[test]
    fn an_underdetermined_replacement_angle_refuses_to_apply() {
        let registry = reg();
        // Find side binds only `(4*pi-theta1)` as a whole; the replacement wants
        // `theta1` alone. Built directly, because the load-time identity check would
        // (rightly) refuse to certify such a rule.
        let rule = SymbolicRule {
            find_before: Pattern::empty(),
            find_after: Pattern::parse_multi("rz(((4*pi)-theta1)) q0;").unwrap(),
            replace_before: Pattern::parse_multi("rz(theta1) q0;").unwrap(),
            replace_after: Pattern::empty(),
            boundary: vec![qcircuit::intern("q0"), qcircuit::intern("q1")],
            // The region below is `cx b, a`: with q0 -> a as the high bit, its
            // boundary action is |a,b> -> |a^b, b>, the permutation [0,3,2,1].
            constraints: vec![BasisPermutation::new(2, vec![0, 3, 2, 1]).unwrap()],
            phase_safe: true,
            id: "underdetermined-test".into(),
        };
        let dag = qasm::parse("cx b, a; rz(0.5) a;").unwrap();
        // The *match* exists — only the application must refuse it, or the test would
        // pass vacuously on a pattern that never matched.
        assert!(
            find_symbolic(
                &dag,
                &rule,
                &SymbolicLimits::default(),
                &registry,
                &mut rng()
            )
            .is_some(),
            "the pattern should match; the guard under test lives in apply"
        );
        let out = apply_symbolic(
            &dag,
            &rule,
            &SymbolicLimits::default(),
            &registry,
            &mut rng(),
        );
        assert!(
            out.is_none(),
            "an underdetermined replacement angle was applied"
        );
    }

    /// A found match carries its whole span in circuit order, so applying it never pays
    /// a circuit-wide topological pass just to order a handful of nodes.
    #[test]
    fn a_match_carries_its_span_in_circuit_order() {
        let rule = SymbolicRule::parse_legacy_with_builtins(SPANNING_RULE).unwrap();
        let dag = qasm::parse("rz(0.5) a; cx a, b; h b;").unwrap();
        let m = find_symbolic(&dag, &rule, &SymbolicLimits::default(), &reg(), &mut rng())
            .expect("this match should be found");

        let position: FxHashMap<NodeIndex, usize> = dag
            .topological_gates()
            .into_iter()
            .enumerate()
            .map(|(i, n)| (n, i))
            .collect();
        let spelled: Vec<usize> = m
            .ordered_span
            .iter()
            .map(|n| *position.get(n).expect("span node is a circuit gate"))
            .collect();
        let mut sorted = spelled.clone();
        sorted.sort_unstable();
        assert_eq!(spelled, sorted, "span is not in circuit order");

        let mut expected: Vec<NodeIndex> = m.span();
        expected.sort_unstable();
        let mut got = m.ordered_span.clone();
        got.sort_unstable();
        assert_eq!(got, expected, "ordered span disagrees with span() as a set");
    }

    /// An elapsed deadline stops the search rather than being noticed only afterwards.
    ///
    /// What this pins is not slowness but *unboundedness*: the symbolic search used to
    /// have no clock at all, so a single application could run for a minute inside a
    /// five-second budget, and neither the per-iteration check nor the one between
    /// applied rewrite sites could see it happening.
    #[test]
    fn an_elapsed_deadline_stops_the_symbolic_search() {
        let rule = SymbolicRule::parse_legacy_with_builtins(SPANNING_RULE).unwrap();
        let dag = qasm::parse("rz(0.5) a; cx a, b; h b;").unwrap();
        let live = SymbolicLimits::default();
        assert!(
            find_symbolic(&dag, &rule, &live, &reg(), &mut rng()).is_some(),
            "this match should be found with time to spare"
        );

        let expired = SymbolicLimits {
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
            ..live
        };
        assert!(find_symbolic(&dag, &rule, &expired, &reg(), &mut rng()).is_none());
    }
}
