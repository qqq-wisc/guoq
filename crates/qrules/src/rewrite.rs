//! Applying a rewrite rule to a circuit.
//!
//! # Structural, not textual
//!
//! A match is applied to the graph directly: gates are removed and inserted by node,
//! and angle and qubit substitution are structural, so nothing depends on the spelling
//! of a printed operand list and nothing round-trips through text on the hot path.
//!
//! Here a match yields bindings, and the replacement is built by walking the replacement
//! pattern and applying those bindings structurally. No text is involved.

use rand::seq::SliceRandom;
use rand::Rng;
use rustc_hash::{FxHashMap, FxHashSet};

use qcircuit::{AngleExpr, Dag, GateOp, NodeIndex, QubitId};

use crate::matcher::{Match, MatchContext, MatchIndex};
use crate::rule::Rule;

/// How to apply a rule.
#[derive(Debug, Clone)]
pub struct ApplyOptions {
    /// Apply only the first match found, rather than every disjoint match.
    pub apply_once: bool,
    /// Try seed candidates in a random order, so repeated application explores.
    pub shuffle: bool,
    /// Stop looking for further sites once this instant has passed.
    ///
    /// Exhaustive application re-matches after every site (see `apply_rule`), which is
    /// linear in the circuit per site -- so a rule that fires thousands of times on a
    /// large circuit can hold the search loop for minutes, long past its wall-clock
    /// budget. Every application leaves a complete, consistent circuit, so stopping
    /// between sites is always sound; the deadline bounds the overshoot to one site's
    /// work. `None` means apply without a clock, which is right for tests and one-shot
    /// tools.
    pub deadline: Option<std::time::Instant>,
}

impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            apply_once: false,
            shuffle: true,
            deadline: None,
        }
    }
}

/// The result of applying a rule.
#[derive(Debug, Clone)]
pub struct Rewritten {
    pub dag: Dag,
    /// How many matches were replaced.
    pub applications: usize,
    /// The nodes this rewrite introduced, and only those.
    ///
    /// A rewrite is a local edit: every gate outside the replaced spans is exactly as it
    /// was. Callers that would otherwise rescan the whole circuit for a property that
    /// only a *new* gate can have -- identity-ness is the one that matters on the search
    /// hot path -- can look at these instead, which turns an O(circuit) pass into an
    /// O(replacement) one.
    pub inserted: Vec<NodeIndex>,
}

/// Apply `rule` to `dag`, returning `None` if it does not match anywhere.
///
/// Matches are found and applied one at a time, each against the *current* graph rather
/// than against a batch computed up front. That is not merely tidier: a rewrite reorders
/// gates, so two matches that are each individually valid on the original circuit can
/// conflict once the first is applied. Applying them blind produces a cyclic graph — see
/// `applying_two_matches_never_creates_a_cycle`.
///
/// Gates written by a previous application are excluded from later matches, which both
/// mirrors the reference's intent and guarantees termination for size-preserving rules
/// that would otherwise swap a pair of gates back and forth forever.
pub fn apply_rule<R: Rng + ?Sized>(
    dag: &Dag,
    rule: &Rule,
    opts: &ApplyOptions,
    rng: &mut R,
) -> Option<Rewritten> {
    let ctx = MatchContext::new(dag);
    apply_rule_with_context(dag, &ctx, rule, opts, rng)
}

/// As [`apply_rule`], reusing a match context the caller already built.
///
/// The search tries many rules against one circuit per iteration, so it builds the
/// context once and hands it in. Rebuilding it inside every call — a topological sort and
/// two index maps over the whole circuit — was the single largest per-iteration cost, and
/// with the reference's default of one sampled transformation per iteration it was pure
/// duplication of work the caller had just done.
///
/// `ctx` must have been built from `dag`; only the *first* match uses it, since applying a
/// rewrite invalidates it.
pub fn apply_rule_with_context<R: Rng + ?Sized>(
    dag: &Dag,
    ctx: &MatchContext<'_>,
    rule: &Rule,
    opts: &ApplyOptions,
    rng: &mut R,
) -> Option<Rewritten> {
    debug_assert!(
        std::ptr::eq(ctx.dag(), dag),
        "context built from another circuit"
    );

    let no_block: FxHashSet<NodeIndex> = FxHashSet::default();
    // Look before cloning: most attempts find nothing, and a clone of the whole circuit
    // for a rule that does not fire is wasted.
    let first = seeded_match(ctx, rule, opts, &no_block, rng)?;

    // Node indices survive a clone, so both the match and the index stay valid against
    // the copy. Carrying the index forward across sites, rather than rebuilding it after
    // each one, is what keeps exhaustive application from being quadratic in the circuit.
    let mut index = ctx.index().clone();
    let mut out = dag.clone();
    let mut blocked: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut inserted: Vec<NodeIndex> = Vec::new();
    let mut applied = 0usize;

    let mut pending = Some(first);
    loop {
        let m = match pending.take() {
            Some(m) => m,
            None => {
                let ctx = MatchContext::with_index(&out, &index);
                match seeded_match(&ctx, rule, opts, &blocked, rng) {
                    Some(m) => m,
                    None => break,
                }
            }
        };
        let removed = m.nodes.clone();
        let new_nodes = apply_match(&mut out, rule, &m);
        index.apply_rewrite(&out, &removed, &new_nodes);
        inserted.extend(new_nodes.iter().copied());
        debug_assert!(
            out.is_acyclic(),
            "rewriting `{}` produced a cyclic graph",
            rule.id
        );
        blocked.extend(new_nodes);
        applied += 1;
        if opts.apply_once {
            break;
        }
        if opts
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            break;
        }
    }

    (applied > 0).then_some(Rewritten {
        dag: out,
        inserted,
        applications: applied,
    })
}

/// The next match of `rule` using an existing context.
fn seeded_match<R: Rng + ?Sized>(
    ctx: &MatchContext<'_>,
    rule: &Rule,
    opts: &ApplyOptions,
    blocked: &FxHashSet<NodeIndex>,
    rng: &mut R,
) -> Option<Match> {
    let pattern = &rule.find;
    // Copy the bucket only to shuffle it: a deterministic scan iterates the index's own
    // slice, which matters because this runs once per applied site — an exhaustive pass
    // over a large circuit calls it thousands of times against a bucket of thousands.
    let borrowed: &[NodeIndex];
    let mut shuffled: Vec<NodeIndex>;
    if opts.shuffle {
        shuffled = ctx.seed_candidates(pattern).to_vec();
        shuffled.shuffle(rng);
        borrowed = &shuffled;
    } else {
        borrowed = ctx.seed_candidates(pattern);
    }
    // A failed attempt costs nothing: no gate is marked consumed until a match is
    // accepted.
    //
    // The clock is checked *inside* this scan, not only around it. Late in an exhaustive
    // application almost every seed is blocked or dead, so the scan that finally reports
    // "no more matches" is the single most expensive step in the pass -- and a check
    // placed between applied sites never runs during it. Giving up early here reports
    // "no match", which is exactly what the caller should do when out of time.
    const CLOCK_STRIDE: usize = 256;
    for (i, &s) in borrowed.iter().enumerate() {
        if i % CLOCK_STRIDE == CLOCK_STRIDE - 1
            && opts
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return None;
        }
        if blocked.contains(&s) {
            continue;
        }
        if let Some(m) = ctx.try_match(pattern, s, blocked) {
            return Some(m);
        }
    }
    None
}

/// Does `rule` match anywhere in the circuit `ctx` was built for?
///
/// A cheap pre-filter. [`apply_rule`] has to build a [`MatchContext`] — which
/// topologically sorts the circuit — and the search tries tens of thousands of rules
/// against the same circuit each iteration, almost none of which match. Building the
/// context once and asking this first turns those sorts from one-per-rule into one per
/// iteration.
pub fn matches_anywhere(ctx: &MatchContext, rule: &Rule) -> bool {
    let blocked = FxHashSet::default();
    ctx.seed_candidates(&rule.find)
        .iter()
        .any(|&seed| ctx.try_match(&rule.find, seed, &blocked).is_some())
}

/// Every match of `rule`'s pattern in `dag`, pairwise disjoint.
///
/// For inspection and testing. Applying more than one of these without re-matching is
/// unsound, because each rewrite can invalidate the others; use [`apply_rule`].
pub fn find_disjoint_matches<R: Rng + ?Sized>(
    dag: &Dag,
    rule: &Rule,
    opts: &ApplyOptions,
    rng: &mut R,
) -> Vec<Match> {
    let ctx = MatchContext::new(dag);
    let pattern = &rule.find;
    let mut seeds: Vec<NodeIndex> = ctx.seed_candidates(pattern).to_vec();
    if opts.shuffle {
        seeds.shuffle(rng);
    }
    let mut blocked: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut out = Vec::new();
    for seed in seeds {
        if blocked.contains(&seed) {
            continue;
        }
        let Some(m) = ctx.try_match(pattern, seed, &blocked) else {
            continue;
        };
        blocked.extend(m.nodes.iter().copied());
        out.push(m);
    }
    out
}

/// `apply_match`, exposed for the index test in `matcher`.
#[cfg(test)]
pub(crate) fn apply_match_for_test(dag: &mut Dag, rule: &Rule, m: &Match) -> Vec<NodeIndex> {
    apply_match(dag, rule, m)
}

/// Enough to put a circuit back exactly as it was before a rewrite.
///
/// The search's elementary move is to try a rewrite and usually reject it. Producing the
/// candidate by cloning the whole circuit makes the *rejected* case cost O(circuit), which
/// on a large circuit is the dominant per-iteration cost and is paid for nothing. Applying
/// in place and rolling back costs O(gates the rewrite touched) either way.
///
/// Restoration is exact but not identical: a restored gate is a new node and gets a new
/// index, because nothing guarantees a graph hands back the index it freed. Everything
/// downstream addresses gates through the index it is given, and `Dag::structural_hash` is
/// built from gate contents rather than node identity, so a rolled-back circuit hashes
/// equal to the original. Anything holding a `NodeIndex` across a rollback, though, is
/// holding a stale one.
#[derive(Debug, Default)]
pub struct Undo {
    /// One entry per applied site, in the order they were applied.
    ///
    /// A chain has to be unwound as a unit rather than a step at a time. Rolling back a
    /// step gives its restored gates *new* node indices, which invalidates the successors
    /// an earlier step recorded — so unwinding step by step, each with its own bookkeeping,
    /// splices gates onto wires they are not on. One renumbering map spanning the whole
    /// unwind is what makes it correct.
    steps: Vec<Step>,
}

/// One removed gate: the node it occupied, the gate itself, and the node that followed it
/// on each of its wires at the moment it was removed.
type Removal = (NodeIndex, GateOp, Vec<(QubitId, NodeIndex)>);

#[derive(Debug)]
struct Step {
    /// The removed gates, newest removal first: the node each occupied, the gate itself,
    /// and the node that followed it on each of its wires when it was removed.
    removed: Vec<Removal>,
    /// Nodes the replacement wrote, in insertion order.
    inserted: Vec<NodeIndex>,
}

impl Undo {
    /// Every node introduced across the chain, for callers normalizing what a rewrite
    /// produced.
    pub fn inserted(&self) -> Vec<NodeIndex> {
        self.steps
            .iter()
            .flat_map(|s| s.inserted.iter().copied())
            .collect()
    }

    /// How many sites were rewritten.
    pub fn applications(&self) -> usize {
        self.steps.len()
    }

    /// `true` if nothing was applied, so rolling back is a no-op.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Apply one match in place, appending to `undo` the record needed to reverse it.
///
/// The circuit is left in a fully consistent state either way; there is no partially
/// applied condition to observe.
pub fn apply_match_undoable(dag: &mut Dag, rule: &Rule, m: &Match, undo: &mut Undo) {
    let matched: FxHashSet<NodeIndex> = m.nodes.iter().copied().collect();

    // The exit nodes have to be read while the match is still intact, since finding them
    // means walking through it.
    let exit = exit_map(dag, &m.nodes, &matched);

    // Record where each gate sat *at the moment it is removed*, not up front. The two
    // differ, and the difference is a correctness bug rather than an optimization.
    //
    // Restoration runs in reverse removal order, which is sound only if every gate's
    // recorded successor is back in place by the time that gate is restored. Reading the
    // successor after the earlier removals guarantees it: whatever follows `n` then is
    // either outside the match, or a gate removed later and so restored earlier.
    //
    // Reading them all up front breaks that. `m.nodes` is the pattern's BFS order, not
    // wire order, and the BFS reaches a gate backwards whenever the pattern has one --
    // `h q0; t q1; cx q0, q1;` visits the `cx` before the `t` that precedes it on `q1`.
    // The `t` then records the `cx` as its successor, is restored first, and splices onto
    // a node that is not there yet.
    let mut removed: Vec<Removal> = Vec::with_capacity(m.nodes.len());
    for &n in &m.nodes {
        let op = dag.gate(n).clone();
        let mut succs = Vec::with_capacity(op.qubits.len());
        for q in &op.qubits {
            let succ = dag.next_on(n, q).expect("every wire runs to a sink");
            succs.push((q.clone(), succ));
        }
        removed.push((n, op, succs));
        dag.remove_gate(n);
    }

    let inserted = write_replacement(dag, rule, m, &exit);

    removed.reverse();
    undo.steps.push(Step { removed, inserted });
}

/// Apply `rule` to `dag` in place, maintaining `index`, and return how to reverse it.
///
/// This is the clone-free counterpart of [`apply_rule`]. The search tries a rewrite and
/// usually rejects it, so producing the candidate by copying the circuit makes the
/// rejected case cost O(circuit) for nothing, and rebuilding the match index afterwards
/// costs it again. Here both the edit and its reversal are O(gates the rewrite touched).
///
/// Returns `None`, having changed nothing, if the rule does not match.
pub fn apply_rule_in_place<R: Rng + ?Sized>(
    dag: &mut Dag,
    index: &mut MatchIndex,
    rule: &Rule,
    opts: &ApplyOptions,
    rng: &mut R,
) -> Option<Undo> {
    let mut undo = Undo::default();
    let mut blocked: FxHashSet<NodeIndex> = FxHashSet::default();

    loop {
        let m = {
            let ctx = MatchContext::with_index(dag, index);
            match seeded_match(&ctx, rule, opts, &blocked, rng) {
                Some(m) => m,
                None => break,
            }
        };
        let removed = m.nodes.clone();
        let before = undo.steps.len();
        apply_match_undoable(dag, rule, &m, &mut undo);
        let inserted: Vec<NodeIndex> = undo.steps[before..]
            .iter()
            .flat_map(|s| s.inserted.iter().copied())
            .collect();
        index.apply_rewrite(dag, &removed, &inserted);
        debug_assert!(
            dag.is_acyclic(),
            "rewriting `{}` produced a cyclic graph",
            rule.id
        );
        blocked.extend(inserted);

        if opts.apply_once {
            break;
        }
        if opts
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            break;
        }
    }

    (!undo.is_empty()).then_some(undo)
}

/// Drop gates among `candidates` that do nothing, recording how to put them back.
///
/// The counterpart of `Dag::drop_identity_gates_among` for the in-place path. Normalizing
/// after a rewrite is part of the same edit: if it were done outside the undo record, a
/// rolled-back circuit would be missing the gates normalization had removed.
pub fn drop_identities_undoable(
    dag: &mut Dag,
    candidates: &[NodeIndex],
    registry: &qcircuit::GateRegistry,
    undo: &mut Undo,
) -> Vec<NodeIndex> {
    let victims: Vec<NodeIndex> = candidates
        .iter()
        .copied()
        .filter(|&n| {
            dag.try_gate(n)
                .is_some_and(|op| qcircuit::is_global_phase(op, registry))
        })
        .collect();
    if victims.is_empty() {
        return victims;
    }

    let mut removed: Vec<Removal> = Vec::with_capacity(victims.len());
    for &n in &victims {
        let op = dag.gate(n).clone();
        let succs = op
            .qubits
            .iter()
            .map(|q| {
                (
                    q.clone(),
                    dag.next_on(n, q).expect("every wire runs to a sink"),
                )
            })
            .collect();
        removed.push((n, op, succs));
        dag.remove_gate(n);
    }
    removed.reverse();
    undo.steps.push(Step {
        removed,
        inserted: Vec::new(),
    });
    victims
}

/// Reverse what [`apply_rule_in_place`] did, restoring `index` along with the circuit.
pub fn rollback_with_index(dag: &mut Dag, index: &mut MatchIndex, undo: Undo) {
    let inserted = undo.inserted();
    let restored: Vec<NodeIndex> = rollback_reporting(dag, undo);
    // From the index's point of view a rollback is just another edit: the replacement's
    // nodes go away and the original gates come back, at whatever indices they landed on.
    index.apply_rewrite(dag, &inserted, &restored);
}

/// Put the circuit back as it was before the rewrites `undo` records.
///
/// Steps unwind newest first, under a single map from each node's original index to the
/// index it was restored at. Both halves consult it: a step's inserted node may since have
/// been removed and restored by a later step, and a step's recorded successor may be a
/// gate a later step has already put back somewhere new.
pub fn rollback(dag: &mut Dag, undo: Undo) {
    let _ = rollback_reporting(dag, undo);
}

/// As [`rollback`], reporting the indices the restored gates landed on.
fn rollback_reporting(dag: &mut Dag, undo: Undo) -> Vec<NodeIndex> {
    let mut moved: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    let mut restored = Vec::new();
    for step in undo.steps.into_iter().rev() {
        for n in step.inserted {
            dag.remove_gate(moved.get(&n).copied().unwrap_or(n));
        }
        for (was, op, succs) in step.removed {
            let before: FxHashMap<QubitId, NodeIndex> = succs
                .into_iter()
                .map(|(q, s)| (q, moved.get(&s).copied().unwrap_or(s)))
                .collect();
            let new = dag.insert_gate_before(op, &before);
            moved.insert(was, new);
            restored.push(new);
        }
    }
    restored
}

/// Splice one match out of `dag` and put the rule's replacement in its place.
///
/// Returns the nodes written by the replacement. The nodes removed are exactly
/// `m.nodes`, so callers maintaining an index have both halves of the edit.
fn apply_match(dag: &mut Dag, rule: &Rule, m: &Match) -> Vec<NodeIndex> {
    let matched: FxHashSet<NodeIndex> = m.nodes.iter().copied().collect();
    let exit = exit_map(dag, &m.nodes, &matched);
    for &n in &m.nodes {
        dag.remove_gate(n);
    }
    write_replacement(dag, rule, m, &exit)
}

/// For each wire the match touches, the node that follows its last matched gate.
///
/// Adjacency guarantees the matched gates on a wire form one contiguous run, so this node
/// is outside the match and survives the removal.
fn exit_map(
    dag: &Dag,
    nodes: &[NodeIndex],
    matched: &FxHashSet<NodeIndex>,
) -> FxHashMap<QubitId, NodeIndex> {
    let mut exit: FxHashMap<QubitId, NodeIndex> = FxHashMap::default();
    for &n in nodes {
        for q in dag.gate(n).qubits.clone() {
            let mut cur = n;
            loop {
                match dag.next_on(cur, &q) {
                    Some(nxt) if matched.contains(&nxt) => cur = nxt,
                    Some(nxt) => {
                        exit.insert(q.clone(), nxt);
                        break;
                    }
                    None => break,
                }
            }
        }
    }
    exit
}

/// Write the rule's replacement into the gap the removed match left.
fn write_replacement(
    dag: &mut Dag,
    rule: &Rule,
    m: &Match,
    exit: &FxHashMap<QubitId, NodeIndex>,
) -> Vec<NodeIndex> {
    let mut inserted = Vec::with_capacity(rule.replace.gate_count());
    for pidx in rule.replace.dag().topological_gates() {
        let pop = rule.replace.dag().gate(pidx);
        let qubits: Vec<QubitId> = pop
            .qubits
            .iter()
            .map(|pq| m.qubits.get(pq).cloned().unwrap_or_else(|| pq.clone()))
            .collect();
        let params: Vec<AngleExpr> = pop
            .params
            .iter()
            .map(|p| substitute_angle(p, &m.angles))
            .collect();

        let before: FxHashMap<QubitId, NodeIndex> = qubits
            .iter()
            .filter_map(|q| exit.get(q).map(|&n| (q.clone(), n)))
            .collect();

        let op = GateOp {
            gate: pop.gate.clone(),
            qubits: qubits.clone(),
            params,
        };
        let idx = if before.len() == qubits.len() {
            dag.insert_gate_before(op, &before)
        } else {
            // A replacement gate can touch a wire the pattern never did only if the rule
            // is malformed; validation rejects those, so this is a defensive fallback.
            dag.push_gate(op)
        };
        inserted.push(idx);
    }
    inserted
}

/// Resolve a replacement-side angle expression against a match's bindings.
///
/// A whole-expression binding wins if there is one (the pattern wrote the same expression
/// on both sides); otherwise each free variable is substituted individually. Both are
/// structural, so nothing depends on iteration order.
pub fn substitute_angle(expr: &AngleExpr, bindings: &FxHashMap<String, AngleExpr>) -> AngleExpr {
    if let Some(bound) = bindings.get(&expr.to_string()) {
        return bound.clone();
    }
    expr.substitute(&|name| bindings.get(name).cloned())
        .simplify()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm::{self, PrintOptions};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0xfeed)
    }

    /// Apply `find | replace` to `circuit` and return the resulting QASM, gates only.
    fn run(circuit: &str, find: &str, replace: &str) -> String {
        run_with(
            circuit,
            find,
            replace,
            &ApplyOptions {
                apply_once: false,
                shuffle: false,
                ..Default::default()
            },
        )
    }

    fn run_with(circuit: &str, find: &str, replace: &str, opts: &ApplyOptions) -> String {
        let dag = qasm::parse(circuit).unwrap();
        let rule = Rule::new(find, replace).unwrap();
        let out = match apply_rule(&dag, &rule, opts, &mut rng()) {
            Some(r) => r.dag,
            None => dag,
        };
        qasm::to_qasm_with(&out, &PrintOptions::bare())
            .lines()
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn removes_a_cancelling_pair() {
        assert_eq!(run("h b; h c; h c; x c;", "h q0; h q0;", ""), "h b; x c;");
    }

    #[test]
    fn reorders_gates() {
        assert_eq!(
            run(
                "x a; x b; cx a, b; cx c, a; cx c, b;",
                "cx q0, q1; cx q2, q0; cx q2, q1;",
                "cx q2, q0; cx q0, q1;"
            ),
            "x a; x b; cx c, a; cx a, b;"
        );
    }

    #[test]
    fn leaves_the_circuit_alone_when_nothing_matches() {
        let before = "x a; x b; cx a, b; cx c, a; cx c, b;";
        assert_eq!(
            run(before, "x d; x b; cx d, b; cx c, e; cx c, b;", ""),
            "x a; x b; cx a, b; cx c, a; cx c, b;"
        );
    }

    #[test]
    fn apply_rule_returns_none_without_a_match() {
        let dag = qasm::parse("h a;").unwrap();
        let rule = Rule::new("x q0; x q0;", "").unwrap();
        assert!(apply_rule(&dag, &rule, &ApplyOptions::default(), &mut rng()).is_none());
    }

    #[test]
    fn applies_every_disjoint_match_by_default() {
        let out = run("h a; h a; h a; h a;", "h q0; h q0;", "");
        assert_eq!(out.trim(), "");
    }

    #[test]
    fn apply_once_stops_after_one() {
        let opts = ApplyOptions {
            apply_once: true,
            shuffle: false,
            ..Default::default()
        };
        let out = run_with("h a; h a; h a; h a;", "h q0; h q0;", "", &opts);
        assert_eq!(out, "h a; h a;");
    }

    #[test]
    fn substitutes_angles_structurally() {
        // Rotation merging: the replacement's `(theta1+theta2)` is evaluated from the two
        // individually bound variables.
        let out = run(
            "rz(0.25) a; rz(0.5) a;",
            "rz(theta1) q0; rz(theta2) q0;",
            "rz((theta1+theta2)) q0;",
        );
        assert_eq!(out, "rz(0.75) a;");
    }

    /// Substituting `theta1` leaves `(theta1+theta2)`'s structure intact whatever the
    /// substitution order.
    #[test]
    fn overlapping_variable_names_substitute_correctly() {
        let mut bindings: FxHashMap<String, AngleExpr> = FxHashMap::default();
        bindings.insert("theta1".into(), AngleExpr::Num(1.0));
        bindings.insert("theta2".into(), AngleExpr::Num(2.0));
        bindings.insert("theta12".into(), AngleExpr::Num(100.0));

        let expr = AngleExpr::add(
            AngleExpr::var("theta1"),
            AngleExpr::add(AngleExpr::var("theta2"), AngleExpr::var("theta12")),
        );
        assert_eq!(substitute_angle(&expr, &bindings).eval(), Some(103.0));
    }

    #[test]
    fn whole_expression_bindings_take_precedence() {
        let mut bindings: FxHashMap<String, AngleExpr> = FxHashMap::default();
        bindings.insert("(theta1+theta2)".into(), AngleExpr::Num(7.0));
        let expr = AngleExpr::add(AngleExpr::var("theta1"), AngleExpr::var("theta2"));
        assert_eq!(substitute_angle(&expr, &bindings).eval(), Some(7.0));
    }

    #[test]
    fn rotation_merging_to_identity_removes_the_gate() {
        use std::f64::consts::PI;
        let out = run(
            &format!("rz({}) a; rz({}) a;", PI / 2.0, -PI / 2.0),
            "rz(theta1) q0; rz(theta2) q0;",
            "rz((theta1+theta2)) q0;",
        );
        // The merged rotation is rz(0), which is retained by the rewriter; dropping it is
        // the job of the explicit identity pass, not a silent parser behaviour.
        assert_eq!(out, "rz(0.0) a;");
        let dag = qasm::parse(&out).unwrap();
        let mut dag2 = dag.clone();
        dag2.drop_identity_gates(&qcircuit::GateRegistry::with_builtins());
        assert_eq!(dag2.gate_count(), 0);
    }

    #[test]
    fn replacement_may_be_larger_than_the_pattern() {
        let out = run("cz a, b;", "cz q0, q1;", "h q1; cx q0, q1; h q1;");
        assert_eq!(out, "h b; cx a, b; h b;");
    }

    #[test]
    fn replacement_on_a_subset_of_the_pattern_qubits() {
        let out = run("cx a, b; cx a, b; h a;", "cx q0, q1; cx q0, q1;", "");
        assert_eq!(out, "h a;");
    }

    #[test]
    fn three_qubit_rules_apply() {
        let out = run("ccz a, b, c; h c;", "ccz q0, q1, q2;", "ccx q0, q1, q2;");
        assert_eq!(out, "ccx a, b, c; h c;");
    }

    #[test]
    fn surrounding_gates_keep_their_order() {
        let out = run("t a; cx a, b; h b; h b; cx a, b; t a;", "h q0; h q0;", "");
        assert_eq!(out, "t a; cx a, b; cx a, b; t a;");
    }

    #[test]
    fn wires_stay_well_formed_after_rewriting() {
        let dag = qasm::parse("h a; cx a, b; h a; h a; t b;").unwrap();
        let rule = Rule::new("h q0; h q0;", "").unwrap();
        let out = apply_rule(&dag, &rule, &ApplyOptions::default(), &mut rng())
            .unwrap()
            .dag;
        for q in out.qubits() {
            let sink = out.sink(q).unwrap();
            let mut n = out.source(q).unwrap();
            let mut steps = 0;
            while n != sink {
                n = out.next_on(n, q).expect("wire is unbroken");
                steps += 1;
                assert!(steps < 20, "wire {q} loops");
            }
        }
    }

    #[test]
    fn applications_are_counted() {
        let dag = qasm::parse("h a; h a; h a; h a;").unwrap();
        let rule = Rule::new("h q0; h q0;", "").unwrap();
        let opts = ApplyOptions {
            apply_once: false,
            shuffle: false,
            ..Default::default()
        };
        let r = apply_rule(&dag, &rule, &opts, &mut rng()).unwrap();
        assert_eq!(r.applications, 2);
        assert_eq!(r.dag.gate_count(), 0);
    }

    /// Two matches that are each valid on the original circuit can conflict once the
    /// first is applied, because a rewrite reorders gates. Applying a pre-computed batch
    /// blind produced a cyclic graph here, which `topological_gates` then silently
    /// reported as a four-gate circuit. Re-matching against the updated graph after each
    /// application is what makes this sound.
    #[test]
    fn applying_two_matches_never_creates_a_cycle() {
        let src = "cx q0,q3; cx q3,q4; t q4; tdg q3; cx q0,q4; \
                   cx q2,q3; cx q0,q3; cx q2,q4; cx q2,q1; tdg q4;";
        let dag = qasm::parse(src).unwrap();
        let rule = Rule::new("cx q2,q0; cx q1,q0", "cx q1,q0; cx q2,q0;").unwrap();

        // There really are two disjoint matches on the original circuit.
        let mut r = rng();
        assert_eq!(
            find_disjoint_matches(&dag, &rule, &ApplyOptions::default(), &mut r).len(),
            2
        );

        for seed in 0..16u64 {
            let mut r = ChaCha8Rng::seed_from_u64(seed);
            let out = apply_rule(&dag, &rule, &ApplyOptions::default(), &mut r)
                .unwrap()
                .dag;
            assert!(out.is_acyclic(), "seed {seed} produced a cycle");
            assert_eq!(out.gate_count(), 10, "seed {seed} lost gates");
            assert_eq!(
                out.topological_gates().len(),
                10,
                "seed {seed}: topological order is incomplete"
            );
        }
    }

    /// A size-preserving rule that swaps two gates must not be applied forever.
    #[test]
    fn size_preserving_rules_terminate() {
        let dag = qasm::parse("cx a, b; cx c, b; cx a, b; cx c, b;").unwrap();
        let rule = Rule::new("cx q0, q1; cx q2, q1;", "cx q2, q1; cx q0, q1;").unwrap();
        let out = apply_rule(&dag, &rule, &ApplyOptions::default(), &mut rng()).unwrap();
        assert_eq!(out.dag.gate_count(), 4);
        assert!(out.applications <= 4, "applied {} times", out.applications);
        assert!(out.dag.is_acyclic());
    }

    #[test]
    fn rewriting_keeps_the_graph_acyclic() {
        let dag = qasm::parse("h a; cx a, b; h b; cx b, c; h c; cx a, c;").unwrap();
        for (find, replace) in [
            ("h q0; cx q0, q1;", "cx q0, q1; h q0;"),
            ("cx q0, q1; h q1;", "h q1; cx q0, q1;"),
        ] {
            let rule = Rule::new(find, replace).unwrap();
            if let Some(out) = apply_rule(&dag, &rule, &ApplyOptions::default(), &mut rng()) {
                assert!(out.dag.is_acyclic(), "`{find}` produced a cycle");
                assert_eq!(out.dag.gate_count(), 6);
            }
        }
    }

    #[test]
    fn the_input_dag_is_not_mutated() {
        let dag = qasm::parse("h a; h a;").unwrap();
        let rule = Rule::new("h q0; h q0;", "").unwrap();
        let r = apply_rule(&dag, &rule, &ApplyOptions::default(), &mut rng()).unwrap();
        assert_eq!(dag.gate_count(), 2, "input must be untouched");
        assert_eq!(r.dag.gate_count(), 0);
    }

    /// Greedy disjoint matching depends on which seed is tried first: starting from the
    /// middle `h` of a run of four leaves the outer two unmatched. That is expected, and
    /// the reference behaves the same way. What must hold is that repeated application
    /// converges, and that the count never increases.
    #[test]
    fn shuffled_application_converges_to_a_fixpoint() {
        let circuit = "h a; h a; h a; h a; h b; h b;";
        let rule = Rule::new("h q0; h q0;", "").unwrap();
        for seed in 0..16u64 {
            let mut r = ChaCha8Rng::seed_from_u64(seed);
            let mut dag = qasm::parse(circuit).unwrap();
            let mut last = dag.gate_count();
            for _ in 0..8 {
                let Some(out) = apply_rule(&dag, &rule, &ApplyOptions::default(), &mut r) else {
                    break;
                };
                assert!(out.dag.gate_count() < last, "seed {seed}: no progress");
                last = out.dag.gate_count();
                dag = out.dag;
            }
            assert_eq!(dag.gate_count(), 0, "seed {seed} did not converge");
        }
    }
    /// Applying a rewrite and rolling it back restores the circuit exactly.
    ///
    /// This is the property the whole in-place scheme rests on, and it is checked by
    /// structural hash rather than by node identity on purpose: a restored gate is a new
    /// node with a new index, so the circuits are equal in content and not in addressing.
    /// `Dag::structural_hash` is built from gate contents and wire structure, never from
    /// node indices, which is exactly the equality that matters here.
    #[test]
    fn a_rewrite_and_its_rollback_leave_the_circuit_unchanged() {
        let cases: &[(&str, &str, &str)] = &[
            // Cancellation: the replacement is empty.
            ("h a; h a; x a;", "h q0; h q0;", ""),
            // Commutation across a two-qubit gate: operands and order both move.
            (
                "rz(0.3) a; cx a, b; h b;",
                "rz(theta1) q0; cx q0, q1;",
                "cx q0, q1; rz(theta1) q0;",
            ),
            // Fusion: two gates become one, with a compound angle.
            (
                "rz(0.3) a; rz(0.4) a; cx a, b;",
                "rz(theta1) q0; rz(theta2) q0;",
                "rz((theta1+theta2)) q0;",
            ),
            // Expansion: one gate becomes several, so rollback must remove more than it
            // restores.
            ("cx a, b; h a;", "cx q0, q1;", "h q1; cz q0, q1; h q1;"),
            // Three-qubit, to check nothing assumes arity.
            (
                "ccz a, b, c; h a;",
                "ccz q0, q1, q2;",
                "h q2; ccx q0, q1, q2; h q2;",
            ),
            // A match at the very end of the circuit, where the exit node is a sink.
            ("x a; h a; h a;", "h q0; h q0;", ""),
            // A pattern whose BFS reaches a gate *backwards*: seeded at `h q0`, the walk
            // goes forward to the `cx` and only then back along `q1` to the `t`. So the
            // matched nodes are not in wire order, and a gate's successor at removal time
            // is not the one it had before any removals. Recording the wrong one splices
            // a restored gate onto a node that is not back yet.
            (
                "h a; t b; cx a, b; x a;",
                "h q0; t q1; cx q0, q1;",
                "cx q0, q1;",
            ),
        ];

        for (circuit, find, replace) in cases {
            let rule = Rule::new(find, replace).unwrap();
            let mut dag = qasm::parse(circuit).unwrap();
            let before_hash = dag.structural_hash();
            let before_text = qasm::to_qasm(&dag);

            let m = {
                let ctx = MatchContext::new(&dag);
                let blocked = FxHashSet::default();
                ctx.seed_candidates(&rule.find)
                    .iter()
                    .find_map(|&s| ctx.try_match(&rule.find, s, &blocked))
                    .unwrap_or_else(|| panic!("`{find}` did not match `{circuit}`"))
            };

            let mut undo = Undo::default();
            apply_match_undoable(&mut dag, &rule, &m, &mut undo);
            assert_ne!(
                dag.structural_hash(),
                before_hash,
                "`{find}` -> `{replace}` left `{circuit}` unchanged, so the test proves nothing"
            );
            assert!(dag.is_acyclic(), "rewrite of `{circuit}` produced a cycle");

            rollback(&mut dag, undo);
            assert_eq!(
                dag.structural_hash(),
                before_hash,
                "rolling back `{find}` -> `{replace}` did not restore `{circuit}`\n\
                 got:      {}\n expected: {before_text}",
                qasm::to_qasm(&dag)
            );
            assert!(dag.is_acyclic(), "rollback of `{circuit}` produced a cycle");
        }
    }

    /// A chain of rewrites accumulated into one record unwinds to the original circuit.
    ///
    /// The chain is the case that a per-step record cannot handle: rolling back a step
    /// gives its restored gates new node indices, so a step recorded earlier finds its
    /// successors moved. One `Undo` spanning the chain carries a single renumbering map
    /// across the whole unwind, which is why applying takes `&mut Undo` rather than
    /// returning one per site.
    #[test]
    fn a_chain_of_rewrites_unwinds_to_the_original() {
        let rule = Rule::new("rz(theta1) q0; cx q0, q1;", "cx q0, q1; rz(theta1) q0;").unwrap();
        let mut dag = qasm::parse(
            "rz(0.1) a; cx a, b; rz(0.2) b; cx b, c; rz(0.3) c; cx c, a; h a; h b; h c;",
        )
        .unwrap();
        let original = dag.structural_hash();

        let mut undo = Undo::default();
        let blocked = FxHashSet::default();
        for _ in 0..6 {
            let m = {
                let ctx = MatchContext::new(&dag);
                ctx.seed_candidates(&rule.find)
                    .iter()
                    .find_map(|&s| ctx.try_match(&rule.find, s, &blocked))
            };
            let Some(m) = m else { break };
            apply_match_undoable(&mut dag, &rule, &m, &mut undo);
        }
        assert!(
            undo.applications() >= 3,
            "expected a chain, got {} rewrites",
            undo.applications()
        );
        assert_ne!(dag.structural_hash(), original);

        rollback(&mut dag, undo);
        assert_eq!(
            dag.structural_hash(),
            original,
            "unwinding the chain did not restore the circuit"
        );
        assert!(dag.is_acyclic());
    }
    #[test]
    fn repro_multisite_rollback() {
        use crate::matcher::MatchIndex;
        use rand::SeedableRng;
        let rule = Rule::new("rz(theta1) q0; cx q0, q1;", "cx q0, q1; rz(theta1) q0;").unwrap();
        let mut src = String::new();
        for i in 0..12 {
            src.push_str(&format!(
                "rz(0.{}) q{}; cx q{}, q{}; h q{};\n",
                i % 9 + 1,
                i % 4,
                i % 4,
                (i + 1) % 4,
                (i + 2) % 4
            ));
        }
        let mut dag = qasm::parse(&src).unwrap();
        let mut index = MatchIndex::build(&dag);
        let original = dag.structural_hash();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let opts = ApplyOptions {
            apply_once: false,
            shuffle: true,
            deadline: None,
        };
        let undo = apply_rule_in_place(&mut dag, &mut index, &rule, &opts, &mut rng).unwrap();
        eprintln!("applied {} sites", undo.applications());
        rollback_with_index(&mut dag, &mut index, undo);
        assert_eq!(
            dag.structural_hash(),
            original,
            "multi-site rollback failed"
        );
        assert!(index.is_topologically_valid(&dag));
    }
}
