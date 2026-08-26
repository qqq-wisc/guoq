//! Reading the reference implementation's rule-file format.
//!
//! Each line is `smaller | larger` for a plain rule, or
//! `smaller | larger | constraints` for a symbolic one. The synthesizer emits each
//! equivalence class's representative first, so the *second* field is what GUOQ searches
//! for and the first is what it writes — see [`Rule`].
//!
//! Keeping this format readable means the 177,380 shipped rules stay usable and the port
//! can be benchmarked against the reference on identical inputs.

use std::path::Path;

use qcircuit::GateRegistry;

use crate::error::{Result, RuleError};
use crate::pattern::Pattern;
use crate::rule::{Rule, RuleKind};

/// Filters applied while loading a rule file.
///
/// These mirror the reference's `Optimizer.getRules` and `validRule`, which took seven
/// positional booleans and integers threaded through three call layers.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Drop rules that leave the gate count unchanged.
    pub drop_size_preserving: bool,
    /// Also load size-preserving rules in the reverse direction.
    pub add_size_preserving_reflection: bool,
    /// Also load rules that increase the gate count, for escaping local minima.
    pub add_size_increasing: bool,
    /// Reject rules acting on more than this many qubits. `None` for no limit.
    pub max_rule_qubits: Option<usize>,
    /// Reject rules whose replacement needs qubit connectivity the pattern lacks.
    pub preserve_mapping: bool,
    /// Reject any rule whose search pattern contains a compound angle expression.
    ///
    /// The reference did this by testing the pattern text for `+` or `-`
    /// (`Optimizer.validRule`), as a blunt way of avoiding replacements that need a
    /// variable the pattern binds only inside a sum. [`Rule`] checks that condition
    /// directly, so this is off by default and admits valid commutation rules such as
    /// `h q0; rz((theta1+theta2)) q0;` that the reference discarded. Turn it on to
    /// reproduce the reference's rule set exactly.
    pub reject_compound_pattern_angles: bool,
}

/// What a rule file yielded.
#[derive(Debug, Default)]
pub struct LoadReport {
    pub rules: Vec<Rule>,
    /// Lines carrying the symbolic hole, kept verbatim for the symbolic reader.
    pub symbolic_lines: Vec<String>,
    /// Lines rejected by the filters, with the reason.
    pub rejected: Vec<(usize, String)>,
}

impl LoadReport {
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// The marker the synthesizer writes for an arbitrary sub-circuit.
pub const SYMBOLIC_HOLE: &str = "symb q";

/// Load rules from a file.
pub fn load_file(path: &Path, opts: &LoadOptions, registry: &GateRegistry) -> Result<LoadReport> {
    let text = std::fs::read_to_string(path).map_err(|source| RuleError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(load_str(&text, opts, registry))
}

/// Load rules from rule-file text.
pub fn load_str(text: &str, opts: &LoadOptions, registry: &GateRegistry) -> LoadReport {
    let mut report = LoadReport::default();
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if line.contains(SYMBOLIC_HOLE) {
            report.symbolic_lines.push(line.to_string());
            continue;
        }
        let fields: Vec<&str> = line.split(" | ").collect();
        if fields.len() < 2 {
            report
                .rejected
                .push((lineno + 1, "no ` | ` separator".into()));
            continue;
        }
        let (smaller, larger) = (fields[0], fields[1]);

        // Primary direction: search for the larger side, write the smaller.
        match build(larger, smaller, opts, registry) {
            Ok(rule) => report.rules.push(rule),
            Err(reason) => report.rejected.push((lineno + 1, reason)),
        }

        // Optional extra directions.
        let same_size = count_gates(smaller) == count_gates(larger);
        if opts.add_size_preserving_reflection && same_size {
            if let Ok(rule) = build(smaller, larger, opts, registry) {
                report.rules.push(rule);
            }
        }
        if opts.add_size_increasing && count_gates(smaller) < count_gates(larger) {
            if let Ok(rule) = build(smaller, larger, opts, registry) {
                report.rules.push(rule);
            }
        }
    }
    report
}

fn count_gates(side: &str) -> usize {
    side.matches(';').count()
}

fn build(
    find: &str,
    replace: &str,
    opts: &LoadOptions,
    registry: &GateRegistry,
) -> std::result::Result<Rule, String> {
    if opts.reject_compound_pattern_angles && (find.contains('+') || find.contains('-')) {
        return Err("pattern contains a compound angle expression".into());
    }

    let rule = Rule::new(find, replace).map_err(|e| e.to_string())?;
    rule.validate(registry).map_err(|e| e.to_string())?;

    if rule.find.is_empty() {
        return Err("empty search pattern".into());
    }
    if opts.drop_size_preserving && rule.is_size_preserving() {
        return Err("size-preserving".into());
    }
    if let Some(max) = opts.max_rule_qubits {
        if rule.find.qubits().len() > max {
            return Err(format!("uses more than {max} qubits"));
        }
    }
    // A rule that rewrites a circuit to itself is a no-op that would still consume a
    // search iteration.
    if rule.find.dag().structural_hash() == rule.replace.dag().structural_hash() {
        return Err("both sides are identical".into());
    }
    if opts.preserve_mapping {
        let have = rule.find.dag().connectivity();
        let need = rule.replace.dag().connectivity();
        // Compare in pattern-qubit space; the match's qubit map is a bijection onto the
        // circuit, so connectivity introduced here is connectivity introduced there.
        if !need.iter().all(|pair| have.contains(pair)) {
            return Err("replacement needs connectivity the pattern lacks".into());
        }
    }
    Ok(rule)
}

/// Parse the symbolic rules a [`LoadReport`] set aside.
///
/// Lines that do not parse are returned as `(line number, reason)` rather than failing the
/// whole load, matching how plain rules are handled.
pub fn parse_symbolic(
    lines: &[String],
    registry: &GateRegistry,
) -> (Vec<crate::symbolic::SymbolicRule>, Vec<(usize, String)>) {
    use rayon::prelude::*;
    // Parallel because parsing a symbolic rule *verifies* it — every constraint is
    // checked numerically against the rule's four half-unitaries, which is
    // most of a run's startup once a generated corpus is the default: the cliffordt set
    // is 64k rules and cost ~5s sequentially. Per-rule work is independent and the
    // collected order is the input order, so loading stays deterministic.
    let results: Vec<std::result::Result<crate::symbolic::SymbolicRule, String>> = lines
        .par_iter()
        .map(|line| {
            crate::symbolic::SymbolicRule::parse_legacy(line, registry)
                .and_then(|rule| rule.validate(registry).map(|()| rule))
                .map_err(|e| e.to_string())
        })
        .collect();
    let mut rules = Vec::new();
    let mut rejected = Vec::new();
    for (i, r) in results.into_iter().enumerate() {
        match r {
            Ok(rule) => rules.push(rule),
            Err(e) => rejected.push((i + 1, e)),
        }
    }
    (rules, rejected)
}

/// Split a symbolic rule line into its three fields.
pub fn split_symbolic(line: &str) -> Result<(&str, &str, &str)> {
    let fields: Vec<&str> = line.split(" | ").collect();
    if fields.len() < 3 {
        return Err(RuleError::Malformed(line.to_string()));
    }
    Ok((fields[0], fields[1], fields[2]))
}

/// Serialize a rule back to the legacy `smaller | larger` form.
pub fn to_legacy_line(rule: &Rule) -> String {
    match rule.kind {
        RuleKind::Plain => format!("{} | {}", rule.replace.text(), rule.find.text()),
        RuleKind::Symbolic => rule.id.clone(),
    }
}

/// Parse one side of a rule file line.
pub fn parse_side(text: &str) -> Result<Pattern> {
    Pattern::parse(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    #[test]
    fn loads_a_plain_rule_in_the_shrinking_direction() {
        let r = load_str(" | h q0; h q0;", &LoadOptions::default(), &reg());
        assert_eq!(r.len(), 1);
        // The file's second field is what gets searched for.
        assert_eq!(r.rules[0].find.gate_count(), 2);
        assert!(r.rules[0].replace.is_empty());
        assert_eq!(r.rules[0].size_delta(), -2);
    }

    #[test]
    fn skips_blank_lines() {
        let r = load_str("\n\n | h q0; h q0;\n\n", &LoadOptions::default(), &reg());
        assert_eq!(r.len(), 1);
        assert!(r.rejected.is_empty());
    }

    #[test]
    fn separates_symbolic_lines() {
        let text = " | h q0; h q0;\nrz(theta1) q0; symb q; | symb q; rz(theta1) q0; | [{}]";
        let r = load_str(text, &LoadOptions::default(), &reg());
        assert_eq!(r.len(), 1);
        assert_eq!(r.symbolic_lines.len(), 1);
    }

    #[test]
    fn reports_malformed_lines() {
        let r = load_str("not a rule", &LoadOptions::default(), &reg());
        assert!(r.is_empty());
        assert_eq!(r.rejected.len(), 1);
        assert!(r.rejected[0].1.contains("separator"));
    }

    #[test]
    fn rejects_identical_sides() {
        let r = load_str("h q0; | h q0;", &LoadOptions::default(), &reg());
        assert!(r.is_empty());
        assert!(r.rejected[0].1.contains("identical"));
    }

    #[test]
    fn rejects_disconnected_patterns() {
        let r = load_str(" | h q0; h q1;", &LoadOptions::default(), &reg());
        assert!(r.is_empty());
        assert!(r.rejected[0].1.contains("disconnected"));
    }

    #[test]
    fn drop_size_preserving() {
        let line = "cx q2, q1; cx q0, q1; | cx q0, q1; cx q2, q1;";
        let keep = load_str(line, &LoadOptions::default(), &reg());
        assert_eq!(keep.len(), 1);

        let opts = LoadOptions {
            drop_size_preserving: true,
            ..LoadOptions::default()
        };
        let drop = load_str(line, &opts, &reg());
        assert!(drop.is_empty());
        assert!(drop.rejected[0].1.contains("size-preserving"));
    }

    #[test]
    fn size_preserving_reflection_adds_the_other_direction() {
        let line = "cx q2, q1; cx q0, q1; | cx q0, q1; cx q2, q1;";
        let opts = LoadOptions {
            add_size_preserving_reflection: true,
            ..LoadOptions::default()
        };
        let r = load_str(line, &opts, &reg());
        assert_eq!(r.len(), 2);
        assert_ne!(r.rules[0].find.text(), r.rules[1].find.text());
    }

    #[test]
    fn size_increasing_adds_the_growing_direction() {
        let line = "cz q0, q1; | h q1; cx q0, q1; h q1;";
        let plain = load_str(line, &LoadOptions::default(), &reg());
        assert_eq!(plain.len(), 1);
        assert_eq!(plain.rules[0].size_delta(), -2);

        let opts = LoadOptions {
            add_size_increasing: true,
            ..LoadOptions::default()
        };
        let r = load_str(line, &opts, &reg());
        assert_eq!(r.len(), 2);
        assert!(r.rules.iter().any(|x| x.size_delta() > 0));
    }

    #[test]
    fn max_rule_qubits_filters() {
        let line = " | ccz q0, q1, q2; ccz q0, q1, q2;";
        assert_eq!(load_str(line, &LoadOptions::default(), &reg()).len(), 1);
        let opts = LoadOptions {
            max_rule_qubits: Some(2),
            ..LoadOptions::default()
        };
        let r = load_str(line, &opts, &reg());
        assert!(r.is_empty());
        assert!(r.rejected[0].1.contains("more than 2"));
    }

    #[test]
    fn preserve_mapping_rejects_new_connectivity() {
        // Replacement introduces a cx between q0 and q2, which the pattern never links.
        let line = "cx q0, q2; cx q0, q1; | cx q0, q1; cx q1, q2;";
        assert_eq!(load_str(line, &LoadOptions::default(), &reg()).len(), 1);
        let opts = LoadOptions {
            preserve_mapping: true,
            ..LoadOptions::default()
        };
        let r = load_str(line, &opts, &reg());
        assert!(r.is_empty(), "expected rejection, got {:?}", r.rules.len());
        assert!(r.rejected[0].1.contains("connectivity"));
    }

    /// The reference discarded any rule whose search pattern contained `+` or `-`. That
    /// also threw away valid commutation rules over a bound compound expression.
    #[test]
    fn compound_pattern_angles_are_accepted_by_default() {
        let line = "rz((theta1+theta2)) q0; h q0; | h q0; rz((theta1+theta2)) q0;";
        let r = load_str(line, &LoadOptions::default(), &reg());
        assert_eq!(r.len(), 1, "this rule is valid and should load");

        let opts = LoadOptions {
            reject_compound_pattern_angles: true,
            ..LoadOptions::default()
        };
        assert!(load_str(line, &opts, &reg()).is_empty());
    }

    /// ...but a rule that genuinely cannot be applied is still rejected, because the
    /// replacement would need variables the pattern binds only inside a sum.
    #[test]
    fn unbindable_rules_are_rejected_on_their_merits() {
        let line = "rz(theta1) q0; rz(theta2) q0; | rz((theta1+theta2)) q0;";
        let r = load_str(line, &LoadOptions::default(), &reg());
        assert!(r.is_empty());
        assert!(r.rejected[0].1.contains("angle"), "{:?}", r.rejected);
    }

    #[test]
    fn unknown_gates_are_rejected() {
        let r = load_str(" | nosuchgate q0; h q0;", &LoadOptions::default(), &reg());
        assert!(r.is_empty());
        assert!(r.rejected[0].1.contains("unknown gate"));
    }

    #[test]
    fn round_trips_to_the_legacy_form() {
        let line = "cz q0, q1; | h q1; cx q0, q1; h q1;";
        let r = load_str(line, &LoadOptions::default(), &reg());
        assert_eq!(to_legacy_line(&r.rules[0]), line);
    }

    #[test]
    fn splits_symbolic_lines() {
        let line = "a q0; symb q; | symb q; b q0; | [{}]";
        let (s, l, c) = split_symbolic(line).unwrap();
        assert_eq!(s, "a q0; symb q;");
        assert_eq!(l, "symb q; b q0;");
        assert_eq!(c, "[{}]");
        assert!(split_symbolic("a | b").is_err());
    }
}
