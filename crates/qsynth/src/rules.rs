//! Turning equivalence classes into rewrite rules.

use qcircuit::GateRegistry;

use crate::enumerate::{Enumerated, EnumerationConfig, EquivalenceClass};
use crate::exact::Side;
use crate::fingerprint::{AngleSample, Equivalence};

/// A rule as the synthesizer produces it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SynthesizedRule {
    /// The class representative: the smaller side.
    pub smaller: String,
    /// The other member: the side an optimizer searches for.
    pub larger: String,
}

impl SynthesizedRule {
    /// The rule-file line, `smaller | larger`.
    pub fn to_line(&self) -> String {
        format!("{} | {}", self.smaller, self.larger)
    }
}

/// Filters applied when turning a class into rules.
#[derive(Debug, Clone)]
pub struct RuleOptions {
    /// Drop a pair whose two sides begin or end with the same gate.
    ///
    /// Such a rule is redundant: the shared gate can be peeled off, leaving a smaller rule
    /// that the enumeration will also have found. The reference calls this
    /// `hasCommonSubcircuit`.
    pub drop_common_ends: bool,
    /// Independent random rational points a pair must survive exact verification at
    /// before being emitted.
    pub verify_rounds: usize,
    pub equivalence: Equivalence,
}

impl Default for RuleOptions {
    fn default() -> Self {
        Self {
            drop_common_ends: true,
            verify_rounds: 2,
            equivalence: Equivalence::Exact,
        }
    }
}

/// `true` if the two circuits share a first or last gate.
pub fn has_common_end(a: &Enumerated, b: &Enumerated) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a.ops.first() == b.ops.first() || a.ops.last() == b.ops.last()
}

/// Emit rules from equivalence classes.
///
/// Every pair is verified **exactly** before being emitted — rational arithmetic at
/// random rational points on the unit circle, zero tolerance (see [`crate::exact`]).
/// The fingerprint is only candidate discovery, so it is free to be fast and floating
/// point: a collision costs a wasted exact check here, never a wrong rule. A pair the
/// exact fragment cannot express (a user-defined composite gate, a decimal angle) is
/// dropped, not trusted.
pub fn gather_rules(
    classes: &[EquivalenceClass],
    config: &EnumerationConfig,
    registry: &GateRegistry,
    sample: &AngleSample,
    options: &RuleOptions,
) -> Vec<SynthesizedRule> {
    let qubits = config.qubit_names();
    let mut out = Vec::new();

    for class in classes {
        let rep = &class.representative;
        for member in &class.members {
            if member.key == rep.key {
                continue;
            }
            if options.drop_common_ends && has_common_end(rep, member) {
                continue;
            }
            let verified = crate::exact::verify(
                &Side::plain(&rep.ops),
                &Side::plain(&member.ops),
                None,
                &qubits,
                registry,
                options.equivalence,
                options.verify_rounds,
                sample.seed(),
            );
            if !matches!(verified, Ok(true)) {
                continue;
            }
            out.push(SynthesizedRule {
                smaller: rep.to_rule_text(),
                larger: member.to_rule_text(),
            });
        }
    }

    out.sort();
    out.dedup();
    out
}

/// Render rules as a rule file.
pub fn to_rule_file(rules: &[SynthesizedRule]) -> String {
    let mut out = String::new();
    for r in rules {
        out.push_str(&r.to_line());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerate::Enumerator;
    use qcircuit::{GateOp, GateSetLibrary};

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn nam_config(max_qubits: usize, max_size: usize) -> EnumerationConfig {
        let lib = GateSetLibrary::builtin();
        EnumerationConfig::from_gate_set(lib.get("nam").unwrap(), max_qubits, max_size)
    }

    fn synthesize(max_qubits: usize, max_size: usize) -> Vec<SynthesizedRule> {
        let registry = reg();
        let config = nam_config(max_qubits, max_size);
        let mut e = Enumerator::new(config.clone(), &registry);
        let classes = e.run();
        let sample = e.sample().clone();
        gather_rules(
            &classes,
            &config,
            &registry,
            &sample,
            &RuleOptions::default(),
        )
    }

    #[test]
    fn common_ends_are_detected() {
        let h = GateOp::new("h", vec!["q0"], []);
        let x = GateOp::new("x", vec!["q0"], []);
        let a = Enumerated::empty().extend(h.clone()).extend(x.clone());
        let b = Enumerated::empty().extend(h.clone()).extend(h.clone());
        assert!(has_common_end(&a, &b), "both start with h");

        let c = Enumerated::empty().extend(x.clone()).extend(x.clone());
        let d = Enumerated::empty().extend(h.clone()).extend(x.clone());
        assert!(has_common_end(&c, &d), "both end with x");

        let e = Enumerated::empty().extend(h.clone());
        let f = Enumerated::empty().extend(x.clone());
        assert!(!has_common_end(&e, &f));
        // The empty circuit shares nothing.
        assert!(!has_common_end(&Enumerated::empty(), &e));
    }

    #[test]
    fn rules_render_in_the_reference_spelling() {
        let r = SynthesizedRule {
            smaller: ";".into(),
            larger: "h q1; h q1;".into(),
        };
        assert_eq!(r.to_line(), "; | h q1; h q1;");
        assert_eq!(to_rule_file(&[r]), "; | h q1; h q1;\n");
    }

    /// Every rule the synthesizer emits must be a genuine equivalence.
    #[test]
    fn every_emitted_rule_is_semantics_preserving() {
        use qcircuit::qasm;
        use qsemantics::{phase_invariant_distance, Unitary};

        let registry = reg();
        let config = nam_config(3, 3);
        let mut e = Enumerator::new(config.clone(), &registry);
        let classes = e.run();
        let sample = e.sample().clone();
        let rules = gather_rules(
            &classes,
            &config,
            &registry,
            &sample,
            &RuleOptions::default(),
        );
        assert!(!rules.is_empty(), "no rules were synthesized");

        let qubits = config.qubit_names();
        for rule in &rules {
            // Bind the symbolic angles to independent concrete values and compare.
            let mut check = sample.resample(9_999);
            let a = qasm::parse(rule.smaller.trim_start_matches(';')).unwrap();
            let b = qasm::parse(&rule.larger).unwrap();
            for dag in [&a, &b] {
                for idx in dag.gate_indices() {
                    for p in &dag.gate(idx).params {
                        for v in p.free_vars() {
                            check.get(v);
                        }
                    }
                }
            }
            let bindings = check.bindings().clone();
            let env = move |name: &str| bindings.get(name).copied();
            let ua = Unitary::from_dag_over_with_env(&a, qubits.clone(), &registry, &env).unwrap();
            let ub = Unitary::from_dag_over_with_env(&b, qubits.clone(), &registry, &env).unwrap();
            let d = phase_invariant_distance(&ua, &ub);
            assert!(
                d < 1e-9,
                "`{}` is not a valid rule (distance {d:.3e})",
                rule.to_line()
            );
        }
        println!("verified {} synthesized rules", rules.len());
    }

    /// The reference's `SynthesizerTest` enumerates the Nam gate set over three qubits at
    /// size five and expects these sixteen rules. Reproducing them is the acceptance
    /// gate for this milestone.
    #[test]
    #[ignore = "slow; run with --include-ignored, ideally in release"]
    fn reproduces_the_reference_rule_set() {
        let expected = [
            "cx q0, q1; x q0; h q0; | x q0; cx q0, q1; h q0; x q1;",
            "cx q0, q1; rz(theta1) q0; | rz(theta1) q0; cx q0, q1;",
            "cx q0, q1; cx q2, q1; | cx q2, q1; cx q0, q1;",
            "cx q2, q0; cx q2, q1; | cx q2, q1; cx q2, q0;",
            "; | cx q1, q0; cx q1, q0;",
            "; | h q1; h q1;",
            "; | h q2; h q2;",
            "; | h q0; h q0;",
            "; | cx q2, q0; cx q2, q0;",
            "; | cx q2, q1; cx q2, q1;",
            "; | cx q0, q1; cx q0, q1;",
            "; | x q1; x q1;",
            "; | x q2; x q2;",
            "; | x q0; x q0;",
            "cx q0, q1; h q0; h q1; | h q0; h q1; cx q1, q0;",
            "h q0; cx q0, q1; h q0; | h q1; cx q1, q0; h q1;",
        ];

        let rules = synthesize(3, 5);
        let lines: std::collections::HashSet<String> = rules.iter().map(|r| r.to_line()).collect();
        println!("synthesized {} rules", lines.len());

        let missing: Vec<&str> = expected
            .iter()
            .filter(|e| !lines.contains(**e))
            .copied()
            .collect();
        assert!(
            missing.is_empty(),
            "{} of the reference's rules were not reproduced:\n{}",
            missing.len(),
            missing.join("\n")
        );
    }

    /// A cheaper version of the same check: the size-2 cancellation rules must all appear.
    #[test]
    fn reproduces_the_cancellation_rules() {
        let rules = synthesize(2, 2);
        let lines: std::collections::HashSet<String> = rules.iter().map(|r| r.to_line()).collect();
        for expected in [
            "; | h q0; h q0;",
            "; | h q1; h q1;",
            "; | x q0; x q0;",
            "; | x q1; x q1;",
            "; | cx q0, q1; cx q0, q1;",
            "; | cx q1, q0; cx q1, q0;",
        ] {
            assert!(
                lines.contains(expected),
                "missing `{expected}`\nhave: {lines:?}"
            );
        }
    }

    /// The commutation rules that make rotation merging work.
    #[test]
    fn reproduces_the_rz_commutation_rule() {
        let rules = synthesize(2, 2);
        let lines: std::collections::HashSet<String> = rules.iter().map(|r| r.to_line()).collect();
        assert!(
            lines.contains("cx q0, q1; rz(theta1) q0; | rz(theta1) q0; cx q0, q1;"),
            "rz on the control must commute with cx; have: {lines:?}"
        );
    }

    /// The common-ends filter turns out to be redundant, and the enumeration is why.
    ///
    /// Suppose two members of one class both begin with gate `g`: `g r` and `g m`. Gates
    /// are unitary, so `g r == g m` implies `r == m`. Suffix pruning only enumerates a
    /// circuit whose drop-first suffix is its class representative, so both `r` and `m`
    /// are representatives of the same class — hence the same circuit, hence `g r` and
    /// `g m` are the same circuit and there is no pair. The argument runs identically for
    /// a shared last gate.
    ///
    /// So the reference's `hasCommonSubcircuit` check cannot ever fire against its own
    /// enumeration. It is kept here because it costs nothing and would matter if the
    /// pruning were ever relaxed, but this test records that it is currently vacuous
    /// rather than silently relying on it.
    #[test]
    fn the_common_ends_filter_is_redundant_under_suffix_pruning() {
        let registry = reg();
        for (qubits, size) in [(2, 3), (3, 3)] {
            let config = nam_config(qubits, size);
            let mut e = Enumerator::new(config.clone(), &registry);
            let classes = e.run();
            let sample = e.sample().clone();

            let with = gather_rules(
                &classes,
                &config,
                &registry,
                &sample,
                &RuleOptions::default(),
            );
            let without = gather_rules(
                &classes,
                &config,
                &registry,
                &sample,
                &RuleOptions {
                    drop_common_ends: false,
                    ..RuleOptions::default()
                },
            );
            assert_eq!(
                with.len(),
                without.len(),
                "q{qubits} s{size}: the filter is expected to be vacuous here"
            );

            // The property the filter would have enforced holds anyway.
            for class in &classes {
                for member in &class.members {
                    if member.key != class.representative.key {
                        assert!(
                            !has_common_end(&class.representative, member),
                            "`{}` and `{}` share an end",
                            class.representative.key,
                            member.key
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn synthesis_is_deterministic() {
        let a = synthesize(2, 3);
        let b = synthesize(2, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn no_rule_has_identical_sides() {
        for rule in synthesize(2, 3) {
            assert_ne!(rule.smaller, rule.larger, "{}", rule.to_line());
        }
    }

    /// Up-to-phase equivalence admits strictly more rules than exact equivalence.
    #[test]
    fn phase_invariant_synthesis_finds_more() {
        let registry = reg();
        let config = nam_config(2, 3);
        let mut e = Enumerator::new(config.clone(), &registry);
        let classes = e.run();
        let sample = e.sample().clone();

        let exact = gather_rules(
            &classes,
            &config,
            &registry,
            &sample,
            &RuleOptions::default(),
        );
        let loose = gather_rules(
            &classes,
            &config,
            &registry,
            &sample,
            &RuleOptions {
                equivalence: Equivalence::UpToPhase,
                ..RuleOptions::default()
            },
        );
        assert!(
            loose.len() >= exact.len(),
            "up-to-phase should not lose rules: {} vs {}",
            loose.len(),
            exact.len()
        );
    }
}
