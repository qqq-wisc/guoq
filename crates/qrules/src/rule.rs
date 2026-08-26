//! Rewrite rules.

use qcircuit::GateRegistry;

use crate::error::{Result, RuleError};
use crate::pattern::Pattern;

/// What kind of rewrite a rule expresses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleKind {
    /// A direct circuit-to-circuit rewrite.
    Plain,
    /// A rewrite valid only when an arbitrary sub-circuit between the two halves
    /// satisfies a classical constraint. Handled in [`crate::symbolic`].
    Symbolic,
}

/// A rewrite rule: find one circuit fragment, write another in its place.
///
/// Note the naming. The shipped rule files are written `smaller | larger`, because the
/// synthesizer emits each equivalence class's representative first. GUOQ optimises by
/// searching for the *larger* side and replacing it with the *smaller*, so the file's
/// right-hand side is what gets matched. Calling the fields `find` and `replace` rather
/// than `lhs` and `rhs` keeps that from being ambiguous; the reference passed
/// `splitRule[0]` as a "replace" string and `rule.getFirst()` as a "pattern" DAG through
/// three call layers, which was easy to misread.
#[derive(Debug, Clone)]
pub struct Rule {
    /// The pattern searched for in the circuit.
    pub find: Pattern,
    /// What is written in its place.
    pub replace: Pattern,
    pub kind: RuleKind,
    /// Stable identity for logging and rule scoring.
    pub id: String,
}

impl Rule {
    /// Build a plain rule from two QASM fragments.
    pub fn new(find: &str, replace: &str) -> Result<Self> {
        let find_p = Pattern::parse(find)?;
        let replace_p = if replace.trim().is_empty() {
            Pattern::empty()
        } else {
            Pattern::parse(replace)?
        };
        let id = format!("{} | {}", find.trim(), replace.trim());
        let rule = Self {
            find: find_p,
            replace: replace_p,
            kind: RuleKind::Plain,
            id,
        };
        rule.check_bindings()?;
        Ok(rule)
    }

    /// Net change in gate count when this rule fires: negative means it shrinks.
    pub fn size_delta(&self) -> isize {
        self.replace.gate_count() as isize - self.find.gate_count() as isize
    }

    /// `true` if the rule leaves the gate count unchanged.
    pub fn is_size_preserving(&self) -> bool {
        self.size_delta() == 0
    }

    /// The reverse rule, searching for what this one writes.
    pub fn reversed(&self) -> Result<Self> {
        let rule = Self {
            find: self.replace.clone(),
            replace: self.find.clone(),
            kind: self.kind.clone(),
            id: format!("{} | {}", self.replace.text(), self.find.text()),
        };
        rule.check_bindings()?;
        Ok(rule)
    }

    /// Every gate mentioned by either side is defined, and both sides are well formed.
    pub fn validate(&self, registry: &GateRegistry) -> Result<()> {
        self.find.validate(registry)?;
        self.replace.validate(registry)?;
        Ok(())
    }

    /// The replacement may only use qubits and angles the pattern binds.
    ///
    /// Without this check a rule can silently introduce an unbound qubit name or leave a
    /// literal `theta1` in the output circuit — which is exactly what the reference did
    /// when a rule was used in the direction that binds a compound expression as a whole.
    fn check_bindings(&self) -> Result<()> {
        for q in self.replace.qubits() {
            if !self.find.qubits().iter().any(|p| p == q) {
                return Err(RuleError::UnboundQubit {
                    qubit: q.to_string(),
                    rule: self.id.clone(),
                });
            }
        }

        let find_keys = self.find.symbolic_param_keys();
        let find_vars = self.find.bound_vars();
        for key in self.replace.symbolic_param_keys() {
            // Either the whole expression is bound, or every variable in it is.
            if find_keys.contains(&key) {
                continue;
            }
            let expr_vars: Vec<String> = self
                .replace
                .free_vars()
                .into_iter()
                .filter(|v| key.contains(v.as_str()))
                .collect();
            let all_bound =
                !expr_vars.is_empty() && expr_vars.iter().all(|v| find_vars.contains(v));
            if !all_bound {
                return Err(RuleError::UnboundAngle {
                    angle: key,
                    rule: self.id.clone(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_plain_rule() {
        let r = Rule::new("h q0; h q0;", "").unwrap();
        assert_eq!(r.find.gate_count(), 2);
        assert!(r.replace.is_empty());
        assert_eq!(r.kind, RuleKind::Plain);
        assert_eq!(r.size_delta(), -2);
        assert!(!r.is_size_preserving());
    }

    #[test]
    fn size_delta_signs() {
        assert_eq!(Rule::new("h q0; h q0;", "").unwrap().size_delta(), -2);
        assert_eq!(
            Rule::new("cz q0, q1;", "h q1; cx q0, q1; h q1;")
                .unwrap()
                .size_delta(),
            2
        );
        let swap = Rule::new("cx q0, q1; cx q2, q1;", "cx q2, q1; cx q0, q1;").unwrap();
        assert_eq!(swap.size_delta(), 0);
        assert!(swap.is_size_preserving());
    }

    #[test]
    fn rejects_a_replacement_with_an_unbound_qubit() {
        let err = Rule::new("h q0; h q0;", "cx q0, q9;").unwrap_err();
        assert!(matches!(err, RuleError::UnboundQubit { .. }));
    }

    #[test]
    fn rejects_a_replacement_with_an_unbound_angle() {
        let err = Rule::new("h q0; h q0;", "rz(theta7) q0;").unwrap_err();
        assert!(matches!(err, RuleError::UnboundAngle { .. }));
    }

    #[test]
    fn accepts_a_replacement_combining_bound_variables() {
        let r = Rule::new("rz(theta1) q0; rz(theta2) q0;", "rz((theta1+theta2)) q0;").unwrap();
        assert_eq!(r.size_delta(), -1);
    }

    #[test]
    fn accepts_a_replacement_reusing_a_whole_bound_expression() {
        let r = Rule::new(
            "rz((theta1+theta2)) q0; h q0;",
            "h q0; rz((theta1+theta2)) q0;",
        )
        .unwrap();
        assert!(r.is_size_preserving());
    }

    #[test]
    fn reversing_a_rule_swaps_the_sides() {
        let r = Rule::new("cz q0, q1;", "h q1; cx q0, q1; h q1;").unwrap();
        let back = r.reversed().unwrap();
        assert_eq!(back.find.gate_count(), 3);
        assert_eq!(back.replace.gate_count(), 1);
        assert_eq!(back.size_delta(), -2);
    }

    #[test]
    fn reversing_can_be_invalid() {
        // Reversed, the replacement would need theta1 and theta2 individually, but the
        // pattern only binds their sum.
        let r = Rule::new("rz(theta1) q0; rz(theta2) q0;", "rz((theta1+theta2)) q0;").unwrap();
        assert!(r.reversed().is_err());
    }

    #[test]
    fn validate_checks_both_sides() {
        let reg = GateRegistry::with_builtins();
        assert!(Rule::new("h q0; h q0;", "").unwrap().validate(&reg).is_ok());
        let bogus = Rule::new("nosuch q0; h q0;", "").unwrap();
        assert!(bogus.validate(&reg).is_err());
    }

    #[test]
    fn a_disconnected_pattern_is_rejected() {
        assert!(matches!(
            Rule::new("h q0; h q1;", "").unwrap_err(),
            RuleError::Disconnected(_)
        ));
    }

    #[test]
    fn rule_id_is_stable() {
        let a = Rule::new("h q0; h q0;", "").unwrap();
        let b = Rule::new("h q0; h q0;", "").unwrap();
        assert_eq!(a.id, b.id);
        assert_eq!(a.id, "h q0; h q0; | ");
    }
}
