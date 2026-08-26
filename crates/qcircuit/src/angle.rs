//! Symbolic angle expressions.
//!
//! Gate parameters in rewrite rules are symbolic (`theta1`, `(theta1+theta2)`), while
//! parameters in concrete circuits are numeric. Both are represented by [`AngleExpr`].
//!
//! # Why this is not just `f64`
//!
//! The reference Java implementation compared angles with
//! `(eval(a) % (4*PI)) == (eval(b) % (4*PI))` (`Optimizer.sameAngle`). That has two
//! defects which this module exists to avoid:
//!
//! 1. Exact `==` on floating point, so `pi/4` computed two different ways compares unequal.
//! 2. Java's `%` is a *remainder*, which keeps the sign of the dividend. `-pi/4 % 4pi`
//!    is `-pi/4`, never equal to `15pi/4` even though the two are the same rotation.
//!    Every rule with a negative angle therefore silently under-matched.
//!
//! [`AngleExpr::approx_eq_mod`] uses `rem_euclid` (always non-negative) and a tolerance,
//! and additionally treats the two ends of the period as adjacent.

use std::fmt;

/// Full period for the angle parameter of a rotation gate.
///
/// Rotation gates are `2`-periodic in the Bloch-sphere sense but `4*PI`-periodic as
/// unitaries: `rz(2*PI) == -I`, which is a global phase but *not* the identity when the
/// gate is a controlled operation. Comparisons are therefore modulo `4*PI`.
pub const FULL_PERIOD: f64 = 4.0 * std::f64::consts::PI;

/// Default tolerance for angle comparison.
pub const ANGLE_EPS: f64 = 1e-9;

/// A symbolic or numeric angle.
#[derive(Debug, Clone, PartialEq)]
pub enum AngleExpr {
    /// A numeric literal, in radians.
    Num(f64),
    /// The constant `pi`.
    Pi,
    /// A free variable, e.g. `theta1`.
    Var(String),
    Neg(Box<AngleExpr>),
    Add(Box<AngleExpr>, Box<AngleExpr>),
    Sub(Box<AngleExpr>, Box<AngleExpr>),
    Mul(Box<AngleExpr>, Box<AngleExpr>),
    Div(Box<AngleExpr>, Box<AngleExpr>),
}

#[allow(clippy::should_implement_trait)] // these are constructors, not operator impls
impl AngleExpr {
    /// A numeric angle.
    pub fn num(v: f64) -> Self {
        AngleExpr::Num(v)
    }

    /// A free variable.
    pub fn var(name: impl Into<String>) -> Self {
        AngleExpr::Var(name.into())
    }

    pub fn add(a: AngleExpr, b: AngleExpr) -> Self {
        AngleExpr::Add(Box::new(a), Box::new(b))
    }

    pub fn sub(a: AngleExpr, b: AngleExpr) -> Self {
        AngleExpr::Sub(Box::new(a), Box::new(b))
    }

    pub fn mul(a: AngleExpr, b: AngleExpr) -> Self {
        AngleExpr::Mul(Box::new(a), Box::new(b))
    }

    pub fn div(a: AngleExpr, b: AngleExpr) -> Self {
        AngleExpr::Div(Box::new(a), Box::new(b))
    }

    pub fn neg(a: AngleExpr) -> Self {
        AngleExpr::Neg(Box::new(a))
    }

    /// `true` if this expression contains no free variables and can be [`eval`]uated.
    ///
    /// [`eval`]: AngleExpr::eval
    pub fn is_concrete(&self) -> bool {
        match self {
            AngleExpr::Num(_) | AngleExpr::Pi => true,
            AngleExpr::Var(_) => false,
            AngleExpr::Neg(a) => a.is_concrete(),
            AngleExpr::Add(a, b)
            | AngleExpr::Sub(a, b)
            | AngleExpr::Mul(a, b)
            | AngleExpr::Div(a, b) => a.is_concrete() && b.is_concrete(),
        }
    }

    /// Numeric value, or `None` if the expression has free variables.
    ///
    /// Unlike the reference implementation's `NodeVisitor.eval`, this never panics on a
    /// free variable; callers decide what to do about symbolic angles.
    pub fn eval(&self) -> Option<f64> {
        match self {
            AngleExpr::Num(v) => Some(*v),
            AngleExpr::Pi => Some(std::f64::consts::PI),
            AngleExpr::Var(_) => None,
            AngleExpr::Neg(a) => Some(-a.eval()?),
            AngleExpr::Add(a, b) => Some(a.eval()? + b.eval()?),
            AngleExpr::Sub(a, b) => Some(a.eval()? - b.eval()?),
            AngleExpr::Mul(a, b) => Some(a.eval()? * b.eval()?),
            AngleExpr::Div(a, b) => Some(a.eval()? / b.eval()?),
        }
    }

    /// Numeric value with free variables resolved through `env`.
    pub fn eval_with(&self, env: &dyn Fn(&str) -> Option<f64>) -> Option<f64> {
        match self {
            AngleExpr::Num(v) => Some(*v),
            AngleExpr::Pi => Some(std::f64::consts::PI),
            AngleExpr::Var(name) => env(name),
            AngleExpr::Neg(a) => Some(-a.eval_with(env)?),
            AngleExpr::Add(a, b) => Some(a.eval_with(env)? + b.eval_with(env)?),
            AngleExpr::Sub(a, b) => Some(a.eval_with(env)? - b.eval_with(env)?),
            AngleExpr::Mul(a, b) => Some(a.eval_with(env)? * b.eval_with(env)?),
            AngleExpr::Div(a, b) => Some(a.eval_with(env)? / b.eval_with(env)?),
        }
    }

    /// Every free variable appearing in the expression, in first-seen order.
    pub fn free_vars(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_vars(&mut out);
        out
    }

    fn collect_vars<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            AngleExpr::Num(_) | AngleExpr::Pi => {}
            AngleExpr::Var(name) => {
                if !out.contains(&name.as_str()) {
                    out.push(name);
                }
            }
            AngleExpr::Neg(a) => a.collect_vars(out),
            AngleExpr::Add(a, b)
            | AngleExpr::Sub(a, b)
            | AngleExpr::Mul(a, b)
            | AngleExpr::Div(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
        }
    }

    /// Replace each free variable using `subst`, leaving unmapped variables in place.
    ///
    /// This is a structural substitution. The reference implementation instead did a
    /// textual `String.replace` over a `HashMap` in unspecified iteration order
    /// (`Optimizer.replaceAngles`), which corrupts `(theta1+theta2)` whenever `theta1`
    /// happens to be substituted first.
    pub fn substitute(&self, subst: &dyn Fn(&str) -> Option<AngleExpr>) -> AngleExpr {
        match self {
            AngleExpr::Num(_) | AngleExpr::Pi => self.clone(),
            AngleExpr::Var(name) => subst(name).unwrap_or_else(|| self.clone()),
            AngleExpr::Neg(a) => AngleExpr::neg(a.substitute(subst)),
            AngleExpr::Add(a, b) => AngleExpr::add(a.substitute(subst), b.substitute(subst)),
            AngleExpr::Sub(a, b) => AngleExpr::sub(a.substitute(subst), b.substitute(subst)),
            AngleExpr::Mul(a, b) => AngleExpr::mul(a.substitute(subst), b.substitute(subst)),
            AngleExpr::Div(a, b) => AngleExpr::div(a.substitute(subst), b.substitute(subst)),
        }
    }

    /// Fold concrete subexpressions into a single [`AngleExpr::Num`] where possible.
    pub fn simplify(&self) -> AngleExpr {
        if let Some(v) = self.eval() {
            return AngleExpr::Num(v);
        }
        match self {
            AngleExpr::Neg(a) => AngleExpr::neg(a.simplify()),
            AngleExpr::Add(a, b) => AngleExpr::add(a.simplify(), b.simplify()),
            AngleExpr::Sub(a, b) => AngleExpr::sub(a.simplify(), b.simplify()),
            AngleExpr::Mul(a, b) => AngleExpr::mul(a.simplify(), b.simplify()),
            AngleExpr::Div(a, b) => AngleExpr::div(a.simplify(), b.simplify()),
            other => other.clone(),
        }
    }

    /// Compare two angles as rotations, modulo `4*PI`, within `eps`.
    ///
    /// Symbolic expressions compare by structural equality after [`simplify`]; only
    /// concrete angles are compared numerically.
    ///
    /// [`simplify`]: AngleExpr::simplify
    pub fn approx_eq_mod(&self, other: &AngleExpr, eps: f64) -> bool {
        match (self.eval(), other.eval()) {
            (Some(a), Some(b)) => angles_equivalent(a, b, eps),
            _ => self.simplify() == other.simplify(),
        }
    }
}

/// Reduce an angle into `[0, 4*PI)`.
///
/// Uses `rem_euclid`, which — unlike `%` — is always non-negative, so a negative angle
/// lands on the same representative as its positive equivalent.
pub fn normalize_angle(a: f64) -> f64 {
    a.rem_euclid(FULL_PERIOD)
}

/// `true` if `a` and `b` are the same rotation modulo `4*PI`, within `eps`.
///
/// Wrap-around is handled explicitly: `0` and `4*PI - 1e-15` are equivalent even though
/// their normalized forms sit at opposite ends of the interval.
pub fn angles_equivalent(a: f64, b: f64, eps: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return a == b;
    }
    let d = (normalize_angle(a) - normalize_angle(b)).abs();
    d <= eps || (FULL_PERIOD - d) <= eps
}

impl fmt::Display for AngleExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AngleExpr::Num(v) => write!(f, "{v}"),
            AngleExpr::Pi => write!(f, "pi"),
            AngleExpr::Var(name) => write!(f, "{name}"),
            AngleExpr::Neg(a) => write!(f, "-{a}"),
            AngleExpr::Add(a, b) => write!(f, "({a}+{b})"),
            AngleExpr::Sub(a, b) => write!(f, "({a}-{b})"),
            AngleExpr::Mul(a, b) => write!(f, "({a}*{b})"),
            AngleExpr::Div(a, b) => write!(f, "({a}/{b})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn eval_basic() {
        assert_eq!(AngleExpr::Num(1.5).eval(), Some(1.5));
        assert_eq!(AngleExpr::Pi.eval(), Some(PI));
        assert_eq!(AngleExpr::var("theta1").eval(), None);
        let e = AngleExpr::div(AngleExpr::Pi, AngleExpr::Num(4.0));
        assert_eq!(e.eval(), Some(PI / 4.0));
    }

    #[test]
    fn normalize_is_never_negative() {
        for a in [-0.5, -PI, -7.0 * PI, -100.0, 0.0, 3.0, 40.0] {
            let n = normalize_angle(a);
            assert!((0.0..FULL_PERIOD).contains(&n), "{a} normalized to {n}");
        }
    }

    /// A negative angle matches its positive representative.
    #[test]
    fn negative_angles_match_positive_representative() {
        assert!(angles_equivalent(-PI / 4.0, 15.0 * PI / 4.0, ANGLE_EPS));
        assert!(angles_equivalent(-PI / 2.0, 7.0 * PI / 2.0, ANGLE_EPS));
        assert!(angles_equivalent(-4.0 * PI, 0.0, ANGLE_EPS));

        // The reference computation, reproduced, to show it disagrees.
        let sign_preserving = (-PI / 4.0) % FULL_PERIOD == (15.0 * PI / 4.0) % FULL_PERIOD;
        assert!(
            !sign_preserving,
            "a sign-preserving remainder would miss this match"
        );
    }

    /// Exact float equality is unreliable for angles that are mathematically equal but
    /// computed by different routes.
    #[test]
    fn angle_comparison_is_tolerant() {
        // The next representable double above pi/4: mathematically the same angle to
        // far beyond any tolerance that matters, but never `==`.
        let a = PI / 4.0;
        let b = f64::from_bits(a.to_bits() + 1);
        assert!(a != b, "expected these to differ in the last bit");
        assert!(angles_equivalent(a, b, ANGLE_EPS));

        // Accumulated summation error over many terms, likewise.
        let c = (0..10).map(|_| 0.1).sum::<f64>();
        assert!(c != 1.0, "expected summation drift");
        assert!(angles_equivalent(c, 1.0, ANGLE_EPS));
    }

    #[test]
    fn wraparound_is_adjacent() {
        assert!(angles_equivalent(0.0, FULL_PERIOD - 1e-15, ANGLE_EPS));
        assert!(angles_equivalent(FULL_PERIOD, 1e-15, ANGLE_EPS));
    }

    #[test]
    fn two_pi_is_not_identity() {
        // rz(2pi) == -I, a global phase, but distinguishable under control.
        assert!(!angles_equivalent(2.0 * PI, 0.0, ANGLE_EPS));
        assert!(angles_equivalent(4.0 * PI, 0.0, ANGLE_EPS));
    }

    /// `theta1` inside `(theta1+theta2)` is untouchable by name collision.
    #[test]
    fn substitution_is_structural() {
        let e = AngleExpr::add(AngleExpr::var("theta1"), AngleExpr::var("theta2"));
        let out = e.substitute(&|name| match name {
            "theta1" => Some(AngleExpr::Num(1.0)),
            "theta2" => Some(AngleExpr::Num(2.0)),
            _ => None,
        });
        assert_eq!(out.eval(), Some(3.0));

        // A textual replace of "theta1" -> "1" over "(theta1+theta2)" would leave
        // "(1+1theta2...)"-style corruption; structurally each Var node is replaced once.
        let nested = AngleExpr::add(
            AngleExpr::var("theta1"),
            AngleExpr::add(AngleExpr::var("theta1"), AngleExpr::var("theta12")),
        );
        let out = nested.substitute(&|name| match name {
            "theta1" => Some(AngleExpr::Num(10.0)),
            "theta12" => Some(AngleExpr::Num(100.0)),
            _ => None,
        });
        assert_eq!(out.eval(), Some(120.0));
    }

    #[test]
    fn free_vars_dedup_in_order() {
        let e = AngleExpr::add(
            AngleExpr::var("theta2"),
            AngleExpr::add(AngleExpr::var("theta1"), AngleExpr::var("theta2")),
        );
        assert_eq!(e.free_vars(), vec!["theta2", "theta1"]);
    }

    #[test]
    fn substitute_leaves_unmapped_vars() {
        let e = AngleExpr::add(AngleExpr::var("theta1"), AngleExpr::var("theta2"));
        let out = e.substitute(&|n| (n == "theta1").then_some(AngleExpr::Num(1.0)));
        assert_eq!(out.free_vars(), vec!["theta2"]);
    }

    #[test]
    fn symbolic_angles_compare_structurally() {
        let a = AngleExpr::var("theta1");
        let b = AngleExpr::var("theta1");
        let c = AngleExpr::var("theta2");
        assert!(a.approx_eq_mod(&b, ANGLE_EPS));
        assert!(!a.approx_eq_mod(&c, ANGLE_EPS));
        // A symbolic angle is never equal to a concrete one.
        assert!(!a.approx_eq_mod(&AngleExpr::Num(0.5), ANGLE_EPS));
    }

    #[test]
    fn display_roundtrips_shape() {
        let e = AngleExpr::add(AngleExpr::var("theta1"), AngleExpr::var("theta2"));
        assert_eq!(e.to_string(), "(theta1+theta2)");
        assert_eq!(
            AngleExpr::div(AngleExpr::Pi, AngleExpr::Num(4.0)).to_string(),
            "(pi/4)"
        );
    }

    #[test]
    fn eval_with_resolves_vars() {
        let e = AngleExpr::add(AngleExpr::var("theta1"), AngleExpr::Num(1.0));
        assert_eq!(e.eval_with(&|n| (n == "theta1").then_some(2.0)), Some(3.0));
        assert_eq!(e.eval_with(&|_| None), None);
    }
}
