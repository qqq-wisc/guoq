//! Distances between unitaries, up to global phase.
//!
//! # Why the optimizer computes this itself
//!
//! The reference implementation delegated the entire question of resynthesis accuracy to
//! whichever backend it called. It passed an `epsilon` to BQSKit (whose
//! `synthesis_epsilon` bounds a Hilbert–Schmidt residual *per synthesis pass*) and the
//! same `epsilon` to Synthetiq (whose `-eps` means something else again), accepted the
//! returned circuit without ever measuring it, and divided the user's total error budget
//! by `MAX_RESYNTH_ALLOWED` — which is `-1` when the user asks for "no limit", making the
//! per-call budget *negative*.
//!
//! Nothing in that chain is checkable, and the two backends' numbers are not comparable
//! with each other, so the end-to-end error of an optimized circuit was simply unknown.
//! This module lets the optimizer measure the error of every resynthesis call itself,
//! against a definition it controls.
//!
//! # Which distance to accumulate
//!
//! Two are provided, and the difference matters:
//!
//! - [`hs_distance`] matches BQSKit's convention, so a backend's claim can be compared
//!   against what it actually delivered. It is **not** subadditive under composition, so
//!   summing it over a chain of substitutions means nothing.
//! - [`phase_invariant_distance`] is `min over phi of ||A - e^{i phi} B||_F`, which has a
//!   closed form and upper-bounds the operator-norm distance
//!   `min over phi of ||A - e^{i phi} B||_2`. Operator-norm distance *is* subadditive:
//!   replacing several disjoint blocks of a circuit incurs at most the sum of the
//!   individual distances. This is the one to accumulate into an error budget, and being
//!   an upper bound makes the accounting conservative rather than optimistic.
//!
//! Both are invariant under global phase, which is correct here: a resynthesized block
//! that differs from the original by a phase changes the whole circuit only by that same
//! phase, which is unobservable.

use num_complex::Complex64;

use crate::unitary::Unitary;

type C = Complex64;

/// `tr(A^dagger B)`, the phase-invariant overlap of two unitaries.
///
/// # Panics
///
/// Panics if the two unitaries have different dimensions.
pub fn trace_overlap(a: &Unitary, b: &Unitary) -> C {
    assert_eq!(a.dim(), b.dim(), "unitaries must have equal dimension");
    let (am, bm) = (a.matrix(), b.matrix());
    let n = a.dim();
    let mut tr = C::default();
    // tr(A^dagger B) = sum over i,j of conj(A[i][j]) * B[i][j].
    for i in 0..n {
        for j in 0..n {
            tr += am[[i, j]].conj() * bm[[i, j]];
        }
    }
    tr
}

/// `|tr(A^dagger B)|`.
pub fn overlap(a: &Unitary, b: &Unitary) -> f64 {
    trace_overlap(a, b).norm()
}

/// `min over phi of ||A - e^{i phi} B||_F`, in `[0, 2*sqrt(d)]`.
///
/// Upper-bounds the phase-invariant operator-norm distance, since `||M||_2 <= ||M||_F`.
/// That makes it safe to accumulate: the true error of a chain of substitutions never
/// exceeds the sum of these values.
///
/// # Numerical note
///
/// There is a closed form, `sqrt(2 * (d - |tr(A^dagger B)|))`, and it is the obvious way
/// to write this. It must not be used. When `A` and `B` are close — the case that matters
/// for accepting a resynthesis result — `|tr(A^dagger B)|` approaches `d` and the
/// subtraction loses every significant digit, leaving a noise floor of about
/// `sqrt(2 * d * eps_machine)`. For a two-qubit block that is `~2e-8`, which is *larger*
/// than GUOQ's default total error budget of `1e-8`: the closed form cannot tell an exact
/// resynthesis from one that has already overspent the budget.
///
/// Aligning the phase first and then taking the norm of the difference elementwise has no
/// cancellation, because each `a_ij - c * b_ij` is small and computed to full relative
/// precision. That is what this does.
pub fn phase_invariant_distance(a: &Unitary, b: &Unitary) -> f64 {
    let tr = trace_overlap(a, b);
    // Minimising ||A - cB||_F^2 = 2d - 2 Re(c * tr(A^dagger B)) over |c| = 1 means
    // maximising Re(c * tr), so c is the *conjugate* of the trace's phase.
    let c = if tr.norm() > 1e-300 {
        tr.conj() / tr.norm()
    } else {
        C::new(1.0, 0.0)
    };
    let (am, bm) = (a.matrix(), b.matrix());
    let n = a.dim();
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += (am[[i, j]] - c * bm[[i, j]]).norm_sqr();
        }
    }
    sum.max(0.0).sqrt()
}

/// Hilbert-Schmidt distance, `sqrt(1 - |tr(A^dagger B)/d|^2)`, in `[0, 1]`.
///
/// This is BQSKit's convention. Use it to report and compare against what a backend
/// claims; do **not** sum it across substitutions, because it is not subadditive under
/// composition.
///
/// Derived from [`phase_invariant_distance`] rather than from the trace directly, for the
/// cancellation reason documented there: with `f` the phase-invariant distance,
/// `1 - t = f^2 / (2d)` and `hs = f * sqrt((1 + t) / (2d))`.
pub fn hs_distance(a: &Unitary, b: &Unitary) -> f64 {
    let d = a.dim() as f64;
    let f = phase_invariant_distance(a, b);
    let t = (1.0 - f * f / (2.0 * d)).clamp(-1.0, 1.0);
    (f * ((1.0 + t) / (2.0 * d)).max(0.0).sqrt()).min(1.0)
}

/// `|tr(A^dagger B)| / d`, in `[0, 1]`; 1 means identical up to global phase.
///
/// Computed via [`phase_invariant_distance`] so that values near 1 stay accurate.
pub fn normalized_overlap(a: &Unitary, b: &Unitary) -> f64 {
    let d = a.dim() as f64;
    let f = phase_invariant_distance(a, b);
    (1.0 - f * f / (2.0 * d)).clamp(-1.0, 1.0)
}

/// `true` if `a` and `b` are the same unitary up to global phase, within `tol`.
pub fn equivalent_up_to_phase(a: &Unitary, b: &Unitary, tol: f64) -> bool {
    a.dim() == b.dim() && phase_invariant_distance(a, b) <= tol
}

/// `true` if `a` and `b` are equal including global phase, within `tol`.
pub fn equivalent_exact(a: &Unitary, b: &Unitary, tol: f64) -> bool {
    a.dim() == b.dim()
        && a.matrix()
            .iter()
            .zip(b.matrix().iter())
            .all(|(x, y)| (x - y).norm() <= tol)
}

/// A measured comparison of an original block against its resynthesized replacement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceReport {
    /// The value to accumulate into an error budget.
    pub phase_invariant: f64,
    /// The backend-comparable Hilbert–Schmidt distance.
    pub hs: f64,
    /// `|tr(A^dagger B)| / d`, in `[0, 1]`; 1 means identical up to phase.
    pub normalized_overlap: f64,
}

impl DistanceReport {
    /// Measure `replacement` against `original`.
    pub fn measure(original: &Unitary, replacement: &Unitary) -> Self {
        let d = original.dim() as f64;
        let f = phase_invariant_distance(original, replacement);
        let t = (1.0 - f * f / (2.0 * d)).clamp(-1.0, 1.0);
        Self {
            phase_invariant: f,
            hs: (f * ((1.0 + t) / (2.0 * d)).max(0.0).sqrt()).min(1.0),
            normalized_overlap: t,
        }
    }

    /// `true` if accepting this substitution keeps the total within `budget`.
    pub fn within(&self, already_spent: f64, budget: f64) -> bool {
        already_spent + self.phase_invariant <= budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::{qasm, GateRegistry};
    use std::f64::consts::PI;

    fn u(src: &str) -> Unitary {
        Unitary::from_dag(&qasm::parse(src).unwrap(), &GateRegistry::with_builtins()).unwrap()
    }

    #[test]
    fn identical_circuits_have_zero_distance() {
        let a = u("qreg q[2];\nh q[0];\ncx q[0], q[1];");
        let b = u("qreg q[2];\nh q[0];\ncx q[0], q[1];");
        assert!(phase_invariant_distance(&a, &b) < 1e-12);
        assert!(hs_distance(&a, &b) < 1e-12);
        assert!((DistanceReport::measure(&a, &b).normalized_overlap - 1.0).abs() < 1e-12);
    }

    #[test]
    fn equivalent_circuits_have_zero_distance() {
        // Two spellings of cz.
        let a = u("qreg q[2];\ncz q[0], q[1];");
        let b = u("qreg q[2];\nh q[1];\ncx q[0], q[1];\nh q[1];");
        assert!(phase_invariant_distance(&a, &b) < 1e-12);
        assert!(equivalent_up_to_phase(&a, &b, 1e-9));
        assert!(equivalent_exact(&a, &b, 1e-9));
    }

    /// The property that makes this usable for resynthesis: a global phase must not count
    /// as error, because it is unobservable in the surrounding circuit.
    #[test]
    fn global_phase_is_free() {
        let a = u("qreg q[1];\nh q[0];");
        // rz(2pi) == -I, so this is -h: the same operation up to phase.
        let b = u("qreg q[1];\nh q[0];\nrz(6.283185307179586) q[0];");
        assert!(phase_invariant_distance(&a, &b) < 1e-9);
        assert!(hs_distance(&a, &b) < 1e-9);
        assert!(equivalent_up_to_phase(&a, &b, 1e-9));
        // ...but they are not equal on the nose.
        assert!(!equivalent_exact(&a, &b, 1e-9));
    }

    #[test]
    fn different_circuits_have_positive_distance() {
        let a = u("qreg q[1];\nh q[0];");
        let b = u("qreg q[1];\nx q[0];");
        assert!(phase_invariant_distance(&a, &b) > 0.1);
        assert!(hs_distance(&a, &b) > 0.1);
    }

    #[test]
    fn distance_grows_with_the_error() {
        let exact = u("qreg q[1];\nrz(0.5) q[0];");
        let mut last = 0.0;
        for eps in [1e-6, 1e-4, 1e-2, 1e-1] {
            let approx = u(&format!("qreg q[1];\nrz({}) q[0];", 0.5 + eps));
            let d = phase_invariant_distance(&exact, &approx);
            assert!(d > last, "distance should increase with eps={eps}");
            last = d;
        }
    }

    #[test]
    fn small_angle_error_gives_proportional_distance() {
        let exact = u("qreg q[1];\nrz(0.5) q[0];");
        let approx = u("qreg q[1];\nrz(0.500001) q[0];");
        let d = phase_invariant_distance(&exact, &approx);
        // For a 1-qubit rz the phase-invariant Frobenius distance is ~ eps for small eps.
        assert!(d < 1e-5, "got {d}");
        assert!(d > 1e-7, "got {d}");
    }

    #[test]
    fn orthogonal_unitaries_are_maximally_distant() {
        let i = u("qreg q[1];");
        let x = u("qreg q[1];\nx q[0];");
        // tr(I^dagger X) = 0, so hs is 1 and phase-invariant is sqrt(2d) = 2.
        assert!((hs_distance(&i, &x) - 1.0).abs() < 1e-12);
        assert!((phase_invariant_distance(&i, &x) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn distance_is_symmetric() {
        let a = u("qreg q[2];\nh q[0];\ncx q[0], q[1];");
        let b = u("qreg q[2];\nrz(0.3) q[1];\ncx q[1], q[0];");
        assert!(
            (phase_invariant_distance(&a, &b) - phase_invariant_distance(&b, &a)).abs() < 1e-12
        );
        assert!((hs_distance(&a, &b) - hs_distance(&b, &a)).abs() < 1e-12);
    }

    /// The property that justifies accumulating `phase_invariant` across calls: it upper
    /// bounds the operator-norm distance, which is subadditive under composition.
    #[test]
    fn frobenius_upper_bounds_operator_norm() {
        // For each of a range of perturbations, check ||A-B||_F >= ||A-B||_2, estimated
        // by power iteration on (A-B)^dagger (A-B).
        for eps in [1e-3, 1e-2, 1e-1, 0.5, 1.0] {
            let a = u("qreg q[2];\ncx q[0], q[1];\nrz(0.4) q[0];");
            let b = u(&format!(
                "qreg q[2];\ncx q[0], q[1];\nrz({}) q[0];",
                0.4 + eps
            ));
            let frob = phase_invariant_distance(&a, &b);
            let spec = spectral_norm_of_difference(&a, &b);
            assert!(
                frob >= spec - 1e-9,
                "Frobenius {frob} must bound spectral {spec} at eps={eps}"
            );
        }
    }

    /// Estimate `min over phi of ||A - e^{i phi} B||_2` by power iteration, for use as a
    /// cross-check on the closed-form Frobenius bound.
    fn spectral_norm_of_difference(a: &Unitary, b: &Unitary) -> f64 {
        let n = a.dim();
        // Optimal phase alignment: phi = -arg(tr(A^dagger B)).
        let mut tr = C::default();
        for i in 0..n {
            for j in 0..n {
                tr += a.matrix()[[i, j]].conj() * b.matrix()[[i, j]];
            }
        }
        let phase = if tr.norm() > 1e-15 {
            tr / tr.norm()
        } else {
            C::new(1.0, 0.0)
        };
        let m = ndarray::Array2::from_shape_fn((n, n), |(i, j)| {
            a.matrix()[[i, j]] - phase * b.matrix()[[i, j]]
        });
        let mh = m.t().mapv(|z| z.conj());
        let g = mh.dot(&m);
        let mut v = ndarray::Array1::from_elem(n, C::new(1.0, 0.0));
        let mut lambda = 0.0;
        for _ in 0..500 {
            let w = g.dot(&v);
            let norm = w.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if norm < 1e-300 {
                return 0.0;
            }
            v = w.mapv(|z| z / norm);
            lambda = norm;
        }
        lambda.max(0.0).sqrt()
    }

    #[test]
    fn budget_accounting_rejects_when_over() {
        let a = u("qreg q[1];\nrz(0.5) q[0];");
        let b = u("qreg q[1];\nrz(0.6) q[0];");
        let r = DistanceReport::measure(&a, &b);
        assert!(r.phase_invariant > 0.0);
        assert!(r.within(0.0, 1.0));
        assert!(!r.within(0.0, r.phase_invariant / 2.0));
        // Accumulated spend is taken into account.
        assert!(!r.within(1.0 - r.phase_invariant / 2.0, 1.0));
        assert!(r.within(1.0 - r.phase_invariant * 2.0, 1.0));
    }

    /// A budget of exactly the measured distance is accepted; anything above is not.
    #[test]
    fn budget_boundary_is_inclusive() {
        let a = u("qreg q[1];\nrz(0.5) q[0];");
        let b = u("qreg q[1];\nrz(0.6) q[0];");
        let r = DistanceReport::measure(&a, &b);
        assert!(r.within(0.0, r.phase_invariant));
        assert!(!r.within(1e-12, r.phase_invariant));
    }

    /// hs and phase-invariant agree on *whether* two unitaries match, and disagree on
    /// scale — which is exactly why the reference could not meaningfully compare a
    /// BQSKit epsilon with a Synthetiq one.
    #[test]
    fn the_two_metrics_are_not_interchangeable() {
        let a = u("qreg q[2];\ncx q[0], q[1];");
        let b = u("qreg q[2];\ncx q[0], q[1];\nrz(0.2) q[0];");
        let hs = hs_distance(&a, &b);
        let pi_ = phase_invariant_distance(&a, &b);
        assert!(hs > 0.0 && pi_ > 0.0);
        // Same zero set...
        assert_eq!(hs < 1e-12, pi_ < 1e-12);
        // ...but different magnitudes, by more than rounding.
        assert!((hs - pi_).abs() > 0.05, "hs={hs} pi={pi_}");
    }

    #[test]
    fn report_fields_are_consistent() {
        let a = u("qreg q[2];\nh q[0];");
        let b = u("qreg q[2];\nh q[0];\nrz(0.1) q[1];");
        let r = DistanceReport::measure(&a, &b);
        assert!((r.hs - hs_distance(&a, &b)).abs() < 1e-15);
        assert!((r.phase_invariant - phase_invariant_distance(&a, &b)).abs() < 1e-15);
        assert!((0.0..=1.0).contains(&r.normalized_overlap));
    }

    #[test]
    fn identity_against_itself_over_many_qubits() {
        for n in 1..=6 {
            let src = format!("qreg q[{n}];");
            let a = u(&src);
            let b = u(&src);
            assert!(phase_invariant_distance(&a, &b) < 1e-12, "n={n}");
        }
    }

    #[test]
    fn t_gate_approximation_error_is_measurable() {
        // A deliberately poor approximation of t by rz.
        let exact = u("qreg q[1];\nt q[0];");
        let good = u(&format!("qreg q[1];\nrz({}) q[0];", PI / 4.0));
        let poor = u(&format!("qreg q[1];\nrz({}) q[0];", PI / 4.0 + 0.05));
        assert!(phase_invariant_distance(&exact, &good) < 1e-9);
        assert!(phase_invariant_distance(&exact, &poor) > 1e-3);
    }

    /// The closed form `sqrt(2 * (d - |tr|))` cannot resolve errors near GUOQ's default
    /// budget of 1e-8, because `d - |tr|` is then below the rounding error of `|tr|`
    /// itself. Depending on which way the last bit falls it reports either an exact zero
    /// (hiding real error) or a spurious `~4e-8` (rejecting an exact result). The
    /// elementwise computation reports the true value.
    #[test]
    fn distance_resolves_errors_at_the_default_budget() {
        let closed_form = |a: &Unitary, b: &Unitary| {
            let d = a.dim() as f64;
            (2.0 * (d - overlap(a, b))).max(0.0).sqrt()
        };

        // A deliberate error of exactly 1e-8 on a two-qubit block.
        let a = u("qreg q[2];\ncz q[0], q[1];");
        let c = u("qreg q[2];\ncz q[0], q[1];\nrz(1e-8) q[0];");

        let measured = phase_invariant_distance(&a, &c);
        assert!(
            (measured - 1e-8).abs() < 1e-10,
            "elementwise should measure ~1e-8, got {measured:.3e}"
        );

        // The closed form is out by at least an order of magnitude, either way.
        let naive = closed_form(&a, &c);
        assert!(
            !(1e-9..=1e-7).contains(&naive),
            "closed form should be unusable here, got {naive:.3e}"
        );

        // On genuinely identical unitaries the elementwise form is exact.
        let b = u("qreg q[2];\nh q[1];\ncx q[0], q[1];\nh q[1];");
        assert!(phase_invariant_distance(&a, &b) < 1e-14);
    }

    /// The measured distance tracks the injected error across many orders of magnitude.
    #[test]
    fn distance_tracks_injected_error_across_scales() {
        let base = u("qreg q[2];\ncz q[0], q[1];");
        for eps in [1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2] {
            let perturbed = u(&format!("qreg q[2];\ncz q[0], q[1];\nrz({eps}) q[0];"));
            let got = phase_invariant_distance(&base, &perturbed);
            let rel = (got - eps).abs() / eps;
            assert!(
                rel < 1e-3,
                "eps={eps:.0e} measured {got:.3e} (rel {rel:.2e})"
            );
        }
    }

    /// Larger blocks make the closed form worse, since the floor grows with `sqrt(d)`.
    #[test]
    fn precision_holds_for_three_qubit_blocks() {
        let a = u("qreg q[3];\nccz q[0], q[1], q[2];");
        let b = u("qreg q[3];\nh q[2];\nccx q[0], q[1], q[2];\nh q[2];");
        assert!(phase_invariant_distance(&a, &b) < 1e-12);
    }

    /// The aligning phase must be the conjugate of the trace's phase; getting this
    /// backwards makes gates that differ only by global phase look maximally distant.
    #[test]
    fn phase_alignment_uses_the_conjugate() {
        // t and rz(pi/4) differ by exactly the global phase e^{i pi / 8}.
        let a = u("qreg q[1];\nt q[0];");
        let b = u(&format!("qreg q[1];\nrz({}) q[0];", PI / 4.0));
        assert!(phase_invariant_distance(&a, &b) < 1e-15);
        // s and rz(pi/2) likewise.
        let c = u("qreg q[1];\ns q[0];");
        let d = u(&format!("qreg q[1];\nrz({}) q[0];", PI / 2.0));
        assert!(phase_invariant_distance(&c, &d) < 1e-15);
    }

    #[test]
    fn normalized_overlap_stays_accurate_near_one() {
        let a = u("qreg q[2];\ncz q[0], q[1];");
        let b = u("qreg q[2];\nh q[1];\ncx q[0], q[1];\nh q[1];");
        assert!((normalized_overlap(&a, &b) - 1.0).abs() < 1e-14);
    }

    #[test]
    #[should_panic(expected = "equal dimension")]
    fn mismatched_dimensions_panic() {
        let a = u("qreg q[1];");
        let b = u("qreg q[2];");
        overlap(&a, &b);
    }
}
