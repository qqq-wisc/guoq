//! Fingerprinting a circuit by what it computes.
//!
//! Rule synthesis works by enumerating small circuits, grouping the ones that compute the
//! same thing, and emitting a rewrite rule for each pair in a group. Grouping needs a
//! cheap key that agrees exactly when two circuits agree.
//!
//! The reference computed this from path sums (`Verifier.hashCode`), evaluating the
//! symbolic amplitude at randomly chosen values for `pi`-free symbols. This does the same
//! thing through the unitary: bind the free angle variables to fixed random values, build
//! the matrix, and quantize it. Two circuits that agree as functions of their angles
//! agree at the sample point; two that differ agree only by coincidence, and the
//! coincidence is ruled out by exact verification ([`crate::exact`]) before any rule is
//! emitted — which is what leaves this fingerprint free to be fast rather than exact.

use ndarray::Array2;
use num_complex::Complex64;
use rustc_hash::FxHashMap;

use qcircuit::{Dag, GateRegistry};
use qsemantics::Unitary;

/// How finely matrix entries are quantized before hashing.
///
/// Coarse enough to absorb floating-point noise between two spellings of the same
/// circuit, fine enough that genuinely different circuits separate.
const QUANTUM: f64 = 1e-9;

/// A 128-bit summary of what a circuit computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint(pub u128);

/// Fixed random values for the free angle variables.
///
/// Sampling once and reusing the same point everywhere is what makes fingerprints
/// comparable. The values are derived from a seed so a synthesis run reproduces.
#[derive(Debug, Clone)]
pub struct AngleSample {
    values: FxHashMap<String, f64>,
    seed: u64,
    next: u64,
}

impl AngleSample {
    pub fn new(seed: u64) -> Self {
        Self {
            values: FxHashMap::default(),
            seed,
            next: seed,
        }
    }

    /// The value bound to `name`, assigning one on first use.
    ///
    /// Values are drawn from `(0, 2)` rather than `(0, 2*pi)`: irrational-looking angles
    /// keep unrelated circuits from colliding, and staying away from multiples of `pi/4`
    /// avoids accidental Clifford coincidences that would merge classes that are not
    /// actually equal for all angles.
    pub fn get(&mut self, name: &str) -> f64 {
        if let Some(v) = self.values.get(name) {
            return *v;
        }
        let v = 0.1 + 1.8 * next_unit(&mut self.next);
        self.values.insert(name.to_string(), v);
        v
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// A second, independent sample point, for confirming a candidate equivalence.
    pub fn resample(&self, nonce: u64) -> Self {
        Self::new(
            self.seed
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(nonce),
        )
    }

    pub fn bindings(&self) -> &FxHashMap<String, f64> {
        &self.values
    }
}

fn next_unit(state: &mut u64) -> f64 {
    // splitmix64
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Build a circuit's unitary at the sample point.
pub fn evaluate(
    dag: &Dag,
    qubits: &[qcircuit::QubitId],
    registry: &GateRegistry,
    sample: &mut AngleSample,
) -> Option<Unitary> {
    // Bind every free variable first, so the closure can borrow immutably.
    for idx in dag.gate_indices() {
        for p in &dag.gate(idx).params {
            for v in p.free_vars() {
                sample.get(v);
            }
        }
    }
    let bindings = sample.bindings().clone();
    let env = move |name: &str| bindings.get(name).copied();
    Unitary::from_dag_over_with_env(dag, qubits.to_vec(), registry, &env).ok()
}

/// Quantize a matrix entry, normalizing `-0.0` to `0.0` so both spellings agree.
fn quantize(part: f64) -> i64 {
    let q = (part / QUANTUM).round();
    if q == 0.0 {
        0
    } else {
        q as i64
    }
}

/// Fingerprint a matrix.
///
/// Two independent FNV-1a streams over `(position, real, imaginary)`, combined into 128
/// bits. Position is fed in explicitly: without it, matrices that are permutations of one
/// another hash alike, which merged `h q1; cx q0, q1;` with `cx q0, q1; h q1;` — circuits
/// that are emphatically not equal — when this was a single stream over values alone.
pub fn fingerprint_matrix(m: &Array2<Complex64>) -> Fingerprint {
    const OFFSET_A: u64 = 0xcbf2_9ce4_8422_2325;
    const OFFSET_B: u64 = 0x9e37_79b9_7f4a_7c15;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut a = OFFSET_A;
    let mut b = OFFSET_B;
    for (index, z) in m.iter().enumerate() {
        for (part_index, part) in [z.re, z.im].into_iter().enumerate() {
            let position = (index as u64) << 1 | part_index as u64;
            for word in [position, quantize(part) as u64] {
                for byte in word.to_le_bytes() {
                    a ^= byte as u64;
                    a = a.wrapping_mul(PRIME);
                    b = b.rotate_left(13) ^ (byte as u64);
                    b = b.wrapping_mul(PRIME).wrapping_add(0x2545_F491_4F6C_DD1D);
                }
            }
        }
    }
    Fingerprint(((a as u128) << 64) | b as u128)
}

/// Fingerprint a matrix up to global phase.
///
/// Symbolic pairing is where the distinction earns its keep: `tdg . P` and
/// `t . P . s` differ by a global phase of `i` for the bit-flip permutation — an
/// identity every consumer of a rule accepts, because matching, verification, and
/// distance are all phase-invariant, but one that exact-phase bucketing can never
/// propose, since the two sides' raw matrices never share a bucket. Nearly half the
/// reference's cliffordt symbolic corpus is of this shape.
///
/// The phase is fixed by rotating a pivot entry onto the positive real axis; a
/// unitary's columns are unit vectors, so the largest entry is at least `1/sqrt(d)`
/// and the rotation is well conditioned. The pivot is the *first* entry within
/// tolerance of the largest, not the largest itself: a diagonal unitary has every
/// entry at magnitude one, and breaking that tie by exact magnitude let floating-point
/// noise pick *different* pivots for two matrices equal up to phase — which split the
/// very buckets this fingerprint exists to merge, and silently cost every parametric
/// (`rz`-carrying) pair. Quantization at a grid boundary can still split a bucket,
/// which costs a candidate pair, never a wrong rule — every pair is verified before it
/// is emitted.
pub fn fingerprint_matrix_up_to_phase(m: &Array2<Complex64>) -> Fingerprint {
    let max = m.iter().map(|z| z.norm_sqr()).fold(0.0f64, f64::max);
    if max < 1e-18 {
        return fingerprint_matrix(m);
    }
    let pivot = m
        .iter()
        .copied()
        .find(|z| z.norm_sqr() >= max - 1e-9)
        .expect("max came from these entries");
    let rot = pivot.conj() / pivot.norm_sqr().sqrt();
    fingerprint_matrix(&m.map(|z| z * rot))
}

/// Fingerprint a circuit.
pub fn fingerprint(
    dag: &Dag,
    qubits: &[qcircuit::QubitId],
    registry: &GateRegistry,
    sample: &mut AngleSample,
) -> Option<Fingerprint> {
    evaluate(dag, qubits, registry, sample).map(|u| fingerprint_matrix(u.matrix()))
}

/// How two circuits are compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Equivalence {
    /// Equal including global phase, which is what the reference's verifier required.
    Exact,
    /// Equal up to a global phase, which is unobservable and so still safe to rewrite by.
    ///
    /// Admits strictly more rules than [`Equivalence::Exact`].
    UpToPhase,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn qubits(n: usize) -> Vec<qcircuit::QubitId> {
        (0..n).map(|i| qcircuit::intern(&format!("q{i}"))).collect()
    }

    fn fp(src: &str, n: usize, seed: u64) -> Fingerprint {
        let mut s = AngleSample::new(seed);
        fingerprint(&qasm::parse(src).unwrap(), &qubits(n), &reg(), &mut s).unwrap()
    }

    #[test]
    fn identical_circuits_fingerprint_alike() {
        assert_eq!(fp("h q0; cx q0, q1;", 2, 7), fp("h q0; cx q0, q1;", 2, 7));
    }

    #[test]
    fn equivalent_circuits_fingerprint_alike() {
        // h h == identity
        assert_eq!(fp("h q0; h q0;", 2, 7), fp("", 2, 7));
        // cz == h(1) cx h(1)
        assert_eq!(fp("cz q0, q1;", 2, 7), fp("h q1; cx q0, q1; h q1;", 2, 7));
        // t t == s
        assert_eq!(fp("t q0; t q0;", 1, 7), fp("s q0;", 1, 7));
    }

    #[test]
    fn different_circuits_fingerprint_differently() {
        assert_ne!(fp("h q0;", 1, 7), fp("x q0;", 1, 7));
        assert_ne!(fp("cx q0, q1;", 2, 7), fp("cx q1, q0;", 2, 7));
        assert_ne!(fp("t q0;", 1, 7), fp("tdg q0;", 1, 7));
    }

    #[test]
    fn symbolic_angles_are_bound_consistently() {
        // The same variable must take the same value throughout a circuit, so a rule that
        // is true for all angles fingerprints alike.
        assert_eq!(
            fp("rz(theta1) q0; cx q0, q1;", 2, 7),
            fp("cx q0, q1; rz(theta1) q0;", 2, 7)
        );
        // ...and one that is not, does not.
        assert_ne!(
            fp("rz(theta1) q1; cx q0, q1;", 2, 7),
            fp("cx q0, q1; rz(theta1) q1;", 2, 7)
        );
    }

    #[test]
    fn distinct_variables_get_distinct_values() {
        let mut s = AngleSample::new(3);
        let a = s.get("theta1");
        let b = s.get("theta2");
        assert_ne!(a, b);
        // ...and the same variable keeps its value.
        assert_eq!(s.get("theta1"), a);
    }

    #[test]
    fn sample_values_avoid_clifford_angles() {
        use std::f64::consts::PI;
        let mut s = AngleSample::new(11);
        for i in 0..50 {
            let v = s.get(&format!("theta{i}"));
            assert!((0.0..2.0).contains(&v));
            // Not near a multiple of pi/4, which would make an rz accidentally Clifford
            // and merge classes that are not equal for all angles.
            let k = (v / (PI / 4.0)).round();
            assert!(
                (v - k * PI / 4.0).abs() > 1e-3,
                "sample {v} is too close to {k} * pi/4"
            );
        }
    }

    #[test]
    fn sampling_is_reproducible() {
        let a: Vec<f64> = {
            let mut s = AngleSample::new(42);
            (0..5).map(|i| s.get(&format!("t{i}"))).collect()
        };
        let b: Vec<f64> = {
            let mut s = AngleSample::new(42);
            (0..5).map(|i| s.get(&format!("t{i}"))).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn resampling_gives_a_different_point() {
        let base = AngleSample::new(5);
        let mut a = base.resample(1);
        let mut b = base.resample(2);
        assert_ne!(a.get("theta1"), b.get("theta1"));
    }

    #[test]
    fn fingerprints_absorb_floating_point_noise() {
        use std::f64::consts::PI;
        let a = fp(&format!("rz({}) q0;", PI / 4.0), 1, 1);
        let noisy = f64::from_bits((PI / 4.0).to_bits() + 1);
        let b = fp(&format!("rz({noisy}) q0;"), 1, 1);
        assert_eq!(a, b);
    }

    /// Position must be part of the hash: without it, a matrix and a rearrangement of it
    /// collide. This is the collision that merged two unequal circuits into one class.
    #[test]
    fn position_is_part_of_the_fingerprint() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::default();
        let a = ndarray::arr2(&[[one, zero], [zero, one]]);
        let b = ndarray::arr2(&[[zero, one], [one, zero]]);
        assert_ne!(fingerprint_matrix(&a), fingerprint_matrix(&b));

        // The specific pair that collided.
        assert_ne!(fp("h q1; cx q0, q1;", 2, 7), fp("cx q0, q1; h q1;", 2, 7));
    }

    /// Every distinct 2-qubit circuit of a small family must get a distinct fingerprint.
    #[test]
    fn small_circuits_do_not_collide() {
        use std::collections::HashMap;
        let mut by_fp: HashMap<Fingerprint, Vec<&str>> = HashMap::new();
        let circuits = [
            "",
            "h q0;",
            "h q1;",
            "x q0;",
            "x q1;",
            "cx q0, q1;",
            "cx q1, q0;",
            "h q0; cx q0, q1;",
            "cx q0, q1; h q0;",
            "h q1; cx q0, q1;",
            "cx q0, q1; h q1;",
            "h q0; h q1;",
            "x q0; cx q0, q1;",
            "cx q0, q1; x q0;",
            "cz q0, q1;",
            "t q0;",
            "tdg q0;",
            "s q0;",
        ];
        for c in circuits {
            by_fp.entry(fp(c, 2, 13)).or_default().push(c);
        }
        // The only expected collision is cz == h(1) cx h(1), which is not in the list.
        for (_, group) in by_fp.iter() {
            assert_eq!(
                group.len(),
                1,
                "these distinct circuits share a fingerprint: {group:?}"
            );
        }
    }

    #[test]
    fn negative_zero_and_zero_agree() {
        let m1 = ndarray::arr2(&[[Complex64::new(0.0, 0.0)]]);
        let m2 = ndarray::arr2(&[[Complex64::new(-0.0, -0.0)]]);
        assert_eq!(fingerprint_matrix(&m1), fingerprint_matrix(&m2));
    }

    #[test]
    fn a_circuit_over_more_qubits_than_it_uses() {
        // Padding with idle wires must not change whether two circuits agree.
        assert_eq!(fp("h q0; h q0;", 3, 4), fp("", 3, 4));
        assert_ne!(fp("h q0;", 3, 4), fp("h q1;", 3, 4));
    }
}
