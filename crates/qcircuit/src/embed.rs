//! Embedding a small gate matrix into a larger register.

use ndarray::Array2;
use num_complex::Complex64;

type C = Complex64;

/// Extract the bit of `index` belonging to qubit `q` of an `n`-qubit register.
///
/// Uses the big-endian convention documented on [`GateDef::matrix`]: qubit 0 is the most
/// significant bit.
///
/// [`GateDef::matrix`]: crate::gate::GateDef::matrix
#[inline]
pub fn bit_of(index: usize, q: usize, n: usize) -> usize {
    (index >> (n - 1 - q)) & 1
}

/// Set the bit of `index` belonging to qubit `q` of an `n`-qubit register.
#[inline]
pub fn with_bit(index: usize, q: usize, n: usize, value: usize) -> usize {
    let mask = 1usize << (n - 1 - q);
    if value == 1 {
        index | mask
    } else {
        index & !mask
    }
}

/// Lift `sub`, a `2^k x 2^k` matrix, onto `operands` within an `n`-qubit register.
///
/// `operands[t]` is the register qubit that plays the role of the sub-matrix's operand
/// `t`. Operands need not be contiguous or ordered, so this also handles the reversed
/// case (`cx q1, q0` versus `cx q0, q1`) with no special casing.
///
/// # Panics
///
/// Panics if `operands` is not a set of distinct indices below `n`, or if `sub` is not
/// square with side `2^operands.len()`.
pub fn embed(sub: &Array2<C>, operands: &[usize], n: usize) -> Array2<C> {
    let k = operands.len();
    let side = 1usize << k;
    assert_eq!(sub.shape(), &[side, side], "sub-matrix shape mismatch");
    assert!(
        operands.iter().all(|&q| q < n),
        "operand out of range for {n}-qubit register"
    );
    for i in 0..k {
        for j in (i + 1)..k {
            assert_ne!(operands[i], operands[j], "duplicate operand");
        }
    }

    let dim = 1usize << n;
    let mut out = Array2::<C>::zeros((dim, dim));

    // Register indices that are zero on every operand qubit; each is the base of one
    // independent sub-block.
    let mask: usize = operands.iter().fold(0, |m, &q| m | (1 << (n - 1 - q)));
    for base in 0..dim {
        if base & mask != 0 {
            continue;
        }
        for si in 0..side {
            let mut ri = base;
            for (t, &q) in operands.iter().enumerate() {
                ri = with_bit(ri, q, n, (si >> (k - 1 - t)) & 1);
            }
            for sj in 0..side {
                let v = sub[[si, sj]];
                if v == C::default() {
                    continue;
                }
                let mut rj = base;
                for (t, &q) in operands.iter().enumerate() {
                    rj = with_bit(rj, q, n, (sj >> (k - 1 - t)) & 1);
                }
                out[[ri, rj]] = v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateRegistry;
    use ndarray::array;

    fn approx_eq(a: &Array2<C>, b: &Array2<C>) -> bool {
        a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() < 1e-12)
    }

    fn eye(n: usize) -> Array2<C> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                C::new(1.0, 0.0)
            } else {
                C::default()
            }
        })
    }

    #[test]
    fn bit_indexing_is_big_endian() {
        // 2-qubit register, index 0b10 => qubit 0 is 1, qubit 1 is 0.
        assert_eq!(bit_of(0b10, 0, 2), 1);
        assert_eq!(bit_of(0b10, 1, 2), 0);
        assert_eq!(with_bit(0b00, 0, 2, 1), 0b10);
        assert_eq!(with_bit(0b00, 1, 2, 1), 0b01);
    }

    #[test]
    fn embedding_identity_is_identity() {
        let reg = GateRegistry::with_builtins();
        let h = reg.get("h").unwrap().matrix(&[], &reg).unwrap();
        let e = embed(&h, &[0], 1);
        assert!(approx_eq(&e, &h));
    }

    #[test]
    fn embed_single_qubit_into_two() {
        let reg = GateRegistry::with_builtins();
        let x = reg.get("x").unwrap().matrix(&[], &reg).unwrap();
        // X on qubit 0 of 2 == X (kron) I
        let got = embed(&x, &[0], 2);
        let want = array![
            [C::default(), C::default(), C::new(1.0, 0.0), C::default()],
            [C::default(), C::default(), C::default(), C::new(1.0, 0.0)],
            [C::new(1.0, 0.0), C::default(), C::default(), C::default()],
            [C::default(), C::new(1.0, 0.0), C::default(), C::default()]
        ];
        assert!(approx_eq(&got, &want));
    }

    #[test]
    fn reversed_operands_swap_control_and_target() {
        let reg = GateRegistry::with_builtins();
        let cx = reg.get("cx").unwrap().matrix(&[], &reg).unwrap();
        let forward = embed(&cx, &[0, 1], 2);
        let reversed = embed(&cx, &[1, 0], 2);
        assert!(approx_eq(&forward, &cx));
        assert!(!approx_eq(&forward, &reversed));
        // Reversed cx maps |01> -> |11>.
        assert!((reversed[[0b11, 0b01]] - C::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn embedding_preserves_unitarity_for_all_operand_orders() {
        let reg = GateRegistry::with_builtins();
        let cx = reg.get("cx").unwrap().matrix(&[], &reg).unwrap();
        for &(a, b) in &[(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)] {
            let m = embed(&cx, &[a, b], 3);
            let adj = m.t().mapv(|z| z.conj());
            assert!(approx_eq(&adj.dot(&m), &eye(8)), "operands {a},{b}");
        }
    }

    #[test]
    fn non_contiguous_operands() {
        let reg = GateRegistry::with_builtins();
        let cx = reg.get("cx").unwrap().matrix(&[], &reg).unwrap();
        // control = qubit 0, target = qubit 3, in a 4-qubit register.
        let m = embed(&cx, &[0, 3], 4);
        // |1000> -> |1001>
        assert!((m[[0b1001, 0b1000]] - C::new(1.0, 0.0)).norm() < 1e-12);
        // |0000> unchanged
        assert!((m[[0b0000, 0b0000]] - C::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn three_qubit_gate_embeds() {
        let reg = GateRegistry::with_builtins();
        let ccz = reg.get("ccz").unwrap().matrix(&[], &reg).unwrap();
        let m = embed(&ccz, &[2, 0, 3], 4);
        let adj = m.t().mapv(|z| z.conj());
        assert!(approx_eq(&adj.dot(&m), &eye(16)));
        // -1 exactly when qubits 2, 0 and 3 are all 1.
        for i in 0..16usize {
            let all = bit_of(i, 2, 4) & bit_of(i, 0, 4) & bit_of(i, 3, 4);
            let want = C::new(if all == 1 { -1.0 } else { 1.0 }, 0.0);
            assert!((m[[i, i]] - want).norm() < 1e-12, "diag {i}");
        }
    }

    #[test]
    #[should_panic(expected = "duplicate operand")]
    fn duplicate_operands_panic() {
        let reg = GateRegistry::with_builtins();
        let cx = reg.get("cx").unwrap().matrix(&[], &reg).unwrap();
        embed(&cx, &[1, 1], 2);
    }
}
