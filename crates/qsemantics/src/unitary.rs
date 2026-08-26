//! Building a circuit's unitary matrix.

use ndarray::Array2;
use num_complex::Complex64;
use qcircuit::embed::with_bit;
use qcircuit::{CircuitError, Dag, GateRegistry, QubitId};

/// Resolves symbolic gate parameters to numeric values.
pub type ParamEnv<'a> = &'a dyn Fn(&str) -> Option<f64>;

type C = Complex64;

/// Default cap on qubit count when materialising a unitary.
///
/// A `k`-qubit unitary is `2^k x 2^k` complex numbers: 12 qubits is 4096x4096, or 256 MB.
/// Resynthesis partitions are far smaller than this (3 qubits by default), so the cap
/// exists to turn a mistake into a clear error rather than an allocation failure.
pub const DEFAULT_MAX_QUBITS: usize = 12;

/// A unitary matrix over an explicit qubit ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct Unitary {
    matrix: Array2<C>,
    qubits: Vec<QubitId>,
}

impl Unitary {
    /// The identity over `qubits`.
    pub fn identity(qubits: Vec<QubitId>) -> Self {
        let dim = 1usize << qubits.len();
        let matrix = Array2::from_shape_fn((dim, dim), |(i, j)| {
            if i == j {
                C::new(1.0, 0.0)
            } else {
                C::default()
            }
        });
        Self { matrix, qubits }
    }

    /// Build the unitary of `dag` over its own qubits, in declaration order.
    pub fn from_dag(dag: &Dag, registry: &GateRegistry) -> Result<Self, CircuitError> {
        Self::from_dag_over(dag, dag.qubits().to_vec(), registry)
    }

    /// Build the unitary of `dag` over an explicit qubit ordering.
    ///
    /// `qubits` may be a superset of the circuit's own qubits; the extra wires are left
    /// untouched. This is what lets two circuits over different qubit subsets — the two
    /// sides of a rewrite rule, say — be compared in a common space.
    pub fn from_dag_over(
        dag: &Dag,
        qubits: Vec<QubitId>,
        registry: &GateRegistry,
    ) -> Result<Self, CircuitError> {
        Self::from_dag_over_limited(dag, qubits, registry, DEFAULT_MAX_QUBITS)
    }

    /// As [`from_dag_over`], resolving symbolic gate parameters through `env`.
    ///
    /// Rewrite rules carry symbolic angles (`theta1`, `(theta1+theta2)`). Binding them to
    /// concrete values lets both sides of a rule be built and compared.
    ///
    /// [`from_dag_over`]: Unitary::from_dag_over
    pub fn from_dag_over_with_env(
        dag: &Dag,
        qubits: Vec<QubitId>,
        registry: &GateRegistry,
        env: ParamEnv<'_>,
    ) -> Result<Self, CircuitError> {
        Self::build(dag, qubits, registry, DEFAULT_MAX_QUBITS, Some(env))
    }

    /// As [`from_dag_over`], with an explicit qubit-count cap.
    ///
    /// [`from_dag_over`]: Unitary::from_dag_over
    pub fn from_dag_over_limited(
        dag: &Dag,
        qubits: Vec<QubitId>,
        registry: &GateRegistry,
        max_qubits: usize,
    ) -> Result<Self, CircuitError> {
        Self::build(dag, qubits, registry, max_qubits, None)
    }

    fn build(
        dag: &Dag,
        qubits: Vec<QubitId>,
        registry: &GateRegistry,
        max_qubits: usize,
        env: Option<ParamEnv<'_>>,
    ) -> Result<Self, CircuitError> {
        if qubits.len() > max_qubits {
            return Err(CircuitError::TooManyQubits {
                qubits: qubits.len(),
                limit: max_qubits,
            });
        }
        let mut u = Self::identity(qubits);
        for idx in dag.topological_gates() {
            let op = dag.gate(idx);
            let def = registry.get(&op.gate)?;
            let params: Vec<f64> = op
                .params
                .iter()
                .map(|p| {
                    match env {
                        Some(e) => p.eval_with(e),
                        None => p.eval(),
                    }
                    .ok_or_else(|| CircuitError::UnresolvedParam {
                        gate: op.gate.to_string(),
                        expr: p.to_string(),
                    })
                })
                .collect::<Result<_, _>>()?;
            let small = def.matrix(&params, registry)?;
            let operands: Vec<usize> =
                op.qubits
                    .iter()
                    .map(|q| {
                        u.qubits.iter().position(|x| x == q).ok_or_else(|| {
                            CircuitError::Other(format!("qubit `{q}` not in ordering"))
                        })
                    })
                    .collect::<Result<_, _>>()?;
            u.apply(&small, &operands);
        }
        Ok(u)
    }

    /// Left-multiply by a `k`-qubit gate acting on `operands`.
    ///
    /// Applied in place, block by block, rather than by materialising the full embedded
    /// matrix and doing a dense multiply. For an `n`-qubit register and a `k`-qubit gate
    /// this is `O(4^n * 2^k)` instead of `O(8^n)`.
    pub fn apply(&mut self, gate: &Array2<C>, operands: &[usize]) {
        let n = self.qubits.len();
        let k = operands.len();
        let side = 1usize << k;
        debug_assert_eq!(gate.shape(), &[side, side]);
        let dim = 1usize << n;
        let mask: usize = operands.iter().fold(0, |m, &q| m | (1 << (n - 1 - q)));

        let mut buf = vec![C::default(); side];
        for base in 0..dim {
            if base & mask != 0 {
                continue;
            }
            // Row indices of this block, in sub-matrix order.
            let rows: Vec<usize> = (0..side)
                .map(|si| {
                    let mut r = base;
                    for (t, &q) in operands.iter().enumerate() {
                        r = with_bit(r, q, n, (si >> (k - 1 - t)) & 1);
                    }
                    r
                })
                .collect();
            for col in 0..dim {
                for (si, slot) in buf.iter_mut().enumerate() {
                    let mut acc = C::default();
                    for (sj, &rj) in rows.iter().enumerate() {
                        let g = gate[[si, sj]];
                        if g != C::default() {
                            acc += g * self.matrix[[rj, col]];
                        }
                    }
                    *slot = acc;
                }
                for (si, &ri) in rows.iter().enumerate() {
                    self.matrix[[ri, col]] = buf[si];
                }
            }
        }
    }

    pub fn matrix(&self) -> &Array2<C> {
        &self.matrix
    }

    pub fn qubits(&self) -> &[QubitId] {
        &self.qubits
    }

    pub fn dim(&self) -> usize {
        self.matrix.shape()[0]
    }

    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// The conjugate transpose.
    pub fn adjoint(&self) -> Self {
        Self {
            matrix: self.matrix.t().mapv(|z| z.conj()),
            qubits: self.qubits.clone(),
        }
    }

    /// `true` if the matrix really is unitary, within `tol`.
    ///
    /// Used as a self-check: a circuit whose unitary is not unitary means a gate
    /// definition is wrong.
    pub fn is_unitary(&self, tol: f64) -> bool {
        let prod = self.adjoint().matrix.dot(&self.matrix);
        let n = self.dim();
        (0..n).all(|i| {
            (0..n).all(|j| {
                let want = if i == j {
                    C::new(1.0, 0.0)
                } else {
                    C::default()
                };
                (prod[[i, j]] - want).norm() <= tol
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;
    use std::f64::consts::FRAC_1_SQRT_2;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn u(src: &str) -> Unitary {
        let dag = qasm::parse(src).unwrap();
        Unitary::from_dag(&dag, &reg()).unwrap()
    }

    fn approx(a: &Unitary, b: &Unitary) -> bool {
        a.dim() == b.dim()
            && a.matrix()
                .iter()
                .zip(b.matrix().iter())
                .all(|(x, y)| (x - y).norm() < 1e-12)
    }

    #[test]
    fn empty_circuit_is_identity() {
        let x = u("qreg q[3];");
        assert_eq!(x.dim(), 8);
        assert!(approx(&x, &Unitary::identity(x.qubits().to_vec())));
    }

    #[test]
    fn single_gate_matches_its_definition() {
        let x = u("qreg q[1];\nh q[0];");
        let want = reg().get("h").unwrap().matrix(&[], &reg()).unwrap();
        assert!(x
            .matrix()
            .iter()
            .zip(want.iter())
            .all(|(a, b)| (a - b).norm() < 1e-12));
    }

    #[test]
    fn gates_compose_in_circuit_order() {
        // h then x, so the matrix is X * H (later gates multiply on the left).
        let got = u("qreg q[1];\nh q[0];\nx q[0];");
        let r = reg();
        let h = r.get("h").unwrap().matrix(&[], &r).unwrap();
        let xm = r.get("x").unwrap().matrix(&[], &r).unwrap();
        let want = xm.dot(&h);
        assert!(got
            .matrix()
            .iter()
            .zip(want.iter())
            .all(|(a, b)| (a - b).norm() < 1e-12));
    }

    #[test]
    fn everything_built_is_unitary() {
        for src in [
            "qreg q[1];\nh q[0];",
            "qreg q[2];\ncx q[0], q[1];\nh q[0];\nrz(0.3) q[1];",
            "qreg q[3];\nccz q[0], q[1], q[2];\nh q[1];",
            "qreg q[4];\nrxx(1.1) q[0], q[3];\nu3(0.1,0.2,0.3) q[2];",
        ] {
            assert!(u(src).is_unitary(1e-10), "not unitary: {src}");
        }
    }

    #[test]
    fn bell_state_preparation() {
        let b = u("qreg q[2];\nh q[0];\ncx q[0], q[1];");
        let s = FRAC_1_SQRT_2;
        // First column: (|00> + |11>)/sqrt(2).
        assert!((b.matrix()[[0b00, 0]].re - s).abs() < 1e-12);
        assert!((b.matrix()[[0b11, 0]].re - s).abs() < 1e-12);
        assert!(b.matrix()[[0b01, 0]].norm() < 1e-12);
        assert!(b.matrix()[[0b10, 0]].norm() < 1e-12);
    }

    #[test]
    fn qubit_order_is_declaration_order() {
        let a = u("qreg q[2];\nx q[0];");
        // X on qubit 0 maps |00> -> |10>.
        assert!((a.matrix()[[0b10, 0b00]] - C::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn extra_qubits_in_the_ordering_are_idle() {
        let dag = qasm::parse("h q0;").unwrap();
        let wide =
            Unitary::from_dag_over(&dag, vec!["q0".into(), "q1".into(), "q2".into()], &reg())
                .unwrap();
        assert_eq!(wide.dim(), 8);
        assert!(wide.is_unitary(1e-12));
        // The h acts only on the first wire.
        let narrow = u("h q0;");
        for i in 0..2 {
            for j in 0..2 {
                let a = narrow.matrix()[[i, j]];
                let b = wide.matrix()[[i * 4, j * 4]];
                assert!((a - b).norm() < 1e-12, "at {i},{j}");
            }
        }
    }

    #[test]
    fn reordering_qubits_permutes_the_matrix() {
        let dag = qasm::parse("cx a, b;").unwrap();
        let ab = Unitary::from_dag_over(&dag, vec!["a".into(), "b".into()], &reg()).unwrap();
        let ba = Unitary::from_dag_over(&dag, vec!["b".into(), "a".into()], &reg()).unwrap();
        assert!(!approx(&ab, &ba));
        // Under the reversed ordering, cx a,b maps |01> (b=0,a=1) to |11>.
        assert!((ba.matrix()[[0b11, 0b01]] - C::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn known_identities_hold() {
        // h x h == z
        let a = u("qreg q[1];\nh q[0];\nx q[0];\nh q[0];");
        let b = u("qreg q[1];\nz q[0];");
        assert!(approx(&a, &b));

        // cx a,b; cx a,b == identity
        let c = u("qreg q[2];\ncx q[0], q[1];\ncx q[0], q[1];");
        assert!(approx(&c, &Unitary::identity(c.qubits().to_vec())));

        // h(1) cx(0,1) h(1) == cz(0,1)
        let d = u("qreg q[2];\nh q[1];\ncx q[0], q[1];\nh q[1];");
        let e = u("qreg q[2];\ncz q[0], q[1];");
        assert!(approx(&d, &e));

        // t t == s
        let f = u("qreg q[1];\nt q[0];\nt q[0];");
        let g = u("qreg q[1];\ns q[0];");
        assert!(approx(&f, &g));
    }

    #[test]
    fn rz_accumulates() {
        let a = u("qreg q[1];\nrz(0.3) q[0];\nrz(0.4) q[0];");
        let b = u("qreg q[1];\nrz(0.7) q[0];");
        assert!(approx(&a, &b));
    }

    #[test]
    fn swap_is_two_cx_sandwich() {
        let a = u("qreg q[2];\ncx q[0], q[1];\ncx q[1], q[0];\ncx q[0], q[1];");
        let b = u("qreg q[2];\nswap q[0], q[1];");
        assert!(approx(&a, &b));
    }

    #[test]
    fn ccx_is_ccz_conjugated_by_h() {
        let a = u("qreg q[3];\nh q[2];\nccz q[0], q[1], q[2];\nh q[2];");
        let b = u("qreg q[3];\nccx q[0], q[1], q[2];");
        assert!(approx(&a, &b));
    }

    #[test]
    fn symbolic_angles_are_rejected() {
        let dag = qasm::parse("rz(theta1) q0;").unwrap();
        let err = Unitary::from_dag(&dag, &reg()).unwrap_err();
        assert!(matches!(err, CircuitError::UnresolvedParam { .. }));
    }

    #[test]
    fn oversized_circuits_are_rejected_cleanly() {
        let dag = qcircuit::Dag::new((0..20).map(|i| format!("q{i}")));
        let qs = dag.qubits().to_vec();
        let err = Unitary::from_dag_over_limited(&dag, qs, &reg(), 12).unwrap_err();
        assert!(matches!(err, CircuitError::TooManyQubits { .. }));
    }

    #[test]
    fn adjoint_inverts() {
        let a = u("qreg q[2];\nh q[0];\ncx q[0], q[1];\nrz(0.3) q[1];");
        let prod = a.adjoint().matrix().dot(a.matrix());
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j {
                    C::new(1.0, 0.0)
                } else {
                    C::default()
                };
                assert!((prod[[i, j]] - want).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn a_four_qubit_gate_is_supported() {
        use qcircuit::{CompositeStep, GateDef, GateSemantics};
        let mut r = reg();
        r.insert(GateDef {
            name: "quad".into(),
            arity: 4,
            num_params: 0,
            is_virtual_z: false,
            is_diagonal: true,
            semantics: GateSemantics::Composite(vec![
                CompositeStep {
                    gate: "cz".into(),
                    operands: vec![0, 3],
                    params: vec![],
                },
                CompositeStep {
                    gate: "cz".into(),
                    operands: vec![1, 2],
                    params: vec![],
                },
            ]),
        });
        let dag = qasm::parse("quad a,b,c,d;").unwrap();
        let x = Unitary::from_dag(&dag, &r).unwrap();
        assert_eq!(x.dim(), 16);
        assert!(x.is_unitary(1e-12));
    }

    #[test]
    fn apply_matches_dense_multiplication() {
        use qcircuit::embed::embed;
        let r = reg();
        let mut got = Unitary::identity(vec!["a".into(), "b".into(), "c".into()]);
        let cx = r.get("cx").unwrap().matrix(&[], &r).unwrap();
        got.apply(&cx, &[2, 0]);
        let want = embed(&cx, &[2, 0], 3);
        assert!(got
            .matrix()
            .iter()
            .zip(want.iter())
            .all(|(a, b)| (a - b).norm() < 1e-12));
    }

    #[test]
    fn global_phase_is_preserved_in_construction() {
        // rz(2pi) is -I, not I: the construction must not normalise phase away.
        let a = u("qreg q[1];\nrz(6.283185307179586) q[0];");
        assert!((a.matrix()[[0, 0]] - C::new(-1.0, 0.0)).norm() < 1e-9);
    }
}
