//! Arity-generic gate definitions and registry.
//!
//! # Why operand roles are indices
//!
//! The reference Java implementation encoded a wire's role in a gate as an enum,
//! `Edge.Label::{CONTROL, CONTROL2, TARGET, NONE}`. That works for one- and two-qubit
//! gates and then stops: supporting `ccz` required a `CONTROL2` variant plus a bespoke
//! `Node.isCCZ()` branch at roughly eight sites (two near-identical 30-line copies of
//! `getEdge`, `patternToCircuitQubit` — marked `// TODO improve` in the original —
//! partition growth, edge relabelling, and the connectivity check). A four-qubit gate
//! would have needed `CONTROL3` and another round of the same edits.
//!
//! Here a wire's role is simply its **operand index** in the gate. `cx q0, q1` puts `q0`
//! at slot 0 and `q1` at slot 1; `ccz a, b, c` uses slots 0, 1, 2; a hypothetical
//! 8-qubit gate uses slots 0..8. Nothing in the DAG, the matcher, or the partitioner
//! needs to know how many qubits a gate has.

use ndarray::{array, Array2};
use num_complex::Complex64;
use rustc_hash::FxHashMap;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::sync::Arc;

use crate::angle::{AngleExpr, ANGLE_EPS};
use crate::error::{CircuitError, Result};

type C = Complex64;

const fn c(re: f64, im: f64) -> C {
    Complex64::new(re, im)
}

const ZERO: C = c(0.0, 0.0);
const ONE: C = c(1.0, 0.0);

/// How a gate's unitary is computed.
#[derive(Debug, Clone)]
pub enum GateSemantics {
    /// A gate with a closed-form matrix.
    Builtin(Builtin),
    /// A gate defined by expanding into a sequence of other gates.
    ///
    /// Each element is `(gate name, operand indices into this gate's operands, params)`.
    /// This is how a gate set can be extended from a config file without writing Rust.
    Composite(Vec<CompositeStep>),
}

/// One step in a [`GateSemantics::Composite`] expansion.
#[derive(Debug, Clone)]
pub struct CompositeStep {
    pub gate: String,
    /// Indices into the *enclosing* gate's operand list.
    pub operands: Vec<usize>,
    pub params: Vec<AngleExpr>,
}

/// Gates with a closed-form unitary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum Builtin {
    I,
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    Sx,
    Sxdg,
    Rx,
    Ry,
    Rz,
    U1,
    U2,
    U3,
    Cx,
    Cz,
    Swap,
    Rxx,
    Ryy,
    Rzz,
    Ccx,
    Ccz,
    /// Trapped-ion single-qubit gate.
    Gpi,
    /// Trapped-ion single-qubit gate.
    Gpi2,
    /// Trapped-ion Mølmer–Sørensen gate.
    Ms,
}

impl Builtin {
    pub fn arity(self) -> usize {
        use Builtin::*;
        match self {
            I | H | X | Y | Z | S | Sdg | T | Tdg | Sx | Sxdg | Rx | Ry | Rz | U1 | U2 | U3
            | Gpi | Gpi2 => 1,
            Cx | Cz | Swap | Rxx | Ryy | Rzz | Ms => 2,
            Ccx | Ccz => 3,
        }
    }

    pub fn num_params(self) -> usize {
        use Builtin::*;
        match self {
            I | H | X | Y | Z | S | Sdg | T | Tdg | Sx | Sxdg | Cx | Cz | Swap | Ccx | Ccz => 0,
            Rx | Ry | Rz | U1 | Rxx | Ryy | Rzz | Gpi | Gpi2 => 1,
            U2 | Ms => 2,
            U3 => 3,
        }
    }

    /// The gate's unitary, in the operand ordering described on [`GateDef::matrix`].
    pub fn matrix(self, p: &[f64]) -> Array2<C> {
        use Builtin::*;
        let s = FRAC_1_SQRT_2;
        match self {
            I => eye(2),
            H => array![[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]],
            X => array![[ZERO, ONE], [ONE, ZERO]],
            Y => array![[ZERO, c(0.0, -1.0)], [c(0.0, 1.0), ZERO]],
            Z => array![[ONE, ZERO], [ZERO, c(-1.0, 0.0)]],
            S => phase_gate(PI / 2.0),
            Sdg => phase_gate(-PI / 2.0),
            T => phase_gate(PI / 4.0),
            Tdg => phase_gate(-PI / 4.0),
            Sx => array![[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]],
            Sxdg => array![[c(0.5, -0.5), c(0.5, 0.5)], [c(0.5, 0.5), c(0.5, -0.5)]],
            Rx => {
                let (ct, st) = half(p[0]);
                array![[c(ct, 0.0), c(0.0, -st)], [c(0.0, -st), c(ct, 0.0)]]
            }
            Ry => {
                let (ct, st) = half(p[0]);
                array![[c(ct, 0.0), c(-st, 0.0)], [c(st, 0.0), c(ct, 0.0)]]
            }
            Rz => {
                let h = p[0] / 2.0;
                array![
                    [C::from_polar(1.0, -h), ZERO],
                    [ZERO, C::from_polar(1.0, h)]
                ]
            }
            U1 => phase_gate(p[0]),
            U2 => {
                let (phi, lam) = (p[0], p[1]);
                array![
                    [c(s, 0.0), -C::from_polar(s, lam)],
                    [C::from_polar(s, phi), C::from_polar(s, phi + lam)]
                ]
            }
            U3 => {
                let (theta, phi, lam) = (p[0], p[1], p[2]);
                let (ct, st) = half(theta);
                array![
                    [c(ct, 0.0), -C::from_polar(st, lam)],
                    [C::from_polar(st, phi), C::from_polar(ct, phi + lam)]
                ]
            }
            Cx => controlled(&Builtin::X.matrix(&[])),
            Cz => controlled(&Builtin::Z.matrix(&[])),
            Swap => array![
                [ONE, ZERO, ZERO, ZERO],
                [ZERO, ZERO, ONE, ZERO],
                [ZERO, ONE, ZERO, ZERO],
                [ZERO, ZERO, ZERO, ONE]
            ],
            Rxx => two_qubit_rotation(p[0], PauliPair::Xx),
            Ryy => two_qubit_rotation(p[0], PauliPair::Yy),
            Rzz => two_qubit_rotation(p[0], PauliPair::Zz),
            Ccx => doubly_controlled(&Builtin::X.matrix(&[])),
            Ccz => doubly_controlled(&Builtin::Z.matrix(&[])),
            Gpi => {
                let phi = p[0];
                array![
                    [ZERO, C::from_polar(1.0, -phi)],
                    [C::from_polar(1.0, phi), ZERO]
                ]
            }
            Gpi2 => {
                let phi = p[0];
                array![
                    [c(s, 0.0), c(0.0, -s) * C::from_polar(1.0, -phi)],
                    [c(0.0, -s) * C::from_polar(1.0, phi), c(s, 0.0)]
                ]
            }
            Ms => {
                let (p0, p1) = (p[0], p[1]);
                let a = c(s, 0.0);
                let b = |ph: f64| c(0.0, -s) * C::from_polar(1.0, ph);
                array![
                    [a, ZERO, ZERO, b(-p0 - p1)],
                    [ZERO, a, b(-p0 + p1), ZERO],
                    [ZERO, b(p0 - p1), a, ZERO],
                    [b(p0 + p1), ZERO, ZERO, a]
                ]
            }
        }
    }
}

enum PauliPair {
    Xx,
    Yy,
    Zz,
}

fn two_qubit_rotation(theta: f64, kind: PauliPair) -> Array2<C> {
    let (ct, st) = half(theta);
    let cd = c(ct, 0.0);
    let msi = c(0.0, -st);
    match kind {
        PauliPair::Xx => array![
            [cd, ZERO, ZERO, msi],
            [ZERO, cd, msi, ZERO],
            [ZERO, msi, cd, ZERO],
            [msi, ZERO, ZERO, cd]
        ],
        PauliPair::Yy => array![
            [cd, ZERO, ZERO, -msi],
            [ZERO, cd, msi, ZERO],
            [ZERO, msi, cd, ZERO],
            [-msi, ZERO, ZERO, cd]
        ],
        PauliPair::Zz => {
            let h = theta / 2.0;
            let m = C::from_polar(1.0, -h);
            let p = C::from_polar(1.0, h);
            array![
                [m, ZERO, ZERO, ZERO],
                [ZERO, p, ZERO, ZERO],
                [ZERO, ZERO, p, ZERO],
                [ZERO, ZERO, ZERO, m]
            ]
        }
    }
}

fn half(theta: f64) -> (f64, f64) {
    ((theta / 2.0).cos(), (theta / 2.0).sin())
}

fn phase_gate(lambda: f64) -> Array2<C> {
    array![[ONE, ZERO], [ZERO, C::from_polar(1.0, lambda)]]
}

fn eye(n: usize) -> Array2<C> {
    Array2::from_shape_fn((n, n), |(i, j)| if i == j { ONE } else { ZERO })
}

/// Lift a 1-qubit matrix to a controlled 2-qubit matrix with operand 0 as the control.
fn controlled(u: &Array2<C>) -> Array2<C> {
    let mut m = eye(4);
    for i in 0..2 {
        for j in 0..2 {
            m[[2 + i, 2 + j]] = u[[i, j]];
        }
    }
    m
}

/// Lift a 1-qubit matrix to a doubly-controlled 3-qubit matrix, operands 0 and 1 control.
fn doubly_controlled(u: &Array2<C>) -> Array2<C> {
    let mut m = eye(8);
    for i in 0..2 {
        for j in 0..2 {
            m[[6 + i, 6 + j]] = u[[i, j]];
        }
    }
    m
}

/// A gate's definition: its name, shape, and semantics.
#[derive(Debug, Clone)]
pub struct GateDef {
    pub name: String,
    /// Number of qubit operands. Any value `>= 1`.
    pub arity: usize,
    /// Number of angle parameters.
    pub num_params: usize,
    pub semantics: GateSemantics,
    /// `true` for gates a backend can implement as a frame change rather than a pulse
    /// (`rz`, `u1`, `vz`). Several cost models charge nothing for these.
    pub is_virtual_z: bool,
    /// `true` if the gate is diagonal in the computational basis, so it commutes with
    /// any other diagonal gate. Used to prune commutation search.
    pub is_diagonal: bool,
}

impl GateDef {
    /// The gate's unitary for the given numeric parameters.
    ///
    /// # Operand ordering
    ///
    /// For a `k`-qubit gate, basis state index `i` encodes operand `j`'s bit at position
    /// `k - 1 - j`; that is, **operand 0 is the most significant bit**. With that
    /// convention `cx` (operand 0 = control) is the textbook
    /// `[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]`.
    ///
    /// Composite gates are expanded recursively against `registry`.
    pub fn matrix(&self, params: &[f64], registry: &GateRegistry) -> Result<Array2<C>> {
        if params.len() != self.num_params {
            return Err(CircuitError::ParamCount {
                gate: self.name.clone(),
                expected: self.num_params,
                got: params.len(),
            });
        }
        match &self.semantics {
            GateSemantics::Builtin(b) => Ok(b.matrix(params)),
            GateSemantics::Composite(steps) => self.expand_composite(steps, params, registry),
        }
    }

    fn expand_composite(
        &self,
        steps: &[CompositeStep],
        params: &[f64],
        registry: &GateRegistry,
    ) -> Result<Array2<C>> {
        let dim = 1usize << self.arity;
        let mut acc = eye(dim);
        for step in steps {
            let def = registry.get(&step.gate)?;
            let step_params: Vec<f64> = step
                .params
                .iter()
                .map(|e| {
                    e.eval_with(&|name| {
                        // Composite steps refer to the enclosing gate's params as p0, p1, ...
                        name.strip_prefix('p')
                            .and_then(|i| i.parse::<usize>().ok())
                            .and_then(|i| params.get(i).copied())
                    })
                    .ok_or_else(|| CircuitError::UnresolvedParam {
                        gate: self.name.clone(),
                        expr: e.to_string(),
                    })
                })
                .collect::<Result<_>>()?;
            let sub = def.matrix(&step_params, registry)?;
            let lifted = crate::embed::embed(&sub, &step.operands, self.arity);
            acc = lifted.dot(&acc);
        }
        Ok(acc)
    }
}

/// A name-to-definition table for gates.
///
/// Built from the embedded builtins, optionally extended from a TOML file so that new
/// gates can be added without recompiling — the reference implementation required editing
/// `PathSum.java`, `Synthesizer.applyGate`, and a `GateSet` enum for every new gate.
#[derive(Debug, Clone)]
pub struct GateRegistry {
    defs: FxHashMap<String, Arc<GateDef>>,
}

impl GateRegistry {
    /// The registry with every builtin gate and the standard aliases.
    pub fn with_builtins() -> Self {
        let mut defs = FxHashMap::default();
        let builtins: &[(&str, Builtin)] = &[
            ("id", Builtin::I),
            ("h", Builtin::H),
            ("x", Builtin::X),
            ("y", Builtin::Y),
            ("z", Builtin::Z),
            ("s", Builtin::S),
            ("sdg", Builtin::Sdg),
            ("t", Builtin::T),
            ("tdg", Builtin::Tdg),
            ("sx", Builtin::Sx),
            ("sxdg", Builtin::Sxdg),
            ("rx", Builtin::Rx),
            ("ry", Builtin::Ry),
            ("rz", Builtin::Rz),
            ("u1", Builtin::U1),
            ("p", Builtin::U1),
            ("u2", Builtin::U2),
            ("u3", Builtin::U3),
            ("u", Builtin::U3),
            ("cx", Builtin::Cx),
            ("cnot", Builtin::Cx),
            ("cz", Builtin::Cz),
            ("swap", Builtin::Swap),
            ("rxx", Builtin::Rxx),
            ("ryy", Builtin::Ryy),
            ("rzz", Builtin::Rzz),
            ("ccx", Builtin::Ccx),
            ("toffoli", Builtin::Ccx),
            ("ccz", Builtin::Ccz),
            ("gpi", Builtin::Gpi),
            ("gpi2", Builtin::Gpi2),
            ("ms", Builtin::Ms),
            // `vz` is the trapped-ion name for a virtual (frame-change) z rotation.
            ("vz", Builtin::Rz),
        ];
        for (name, b) in builtins {
            let is_virtual_z = matches!(*name, "rz" | "u1" | "p" | "vz");
            let is_diagonal = matches!(
                b,
                Builtin::I
                    | Builtin::Z
                    | Builtin::S
                    | Builtin::Sdg
                    | Builtin::T
                    | Builtin::Tdg
                    | Builtin::Rz
                    | Builtin::U1
                    | Builtin::Cz
                    | Builtin::Rzz
                    | Builtin::Ccz
            );
            defs.insert(
                (*name).to_string(),
                Arc::new(GateDef {
                    name: (*name).to_string(),
                    arity: b.arity(),
                    num_params: b.num_params(),
                    semantics: GateSemantics::Builtin(*b),
                    is_virtual_z,
                    is_diagonal,
                }),
            );
        }
        Self { defs }
    }

    pub fn get(&self, name: &str) -> Result<&Arc<GateDef>> {
        self.defs
            .get(name)
            .ok_or_else(|| CircuitError::UnknownGate(name.to_string()))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.defs.contains_key(name)
    }

    /// Register a gate, replacing any previous definition of the same name.
    pub fn insert(&mut self, def: GateDef) {
        self.defs.insert(def.name.clone(), Arc::new(def));
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.defs.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }
}

impl Default for GateRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

/// `true` if the gate is a T or T-dagger, including an `rz` at an odd multiple of `pi/4`.
///
/// The reference implementation wrote this as
/// `Math.abs((eval(angle) / Math.PI) % 0.5) == 0.25`, an exact float comparison that also
/// panicked on symbolic angles because `eval` threw on any non-`pi` symbol.
pub fn is_t_gate(name: &str, params: &[AngleExpr]) -> bool {
    match name {
        "t" | "tdg" => true,
        "rz" | "u1" | "p" | "vz" => params.first().is_some_and(|a| {
            a.eval().is_some_and(|v| {
                let r = (v / (PI / 4.0)).round();
                (v - r * PI / 4.0).abs() <= ANGLE_EPS && (r as i64).rem_euclid(2) == 1
            })
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: &Array2<C>, b: &Array2<C>) -> bool {
        a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() < 1e-12)
    }

    fn is_unitary(m: &Array2<C>) -> bool {
        let n = m.shape()[0];
        let adj = m.t().mapv(|z| z.conj());
        approx_eq(&adj.dot(m), &eye(n))
    }

    #[test]
    fn every_builtin_is_unitary() {
        let reg = GateRegistry::with_builtins();
        // Deliberately awkward parameter values, not multiples of pi/2.
        let sample = [0.7391, 1.2345, 2.4813];
        for name in reg.names() {
            let def = reg.get(name).unwrap();
            let params = &sample[..def.num_params];
            let m = def.matrix(params, &reg).unwrap();
            assert_eq!(m.shape(), &[1 << def.arity, 1 << def.arity], "{name}");
            assert!(is_unitary(&m), "{name} is not unitary");
        }
    }

    #[test]
    fn cx_uses_operand_zero_as_control() {
        let reg = GateRegistry::with_builtins();
        let m = reg.get("cx").unwrap().matrix(&[], &reg).unwrap();
        let expect = array![
            [ONE, ZERO, ZERO, ZERO],
            [ZERO, ONE, ZERO, ZERO],
            [ZERO, ZERO, ZERO, ONE],
            [ZERO, ZERO, ONE, ZERO]
        ];
        assert!(approx_eq(&m, &expect));
    }

    #[test]
    fn ccz_is_diagonal_with_single_minus_one() {
        let reg = GateRegistry::with_builtins();
        let m = reg.get("ccz").unwrap().matrix(&[], &reg).unwrap();
        for i in 0..8 {
            for j in 0..8 {
                let want = if i != j {
                    ZERO
                } else if i == 7 {
                    c(-1.0, 0.0)
                } else {
                    ONE
                };
                assert!((m[[i, j]] - want).norm() < 1e-12, "at {i},{j}");
            }
        }
    }

    #[test]
    fn rotation_gates_are_four_pi_periodic() {
        let reg = GateRegistry::with_builtins();
        for name in ["rx", "ry", "rz", "rxx", "ryy", "rzz"] {
            let def = reg.get(name).unwrap();
            let a = def.matrix(&[0.3], &reg).unwrap();
            let b = def.matrix(&[0.3 + 4.0 * PI], &reg).unwrap();
            assert!(approx_eq(&a, &b), "{name} not 4pi-periodic");
            // ...and 2pi gives a global phase of -1, not the identity.
            let d = def.matrix(&[0.3 + 2.0 * PI], &reg).unwrap();
            assert!(!approx_eq(&a, &d), "{name} should differ at 2pi");
        }
    }

    #[test]
    fn named_gates_agree_with_their_rotation_forms() {
        let reg = GateRegistry::with_builtins();
        let g = |n: &str, p: &[f64]| reg.get(n).unwrap().matrix(p, &reg).unwrap();
        // t == u1(pi/4), s == u1(pi/2), z == u1(pi)
        assert!(approx_eq(&g("t", &[]), &g("u1", &[PI / 4.0])));
        assert!(approx_eq(&g("tdg", &[]), &g("u1", &[-PI / 4.0])));
        assert!(approx_eq(&g("s", &[]), &g("u1", &[PI / 2.0])));
        assert!(approx_eq(&g("sdg", &[]), &g("u1", &[-PI / 2.0])));
        assert!(approx_eq(&g("z", &[]), &g("u1", &[PI])));
        // h == u2(0, pi)
        assert!(approx_eq(&g("h", &[]), &g("u2", &[0.0, PI])));
        // sx and sxdg are inverses
        let prod = g("sx", &[]).dot(&g("sxdg", &[]));
        assert!(approx_eq(&prod, &eye(2)));
    }

    #[test]
    fn dagger_pairs_are_inverses() {
        let reg = GateRegistry::with_builtins();
        let g = |n: &str| reg.get(n).unwrap().matrix(&[], &reg).unwrap();
        for (a, b) in [("t", "tdg"), ("s", "sdg"), ("sx", "sxdg")] {
            assert!(approx_eq(&g(a).dot(&g(b)), &eye(2)), "{a}/{b}");
        }
    }

    #[test]
    fn self_inverse_gates() {
        let reg = GateRegistry::with_builtins();
        for name in ["h", "x", "y", "z", "cx", "cz", "swap", "ccx", "ccz"] {
            let def = reg.get(name).unwrap();
            let m = def.matrix(&[], &reg).unwrap();
            assert!(approx_eq(&m.dot(&m), &eye(1 << def.arity)), "{name}");
        }
    }

    #[test]
    fn arity_and_param_counts_are_consistent() {
        let reg = GateRegistry::with_builtins();
        for name in reg.names() {
            let def = reg.get(name).unwrap();
            if let GateSemantics::Builtin(b) = &def.semantics {
                assert_eq!(def.arity, b.arity(), "{name} arity");
                assert_eq!(def.num_params, b.num_params(), "{name} params");
            }
            assert!(def.arity >= 1, "{name} must act on at least one qubit");
        }
    }

    #[test]
    fn wrong_param_count_is_an_error() {
        let reg = GateRegistry::with_builtins();
        let err = reg.get("rz").unwrap().matrix(&[], &reg).unwrap_err();
        assert!(matches!(err, CircuitError::ParamCount { .. }));
    }

    #[test]
    fn unknown_gate_is_an_error() {
        let reg = GateRegistry::with_builtins();
        assert!(matches!(
            reg.get("nope").unwrap_err(),
            CircuitError::UnknownGate(_)
        ));
    }

    /// The registry supports arities beyond the 1/2/3 the reference hardcoded.
    #[test]
    fn composite_gates_of_arbitrary_arity() {
        let mut reg = GateRegistry::with_builtins();
        // A 4-qubit gate: cz between operands (0,3) then (1,2).
        reg.insert(GateDef {
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
        let m = reg.get("quad").unwrap().matrix(&[], &reg).unwrap();
        assert_eq!(m.shape(), &[16, 16]);
        assert!(is_unitary(&m));
        // Diagonal entry is -1 exactly when (b0 & b3) xor-parity (b1 & b2) is odd.
        for i in 0..16usize {
            let b = |k: usize| (i >> (3 - k)) & 1;
            let sign = if (b(0) & b(3)) ^ (b(1) & b(2)) == 1 {
                -1.0
            } else {
                1.0
            };
            let neg = (b(0) & b(3)) + (b(1) & b(2));
            let want = c(if neg % 2 == 1 { -1.0 } else { 1.0 }, 0.0);
            let _ = sign;
            assert!((m[[i, i]] - want).norm() < 1e-12, "diag {i}");
        }
    }

    #[test]
    fn composite_expansion_matches_builtin() {
        let mut reg = GateRegistry::with_builtins();
        // ccz built the way the reference did it, as a cx/rz ladder.
        let q = |v: f64| AngleExpr::Num(v);
        reg.insert(GateDef {
            name: "ccz_ladder".into(),
            arity: 3,
            num_params: 0,
            is_virtual_z: false,
            is_diagonal: true,
            semantics: GateSemantics::Composite(vec![
                CompositeStep {
                    gate: "cx".into(),
                    operands: vec![1, 2],
                    params: vec![],
                },
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![2],
                    params: vec![q(-PI / 4.0)],
                },
                CompositeStep {
                    gate: "cx".into(),
                    operands: vec![0, 2],
                    params: vec![],
                },
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![2],
                    params: vec![q(PI / 4.0)],
                },
                CompositeStep {
                    gate: "cx".into(),
                    operands: vec![1, 2],
                    params: vec![],
                },
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![2],
                    params: vec![q(-PI / 4.0)],
                },
                CompositeStep {
                    gate: "cx".into(),
                    operands: vec![0, 2],
                    params: vec![],
                },
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![2],
                    params: vec![q(PI / 4.0)],
                },
                CompositeStep {
                    gate: "cx".into(),
                    operands: vec![0, 1],
                    params: vec![],
                },
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![1],
                    params: vec![q(-PI / 4.0)],
                },
                CompositeStep {
                    gate: "cx".into(),
                    operands: vec![0, 1],
                    params: vec![],
                },
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![1],
                    params: vec![q(PI / 4.0)],
                },
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![0],
                    params: vec![q(PI / 4.0)],
                },
            ]),
        });
        let ladder = reg.get("ccz_ladder").unwrap().matrix(&[], &reg).unwrap();
        let direct = reg.get("ccz").unwrap().matrix(&[], &reg).unwrap();
        // Equal up to global phase.
        let ratio = ladder[[0, 0]] / direct[[0, 0]];
        let scaled = direct.mapv(|z| z * ratio);
        assert!(approx_eq(&ladder, &scaled), "ladder != ccz up to phase");
    }

    #[test]
    fn composite_params_thread_through() {
        let mut reg = GateRegistry::with_builtins();
        reg.insert(GateDef {
            name: "rz_twice".into(),
            arity: 1,
            num_params: 1,
            is_virtual_z: true,
            is_diagonal: true,
            semantics: GateSemantics::Composite(vec![
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![0],
                    params: vec![AngleExpr::div(AngleExpr::var("p0"), AngleExpr::Num(2.0))],
                },
                CompositeStep {
                    gate: "rz".into(),
                    operands: vec![0],
                    params: vec![AngleExpr::div(AngleExpr::var("p0"), AngleExpr::Num(2.0))],
                },
            ]),
        });
        let a = reg.get("rz_twice").unwrap().matrix(&[1.3], &reg).unwrap();
        let b = reg.get("rz").unwrap().matrix(&[1.3], &reg).unwrap();
        assert!(approx_eq(&a, &b));
    }

    #[test]
    fn t_gate_detection() {
        assert!(is_t_gate("t", &[]));
        assert!(is_t_gate("tdg", &[]));
        assert!(is_t_gate("rz", &[AngleExpr::Num(PI / 4.0)]));
        assert!(is_t_gate("rz", &[AngleExpr::Num(-PI / 4.0)]));
        assert!(is_t_gate("rz", &[AngleExpr::Num(3.0 * PI / 4.0)]));
        assert!(!is_t_gate("rz", &[AngleExpr::Num(PI / 2.0)]));
        assert!(!is_t_gate("rz", &[AngleExpr::Num(PI)]));
        assert!(!is_t_gate("h", &[]));
        // A symbolic angle must not panic, unlike the reference's `eval`.
        assert!(!is_t_gate("rz", &[AngleExpr::var("theta1")]));
    }

    #[test]
    fn virtual_z_flags() {
        let reg = GateRegistry::with_builtins();
        for n in ["rz", "u1", "p", "vz"] {
            assert!(reg.get(n).unwrap().is_virtual_z, "{n}");
        }
        for n in ["h", "x", "cx", "sx", "u3"] {
            assert!(!reg.get(n).unwrap().is_virtual_z, "{n}");
        }
    }

    #[test]
    fn aliases_resolve_to_same_semantics() {
        let reg = GateRegistry::with_builtins();
        let a = reg.get("cx").unwrap().matrix(&[], &reg).unwrap();
        let b = reg.get("cnot").unwrap().matrix(&[], &reg).unwrap();
        assert!(approx_eq(&a, &b));
        let a = reg.get("u1").unwrap().matrix(&[0.5], &reg).unwrap();
        let b = reg.get("p").unwrap().matrix(&[0.5], &reg).unwrap();
        assert!(approx_eq(&a, &b));
    }
}
