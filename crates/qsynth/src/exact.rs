//! Exact rule verification: rational arithmetic, zero tolerance.
//!
//! This is the QUESO paper's verification discipline (its `--useRational` mode): decide
//! whether two circuits are equal by evaluating both on every basis input and comparing
//! amplitudes **exactly**, with every number a rational — no floats, no epsilon. Angle
//! variables are handled by polynomial identity testing: each is instantiated at a
//! *random rational point on the unit circle*, so a true identity always passes, and a
//! false one passes only if the random point happens to be a root of the difference —
//! a probability-zero event repeated over independent rounds.
//!
//! # The number field
//!
//! Amplitudes live in `K = Q(i, sqrt(2))`, represented as `a + b*sqrt(2)` with `a`, `b`
//! complex rationals. Every constant the synthesis gate sets produce is in `K`: `h`'s
//! `1/sqrt(2)`, `t`'s `e^{i*pi/4} = (1+i)/sqrt(2)`, and every phase on the `pi/4` grid.
//! The reference tracked `1/sqrt(2)` factors by pairing them into halves and juggling
//! the odd one out symbolically; a field closed under the arithmetic needs no juggling.
//!
//! # Angles as rational points on the circle
//!
//! For angle variable `v`, the *half-angle* factor `u = e^{iv/2}` is sampled by the
//! tangent-half-angle parametrization: for a random rational `r`, the point
//! `((1-r^2) + 2ir) / (1+r^2)` is exactly unimodular and exactly rational. Everything a
//! gate needs is then exact arithmetic on `u`: `e^{iv} = u^2`, `e^{-iv/2} = conj(u)`
//! (conjugate is inverse on the circle), and `cos(v/2), sin(v/2) = (u ± conj(u))/2`.
//! Sampling the half-angle first is what lets `rz`-style gates take exact half-phases;
//! the reference reconstructed this relationship after the fact and could not evaluate
//! `cos`/`sin` at all, so its rational mode refused `rx`-family gates. Here they are
//! exact like everything else.
//!
//! # Holes and the strong contract
//!
//! A symbolic rule's constraint promises equality *provided* the hole acts on the
//! boundary like a listed permutation. "Acts like" is read strictly here: the hole is
//! modeled as the permutation times an **arbitrary phase per boundary basis state**,
//! each sampled as an independent random unit point. A rule that verifies under free
//! per-state phases holds for *every* region whose boundary support is the permutation
//! — which is exactly the condition the optimizer's support-tier region check
//! establishes at application time, so no further per-application soundness check is
//! needed. Rules that hold only for the bare permutation (the reference's float mode
//! admitted them) are deliberately not emitted; see `docs/PORTING-NOTES.md`.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

use qcircuit::{AngleExpr, GateOp, GateRegistry, QubitId};
use qrules::BasisPermutation;

use crate::fingerprint::Equivalence;

/// Why a circuit could not be verified exactly.
///
/// Exact verification is total over the builtin gate sets; this exists for user-defined
/// gates and angle spellings outside the exact fragment. A candidate that cannot be
/// checked is *dropped*, never emitted on trust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub String);

type ExactResult<T> = Result<T, Unsupported>;

fn unsupported<T>(what: impl Into<String>) -> ExactResult<T> {
    Err(Unsupported(what.into()))
}

fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

/// A complex rational: the field `Q(i)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CRat {
    re: BigRational,
    im: BigRational,
}

impl CRat {
    fn new(re: BigRational, im: BigRational) -> Self {
        Self { re, im }
    }

    fn zero() -> Self {
        Self::new(BigRational::zero(), BigRational::zero())
    }

    fn one() -> Self {
        Self::new(BigRational::one(), BigRational::zero())
    }

    fn i() -> Self {
        Self::new(BigRational::zero(), BigRational::one())
    }

    fn from_int(n: i64) -> Self {
        Self::new(rat(n), BigRational::zero())
    }

    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    fn add(&self, o: &Self) -> Self {
        Self::new(&self.re + &o.re, &self.im + &o.im)
    }

    fn sub(&self, o: &Self) -> Self {
        Self::new(&self.re - &o.re, &self.im - &o.im)
    }

    fn mul(&self, o: &Self) -> Self {
        Self::new(
            &self.re * &o.re - &self.im * &o.im,
            &self.re * &o.im + &self.im * &o.re,
        )
    }

    fn neg(&self) -> Self {
        Self::new(-self.re.clone(), -self.im.clone())
    }

    fn conj(&self) -> Self {
        Self::new(self.re.clone(), -self.im.clone())
    }

    fn scale(&self, s: &BigRational) -> Self {
        Self::new(&self.re * s, &self.im * s)
    }

    /// `1/self`; `self` must be nonzero.
    fn inv(&self) -> Self {
        let n = &self.re * &self.re + &self.im * &self.im;
        Self::new(&self.re / &n, -(&self.im / &n))
    }

    /// Integer power, negative allowed for nonzero values.
    fn pow(&self, e: i64) -> Self {
        let base = if e < 0 { self.inv() } else { self.clone() };
        let mut out = Self::one();
        for _ in 0..e.unsigned_abs() {
            out = out.mul(&base);
        }
        out
    }
}

/// An element of `K = Q(i, sqrt(2))`, as `a + b*sqrt(2)`.
///
/// `sqrt(2)` is irrational, so the representation is unique and equality is exact
/// componentwise equality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Amp {
    a: CRat,
    b: CRat,
}

impl Amp {
    fn zero() -> Self {
        Self {
            a: CRat::zero(),
            b: CRat::zero(),
        }
    }

    fn one() -> Self {
        Self {
            a: CRat::one(),
            b: CRat::zero(),
        }
    }

    fn from_crat(a: CRat) -> Self {
        Self { a, b: CRat::zero() }
    }

    /// `1/sqrt(2)`, i.e. `sqrt(2)/2`.
    fn isq2() -> Self {
        Self {
            a: CRat::zero(),
            b: CRat::new(
                BigRational::new(BigInt::from(1), BigInt::from(2)),
                BigRational::zero(),
            ),
        }
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            a: self.a.add(&o.a),
            b: self.b.add(&o.b),
        }
    }

    fn mul(&self, o: &Self) -> Self {
        // (a1 + b1 s)(a2 + b2 s) = (a1 a2 + 2 b1 b2) + (a1 b2 + b1 a2) s, s = sqrt(2).
        Self {
            a: self.a.mul(&o.a).add(&self.b.mul(&o.b).scale(&rat(2))),
            b: self.a.mul(&o.b).add(&self.b.mul(&o.a)),
        }
    }

    fn neg(&self) -> Self {
        Self {
            a: self.a.neg(),
            b: self.b.neg(),
        }
    }

    /// Complex conjugate; `sqrt(2)` is real, so it conjugates componentwise.
    fn conj(&self) -> Self {
        Self {
            a: self.a.conj(),
            b: self.b.conj(),
        }
    }

    fn scale(&self, s: &CRat) -> Self {
        Self {
            a: self.a.mul(s),
            b: self.b.mul(s),
        }
    }

    /// `1/self`; `self` must be nonzero.
    fn inv(&self) -> Self {
        // Rationalize the sqrt(2): (a + b s)(a - b s) = a^2 - 2 b^2 in Q(i).
        let d = self.a.mul(&self.a).sub(&self.b.mul(&self.b).scale(&rat(2)));
        if d.is_zero() {
            // a = ±b*sqrt(2) is impossible for a, b rational complex unless both zero.
            unreachable!("a^2 = 2 b^2 has no nonzero rational solution");
        }
        let di = d.inv();
        Self {
            a: self.a.mul(&di),
            b: self.b.neg().mul(&di),
        }
    }

    fn div(&self, o: &Self) -> Self {
        self.mul(&o.inv())
    }

    /// `e^{i*pi*k/4}`: the eighth roots of unity, all in `K`.
    fn eighth_root(k: i64) -> Self {
        let half = BigRational::new(BigInt::from(1), BigInt::from(2));
        let pos = CRat::new(half.clone(), half.clone()); // (1+i)/2, times sqrt(2) = e^{i pi/4}
        let neg = CRat::new(half.clone(), -half); // (1-i)/2, times sqrt(2) = e^{-i pi/4}
        match k.rem_euclid(8) {
            0 => Self::one(),
            1 => Self {
                a: CRat::zero(),
                b: pos,
            },
            2 => Self::from_crat(CRat::i()),
            3 => Self {
                a: CRat::zero(),
                b: neg.neg(),
            },
            4 => Self::from_crat(CRat::from_int(-1)),
            5 => Self {
                a: CRat::zero(),
                b: pos.neg(),
            },
            6 => Self::from_crat(CRat::i().neg()),
            7 => Self {
                a: CRat::zero(),
                b: neg,
            },
            _ => unreachable!(),
        }
    }
}

/// An angle as a rational-linear form: `sum(c_v * v) + c_pi * pi + c_1`.
///
/// This is the whole angle language rule synthesis uses — variables, sums,
/// differences, and rational multiples of `pi` — and the fragment on which exact phase
/// evaluation is possible. A nonzero plain-number constant has no exact phase and is
/// refused.
#[derive(Debug, Clone, Default)]
struct Linear {
    vars: BTreeMap<String, BigRational>,
    pi: BigRational,
    constant: BigRational,
}

impl Linear {
    fn add(mut self, o: Linear) -> Linear {
        for (v, c) in o.vars {
            let entry = self.vars.entry(v).or_insert_with(BigRational::zero);
            *entry += c;
        }
        self.pi += o.pi;
        self.constant += o.constant;
        self
    }

    fn scale(mut self, s: &BigRational) -> Linear {
        for c in self.vars.values_mut() {
            *c *= s;
        }
        self.pi *= s;
        self.constant *= s;
        self
    }

    fn neg(self) -> Linear {
        self.scale(&rat(-1))
    }

    /// `Some` if this is a pure number with no variables and no `pi`.
    fn as_constant(&self) -> Option<&BigRational> {
        (self.vars.values().all(Zero::is_zero) && self.pi.is_zero()).then_some(&self.constant)
    }
}

/// Normalize an [`AngleExpr`] into a [`Linear`] form.
///
/// Numeric literals must be integers: they only ever appear as coefficients
/// (`4*pi - theta1`, `pi/2`), and an arbitrary decimal has no exact phase.
fn linearize(e: &AngleExpr) -> ExactResult<Linear> {
    match e {
        AngleExpr::Num(v) => {
            if v.fract() == 0.0 && v.abs() < 9e15 {
                Ok(Linear {
                    constant: rat(*v as i64),
                    ..Default::default()
                })
            } else {
                unsupported(format!("non-integer numeric angle literal {v}"))
            }
        }
        AngleExpr::Pi => Ok(Linear {
            pi: BigRational::one(),
            ..Default::default()
        }),
        AngleExpr::Var(name) => {
            let mut vars = BTreeMap::new();
            vars.insert(name.clone(), BigRational::one());
            Ok(Linear {
                vars,
                ..Default::default()
            })
        }
        AngleExpr::Neg(x) => Ok(linearize(x)?.neg()),
        AngleExpr::Add(x, y) => Ok(linearize(x)?.add(linearize(y)?)),
        AngleExpr::Sub(x, y) => Ok(linearize(x)?.add(linearize(y)?.neg())),
        AngleExpr::Mul(x, y) => {
            let (lx, ly) = (linearize(x)?, linearize(y)?);
            if let Some(c) = lx.as_constant() {
                let c = c.clone();
                Ok(ly.scale(&c))
            } else if let Some(c) = ly.as_constant() {
                let c = c.clone();
                Ok(lx.scale(&c))
            } else {
                unsupported("product of two non-constant angles")
            }
        }
        AngleExpr::Div(x, y) => {
            let (lx, ly) = (linearize(x)?, linearize(y)?);
            match ly.as_constant() {
                Some(c) if !c.is_zero() => {
                    let inv = c.recip();
                    Ok(lx.scale(&inv))
                }
                _ => unsupported("division by a non-constant angle"),
            }
        }
    }
}

/// The random rational sample points of one verification round.
pub(crate) struct Samples {
    /// Per angle variable, the *half-angle* unit point `e^{iv/2}`.
    half_points: FxHashMap<String, CRat>,
    /// Per boundary basis state, the hole's free phase.
    hole_phases: FxHashMap<usize, Amp>,
    state: u64,
}

impl Samples {
    fn new(seed: u64) -> Self {
        Self {
            half_points: FxHashMap::default(),
            hole_phases: FxHashMap::default(),
            state: seed ^ 0x243f_6a88_85a3_08d3,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A random rational point on the unit circle, exactly unimodular.
    ///
    /// The tangent-half-angle parametrization: for rational `r`,
    /// `((1 - r^2) + 2ir)/(1 + r^2)` has norm exactly 1. The denominator is kept
    /// moderate so products across a whole circuit stay a few hundred bits.
    fn unit_point(&mut self) -> CRat {
        let n = 2 + (self.next_u64() % ((1 << 10) - 2)) as i64;
        let r = BigRational::new(BigInt::from(1), BigInt::from(n));
        let r2 = &r * &r;
        let den = BigRational::one() + &r2;
        CRat::new((BigRational::one() - &r2) / &den, (rat(2) * &r) / &den)
    }

    fn half_point(&mut self, var: &str) -> CRat {
        if let Some(u) = self.half_points.get(var) {
            return u.clone();
        }
        let u = self.unit_point();
        self.half_points.insert(var.to_string(), u.clone());
        u
    }

    fn hole_phase(&mut self, state: usize) -> Amp {
        if let Some(p) = self.hole_phases.get(&state) {
            return p.clone();
        }
        let p = Amp::from_crat(self.unit_point());
        self.hole_phases.insert(state, p.clone());
        p
    }

    /// `e^{i * l}`, exactly.
    ///
    /// Variable coefficients must be multiples of `1/2` (the half-angle points are the
    /// finest granularity sampled) and the `pi` coefficient a multiple of `1/4` (the
    /// eighth roots of unity are the finest roots in `K`).
    fn phase(&mut self, l: &Linear) -> ExactResult<Amp> {
        if !l.constant.is_zero() {
            return unsupported("angle with a non-pi numeric term");
        }
        let quarter_pi = &l.pi * rat(4);
        if !quarter_pi.is_integer() {
            return unsupported(format!("pi coefficient {} is not a multiple of 1/4", l.pi));
        }
        let mut out = Amp::eighth_root(to_i64(&quarter_pi.to_integer())?);
        for (v, c) in &l.vars {
            if c.is_zero() {
                continue;
            }
            let doubled = c * rat(2);
            if !doubled.is_integer() {
                return unsupported(format!("coefficient {c} of {v} is not a multiple of 1/2"));
            }
            let e = to_i64(&doubled.to_integer())?;
            let u = self.half_point(v);
            // Unimodular, so the inverse is the conjugate: negative powers are exact.
            let factor = if e >= 0 { u.pow(e) } else { u.conj().pow(-e) };
            out = out.mul(&Amp::from_crat(factor));
        }
        Ok(out)
    }

    /// `cos(l)` and `sin(l)`, exactly, via `e^{±il}`.
    fn cos_sin(&mut self, l: &Linear) -> ExactResult<(Amp, Amp)> {
        let e = self.phase(l)?;
        let ec = e.conj();
        let half = CRat::new(
            BigRational::new(BigInt::from(1), BigInt::from(2)),
            BigRational::zero(),
        );
        let cos = e.add(&ec).scale(&half);
        // (e - conj e) / 2i = (e - conj e) * (-i/2)
        let neg_half_i = CRat::new(
            BigRational::zero(),
            BigRational::new(BigInt::from(-1), BigInt::from(2)),
        );
        let sin = e.add(&ec.neg()).scale(&neg_half_i);
        Ok((cos, sin))
    }
}

fn to_i64(v: &BigInt) -> ExactResult<i64> {
    use num_traits::ToPrimitive;
    v.to_i64()
        .ok_or_else(|| Unsupported(format!("coefficient {v} out of range")))
}

/// A gate's exact unitary over `K`, in the operand ordering of
/// [`qcircuit::gate::Builtin::matrix`] (operand 0 is the most significant bit).
fn gate_matrix(
    op: &GateOp,
    samples: &mut Samples,
    registry: &GateRegistry,
) -> ExactResult<Vec<Vec<Amp>>> {
    use qcircuit::gate::{Builtin, GateSemantics};

    let def = registry
        .get(&op.gate)
        .map_err(|_| Unsupported(format!("unknown gate `{}`", op.gate)))?;
    let GateSemantics::Builtin(b) = &def.semantics else {
        return unsupported(format!("composite gate `{}`", op.gate));
    };

    let angle = |i: usize| -> ExactResult<Linear> { linearize(&op.params[i]) };
    let half = |l: Linear| l.scale(&BigRational::new(BigInt::from(1), BigInt::from(2)));

    let o = Amp::zero;
    let l = Amp::one;
    let i_amp = || Amp::from_crat(CRat::i());
    let neg_i = || Amp::from_crat(CRat::i().neg());
    let m1 = || Amp::from_crat(CRat::from_int(-1));

    let m: Vec<Vec<Amp>> = match b {
        Builtin::I => vec![vec![l(), o()], vec![o(), l()]],
        Builtin::H => {
            let s = Amp::isq2();
            vec![vec![s.clone(), s.clone()], vec![s.clone(), s.neg()]]
        }
        Builtin::X => vec![vec![o(), l()], vec![l(), o()]],
        Builtin::Y => vec![vec![o(), neg_i()], vec![i_amp(), o()]],
        Builtin::Z => vec![vec![l(), o()], vec![o(), m1()]],
        Builtin::S => vec![vec![l(), o()], vec![o(), i_amp()]],
        Builtin::Sdg => vec![vec![l(), o()], vec![o(), neg_i()]],
        Builtin::T => vec![vec![l(), o()], vec![o(), Amp::eighth_root(1)]],
        Builtin::Tdg => vec![vec![l(), o()], vec![o(), Amp::eighth_root(-1)]],
        Builtin::Sx | Builtin::Sxdg => {
            let half_r = BigRational::new(BigInt::from(1), BigInt::from(2));
            let p = Amp::from_crat(CRat::new(half_r.clone(), half_r.clone()));
            let q = Amp::from_crat(CRat::new(half_r.clone(), -half_r));
            if matches!(b, Builtin::Sx) {
                vec![vec![p.clone(), q.clone()], vec![q, p]]
            } else {
                vec![vec![q.clone(), p.clone()], vec![p, q]]
            }
        }
        Builtin::Rx => {
            let (ct, st) = samples.cos_sin(&half(angle(0)?))?;
            let mist = st.mul(&neg_i());
            vec![vec![ct.clone(), mist.clone()], vec![mist, ct]]
        }
        Builtin::Ry => {
            let (ct, st) = samples.cos_sin(&half(angle(0)?))?;
            vec![vec![ct.clone(), st.neg()], vec![st, ct]]
        }
        Builtin::Rz => {
            let h = half(angle(0)?);
            let p = samples.phase(&h)?;
            vec![vec![p.conj(), o()], vec![o(), p]]
        }
        Builtin::U1 => {
            let p = samples.phase(&angle(0)?)?;
            vec![vec![l(), o()], vec![o(), p]]
        }
        Builtin::U2 => {
            let s = Amp::isq2();
            let (ephi, elam) = (samples.phase(&angle(0)?)?, samples.phase(&angle(1)?)?);
            vec![
                vec![s.clone(), elam.mul(&s).neg()],
                vec![ephi.mul(&s), ephi.mul(&elam).mul(&s)],
            ]
        }
        Builtin::U3 => {
            let (ct, st) = samples.cos_sin(&half(angle(0)?))?;
            let (ephi, elam) = (samples.phase(&angle(1)?)?, samples.phase(&angle(2)?)?);
            vec![
                vec![ct.clone(), elam.mul(&st).neg()],
                vec![ephi.mul(&st), ephi.mul(&elam).mul(&ct)],
            ]
        }
        Builtin::Cx => controlled(&[[o(), l()], [l(), o()]]),
        Builtin::Cz => controlled(&[[l(), o()], [o(), m1()]]),
        Builtin::Swap => vec![
            vec![l(), o(), o(), o()],
            vec![o(), o(), l(), o()],
            vec![o(), l(), o(), o()],
            vec![o(), o(), o(), l()],
        ],
        Builtin::Rxx | Builtin::Ryy => {
            let (ct, st) = samples.cos_sin(&half(angle(0)?))?;
            let mist = st.mul(&neg_i());
            let corner = if matches!(b, Builtin::Ryy) {
                mist.neg()
            } else {
                mist.clone()
            };
            vec![
                vec![ct.clone(), o(), o(), corner.clone()],
                vec![o(), ct.clone(), mist.clone(), o()],
                vec![o(), mist, ct.clone(), o()],
                vec![corner, o(), o(), ct],
            ]
        }
        Builtin::Rzz => {
            let p = samples.phase(&half(angle(0)?))?;
            let m = p.conj();
            vec![
                vec![m.clone(), o(), o(), o()],
                vec![o(), p.clone(), o(), o()],
                vec![o(), o(), p, o()],
                vec![o(), o(), o(), m],
            ]
        }
        Builtin::Ccx => {
            let mut m = eye(8);
            m[6][6] = o();
            m[6][7] = l();
            m[7][6] = l();
            m[7][7] = o();
            m
        }
        Builtin::Ccz => {
            let mut m = eye(8);
            m[7][7] = m1();
            m
        }
        Builtin::Gpi => {
            let p = samples.phase(&angle(0)?)?;
            vec![vec![o(), p.conj()], vec![p, o()]]
        }
        Builtin::Gpi2 => {
            let s = Amp::isq2();
            let p = samples.phase(&angle(0)?)?;
            vec![
                vec![s.clone(), p.conj().mul(&neg_i()).mul(&s)],
                vec![p.mul(&neg_i()).mul(&s), s],
            ]
        }
        Builtin::Ms => {
            let s = Amp::isq2();
            let (p0, p1) = (angle(0)?, angle(1)?);
            let mut ph =
                |l: Linear| -> ExactResult<Amp> { Ok(samples.phase(&l)?.mul(&neg_i()).mul(&s)) };
            let b_mm = ph(p0.clone().neg().add(p1.clone().neg()))?;
            let b_mp = ph(p0.clone().neg().add(p1.clone()))?;
            let b_pm = ph(p0.clone().add(p1.clone().neg()))?;
            let b_pp = ph(p0.add(p1))?;
            vec![
                vec![s.clone(), o(), o(), b_mm],
                vec![o(), s.clone(), b_mp, o()],
                vec![o(), b_pm, s.clone(), o()],
                vec![b_pp, o(), o(), s],
            ]
        }
    };
    Ok(m)
}

fn eye(n: usize) -> Vec<Vec<Amp>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { Amp::one() } else { Amp::zero() })
                .collect()
        })
        .collect()
}

fn controlled(u: &[[Amp; 2]; 2]) -> Vec<Vec<Amp>> {
    let mut m = eye(4);
    for i in 0..2 {
        for j in 0..2 {
            m[2 + i][2 + j] = u[i][j].clone();
        }
    }
    m
}

/// One path of the evaluation: an exact amplitude and the concrete basis state.
struct Term {
    amp: Amp,
    /// Bit `i` is the state of qubit index `i`.
    bits: usize,
}

/// Apply one gate to every term, splitting where the gate does.
fn apply_gate(
    terms: Vec<Term>,
    op: &GateOp,
    qubit_index: &FxHashMap<QubitId, usize>,
    samples: &mut Samples,
    registry: &GateRegistry,
) -> ExactResult<Vec<Term>> {
    let m = gate_matrix(op, samples, registry)?;
    let arity = op.qubits.len();
    let positions: Vec<usize> = op
        .qubits
        .iter()
        .map(|q| {
            qubit_index
                .get(q)
                .copied()
                .ok_or_else(|| Unsupported(format!("gate on unknown qubit `{q}`")))
        })
        .collect::<ExactResult<_>>()?;

    let mut out = Vec::with_capacity(terms.len());
    for t in terms {
        // Operand 0 is the most significant bit of the matrix index.
        let mut col = 0usize;
        for (k, &pos) in positions.iter().enumerate() {
            col |= ((t.bits >> pos) & 1) << (arity - 1 - k);
        }
        for (row, m_row) in m.iter().enumerate() {
            let entry = &m_row[col];
            if entry.is_zero() {
                continue;
            }
            let mut bits = t.bits;
            for (k, &pos) in positions.iter().enumerate() {
                let bit = (row >> (arity - 1 - k)) & 1;
                bits = (bits & !(1 << pos)) | (bit << pos);
            }
            out.push(Term {
                amp: t.amp.mul(entry),
                bits,
            });
        }
    }
    Ok(out)
}

/// Evaluate a circuit — optionally with one hole between two halves — on one basis
/// input, exactly. Returns the output amplitude per basis state.
///
/// The hole applies the permutation to the boundary qubits (indices `0..width`, `q0`
/// as the most significant constraint bit, matching [`BasisPermutation`]) and
/// multiplies by the free phase sampled for the boundary state it saw — the strong
/// contract's arbitrary-per-state phase, keyed by the state *at the hole*, which on
/// circuits with `h` before the hole is finer than keying by the circuit input.
#[allow(clippy::too_many_arguments)]
fn eval_on_input(
    before: &[GateOp],
    hole: Option<(&BasisPermutation, usize)>,
    after: &[GateOp],
    qubits: &[QubitId],
    input: usize,
    samples: &mut Samples,
    registry: &GateRegistry,
) -> ExactResult<Vec<Amp>> {
    let qubit_index: FxHashMap<QubitId, usize> = qubits
        .iter()
        .enumerate()
        .map(|(i, q)| (q.clone(), i))
        .collect();

    let mut terms = vec![Term {
        amp: Amp::one(),
        bits: input,
    }];
    for op in before {
        terms = apply_gate(terms, op, &qubit_index, samples, registry)?;
    }
    if let Some((perm, width)) = hole {
        for t in &mut terms {
            let mut state = 0usize;
            for j in 0..width {
                state |= ((t.bits >> j) & 1) << (width - 1 - j);
            }
            t.amp = t.amp.mul(&samples.hole_phase(state));
            let image = perm.apply(state);
            for j in 0..width {
                let bit = (image >> (width - 1 - j)) & 1;
                t.bits = (t.bits & !(1 << j)) | (bit << j);
            }
        }
    }
    for op in after {
        terms = apply_gate(terms, op, &qubit_index, samples, registry)?;
    }

    let mut out = vec![Amp::zero(); 1 << qubits.len()];
    for t in terms {
        out[t.bits] = out[t.bits].add(&t.amp);
    }
    Ok(out)
}

/// One side of a comparison: a circuit, possibly split around a hole.
pub(crate) struct Side<'a> {
    pub before: &'a [GateOp],
    pub after: &'a [GateOp],
}

impl<'a> Side<'a> {
    pub fn plain(ops: &'a [GateOp]) -> Self {
        Self {
            before: ops,
            after: &[],
        }
    }
}

/// Compare two circuits exactly at `rounds` independent random rational points.
///
/// Returns `Ok(true)` only if every amplitude of both sides agrees exactly, on every
/// basis input, in every round — under [`Equivalence::UpToPhase`], after dividing out
/// one global phase whose unimodularity is itself checked exactly. `Err` means a gate
/// or angle fell outside the exact fragment; the caller must treat that as "not
/// verified", never as "probably fine".
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify(
    a: &Side<'_>,
    b: &Side<'_>,
    hole: Option<(&BasisPermutation, usize)>,
    qubits: &[QubitId],
    registry: &GateRegistry,
    equivalence: Equivalence,
    rounds: usize,
    seed: u64,
) -> ExactResult<bool> {
    let n = qubits.len();
    for round in 0..rounds.max(1) {
        let mut samples = Samples::new(
            seed.wrapping_mul(0x2545_f491_4f6c_dd1d)
                .wrapping_add(round as u64 + 1),
        );
        // One global phase for the whole matrix, fixed by the first nonzero entry.
        let mut ratio: Option<Amp> = None;
        for input in 0..(1usize << n) {
            let va = eval_on_input(
                a.before,
                hole,
                a.after,
                qubits,
                input,
                &mut samples,
                registry,
            )?;
            let vb = eval_on_input(
                b.before,
                hole,
                b.after,
                qubits,
                input,
                &mut samples,
                registry,
            )?;
            for (x, y) in va.iter().zip(vb.iter()) {
                match equivalence {
                    Equivalence::Exact => {
                        if x != y {
                            return Ok(false);
                        }
                    }
                    Equivalence::UpToPhase => match &ratio {
                        Some(r) => {
                            if y != &x.mul(r) {
                                return Ok(false);
                            }
                        }
                        None => {
                            match (x.is_zero(), y.is_zero()) {
                                (true, true) => {}
                                (false, false) => {
                                    let r = y.div(x);
                                    // The phase must be exactly unimodular.
                                    if r.mul(&r.conj()) != Amp::one() {
                                        return Ok(false);
                                    }
                                    ratio = Some(r);
                                }
                                _ => return Ok(false),
                            }
                        }
                    },
                }
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcircuit::qasm;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn qubits(n: usize) -> Vec<QubitId> {
        (0..n).map(|i| qcircuit::intern(&format!("q{i}"))).collect()
    }

    fn ops(src: &str) -> Vec<GateOp> {
        let dag = qasm::parse(src).unwrap();
        dag.topological_gates()
            .into_iter()
            .map(|n| dag.gate(n).clone())
            .collect()
    }

    fn check(a: &str, b: &str, n: usize, eq: Equivalence) -> bool {
        let (oa, ob) = (ops(a), ops(b));
        verify(
            &Side::plain(&oa),
            &Side::plain(&ob),
            None,
            &qubits(n),
            &reg(),
            eq,
            2,
            7,
        )
        .unwrap()
    }

    #[test]
    fn field_arithmetic_is_exact() {
        // (1/sqrt2)^2 = 1/2, with no epsilon anywhere.
        let half = Amp::isq2().mul(&Amp::isq2());
        assert_eq!(
            half,
            Amp::from_crat(CRat::new(
                BigRational::new(BigInt::from(1), BigInt::from(2)),
                BigRational::zero()
            ))
        );
        // Eighth roots multiply as exponents add, and the eighth power is 1.
        let w = Amp::eighth_root(1);
        let mut acc = Amp::one();
        for k in 1..=8 {
            acc = acc.mul(&w);
            assert_eq!(acc, Amp::eighth_root(k));
        }
        assert_eq!(acc, Amp::one());
        // Field inverse round-trips.
        let x = Amp::eighth_root(3).add(&Amp::isq2());
        assert_eq!(x.mul(&x.inv()), Amp::one());
    }

    #[test]
    fn sampled_unit_points_are_exactly_unimodular() {
        let mut s = Samples::new(99);
        for _ in 0..10 {
            let u = s.unit_point();
            assert_eq!(u.mul(&u.conj()), CRat::one());
        }
    }

    /// The exact gate table must agree with the registry's floating-point matrices —
    /// this is what pins the two semantics together, so a convention drift (a global
    /// phase, a transposed operand) cannot open a gap between what the optimizer
    /// applies and what synthesis verified.
    #[test]
    fn exact_gate_matrices_agree_with_the_registry() {
        use num_traits::ToPrimitive;
        let registry = reg();
        let mut samples = Samples::new(3);
        let theta = AngleExpr::var("theta1");
        let phi = AngleExpr::var("theta2");
        let lam = AngleExpr::var("theta3");
        let cases: Vec<GateOp> = vec![
            GateOp::new("h", vec!["q0"], []),
            GateOp::new("x", vec!["q0"], []),
            GateOp::new("y", vec!["q0"], []),
            GateOp::new("z", vec!["q0"], []),
            GateOp::new("s", vec!["q0"], []),
            GateOp::new("sdg", vec!["q0"], []),
            GateOp::new("t", vec!["q0"], []),
            GateOp::new("tdg", vec!["q0"], []),
            GateOp::new("sx", vec!["q0"], []),
            GateOp::new("sxdg", vec!["q0"], []),
            GateOp::new("rx", vec!["q0"], [theta.clone()]),
            GateOp::new("ry", vec!["q0"], [theta.clone()]),
            GateOp::new("rz", vec!["q0"], [theta.clone()]),
            GateOp::new("u1", vec!["q0"], [theta.clone()]),
            GateOp::new("u2", vec!["q0"], [phi.clone(), lam.clone()]),
            GateOp::new("u3", vec!["q0"], [theta.clone(), phi.clone(), lam.clone()]),
            GateOp::new("cx", vec!["q0", "q1"], []),
            GateOp::new("cz", vec!["q0", "q1"], []),
            GateOp::new("swap", vec!["q0", "q1"], []),
            GateOp::new("rxx", vec!["q0", "q1"], [theta.clone()]),
            GateOp::new("ryy", vec!["q0", "q1"], [theta.clone()]),
            GateOp::new("rzz", vec!["q0", "q1"], [theta.clone()]),
            GateOp::new("ccx", vec!["q0", "q1", "q2"], []),
            GateOp::new("ccz", vec!["q0", "q1", "q2"], []),
            GateOp::new("gpi", vec!["q0"], [theta.clone()]),
            GateOp::new("gpi2", vec!["q0"], [theta.clone()]),
            GateOp::new("ms", vec!["q0", "q1"], [theta.clone(), phi.clone()]),
        ];
        for op in cases {
            let exact = gate_matrix(&op, &mut samples, &registry).unwrap();
            // The float angles the samples correspond to: recover theta from the
            // sampled half point u = e^{i theta/2}.
            let mut angle_of = |name: &str| -> f64 {
                let u = samples.half_point(name);
                2.0 * f64::atan2(u.im.to_f64().unwrap(), u.re.to_f64().unwrap())
            };
            let params: Vec<f64> = op
                .params
                .iter()
                .map(|p| match p {
                    AngleExpr::Var(v) => angle_of(v),
                    _ => unreachable!(),
                })
                .collect();
            let float = registry
                .get(&op.gate)
                .unwrap()
                .matrix(&params, &registry)
                .unwrap();
            for (i, row) in exact.iter().enumerate() {
                for (j, e) in row.iter().enumerate() {
                    let sqrt2 = 2f64.sqrt();
                    let re = e.a.re.to_f64().unwrap() + sqrt2 * e.b.re.to_f64().unwrap();
                    let im = e.a.im.to_f64().unwrap() + sqrt2 * e.b.im.to_f64().unwrap();
                    let f = float[[i, j]];
                    assert!(
                        (re - f.re).abs() < 1e-9 && (im - f.im).abs() < 1e-9,
                        "`{}`[{i}][{j}]: exact ({re}, {im}) vs registry ({}, {})",
                        op.gate,
                        f.re,
                        f.im
                    );
                }
            }
        }
    }

    #[test]
    fn accepts_true_identities() {
        assert!(check("h q0; h q0;", "", 1, Equivalence::Exact));
        assert!(check("t q0; t q0;", "s q0;", 1, Equivalence::Exact));
        assert!(check(
            "cz q0, q1;",
            "h q1; cx q0, q1; h q1;",
            2,
            Equivalence::Exact
        ));
        assert!(check(
            "rz(theta1) q0; cx q0, q1;",
            "cx q0, q1; rz(theta1) q0;",
            2,
            Equivalence::Exact
        ));
        assert!(check(
            "rz(theta1) q0; rz(theta2) q0;",
            "rz((theta1+theta2)) q0;",
            1,
            Equivalence::Exact
        ));
        // The rx family, which the reference's rational mode could not evaluate.
        assert!(check(
            "rx(theta1) q0; rx(theta2) q0;",
            "rx((theta1+theta2)) q0;",
            1,
            Equivalence::Exact
        ));
    }

    #[test]
    fn rejects_non_identities() {
        assert!(!check("h q0;", "x q0;", 1, Equivalence::Exact));
        assert!(!check("t q0;", "tdg q0;", 1, Equivalence::Exact));
        assert!(!check(
            "rz(theta1) q0;",
            "rz(theta2) q0;",
            1,
            Equivalence::Exact
        ));
        assert!(!check("cx q0, q1;", "cx q1, q0;", 2, Equivalence::Exact));
    }

    /// `s` and `rz(pi/2)` differ by exactly `e^{i pi/4}`: the equivalence notions must
    /// split them, and the up-to-phase ratio check must confirm unimodularity exactly.
    #[test]
    fn global_phase_distinguishes_the_equivalence_notions() {
        let a = "s q0;";
        let b = "rz((pi/2)) q0;";
        assert!(!check(a, b, 1, Equivalence::Exact));
        assert!(check(a, b, 1, Equivalence::UpToPhase));
    }

    /// The strong contract at work: a rule that survives arbitrary per-state hole
    /// phases holds for every region with the permutation's support, so it needs no
    /// per-application soundness check. `rz` commutes with any such region on its own
    /// wire's diagonal; `cx` does not.
    #[test]
    fn hole_phases_enforce_the_strong_contract() {
        let registry = reg();
        let q = qubits(2);
        let identity = BasisPermutation::identity(2);

        // rz(theta1) q0; hole  ==  hole; rz(theta1) q0  under the identity permutation:
        // safe for any diagonal-carrying region.
        let rz = ops("rz(theta1) q0;");
        let a = Side {
            before: &rz,
            after: &[],
        };
        let b = Side {
            before: &[],
            after: &rz,
        };
        assert!(verify(
            &a,
            &b,
            Some((&identity, 2)),
            &q,
            &registry,
            Equivalence::UpToPhase,
            2,
            11
        )
        .unwrap());

        // cx; hole == hole; cx under the identity permutation is NOT phase-safe: a
        // diagonal region (cz) with identity support breaks it. Free per-state phases
        // must refute it.
        let cx = ops("cx q0, q1;");
        let a = Side {
            before: &cx,
            after: &[],
        };
        let b = Side {
            before: &[],
            after: &cx,
        };
        assert!(!verify(
            &a,
            &b,
            Some((&identity, 2)),
            &q,
            &registry,
            Equivalence::UpToPhase,
            2,
            11
        )
        .unwrap());
    }

    #[test]
    fn unsupported_angles_are_refused_not_guessed() {
        let o = ops("rz(0.5) q0;");
        let err = verify(
            &Side::plain(&o),
            &Side::plain(&o),
            None,
            &qubits(1),
            &reg(),
            Equivalence::Exact,
            1,
            1,
        );
        assert!(err.is_err(), "a decimal angle has no exact phase");
    }
}
