//! QASM serialization.

use crate::angle::AngleExpr;
use crate::dag::{Dag, GateOp, QubitId};
use std::fmt::Write as _;
use std::sync::Arc;

/// A qubit renaming applied while printing.
pub type Renamer = Arc<dyn Fn(&str) -> Option<QubitId> + Send + Sync>;

/// How to render a circuit back to QASM text.
#[derive(Clone)]
pub struct PrintOptions {
    /// Emit `OPENQASM`/`include`/`qreg` lines. Rewrite rules are printed without them.
    pub header: bool,
    /// Rename each qubit on the way out.
    pub rename: Option<Renamer>,
}

impl std::fmt::Debug for PrintOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrintOptions")
            .field("header", &self.header)
            .field("rename", &self.rename.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            header: true,
            rename: None,
        }
    }
}

impl PrintOptions {
    /// Gates only, no header — the form used for rewrite rules.
    pub fn bare() -> Self {
        Self {
            header: false,
            rename: None,
        }
    }
}

/// Render `dag` as QASM with default options.
pub fn to_qasm(dag: &Dag) -> String {
    to_qasm_with(dag, &PrintOptions::default())
}

/// Render `dag` as QASM.
pub fn to_qasm_with(dag: &Dag, opts: &PrintOptions) -> String {
    let mut out = String::new();
    if opts.header {
        let version = dag.header.version.clone().unwrap_or_else(|| "2.0".into());
        let _ = writeln!(out, "OPENQASM {version};");
        for inc in &dag.header.includes {
            let _ = writeln!(out, "include \"{inc}\";");
        }
        if dag.header.qregs.is_empty() {
            if dag.num_qubits() > 0 {
                let _ = writeln!(out, "qreg q[{}];", dag.num_qubits());
            }
        } else {
            for (name, size) in &dag.header.qregs {
                let _ = writeln!(out, "qreg {name}[{size}];");
            }
        }
        for (name, size) in &dag.header.cregs {
            let _ = writeln!(out, "creg {name}[{size}];");
        }
    }
    for idx in dag.topological_gates() {
        let _ = writeln!(out, "{};", format_gate(dag.gate(idx), opts));
    }
    out
}

/// Render one gate call without its trailing semicolon.
///
/// Handles any arity. The reference implementation's `gateNodeToQasm` enumerated the
/// cases by hand — one branch for `isCX`, one for `isCCZ`, and three more for one-qubit
/// gates with one, two, or three angles — and threw `RuntimeException("angles")` for
/// anything else.
pub fn format_gate(op: &GateOp, opts: &PrintOptions) -> String {
    let mut s = String::new();
    s.push_str(&op.gate);
    if !op.params.is_empty() {
        s.push('(');
        for (i, p) in op.params.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format_angle(p));
        }
        s.push(')');
    }
    s.push(' ');
    for (i, q) in op.qubits.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let name = match &opts.rename {
            Some(f) => f(q).unwrap_or_else(|| q.clone()),
            None => q.clone(),
        };
        s.push_str(&name);
    }
    s
}

/// Render an angle so that it re-parses to the same value.
///
/// A numeric literal prints as the shortest decimal that round-trips exactly (`{:?}`).
/// Anything with structure keeps its structure: `pi/2` stays `(pi/2)` rather than
/// becoming `1.5707963267948966`. Folding constants to decimals read worse and cost
/// real coverage — the synthesizer's output is compared textually against the
/// reference's rule files, whose fixed angles are spelled symbolically, so every
/// `rx`-carrying rigetti rule looked novel when it was actually a reproduction.
pub fn format_angle(a: &AngleExpr) -> String {
    match a {
        AngleExpr::Num(v) => format!("{v:?}"),
        _ => a.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qasm::parse;
    use std::f64::consts::PI;

    #[test]
    fn round_trips_a_circuit() {
        let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nh q[0];\ncx q[0], q[1];\nrz(0.25) q[2];\n";
        let a = parse(src).unwrap();
        let text = to_qasm(&a);
        let b = parse(&text).unwrap();
        assert_eq!(to_qasm(&b), text, "printing is idempotent");
        assert_eq!(a.gate_count(), b.gate_count());
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn angles_round_trip_exactly() {
        for v in [PI / 4.0, -PI / 4.0, 1.0 / 3.0, 1e-12, 12345.6789, 0.0] {
            let src = format!("rz({v:?}) q0;");
            let dag = parse(&src).unwrap();
            let got = dag.gate(dag.topological_gates()[0]).params[0]
                .eval()
                .unwrap();
            assert_eq!(got, v, "angle {v} did not round-trip");
        }
    }

    #[test]
    fn symbolic_angles_round_trip() {
        let dag = parse("rz((theta1+theta2)) q0;").unwrap();
        let text = to_qasm_with(&dag, &PrintOptions::bare());
        assert_eq!(text.trim(), "rz((theta1+theta2)) q0;");
        let back = parse(&text).unwrap();
        assert_eq!(
            back.gate(back.topological_gates()[0]).params[0].free_vars(),
            vec!["theta1", "theta2"]
        );
    }

    #[test]
    fn any_arity_prints() {
        let dag = parse("ccz a,b,c; wide a,b,c,d,e,f;").unwrap();
        let text = to_qasm_with(&dag, &PrintOptions::bare());
        assert!(text.contains("ccz a, b, c;"));
        assert!(text.contains("wide a, b, c, d, e, f;"));
    }

    #[test]
    fn bare_mode_omits_header() {
        let dag = parse("OPENQASM 2.0;\nqreg q[1];\nh q[0];").unwrap();
        let text = to_qasm_with(&dag, &PrintOptions::bare());
        assert_eq!(text.trim(), "h q[0];");
    }

    #[test]
    fn header_is_reconstructed() {
        let dag = parse("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];")
            .unwrap();
        let text = to_qasm(&dag);
        assert!(
            text.starts_with("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n")
        );
    }

    #[test]
    fn rename_applies_on_output() {
        let dag = parse("cx q[0], q[1];").unwrap();
        let opts = PrintOptions {
            header: false,
            rename: Some(Arc::new(|q: &str| match q {
                "q[0]" => Some(crate::dag::intern("a")),
                "q[1]" => Some(crate::dag::intern("b")),
                _ => None,
            })),
        };
        assert_eq!(to_qasm_with(&dag, &opts).trim(), "cx a, b;");
    }

    #[test]
    fn empty_circuit_prints_header_only() {
        let dag = parse("OPENQASM 2.0;\nqreg q[2];").unwrap();
        let text = to_qasm(&dag);
        assert_eq!(text, "OPENQASM 2.0;\nqreg q[2];\n");
    }
}
