//! Recursive-descent parser for the OpenQASM 2.0 / 3.0 subset GUOQ uses.
//!
//! Replaces the reference implementation's two ANTLR grammar files and their generated
//! visitors. Besides removing a code-generation step, this parser is *neutral*: it does
//! not silently drop gates. The reference's `NodeVisitor` returned `null` for any gate
//! whose angles were all multiples of `4*PI`, which quietly rewrote rule patterns as they
//! were read and forced `u2` to be exempted by name. Identity removal is now an explicit
//! pass, [`Dag::drop_identity_gates`].
//!
//! [`Dag::drop_identity_gates`]: crate::dag::Dag::drop_identity_gates

use super::lexer::{lex, Tok, Token};
use crate::angle::AngleExpr;
use crate::dag::{intern, Dag, GateOp, QasmHeader, QubitId};
use crate::error::{CircuitError, Result};

/// A qubit renaming applied while parsing operands.
pub type Renamer<'a> = &'a dyn Fn(&str) -> Option<QubitId>;

/// Parse a QASM program into a [`Dag`].
///
/// Bare identifiers are accepted as qubit names so that rewrite rules such as
/// `cx q0, q1; h q1;` parse with the same code path as full circuits.
pub fn parse(src: &str) -> Result<Dag> {
    Parser::new(src)?.program()
}

/// Parse with a qubit renaming applied as operands are read.
///
/// Used when a resynthesis backend returns a circuit over its own register naming that
/// must be mapped back onto the original circuit's qubits.
pub fn parse_with_rename(src: &str, rename: Renamer<'_>) -> Result<Dag> {
    let mut p = Parser::new(src)?;
    p.rename = Some(rename);
    p.program()
}

struct Parser<'a> {
    toks: Vec<Token>,
    pos: usize,
    rename: Option<Renamer<'a>>,
}

impl<'a> Parser<'a> {
    fn new(src: &str) -> Result<Self> {
        Ok(Self {
            toks: lex(src)?,
            pos: 0,
            rename: None,
        })
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        let t = &self.toks[self.pos];
        Err(CircuitError::Parse {
            line: t.line,
            col: t.col,
            msg: msg.into(),
        })
    }

    fn expect(&mut self, want: &Tok) -> Result<()> {
        if self.peek() == want {
            self.bump();
            Ok(())
        } else {
            self.err(format!("expected {want:?}, found {:?}", self.peek()))
        }
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if self.peek() == want {
            self.bump();
            true
        } else {
            false
        }
    }

    fn program(mut self) -> Result<Dag> {
        let mut header = QasmHeader::default();
        let mut ops: Vec<GateOp> = Vec::new();

        loop {
            match self.peek().clone() {
                Tok::Eof => break,
                Tok::Semi => {
                    self.bump();
                }
                Tok::Ident(kw) => match kw.as_str() {
                    "OPENQASM" => {
                        self.bump();
                        header.version = Some(self.version_text()?);
                        self.eat(&Tok::Semi);
                    }
                    "include" => {
                        self.bump();
                        match self.bump() {
                            Tok::String(s) => header.includes.push(s),
                            other => {
                                return self.err(format!("expected include path, found {other:?}"))
                            }
                        }
                        self.eat(&Tok::Semi);
                    }
                    "qreg" | "creg" => {
                        self.bump();
                        let (name, size) = self.register_decl()?;
                        if kw == "qreg" {
                            header.qregs.push((name, size));
                        } else {
                            header.cregs.push((name, size));
                        }
                    }
                    // OpenQASM 3 spellings: `qubit[5] q;` / `bit[5] c;`
                    "qubit" | "bit" => {
                        self.bump();
                        let (name, size) = self.typed_decl()?;
                        if kw == "qubit" {
                            header.qregs.push((name, size));
                        } else {
                            header.cregs.push((name, size));
                        }
                    }
                    // Statements we accept and skip: they do not affect the unitary that
                    // GUOQ optimises, but rejecting them would break real input files.
                    "barrier" | "measure" | "reset" | "if" | "gate" | "opaque" => {
                        self.skip_statement();
                    }
                    _ => {
                        let op = self.gate_call()?;
                        ops.push(op);
                    }
                },
                other => return self.err(format!("unexpected token {other:?}")),
            }
        }

        // Declared qubits first, in declaration order, so an empty circuit still knows
        // its width and the printer can reproduce the register layout.
        let mut dag = Dag::new(Vec::<String>::new());
        for (name, size) in &header.qregs {
            for i in 0..*size {
                let raw = format!("{name}[{i}]");
                dag.ensure_qubit(&self.apply_rename(&raw));
            }
        }
        for op in ops {
            dag.push_gate(op);
        }
        dag.header = header;
        Ok(dag)
    }

    fn apply_rename(&self, raw: &str) -> QubitId {
        match self.rename {
            Some(f) => f(raw).unwrap_or_else(|| intern(raw)),
            None => intern(raw),
        }
    }

    /// `OPENQASM 2.0;` / `OPENQASM 3;`
    ///
    /// The lexer reads `2.0` as a single float, so the dotted form is reconstructed
    /// here rather than being reassembled from separate tokens downstream.
    fn version_text(&mut self) -> Result<String> {
        match self.peek().clone() {
            Tok::Int(v) => {
                self.bump();
                Ok(v.to_string())
            }
            Tok::Float(v) => {
                self.bump();
                Ok(if v.fract() == 0.0 {
                    format!("{v:.1}")
                } else {
                    v.to_string()
                })
            }
            _ => self.err("expected a version number after OPENQASM"),
        }
    }

    /// `qreg q[5];`
    fn register_decl(&mut self) -> Result<(String, usize)> {
        let name = match self.bump() {
            Tok::Ident(n) => n,
            other => return self.err(format!("expected register name, found {other:?}")),
        };
        let mut size = 1;
        if self.eat(&Tok::LBracket) {
            size = match self.bump() {
                Tok::Int(v) => v as usize,
                other => return self.err(format!("expected register size, found {other:?}")),
            };
            self.expect(&Tok::RBracket)?;
        }
        self.eat(&Tok::Semi);
        Ok((name, size))
    }

    /// `qubit[5] q;`
    fn typed_decl(&mut self) -> Result<(String, usize)> {
        let mut size = 1;
        if self.eat(&Tok::LBracket) {
            size = match self.bump() {
                Tok::Int(v) => v as usize,
                other => return self.err(format!("expected register size, found {other:?}")),
            };
            self.expect(&Tok::RBracket)?;
        }
        let name = match self.bump() {
            Tok::Ident(n) => n,
            other => return self.err(format!("expected register name, found {other:?}")),
        };
        self.eat(&Tok::Semi);
        Ok((name, size))
    }

    fn skip_statement(&mut self) {
        // `gate`/`opaque` bodies are brace-delimited in QASM but the lexer does not emit
        // braces, so those declarations are skipped to the next semicolon like the rest.
        while !matches!(self.peek(), Tok::Semi | Tok::Eof) {
            self.bump();
        }
        self.eat(&Tok::Semi);
    }

    /// `name(params) operand, operand;`
    fn gate_call(&mut self) -> Result<GateOp> {
        let gate = match self.bump() {
            Tok::Ident(n) => n,
            other => return self.err(format!("expected gate name, found {other:?}")),
        };

        let mut params = Vec::new();
        if self.eat(&Tok::LParen) && !self.eat(&Tok::RParen) {
            loop {
                params.push(self.expr()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                self.expect(&Tok::RParen)?;
                break;
            }
        }

        let mut qubits = Vec::new();
        loop {
            if matches!(self.peek(), Tok::Semi | Tok::Eof) {
                break;
            }
            qubits.push(self.operand()?);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.eat(&Tok::Semi);

        if qubits.is_empty() {
            return self.err(format!("gate `{gate}` has no operands"));
        }
        Ok(GateOp {
            gate: intern(&gate),
            qubits,
            params,
        })
    }

    /// `q[3]` or a bare `q0` (the form rewrite rules use).
    fn operand(&mut self) -> Result<QubitId> {
        let name = match self.bump() {
            Tok::Ident(n) => n,
            other => return self.err(format!("expected qubit operand, found {other:?}")),
        };
        let raw = if self.eat(&Tok::LBracket) {
            let idx = match self.bump() {
                Tok::Int(v) => v,
                other => return self.err(format!("expected qubit index, found {other:?}")),
            };
            self.expect(&Tok::RBracket)?;
            format!("{name}[{idx}]")
        } else {
            name
        };
        Ok(self.apply_rename(&raw))
    }

    // Expression grammar, lowest precedence first.
    fn expr(&mut self) -> Result<AngleExpr> {
        self.additive()
    }

    fn additive(&mut self) -> Result<AngleExpr> {
        let mut lhs = self.multiplicative()?;
        loop {
            if self.eat(&Tok::Plus) {
                lhs = AngleExpr::add(lhs, self.multiplicative()?);
            } else if self.eat(&Tok::Minus) {
                lhs = AngleExpr::sub(lhs, self.multiplicative()?);
            } else {
                return Ok(lhs);
            }
        }
    }

    fn multiplicative(&mut self) -> Result<AngleExpr> {
        let mut lhs = self.unary()?;
        loop {
            if self.eat(&Tok::Star) {
                lhs = AngleExpr::mul(lhs, self.unary()?);
            } else if self.eat(&Tok::Slash) {
                lhs = AngleExpr::div(lhs, self.unary()?);
            } else {
                return Ok(lhs);
            }
        }
    }

    fn unary(&mut self) -> Result<AngleExpr> {
        if self.eat(&Tok::Minus) {
            return Ok(AngleExpr::neg(self.unary()?));
        }
        if self.eat(&Tok::Plus) {
            return self.unary();
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<AngleExpr> {
        match self.bump() {
            Tok::Int(v) => Ok(AngleExpr::Num(v as f64)),
            Tok::Float(v) => Ok(AngleExpr::Num(v)),
            Tok::Ident(name) => match name.as_str() {
                "pi" | "PI" | "Pi" => Ok(AngleExpr::Pi),
                "tau" | "TAU" => Ok(AngleExpr::mul(AngleExpr::Num(2.0), AngleExpr::Pi)),
                "euler" | "E" => Ok(AngleExpr::Num(std::f64::consts::E)),
                _ => Ok(AngleExpr::var(name)),
            },
            Tok::LParen => {
                let e = self.expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            other => {
                self.pos = self.pos.saturating_sub(1);
                self.err(format!("expected an expression, found {other:?}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn gates(dag: &Dag) -> Vec<String> {
        dag.topological_gates()
            .into_iter()
            .map(|i| {
                let g = dag.gate(i);
                let p = if g.params.is_empty() {
                    String::new()
                } else {
                    format!(
                        "({})",
                        g.params
                            .iter()
                            .map(|e| format!("{:.6}", e.eval().unwrap_or(f64::NAN)))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                };
                format!("{}{} {}", g.gate, p, g.qubits.join(","))
            })
            .collect()
    }

    #[test]
    fn parses_a_full_qasm2_file() {
        let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nh q[0];\ncx q[0], q[1];\n";
        let dag = parse(src).unwrap();
        assert_eq!(dag.header.version.as_deref(), Some("2.0"));
        assert_eq!(dag.header.includes, vec!["qelib1.inc".to_string()]);
        assert_eq!(dag.header.qregs, vec![("q".to_string(), 3)]);
        assert_eq!(dag.num_qubits(), 3);
        assert_eq!(gates(&dag), vec!["h q[0]", "cx q[0],q[1]"]);
    }

    #[test]
    fn parses_bare_qubit_names_used_by_rules() {
        let dag = parse("cx q0, q1; h q1;").unwrap();
        assert_eq!(gates(&dag), vec!["cx q0,q1", "h q1"]);
        assert_eq!(dag.num_qubits(), 2);
    }

    #[test]
    fn parses_angle_expressions() {
        let dag = parse("rz(-pi/4) q[0]; rz(pi*3/2) q[0]; rz(2) q[0];").unwrap();
        let g = gates(&dag);
        assert_eq!(g[0], format!("rz({:.6}) q[0]", -PI / 4.0));
        assert_eq!(g[1], format!("rz({:.6}) q[0]", PI * 3.0 / 2.0));
        assert_eq!(g[2], "rz(2.000000) q[0]");
    }

    #[test]
    fn operator_precedence_and_associativity() {
        let one = |s: &str| {
            let d = parse(s).unwrap();
            let i = d.topological_gates()[0];
            d.gate(i).params[0].eval().unwrap()
        };
        assert!((one("rz(1+2*3) q0;") - 7.0).abs() < 1e-12);
        assert!((one("rz((1+2)*3) q0;") - 9.0).abs() < 1e-12);
        assert!((one("rz(8/4/2) q0;") - 1.0).abs() < 1e-12);
        assert!((one("rz(8-4-2) q0;") - 2.0).abs() < 1e-12);
        assert!((one("rz(-2+5) q0;") - 3.0).abs() < 1e-12);
        assert!((one("rz(--3) q0;") - 3.0).abs() < 1e-12);
    }

    #[test]
    fn parses_symbolic_rule_angles() {
        let dag = parse("rz(theta1) q0; rz((theta1+theta2)) q1;").unwrap();
        let idx = dag.topological_gates();
        assert_eq!(dag.gate(idx[0]).params[0].free_vars(), vec!["theta1"]);
        assert_eq!(
            dag.gate(idx[1]).params[0].free_vars(),
            vec!["theta1", "theta2"]
        );
    }

    #[test]
    fn parses_multi_param_gates() {
        let dag = parse("u3(0.1,0.2,0.3) q[0]; u2(0,pi) q[1];").unwrap();
        let idx = dag.topological_gates();
        assert_eq!(dag.gate(idx[0]).params.len(), 3);
        assert_eq!(dag.gate(idx[1]).params.len(), 2);
    }

    #[test]
    fn parses_arbitrary_arity_gate_calls() {
        let dag = parse("ccz q[0], q[1], q[2]; wide a,b,c,d,e;").unwrap();
        let idx = dag.topological_gates();
        assert_eq!(dag.gate(idx[0]).qubits.len(), 3);
        assert_eq!(dag.gate(idx[1]).qubits.len(), 5);
    }

    /// The parser is semantics-neutral: no gate is dropped, however trivial its angles.
    #[test]
    fn parser_does_not_drop_gates() {
        let dag = parse("rz(0) q[0]; u2(0,0) q[0]; rz(2*pi) q[0];").unwrap();
        assert_eq!(dag.gate_count(), 3, "parser must be semantics-neutral");
    }

    #[test]
    fn skips_non_unitary_statements() {
        let src = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh q[0];\nbarrier q[0], q[1];\nmeasure q[0] -> c[0];\nreset q[1];\n";
        let dag = parse(src).unwrap();
        assert_eq!(gates(&dag), vec!["h q[0]"]);
        assert_eq!(dag.header.cregs, vec![("c".to_string(), 2)]);
    }

    #[test]
    fn handles_qasm3_declarations() {
        let dag = parse("OPENQASM 3.0;\nqubit[4] q;\nbit[4] c;\nh q[2];").unwrap();
        assert_eq!(dag.header.qregs, vec![("q".to_string(), 4)]);
        assert_eq!(dag.num_qubits(), 4);
        assert_eq!(gates(&dag), vec!["h q[2]"]);
    }

    #[test]
    fn declared_qubits_exist_even_without_gates() {
        let dag = parse("OPENQASM 2.0;\nqreg q[5];\n").unwrap();
        assert_eq!(dag.num_qubits(), 5);
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn qubit_order_follows_declaration() {
        let dag = parse("qreg q[3];\nh q[2];").unwrap();
        assert_eq!(
            dag.qubits(),
            &[intern("q[0]"), intern("q[1]"), intern("q[2]")]
        );
    }

    #[test]
    fn empty_and_whitespace_inputs() {
        assert_eq!(parse("").unwrap().gate_count(), 0);
        assert_eq!(parse("   \n\t ").unwrap().gate_count(), 0);
        assert_eq!(parse(";").unwrap().gate_count(), 0);
        assert_eq!(parse(";;;").unwrap().gate_count(), 0);
    }

    #[test]
    fn trailing_statement_without_semicolon() {
        let dag = parse("h q0; x q1").unwrap();
        assert_eq!(gates(&dag), vec!["h q0", "x q1"]);
    }

    #[test]
    fn rename_maps_operands() {
        let map = |s: &str| match s {
            "q[0]" => Some(intern("orig[7]")),
            "q[1]" => Some(intern("orig[9]")),
            _ => None,
        };
        let dag = parse_with_rename("qreg q[2];\ncx q[0], q[1];", &map).unwrap();
        assert_eq!(gates(&dag), vec!["cx orig[7],orig[9]"]);
        assert_eq!(dag.qubits(), &[intern("orig[7]"), intern("orig[9]")]);
    }

    #[test]
    fn errors_carry_position() {
        let e = parse("h q[0]; cx )").unwrap_err();
        match e {
            CircuitError::Parse { line, col, .. } => {
                assert_eq!(line, 1);
                assert!(col > 8);
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(parse("rz( q0;").is_err());
        assert!(parse("qreg q[;").is_err());
        assert!(parse("h q[0").is_err());
    }

    #[test]
    fn comments_are_ignored() {
        let dag = parse("// leading\nh q0; // trailing\n/* block */ x q0;").unwrap();
        assert_eq!(gates(&dag), vec!["h q0", "x q0"]);
    }
}
