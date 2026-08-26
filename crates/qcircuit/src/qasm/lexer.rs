//! Tokenizer for the OpenQASM subset used by GUOQ.

use crate::error::{CircuitError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(u64),
    Float(f64),
    String(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Arrow,
    Eq,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
    pub col: usize,
}

pub fn lex(src: &str) -> Result<Vec<Token>> {
    let mut out = Vec::new();
    let bytes: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut line = 1;
    let mut col = 1;

    macro_rules! push {
        ($t:expr, $len:expr) => {{
            out.push(Token { tok: $t, line, col });
            i += $len;
            col += $len;
        }};
    }

    while i < bytes.len() {
        let ch = bytes[i];
        match ch {
            '\n' => {
                i += 1;
                line += 1;
                col = 1;
            }
            c if c.is_whitespace() => {
                i += 1;
                col += 1;
            }
            '/' if bytes.get(i + 1) == Some(&'/') => {
                while i < bytes.len() && bytes[i] != '\n' {
                    i += 1;
                }
            }
            '/' if bytes.get(i + 1) == Some(&'*') => {
                i += 2;
                col += 2;
                loop {
                    if i + 1 >= bytes.len() {
                        return Err(CircuitError::Parse {
                            line,
                            col,
                            msg: "unterminated block comment".into(),
                        });
                    }
                    if bytes[i] == '*' && bytes[i + 1] == '/' {
                        i += 2;
                        col += 2;
                        break;
                    }
                    if bytes[i] == '\n' {
                        line += 1;
                        col = 1;
                    } else {
                        col += 1;
                    }
                    i += 1;
                }
            }
            '(' => push!(Tok::LParen, 1),
            ')' => push!(Tok::RParen, 1),
            '[' => push!(Tok::LBracket, 1),
            ']' => push!(Tok::RBracket, 1),
            ',' => push!(Tok::Comma, 1),
            ';' => push!(Tok::Semi, 1),
            '+' => push!(Tok::Plus, 1),
            '*' => push!(Tok::Star, 1),
            '/' => push!(Tok::Slash, 1),
            '^' => push!(Tok::Caret, 1),
            '=' => push!(Tok::Eq, 1),
            '-' => {
                if bytes.get(i + 1) == Some(&'>') {
                    push!(Tok::Arrow, 2)
                } else {
                    push!(Tok::Minus, 1)
                }
            }
            '"' | '\'' => {
                let quote = ch;
                let start_col = col;
                let mut j = i + 1;
                let mut s = String::new();
                while j < bytes.len() && bytes[j] != quote {
                    s.push(bytes[j]);
                    j += 1;
                }
                if j >= bytes.len() {
                    return Err(CircuitError::Parse {
                        line,
                        col: start_col,
                        msg: "unterminated string".into(),
                    });
                }
                out.push(Token {
                    tok: Tok::String(s),
                    line,
                    col: start_col,
                });
                col += j + 1 - i;
                i = j + 1;
            }
            c if c.is_ascii_digit()
                || (c == '.' && bytes.get(i + 1).is_some_and(|d| d.is_ascii_digit())) =>
            {
                let start = i;
                let start_col = col;
                let mut seen_dot = false;
                let mut seen_exp = false;
                while i < bytes.len() {
                    let d = bytes[i];
                    if d.is_ascii_digit() {
                        i += 1;
                    } else if d == '.' && !seen_dot && !seen_exp {
                        seen_dot = true;
                        i += 1;
                    } else if (d == 'e' || d == 'E') && !seen_exp {
                        seen_exp = true;
                        i += 1;
                        if matches!(bytes.get(i), Some('+') | Some('-')) {
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }
                let text: String = bytes[start..i].iter().collect();
                col = start_col + (i - start);
                let tok = if seen_dot || seen_exp {
                    Tok::Float(text.parse().map_err(|_| CircuitError::Parse {
                        line,
                        col: start_col,
                        msg: format!("bad float literal `{text}`"),
                    })?)
                } else {
                    match text.parse::<u64>() {
                        Ok(v) => Tok::Int(v),
                        Err(_) => Tok::Float(text.parse().map_err(|_| CircuitError::Parse {
                            line,
                            col: start_col,
                            msg: format!("bad numeric literal `{text}`"),
                        })?),
                    }
                };
                out.push(Token {
                    tok,
                    line,
                    col: start_col,
                });
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                let start_col = col;
                while i < bytes.len() && (bytes[i].is_alphanumeric() || bytes[i] == '_') {
                    i += 1;
                }
                let text: String = bytes[start..i].iter().collect();
                col = start_col + (i - start);
                out.push(Token {
                    tok: Tok::Ident(text),
                    line,
                    col: start_col,
                });
            }
            other => {
                return Err(CircuitError::Parse {
                    line,
                    col,
                    msg: format!("unexpected character `{other}`"),
                })
            }
        }
    }
    out.push(Token {
        tok: Tok::Eof,
        line,
        col,
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<Tok> {
        lex(s).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn lexes_a_gate_call() {
        assert_eq!(
            toks("cx q[0], q[1];"),
            vec![
                Tok::Ident("cx".into()),
                Tok::Ident("q".into()),
                Tok::LBracket,
                Tok::Int(0),
                Tok::RBracket,
                Tok::Comma,
                Tok::Ident("q".into()),
                Tok::LBracket,
                Tok::Int(1),
                Tok::RBracket,
                Tok::Semi,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn lexes_numbers() {
        assert_eq!(
            toks("1 2.5 1e3 1.5e-2 .5")[..5].to_vec(),
            vec![
                Tok::Int(1),
                Tok::Float(2.5),
                Tok::Float(1000.0),
                Tok::Float(0.015),
                Tok::Float(0.5),
            ]
        );
    }

    #[test]
    fn lexes_negative_via_minus_token() {
        assert_eq!(
            toks("-pi/4"),
            vec![
                Tok::Minus,
                Tok::Ident("pi".into()),
                Tok::Slash,
                Tok::Int(4),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn skips_comments() {
        // h q ; x q ; Eof
        assert_eq!(
            toks("// hi\nh q; /* block\ncomment */ x q;"),
            vec![
                Tok::Ident("h".into()),
                Tok::Ident("q".into()),
                Tok::Semi,
                Tok::Ident("x".into()),
                Tok::Ident("q".into()),
                Tok::Semi,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn tracks_line_numbers_across_comments() {
        let ts = lex("// one\n/* two\nthree */\nh q;").unwrap();
        assert_eq!(ts[0].line, 4);
    }

    #[test]
    fn lexes_include_string() {
        assert_eq!(
            toks("include \"qelib1.inc\";"),
            vec![
                Tok::Ident("include".into()),
                Tok::String("qelib1.inc".into()),
                Tok::Semi,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn rejects_unterminated_string() {
        assert!(lex("include \"oops;").is_err());
    }

    #[test]
    fn rejects_unterminated_block_comment() {
        assert!(lex("/* oops").is_err());
    }

    #[test]
    fn rejects_stray_character() {
        let e = lex("h q; @").unwrap_err();
        assert!(matches!(e, CircuitError::Parse { .. }));
    }
}
