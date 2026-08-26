//! Corpus tests: parse and print every benchmark circuit and every rewrite rule.
//!
//! The reference Java suite had a single `ParserTest` with no assertions at all — it
//! parsed six lines and called `System.out.println`. These tests instead run the parser
//! and printer over the whole in-repo corpus (hundreds of circuits, and every rule file),
//! which is what actually catches grammar gaps and printing bugs.

use qcircuit::qasm::{self, PrintOptions};
use qcircuit::{Dag, GateRegistry};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<name> has a grandparent")
        .to_path_buf()
}

fn qasm_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let root = repo_root().join("benchmarks");
    let Ok(dirs) = std::fs::read_dir(&root) else {
        panic!("benchmarks directory missing at {}", root.display());
    };
    for d in dirs.flatten() {
        if !d.path().is_dir() {
            continue;
        }
        for f in std::fs::read_dir(d.path()).into_iter().flatten().flatten() {
            if f.path().extension().is_some_and(|e| e == "qasm") {
                out.push(f.path());
            }
        }
    }
    out.sort();
    assert!(!out.is_empty(), "no benchmark circuits found");
    out
}

#[test]
fn every_benchmark_parses() {
    let files = qasm_files();
    let mut total_gates = 0usize;
    for path in &files {
        let src = std::fs::read_to_string(path).unwrap();
        let dag =
            qasm::parse(&src).unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));
        assert!(
            dag.gate_count() > 0,
            "{} parsed to an empty circuit",
            path.display()
        );
        total_gates += dag.gate_count();
    }
    println!("parsed {} circuits, {total_gates} gates", files.len());
}

#[test]
fn every_benchmark_round_trips() {
    for path in qasm_files() {
        let src = std::fs::read_to_string(&path).unwrap();
        let a = qasm::parse(&src).unwrap();
        let text = qasm::to_qasm(&a);
        let b = qasm::parse(&text)
            .unwrap_or_else(|e| panic!("reprinting {} did not reparse: {e}", path.display()));

        assert_eq!(
            a.gate_count(),
            b.gate_count(),
            "gate count changed for {}",
            path.display()
        );
        assert_eq!(
            a.num_qubits(),
            b.num_qubits(),
            "qubit count changed for {}",
            path.display()
        );
        assert_eq!(
            a.structural_hash(),
            b.structural_hash(),
            "structure changed for {}",
            path.display()
        );
        // Printing must be a fixed point: print(parse(print(x))) == print(x).
        assert_eq!(
            qasm::to_qasm(&b),
            text,
            "printing not idempotent for {}",
            path.display()
        );
    }
}

#[test]
fn every_benchmark_gate_is_known_and_well_formed() {
    let reg = GateRegistry::with_builtins();
    let mut unknown: Vec<String> = Vec::new();
    for path in qasm_files() {
        let src = std::fs::read_to_string(&path).unwrap();
        let dag = qasm::parse(&src).unwrap();
        for idx in dag.gate_indices() {
            let op = dag.gate(idx);
            match op.validate(&reg) {
                Ok(()) => {}
                Err(qcircuit::CircuitError::UnknownGate(g)) => {
                    if !unknown.contains(&g) {
                        unknown.push(g);
                    }
                }
                Err(e) => panic!("{} in {}", e, path.display()),
            }
        }
    }
    assert!(
        unknown.is_empty(),
        "benchmarks use gates the registry does not define: {unknown:?}"
    );
}

/// Every qubit wire in every benchmark runs unbroken from its source to its sink.
#[test]
fn every_benchmark_has_well_formed_wires() {
    for path in qasm_files() {
        let src = std::fs::read_to_string(&path).unwrap();
        let dag = qasm::parse(&src).unwrap();
        for q in dag.qubits() {
            let sink = dag.sink(q).unwrap();
            let mut n = dag.source(q).unwrap();
            let mut steps = 0;
            while n != sink {
                n = dag
                    .next_on(n, q)
                    .unwrap_or_else(|| panic!("wire {q} breaks in {}", path.display()));
                steps += 1;
                assert!(
                    steps <= dag.gate_count() + 1,
                    "wire {q} loops in {}",
                    path.display()
                );
            }
        }
    }
}

fn rule_files() -> Vec<PathBuf> {
    let root = repo_root().join("rules");
    let mut out: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("rules directory missing at {}: {e}", root.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    out.sort();
    assert!(!out.is_empty(), "no rule files found");
    out
}

/// Both sides of every rewrite rule, in every shipped rule file, must parse.
///
/// Rule files use the bare-identifier operand form (`cx q0, q1`) and symbolic angles
/// (`rz((theta1+theta2)) q0`), so this exercises a different part of the grammar than the
/// benchmark circuits do. Milestone 3 extends this to check that the two sides are
/// actually equivalent unitaries.
#[test]
fn every_rule_side_parses() {
    let mut rules = 0usize;
    let mut sides = 0usize;
    for path in rule_files() {
        let text = std::fs::read_to_string(&path).unwrap();
        for (lineno, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            rules += 1;
            // `lhs | rhs` for plain rules, `lhs | rhs | constraints` for symbolic ones.
            for side in line.split(" | ").take(2) {
                // The symbolic hole `symb q` is not a gate; it is handled by the rule
                // reader in the qrules crate. Skip sides containing it here.
                if side.contains("symb q") {
                    continue;
                }
                sides += 1;
                let dag: Dag = qasm::parse(side).unwrap_or_else(|e| {
                    panic!(
                        "{}:{}: failed to parse `{side}`: {e}",
                        path.display(),
                        lineno + 1
                    )
                });
                // Round-trip the bare form too.
                let printed = qasm::to_qasm_with(&dag, &PrintOptions::bare());
                let back = qasm::parse(&printed).unwrap_or_else(|e| {
                    panic!(
                        "{}:{}: reprint did not reparse: {e}",
                        path.display(),
                        lineno + 1
                    )
                });
                assert_eq!(
                    dag.gate_count(),
                    back.gate_count(),
                    "{}:{}: gate count changed on round-trip",
                    path.display(),
                    lineno + 1
                );
            }
        }
    }
    println!("parsed {rules} rules, {sides} non-symbolic sides");
    assert!(
        rules > 100_000,
        "expected the full rule corpus, saw {rules}"
    );
}
