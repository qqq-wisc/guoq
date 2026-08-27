//! Independent equivalence checking of optimizer output, via MQT QCEC.
//!
//! The optimizer's own end-to-end check (`--verify-final`, and `cli.rs`'s
//! `optimization_preserves_semantics`) builds dense unitaries, which caps it at about
//! twelve qubits; everything wider rests on the compositional argument that every local
//! rewrite is verified on a convex span. These tests close that gap with a checker that
//! shares none of this codebase's assumptions: QCEC decides equivalence symbolically,
//! so a 16-qubit circuit — beyond any dense check — is verified directly, by an
//! implementation that cannot inherit a bug from the code it is checking.
//!
//! These tests are deliberately not optional: they fail, loudly, when `python3` or
//! `mqt.qcec` is missing (`pip install mqt.qcec`), because an equivalence check that
//! silently skips is indistinguishable from one that passes. The three core tests run
//! in every build; the per-gate-set benchmark sweeps at the bottom are `#[ignore]`d
//! like the corpus sweeps — slow in debug, quick in release, run by CI — and they too
//! fail rather than skip when qcec is absent.

use std::path::{Path, PathBuf};

use guoq_cli::{optimize, Cli};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<name> has a grandparent")
        .to_path_buf()
}

/// Run the QCEC harness on two circuit files and return its verdict line.
///
/// Exit 0 is "equivalent" (a global phase is no difference), 1 is "not equivalent", and
/// anything else — python missing, qcec missing, the checker giving up — is a test
/// failure here, never a skip.
fn qcec_verdict(a: &Path, b: &Path) -> bool {
    let script = repo_root().join("py/qcec_check.py");
    let out = std::process::Command::new("python3")
        .arg(&script)
        .arg(a)
        .arg(b)
        .output()
        .expect("python3 must be runnable for the equivalence tests");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    match out.status.code() {
        Some(0) => true,
        Some(1) => false,
        _ => panic!(
            "the QCEC check itself failed (these tests require `pip install mqt.qcec`):\n\
             {stdout}{stderr}"
        ),
    }
}

struct Run {
    out_dir: tempfile::TempDir,
    cli: Cli,
}

/// Build a CLI invocation against a circuit file, mirroring `cli.rs`.
fn run_args(circuit: &Path, extra: &[&str]) -> Run {
    let out_dir = tempfile::tempdir().unwrap();
    let rules_dir = repo_root().join("rules");
    let mut args: Vec<String> = vec!["guoq".into()];
    args.extend(extra.iter().map(|s| s.to_string()));
    args.push("--rules-dir".into());
    args.push(rules_dir.display().to_string());
    args.push("-out".into());
    args.push(out_dir.path().display().to_string());
    args.push(circuit.display().to_string());
    let cli = Cli::parse_from_args(args).unwrap();
    Run { out_dir, cli }
}

/// The full pipeline on a real benchmark, judged by the external checker.
#[test]
fn qcec_confirms_an_optimized_benchmark() {
    let circuit = repo_root().join("benchmarks/nam_rz/tof_3.qasm");
    // The shipped symbolic file rather than the generated default: loading the 26k-rule
    // default verifies every constraint numerically, which costs minutes in a debug
    // build, and the corpus is not what this test is about.
    let symb = repo_root().join("rules/rules_q3_s3_nam_symb.txt");
    let r = run_args(
        &circuit,
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM",
            "-temp",
            "0",
            "-resynth",
            "NONE",
            "--max-iters",
            "20",
            "-sr",
            symb.to_str().unwrap(),
        ],
    );
    optimize(&r.cli).unwrap();
    let optimized = r.out_dir.path().join("optimized__tof_3.qasm");
    assert!(
        qcec_verdict(&circuit, &optimized),
        "QCEC says the optimized circuit is not equivalent to the input"
    );
}

/// A deterministic 16-qubit Clifford+T circuit, far too wide for any dense check.
///
/// Seeded cancellation pairs guarantee the optimizer changes the circuit, so the
/// equivalence claim is about a real transformation rather than a no-op.
fn wide_circuit(path: &Path) {
    let mut src = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[16];\n");
    let gates = ["h", "t", "tdg", "s", "sdg", "x"];
    // xorshift64, fixed seed: the same circuit every run, no RNG dependency.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..1200u64 {
        let a = (next() % 16) as usize;
        if i % 5 == 0 {
            let b = (a + 1 + (next() % 15) as usize) % 16;
            src.push_str(&format!("cx q[{a}], q[{b}];\n"));
        } else if i % 7 == 0 {
            // An adjacent self-inverse pair, so reduction has guaranteed prey.
            src.push_str(&format!("h q[{a}];\nh q[{a}];\n"));
        } else {
            let g = gates[(next() % gates.len() as u64) as usize];
            src.push_str(&format!("{g} q[{a}];\n"));
        }
    }
    std::fs::write(path, src).unwrap();
}

/// The check the native harness cannot do: 16 qubits, REDUCE, verified externally.
#[test]
fn qcec_verifies_a_circuit_too_wide_for_the_native_check() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("wide.qasm");
    wide_circuit(&input);

    // A minimal rule file rather than the shipped corpora: this test's subject is
    // external verification of a real transformation, not corpus power, and the full
    // cliffordt sets cost minutes of load and fixpoint per run in debug builds.
    let rules = dir.path().join("rules.txt");
    std::fs::write(
        &rules,
        " | h q0; h q0;\n | x q0; x q0;\n | t q0; tdg q0;\n | tdg q0; t q0;\n \
         | s q0; sdg q0;\n | sdg q0; s q0;\n | cx q0, q1; cx q0, q1;\ns q0; | t q0; t q0;\n",
    )
    .unwrap();
    let symb = repo_root().join("rules/rules_q3_s3_cliffordt_symb.txt");
    let r = run_args(
        &input,
        &[
            "-g",
            "CLIFFORDT",
            "-opt",
            "TOTAL",
            "-search",
            "REDUCE",
            "-resynth",
            "NONE",
            "--timeout",
            "3",
            "-r",
            rules.to_str().unwrap(),
            "-sr",
            symb.to_str().unwrap(),
        ],
    );
    let before = qcircuit::qasm::parse(&std::fs::read_to_string(&input).unwrap())
        .unwrap()
        .gate_count();
    optimize(&r.cli).unwrap();
    let optimized = r.out_dir.path().join("optimized__wide.qasm");
    let after = qcircuit::qasm::parse(&std::fs::read_to_string(&optimized).unwrap())
        .unwrap()
        .gate_count();
    assert!(
        after < before,
        "the optimizer changed nothing ({before} gates), so the check would be vacuous"
    );
    assert!(
        qcec_verdict(&input, &optimized),
        "QCEC says the optimized wide circuit is not equivalent to the input"
    );
}

/// The harness itself must be falsifiable: a circuit that genuinely differs — one `t`
/// turned into a `tdg` — must be reported as inequivalent, or every green result above
/// is meaningless.
///
/// A small circuit on purpose: *disproving* equivalence of wide random circuits is
/// QCEC's hard case (a 16-qubit mutated pair ran for a quarter of an hour), and the
/// falsifiability of the harness is established just as well at five qubits in
/// milliseconds.
#[test]
fn qcec_rejects_an_inequivalent_circuit() {
    let dir = tempfile::tempdir().unwrap();
    let input = repo_root().join("benchmarks/nam_t_tdg/tof_3.qasm");
    let mutated = dir.path().join("mutated.qasm");
    let src = std::fs::read_to_string(&input).unwrap();
    let broken = src.replacen("t q[", "tdg q[", 1);
    assert_ne!(src, broken, "the mutation must actually change a gate");
    std::fs::write(&mutated, broken).unwrap();
    assert!(
        !qcec_verdict(&input, &mutated),
        "QCEC accepted a circuit with a deliberately flipped gate"
    );
}

/// Sweep the smallest benchmarks of one gate set through both search strategies and
/// verify every optimized circuit with QCEC (`py/qcec_sweep.py`, modeled on the sweep
/// in qqq-wisc/tzap). The script includes a tamper negative-control per run, so a
/// checker that stopped rejecting anything fails the sweep itself.
fn qcec_sweep(bench_dir: &str, gate_set: &str, objective: &str, count: usize) {
    let script = repo_root().join("py/qcec_sweep.py");
    let out = std::process::Command::new("python3")
        .arg(&script)
        .arg(env!("CARGO_BIN_EXE_guoq"))
        .arg(repo_root().join("rules"))
        .arg(repo_root().join("benchmarks").join(bench_dir))
        .arg(gate_set)
        .arg(objective)
        .arg(count.to_string())
        .output()
        .expect("python3 must be runnable for the equivalence tests");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    println!("{stdout}");
    assert!(
        out.status.success(),
        "QCEC sweep over {bench_dir} failed (these tests require `pip install mqt.qcec`):\n\
         {stdout}{stderr}"
    );
}

// One sweep per gate set with a regenerated (exactly verified) default corpus. The ion
// and ibm(old) sets are absent deliberately: QCEC's parser has no `gpi`/`gpi2`/`ms`,
// and the `ibm` set's shipped `u2`/`u3` corpus is documented as not regenerable.

#[test]
#[ignore = "sweeps many circuits through the optimizer and QCEC; run in release"]
fn qcec_sweeps_the_nam_benchmarks() {
    qcec_sweep("nam_rz", "NAM", "TOTAL", 6);
}

#[test]
#[ignore = "sweeps many circuits through the optimizer and QCEC; run in release"]
fn qcec_sweeps_the_cliffordt_benchmarks() {
    qcec_sweep("nam_t_tdg", "CLIFFORDT", "FT", 6);
}

#[test]
#[ignore = "sweeps many circuits through the optimizer and QCEC; run in release"]
fn qcec_sweeps_the_ibmnew_benchmarks() {
    qcec_sweep("ibmnew", "IBMN", "TOTAL", 6);
}

#[test]
#[ignore = "sweeps many circuits through the optimizer and QCEC; run in release"]
fn qcec_sweeps_the_rigetti_benchmarks() {
    // Rigetti's smallest benchmarks are an order of magnitude larger, so fewer of them.
    qcec_sweep("rigetti", "RIGETTI", "TOTAL", 4);
}
