//! Gate-set inference: `-g` is optional.
//!
//! Without it, the run uses the smallest known gate set covering every gate the circuit
//! uses; when none covers it, rules are synthesized for the circuit's own gates and
//! cached, so the cost is paid once per gate combination. An explicit `-g` is used as
//! given, whatever gates the circuit holds — the reference's (only) behavior.

use std::path::{Path, PathBuf};

use guoq_cli::{optimize, Cli, RunOutcome};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<name> has a grandparent")
        .to_path_buf()
}

/// Run `guoq` rules-only with the given gate-set arguments (empty for inference).
fn run(gate_args: &[&str], rules_dir: &Path, circuit: &Path) -> anyhow::Result<RunOutcome> {
    let out = tempfile::tempdir().unwrap();
    let args: Vec<String> = ["guoq", "-opt", "TOTAL", "-search", "BEAM_MCMC"]
        .into_iter()
        .chain(gate_args.iter().copied())
        .map(String::from)
        .chain([
            "-resynth".into(),
            "NONE".into(),
            "--max-iters".into(),
            "20".into(),
            "--seed".into(),
            "1".into(),
            "--rules-dir".into(),
            rules_dir.display().to_string(),
            "-out".into(),
            out.path().display().to_string(),
            circuit.display().to_string(),
        ])
        .collect();
    optimize(&Cli::parse_from_args(args)?)
}

#[test]
fn the_gate_set_is_inferred_from_the_circuits_gates() {
    let root = repo_root();
    let outcome = run(
        &[],
        &root.join("rules"),
        &root.join("benchmarks/nam_rz/tof_3.qasm"),
    )
    .unwrap();
    assert_eq!(
        outcome.gate_set, "nam",
        "an h/rz/x/cx circuit should land on nam"
    );
    assert!(outcome.result.best.dag.gate_count() <= outcome.original_gates);
}

#[test]
fn an_explicit_gate_set_wins_whatever_the_circuit_holds() {
    let root = repo_root();
    // A nam circuit under -g CLIFFORDT: rz is not even in that set, and the run still
    // uses cliffordt as told.
    let outcome = run(
        &["-g", "CLIFFORDT"],
        &root.join("rules"),
        &root.join("benchmarks/nam_rz/tof_3.qasm"),
    )
    .unwrap();
    assert_eq!(outcome.gate_set, "cliffordt");
}

/// No shipped set contains both `sx` and `cz`, so the first run synthesizes a rule file
/// for exactly that combination and later runs reuse it untouched.
#[test]
fn an_uncovered_circuit_synthesizes_its_own_rules_once() {
    let dir = tempfile::tempdir().unwrap();
    let circuit = dir.path().join("odd.qasm");
    std::fs::write(
        &circuit,
        "OPENQASM 2.0;\nqreg q[2];\nsx q[0];\ncz q[0], q[1];\nsx q[1];\nsx q[1];\n",
    )
    .unwrap();
    let rules_dir = dir.path().join("rules");
    std::fs::create_dir_all(&rules_dir).unwrap();

    let outcome = run(&[], &rules_dir, &circuit).unwrap();
    assert_eq!(outcome.gate_set, "auto-cz-sx");
    let cache = rules_dir.join("auto/rules_q3_s3_auto-cz-sx.txt");
    assert!(cache.is_file(), "the synthesized rules were not cached");
    let synthesized = std::fs::read_to_string(&cache).unwrap();
    assert!(
        !synthesized.is_empty(),
        "cz.cz = id alone should yield rules"
    );

    // Replace the cache with a sentinel; a second run must load it rather than
    // regenerate it, or the file would be overwritten.
    std::fs::write(&cache, "").unwrap();
    let again = run(&[], &rules_dir, &circuit).unwrap();
    assert_eq!(again.gate_set, "auto-cz-sx");
    assert_eq!(
        std::fs::read_to_string(&cache).unwrap(),
        "",
        "a second run re-synthesized rules that were already cached"
    );
}

#[test]
fn an_unknown_gate_without_a_gate_set_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let circuit = dir.path().join("mystery.qasm");
    std::fs::write(&circuit, "OPENQASM 2.0;\nqreg q[1];\nfrobnicate q[0];\n").unwrap();
    let err = run(&[], dir.path(), &circuit).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("frobnicate") && msg.contains("-g"),
        "unhelpful message: {msg}"
    );
}
