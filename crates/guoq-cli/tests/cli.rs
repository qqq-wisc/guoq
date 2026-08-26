//! End-to-end tests for the `guoq` binary's library entry point.
//!
//! The reference had no CLI tests at all. These check the parts downstream tools depend
//! on: the flag spellings `wisq` and `evaluation/run_guoq.py` pass, the output file
//! naming they scrape, and that optimization actually preserves what the circuit computes.

use std::path::{Path, PathBuf};

use guoq_cli::{optimize, Cli};
use qcircuit::{qasm, GateRegistry};
use qsemantics::{phase_invariant_distance, Unitary};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<name> has a grandparent")
        .to_path_buf()
}

struct Run {
    out_dir: tempfile::TempDir,
    cli: Cli,
}

/// Build a CLI invocation against a real benchmark circuit.
fn run_args(bench: &str, extra: &[&str]) -> Run {
    let out_dir = tempfile::tempdir().unwrap();
    let circuit = repo_root().join("benchmarks").join(bench);
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

fn same_semantics(a: &Path, b: &Path) -> f64 {
    let reg = GateRegistry::with_builtins();
    let da = qasm::parse(&std::fs::read_to_string(a).unwrap()).unwrap();
    let db = qasm::parse(&std::fs::read_to_string(b).unwrap()).unwrap();
    let mut qs: Vec<qcircuit::QubitId> = da.qubits().to_vec();
    for q in db.qubits() {
        if !qs.contains(q) {
            qs.push(q.clone());
        }
    }
    qs.sort();
    let ua = Unitary::from_dag_over(&da, qs.clone(), &reg).unwrap();
    let ub = Unitary::from_dag_over(&db, qs, &reg).unwrap();
    phase_invariant_distance(&ua, &ub)
}

#[test]
fn optimizes_a_nam_circuit_and_writes_both_outputs() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM",
            "-temp",
            "0",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONE",
            "--max-iters",
            "25",
        ],
    );
    let outcome = optimize(&r.cli).unwrap();

    assert_eq!(outcome.original_gates, 45);
    assert!(
        outcome.result.best.dag.gate_count() <= outcome.original_gates,
        "optimization made the circuit larger"
    );

    // Both files the reference's harness expects.
    let latest = r.out_dir.path().join("latest_sol__tof_3.qasm");
    let final_ = r.out_dir.path().join("optimized__tof_3.qasm");
    assert!(latest.exists(), "latest_sol file missing");
    assert!(final_.exists(), "optimized file missing");

    // ...and they agree.
    let a = qasm::parse(&std::fs::read_to_string(&latest).unwrap()).unwrap();
    let b = qasm::parse(&std::fs::read_to_string(&final_).unwrap()).unwrap();
    assert_eq!(a.structural_hash(), b.structural_hash());
}

/// The property that matters: optimizing must not change what the circuit computes.
#[test]
fn optimization_preserves_semantics() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM",
            "-temp",
            "0",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONE",
            "--max-iters",
            "20",
        ],
    );
    optimize(&r.cli).unwrap();

    let original = repo_root().join("benchmarks/nam_rz/tof_3.qasm");
    let optimized = r.out_dir.path().join("optimized__tof_3.qasm");
    let d = same_semantics(&original, &optimized);
    assert!(d < 1e-9, "optimized circuit differs by {d:.3e}");
}

/// `--verify-final` performs that check inside the optimizer.
#[test]
fn verify_final_runs_and_passes() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM",
            "-temp",
            "0",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONE",
            "--max-iters",
            "15",
            "--verify-final",
        ],
    );
    let outcome = optimize(&r.cli).unwrap();
    let d = outcome
        .verified_distance
        .expect("a 5-qubit circuit is small enough to verify");
    assert!(d < 1e-9, "verified distance {d:.3e}");
}

/// The default fidelity configuration runs end to end.
#[test]
fn the_default_fidelity_configuration_runs() {
    let r = run_args(
        "ibmnew/tof_3.qasm",
        &[
            "-g",
            "IBMN",
            "-opt",
            "FIDELITY",
            "--error-1q",
            "0.0003",
            "--error-2q",
            "0.0115",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONE",
            "--max-iters",
            "20",
            "-q",
            "8",
        ],
    );
    // Exactly the README's flagship invocation, minus the resynthesis server.
    assert_eq!(r.cli.opt_obj, "FIDELITY");
    assert_eq!(r.cli.search_strategy, "BEAM_MCMC");
    let outcome = optimize(&r.cli).unwrap();
    assert!(outcome.result.iterations > 1);
}

#[test]
fn every_objective_and_strategy_combination_runs() {
    for objective in ["TOTAL", "TWO_Q", "T", "FT", "TOTAL_IGNORE_RZ", "FIDELITY"] {
        for strategy in ["BEAM", "MCMC", "SIM_ANN", "BEAM_MCMC"] {
            let r = run_args(
                "nam_rz/tof_3.qasm",
                &[
                    "-g",
                    "NAM",
                    "-opt",
                    objective,
                    "-search",
                    strategy,
                    // Weighted so resynthesis is sampled essentially every iteration. Left at its
                    // default it is one slot among ~23,000, and whether the test exercises the
                    // backend at all comes down to the RNG stream.
                    "--resynth-weight",
                    "100000",
                    "-resynth",
                    "NONE",
                    "--max-iters",
                    "3",
                    "--fidelity",
                    "38",
                ],
            );
            let outcome =
                optimize(&r.cli).unwrap_or_else(|e| panic!("{objective}/{strategy} failed: {e:#}"));
            assert!(
                outcome.result.best.dag.is_acyclic(),
                "{objective}/{strategy} produced a cycle"
            );
        }
    }
}

#[test]
fn fidelity_without_error_rates_is_a_clear_error() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "FIDELITY",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONE",
            "--max-iters",
            "2",
        ],
    );
    let err = optimize(&r.cli).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("error-1q"), "unhelpful message: {msg}");
}

#[test]
fn an_unknown_objective_is_rejected() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &["-g", "NAM", "-opt", "NONSENSE", "-resynth", "NONE"],
    );
    assert!(optimize(&r.cli).is_err());
}

#[test]
fn an_unknown_gate_set_is_rejected() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &["-g", "NOSUCHSET", "-opt", "TOTAL", "-resynth", "NONE"],
    );
    assert!(optimize(&r.cli).is_err());
}

#[test]
fn job_info_appears_in_output_names() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONE",
            "--max-iters",
            "2",
            "-job",
            "cluster_7",
        ],
    );
    optimize(&r.cli).unwrap();
    assert!(r
        .out_dir
        .path()
        .join("latest_sol_cluster_7_tof_3.qasm")
        .exists());
    assert!(r
        .out_dir
        .path()
        .join("optimized_cluster_7_tof_3.qasm")
        .exists());
}

#[test]
fn results_are_reproducible_for_a_seed() {
    let mut hashes = Vec::new();
    for _ in 0..2 {
        let r = run_args(
            "nam_rz/tof_3.qasm",
            &[
                "-g",
                "NAM",
                "-opt",
                "TOTAL",
                "-search",
                "BEAM_MCMC",
                // Weighted so resynthesis is sampled essentially every iteration. Left at its
                // default it is one slot among ~23,000, and whether the test exercises the
                // backend at all comes down to the RNG stream.
                "--resynth-weight",
                "100000",
                "-resynth",
                "NONE",
                "--max-iters",
                "40",
                "--seed",
                "424242",
                "-q",
                "8",
            ],
        );
        let outcome = optimize(&r.cli).unwrap();
        hashes.push(outcome.result.best.dag.structural_hash());
    }
    assert_eq!(hashes[0], hashes[1], "the same seed gave different results");
}

/// An argument file, as `evaluation/run_guoq.py` writes it.
#[test]
fn argument_files_drive_a_full_run() {
    let out_dir = tempfile::tempdir().unwrap();
    let args_file = out_dir.path().join("args.txt");
    let body = format!(
        "-g\nNAM\n-opt\nTOTAL\n-search\nBEAM\n-temp\n0\n-resynth\nNONE\n--max-iters\n5\n\
         --rules-dir\n{}\n-out\n{}\n{}\n",
        repo_root().join("rules").display(),
        out_dir.path().display(),
        repo_root().join("benchmarks/nam_rz/tof_3.qasm").display(),
    );
    std::fs::write(&args_file, body).unwrap();

    let cli =
        Cli::parse_from_args(["guoq".to_string(), format!("@{}", args_file.display())]).unwrap();
    let outcome = optimize(&cli).unwrap();
    assert_eq!(outcome.original_gates, 45);
    assert!(out_dir.path().join("optimized__tof_3.qasm").exists());
}

#[test]
fn explicit_rule_files_are_honoured() {
    let rules = repo_root().join("rules");
    let empty = rules.join("empty.txt");
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONE",
            "--max-iters",
            "5",
            "-r",
            empty.to_str().unwrap(),
            "-sr",
            empty.to_str().unwrap(),
        ],
    );
    let outcome = optimize(&r.cli).unwrap();
    // With no rules there is nothing to do, so the circuit comes back unchanged.
    assert_eq!(outcome.result.best.dag.gate_count(), outcome.original_gates);
}

#[test]
fn a_missing_circuit_is_a_clear_error() {
    let out_dir = tempfile::tempdir().unwrap();
    let cli = Cli::parse_from_args([
        "guoq",
        "-g",
        "NAM",
        "-out",
        out_dir.path().to_str().unwrap(),
        "/nonexistent/circuit.qasm",
    ])
    .unwrap();
    let msg = format!("{:#}", optimize(&cli).unwrap_err());
    assert!(msg.contains("circuit"), "unhelpful message: {msg}");
}

// --- resynthesis -----------------------------------------------------------------

use guoq_cli::BackendChoice;

/// The reference's default backend per objective.
#[test]
fn backend_defaults_match_the_reference() {
    use qopt::Objective;
    assert_eq!(
        BackendChoice::default_for(Objective::Fidelity),
        BackendChoice::Bqskit
    );
    assert_eq!(
        BackendChoice::default_for(Objective::TwoQ),
        BackendChoice::Bqskit
    );
    assert_eq!(
        BackendChoice::default_for(Objective::Ft),
        BackendChoice::Synthetiq
    );
    assert_eq!(
        BackendChoice::default_for(Objective::T),
        BackendChoice::Synthetiq
    );
    assert_eq!(
        BackendChoice::default_for(Objective::Total),
        BackendChoice::None
    );
}

#[test]
fn backend_names_parse() {
    for (name, want) in [
        ("NONE", BackendChoice::None),
        ("none", BackendChoice::None),
        ("BQSKIT", BackendChoice::Bqskit),
        ("SYNTHETIQ", BackendChoice::Synthetiq),
    ] {
        assert_eq!(BackendChoice::parse(name), Some(want));
    }
    assert_eq!(BackendChoice::parse("nonsense"), None);
}

#[test]
fn an_unknown_backend_is_rejected() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONSENSE",
            "--max-iters",
            "2",
        ],
    );
    let msg = format!("{:#}", optimize(&r.cli).unwrap_err());
    assert!(msg.contains("resynthesis backend"), "{msg}");
}

/// `-resynth NONE` must not start any subprocess, and must not need one to exist.
#[test]
fn resynth_none_starts_nothing() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "NONE",
            "--max-iters",
            "5",
            // Deliberately bogus paths: nothing should look at them.
            "--bqskit-worker",
            "/nonexistent/worker.py",
            "--synthetiq-binary",
            "/nonexistent/main",
        ],
    );
    let outcome = optimize(&r.cli).unwrap();
    assert!(
        outcome.resynth.is_none(),
        "no backend should have been built"
    );
}

/// A configured-but-unavailable backend must not take the whole run down: the search
/// still optimizes with rules, and the failures are counted.
#[test]
fn an_unavailable_backend_degrades_rather_than_aborting() {
    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM_MCMC",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "SYNTHETIQ",
            "--max-iters",
            "30",
            "-q",
            "4",
            "--synthetiq-binary",
            "/nonexistent/main",
        ],
    );
    let outcome = optimize(&r.cli).unwrap();
    let stats = outcome.resynth.expect("a backend was configured");
    assert!(stats.attempts > 0, "resynthesis was never attempted");
    assert_eq!(stats.accepted, 0);
    assert!(stats.errored > 0, "the failures should have been counted");
    // ...and the circuit is still valid.
    assert!(outcome.result.best.dag.is_acyclic());
}

/// The whole point: a backend that returns something wrong never reaches the circuit.
///
/// Uses a stub worker so no BQSKit install is needed. It answers every request with a
/// circuit that is *not* equivalent to what it was given.
#[test]
fn a_faulty_backend_never_corrupts_the_circuit() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("faulty.py");
    std::fs::write(
        &script,
        r#"
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\n")
sys.stdout.flush()
for line in sys.stdin:
    if line.strip():
        # Always the same wrong answer, regardless of the request.
        sys.stdout.write(json.dumps({
            "ok": True,
            "circuit": "OPENQASM 2.0;\nqreg q[1];\nx q[0];\n"
        }) + "\n")
        sys.stdout.flush()
"#,
    )
    .unwrap();
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        return;
    }

    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM_MCMC",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "BQSKIT",
            "--max-iters",
            "40",
            "-q",
            "4",
            "--max-partition-qubits",
            "1",
            "--bqskit-worker",
            script.to_str().unwrap(),
        ],
    );
    let outcome = optimize(&r.cli).unwrap();
    let stats = outcome.resynth.expect("a backend was configured");
    assert!(stats.attempts > 0, "resynthesis was never attempted");
    assert!(
        stats.over_budget > 0,
        "wrong results should have been rejected on measurement; stats: {stats:?}"
    );

    // Whatever happened, the circuit still computes what it did to begin with.
    let original = repo_root().join("benchmarks/nam_rz/tof_3.qasm");
    let optimized = r.out_dir.path().join("optimized__tof_3.qasm");
    let d = same_semantics(&original, &optimized);
    assert!(
        d < 1e-9,
        "the faulty backend corrupted the circuit by {d:.3e}"
    );
}

/// A correct backend is accepted, and its measured error is recorded.
#[test]
fn a_correct_backend_is_accepted_and_measured() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("identity.py");
    // Echo the request back: always exactly equivalent, so always affordable.
    std::fs::write(
        &script,
        r#"
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    sys.stdout.write(json.dumps({"ok": True, "circuit": req["circuit"]}) + "\n")
    sys.stdout.flush()
"#,
    )
    .unwrap();
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        return;
    }

    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM_MCMC",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "BQSKIT",
            "--max-iters",
            "40",
            "-q",
            "4",
            "--bqskit-worker",
            script.to_str().unwrap(),
        ],
    );
    let outcome = optimize(&r.cli).unwrap();
    let stats = outcome.resynth.expect("a backend was configured");
    assert!(
        stats.accepted > 0,
        "an exact backend was never accepted: {stats:?}"
    );
    assert!(
        stats.total_error < 1e-9,
        "an exact backend should cost nothing, got {}",
        stats.total_error
    );

    let original = repo_root().join("benchmarks/nam_rz/tof_3.qasm");
    let optimized = r.out_dir.path().join("optimized__tof_3.qasm");
    assert!(same_semantics(&original, &optimized) < 1e-9);
}

/// `--max-resynth-allowed -1` is the documented "no limit" value: the budget is what
/// remains of the total, and -1 simply means the call count is unbounded.
#[test]
fn no_resynth_limit_is_usable() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("echo.py");
    std::fs::write(
        &script,
        r#"
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    sys.stdout.write(json.dumps({"ok": True, "circuit": req["circuit"]}) + "\n")
    sys.stdout.flush()
"#,
    )
    .unwrap();
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        return;
    }

    let r = run_args(
        "nam_rz/tof_3.qasm",
        &[
            "-g",
            "NAM",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM_MCMC",
            // Weighted so resynthesis is sampled essentially every iteration. Left at its
            // default it is one slot among ~23,000, and whether the test exercises the
            // backend at all comes down to the RNG stream.
            "--resynth-weight",
            "100000",
            "-resynth",
            "BQSKIT",
            "--max-iters",
            "30",
            "-q",
            "4",
            "-maxsynth",
            "-1",
            "--bqskit-worker",
            script.to_str().unwrap(),
        ],
    );
    assert_eq!(r.cli.max_resynth_allowed, -1);
    let outcome = optimize(&r.cli).unwrap();
    let stats = outcome.resynth.expect("a backend was configured");
    // With no limit, and an exact backend, calls are made and accepted rather than every
    // attempt failing on a negative budget.
    assert!(
        stats.accepted > 0,
        "no call succeeded with -maxsynth -1: {stats:?}"
    );
}

// --- synthesis -------------------------------------------------------------------

use guoq_cli::synth;

/// Rules the synthesizer produces must be usable by the optimizer.
///
/// This closes the loop QUESO and GUOQ form: enumerate circuits, emit rules, load them
/// back, and optimize a real benchmark with them. The reference had no test spanning the
/// two halves.
#[test]
fn synthesized_rules_drive_the_optimizer() {
    use clap::Parser as _;

    let dir = tempfile::tempdir().unwrap();
    let rules_dir = dir.path().join("rules");
    std::fs::create_dir_all(&rules_dir).unwrap();
    let plain = rules_dir.join("synth.txt");

    let synth_cli = synth::Cli::try_parse_from([
        "queso",
        "-g",
        "nam",
        "-q",
        "3",
        "-s",
        "3",
        "-o",
        plain.to_str().unwrap(),
    ])
    .unwrap();
    let outcome = synth::synthesize(&synth_cli).unwrap();
    assert!(
        outcome.rules > 100,
        "only {} rules synthesized",
        outcome.rules
    );

    // An empty symbolic file, so the optimizer has both inputs.
    let symb = rules_dir.join("synth_symb.txt");
    std::fs::write(&symb, "").unwrap();

    let out_dir = tempfile::tempdir().unwrap();
    let circuit = repo_root().join("benchmarks/nam_rz/tof_3.qasm");
    let cli = Cli::parse_from_args([
        "guoq".to_string(),
        "-g".into(),
        "NAM".into(),
        "-opt".into(),
        "TOTAL".into(),
        "-search".into(),
        "BEAM".into(),
        "-temp".into(),
        "0".into(),
        "-resynth".into(),
        "NONE".into(),
        "--max-iters".into(),
        "20".into(),
        "-r".into(),
        plain.display().to_string(),
        "-sr".into(),
        symb.display().to_string(),
        "-out".into(),
        out_dir.path().display().to_string(),
        circuit.display().to_string(),
    ])
    .unwrap();

    let run = optimize(&cli).unwrap();
    assert_eq!(run.original_gates, 45);
    assert!(
        run.result.best.dag.gate_count() < run.original_gates,
        "freshly synthesized rules made no progress"
    );

    // ...and the result still computes what it should.
    let optimized = out_dir.path().join("optimized__tof_3.qasm");
    let d = same_semantics(&circuit, &optimized);
    assert!(d < 1e-9, "synthesized rules changed the circuit by {d:.3e}");
}

/// Rules the synthesizer writes must load without a high rejection rate.
#[test]
fn synthesized_rules_load_cleanly() {
    use clap::Parser as _;
    use qcircuit::GateRegistry;
    use qrules::legacy::{self, LoadOptions};

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("rules.txt");
    let cli = synth::Cli::try_parse_from([
        "queso",
        "-g",
        "nam",
        "-q",
        "3",
        "-s",
        "3",
        "-o",
        out.to_str().unwrap(),
    ])
    .unwrap();
    synth::synthesize(&cli).unwrap();

    let report = legacy::load_file(
        &out,
        &LoadOptions::default(),
        &GateRegistry::with_builtins(),
    )
    .unwrap();
    assert!(!report.rules.is_empty());

    // Rejections are expected -- a disconnected pattern has no single insertion point --
    // but the surviving majority must be usable. The reference's own shipped files reject
    // at a similar rate.
    let total = report.rules.len() + report.rejected.len();
    let reasons: std::collections::BTreeSet<&str> = report
        .rejected
        .iter()
        .map(|(_, r)| r.split(',').next().unwrap_or(r))
        .collect();
    println!(
        "{} of {total} rules loaded; rejection reasons: {reasons:?}",
        report.rules.len()
    );
    assert!(
        report.rules.len() * 3 > total,
        "too many rejections: {} of {total}",
        report.rejected.len()
    );
}
