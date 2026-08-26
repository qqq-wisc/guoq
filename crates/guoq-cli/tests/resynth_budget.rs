//! The resynthesis error budget, checked against reality.
//!
//! The port's headline claim about resynthesis is arithmetic: every accepted result's
//! *measured* error is charged to the candidate's `accumulated_error`, and a result is
//! accepted only while the total stays within `--epsilon`. The existing backend tests
//! prove the two edges — wrong results are rejected, exact results cost nothing — but
//! neither exercises the budget with a backend that is genuinely *approximate*, the
//! case the accounting exists for. These tests do, and then hold the optimizer to both
//! sides of its claim:
//!
//! - the claimed `accumulated_error` never exceeds `--epsilon`, and
//! - on circuits small enough to build dense unitaries, the *actual* `hs_distance`
//!   between input and output is within `--epsilon`, and within the claim itself.
//!
//! `hs_distance` is the right global metric here, and the phase-invariant Frobenius
//! distance deliberately is not asserted: Frobenius grows by `sqrt(2^(n-k))` when a
//! `k`-qubit partition's error is embedded in the full `n`-qubit space (see
//! `distance_scales_with_partition_width` in `qresynth`), so the sum of
//! partition-level charges bounds the global *operator* distance — which `hs_distance`
//! sits below — but not the global Frobenius. Asserting the Frobenius against the
//! claim fails by exactly that embedding factor, which is how this file found the
//! distinction in the first place.
//!
//! Two kinds of backend run under the same harness, because they answer different
//! questions:
//!
//! - A **stub** worker that strips the dust deterministically. Real synthesizers
//!   cannot promise nonzero error — an exactly-representable block comes back exact —
//!   so only the stub can *guarantee* the budget gets spent, which is what pins the
//!   accounting itself (`claimed > 0`, every acceptance charged).
//! - The **real backends**, BQSKit and Synthetiq, end to end. These prove the whole
//!   production path — worker protocol, target-basis synthesis, Synthetiq's on-disk
//!   unitary format and qubit-ordering convention, measurement, splice — against the
//!   actual tools. The Synthetiq test exists because its first run caught the port
//!   encoding the target unitary in the wrong qubit order (see
//!   `the_target_is_encoded_in_synthetiqs_qubit_order` in `qresynth`).
//!
//! Like the QCEC sweeps, the multi-circuit sweep and the real-backend tests are
//! `#[ignore]`d (slow in debug, run by CI in release); one single-circuit stub test
//! runs in every build. The real-backend tests skip quietly when the backend is not
//! installed — CI installs both, so they always run there.

use std::path::{Path, PathBuf};

use guoq_cli::{optimize, Cli, RunOutcome};
use qcircuit::{qasm, GateRegistry, QubitId};
use qsemantics::{hs_distance, phase_invariant_distance, Unitary};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<name> has a grandparent")
        .to_path_buf()
}

fn python3_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_ok()
}

/// Whether the real BQSKit is importable, not merely whether Python runs.
fn bqskit_available() -> bool {
    std::process::Command::new("python3")
        .args(["-c", "import bqskit"])
        .output()
        .is_ok_and(|o| o.status.success())
}

/// The Synthetiq binary, if one is installed: `GUOQ_SYNTHETIQ_BIN`, or the reference's
/// conventional location `lib/synthetiq/bin/main` under the repository root.
fn synthetiq_binary() -> Option<PathBuf> {
    let candidate = match std::env::var_os("GUOQ_SYNTHETIQ_BIN") {
        Some(path) => PathBuf::from(path),
        None => repo_root().join("lib/synthetiq/bin/main"),
    };
    candidate.is_file().then_some(candidate)
}

/// The near-identity rotation the dusted inputs carry and the worker strips.
///
/// Small enough that dropping a few stays far inside the test's epsilon; large enough
/// that the charged error is unmistakably nonzero. Written and matched textually, so
/// the exact spelling is the contract between [`dusted`] and [`approximate_worker`].
const DUST: &str = "rz(0.0001)";

/// A copy of `bench` with a few near-identity `rz` gates sprinkled onto its wires.
///
/// The dust is what makes an *approximate* backend able to win: no rewrite rule can
/// remove an `rz(0.0001)` (it is not an identity), so the only way the search can shed
/// one is a resynthesis result that drops it — a genuinely smaller circuit that is
/// genuinely not equivalent, off by exactly the kind of small error the budget exists
/// to meter. A backend that only *adds* error without shrinking anything never enters
/// the best lineage at all, which is why the first version of this test proved nothing.
fn dusted(bench: &str, dir: &Path) -> PathBuf {
    let source = std::fs::read_to_string(repo_root().join("benchmarks").join(bench)).unwrap();
    let mut out = String::new();
    let mut dust_left = 3usize;
    for line in source.lines() {
        out.push_str(line);
        out.push('\n');
        // After a register declaration, one dust gate on each of its first wires.
        if line.trim_start().starts_with("qreg") && dust_left > 0 {
            let reg = line
                .trim_start()
                .trim_start_matches("qreg")
                .trim()
                .trim_end_matches(';');
            let name = reg.split('[').next().unwrap().trim();
            let width: usize = reg
                .split('[')
                .nth(1)
                .and_then(|w| w.trim_end_matches(']').parse().ok())
                .unwrap_or(1);
            for i in 0..width.min(dust_left) {
                out.push_str(&format!("{DUST} {name}[{i}];\n"));
            }
            dust_left = dust_left.saturating_sub(width);
        }
    }
    let path = dir.join(Path::new(bench).file_name().unwrap());
    std::fs::write(&path, out).unwrap();
    path
}

/// A worker that answers every request with the request's own circuit minus the dust —
/// a backend whose results are smaller (so the search keeps them) and slightly wrong
/// (so every acceptance charges real, nonzero error to the budget).
fn approximate_worker(dir: &Path) -> PathBuf {
    let script = dir.join("approximate.py");
    let body = r#"
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    kept = [l for l in req["circuit"].splitlines() if "DUST" not in l]
    sys.stdout.write(json.dumps({"ok": True, "circuit": "\n".join(kept) + "\n"}) + "\n")
    sys.stdout.flush()
"#;
    std::fs::write(&script, body.replace("DUST", DUST)).unwrap();
    script
}

struct Checked {
    outcome: RunOutcome,
    claimed: f64,
    measured_phase_invariant: f64,
    measured_hs: f64,
}

/// Optimize the prepared `circuit` with a resynthesis backend under `epsilon`, and
/// check every side of the budget claim. `gate_set`/`objective` pick the configuration
/// and `backend` is the `-resynth ...` argument tail naming and configuring the
/// backend. Returns the numbers so callers can assert sweep-level facts.
fn check_budget(
    circuit: &Path,
    gate_set: &str,
    objective: &str,
    epsilon: f64,
    max_iters: u32,
    backend: &[&str],
) -> Checked {
    let out_dir = tempfile::tempdir().unwrap();
    let bench = circuit.display();
    let mut args: Vec<String> = [
        "guoq",
        "-g",
        gate_set,
        "-opt",
        objective,
        "-search",
        "BEAM_MCMC",
        // Weighted so resynthesis is sampled essentially every iteration, as the other
        // backend tests do; at the default weight whether it fires at all is up to the
        // RNG stream.
        "--resynth-weight",
        "100000",
    ]
    .into_iter()
    .chain(backend.iter().copied())
    .map(String::from)
    .collect();
    args.extend([
        "-eps".into(),
        epsilon.to_string(),
        "--max-iters".into(),
        max_iters.to_string(),
        "-q".into(),
        "4".into(),
        "--rules-dir".into(),
        repo_root().join("rules").to_str().unwrap().into(),
        "-out".into(),
        out_dir.path().to_str().unwrap().into(),
        circuit.display().to_string(),
    ]);
    let cli = Cli::parse_from_args(args).unwrap();
    let outcome = optimize(&cli).unwrap();

    let name = format!(
        "optimized__{}",
        circuit.file_name().unwrap().to_string_lossy()
    );
    let optimized = out_dir.path().join(name);

    let claimed = outcome.result.best.accumulated_error;
    assert!(
        claimed <= epsilon,
        "{bench}: claimed accumulated_error {claimed:.3e} exceeds epsilon {epsilon:.3e}"
    );

    // The actual distance, measured independently of everything the optimizer tracked.
    // The subadditive chain is: global operator distance <= sum of per-call operator
    // distances <= sum of per-call Frobenius charges = `accumulated_error`; and
    // `hs_distance` sits below the operator distance. So `hs <= claimed <= epsilon`
    // is the guarantee. The global Frobenius distance is reported to callers but not
    // bounded by the claim (see the module doc).
    let registry = GateRegistry::with_builtins();
    let a = qasm::parse(&std::fs::read_to_string(circuit).unwrap()).unwrap();
    let b = qasm::parse(&std::fs::read_to_string(&optimized).unwrap()).unwrap();
    // Over the *used* wires of either circuit, not the declared register: these
    // benchmarks declare `qreg q[16]` however few wires they touch, and an idle wire
    // is an identity tensor factor that changes no distance here.
    let mut qs: Vec<QubitId> = Vec::new();
    for dag in [&a, &b] {
        for idx in dag.gate_indices() {
            for q in &dag.gate(idx).qubits {
                if !qs.contains(q) {
                    qs.push(q.clone());
                }
            }
        }
    }
    qs.sort();
    assert!(
        qs.len() <= 10,
        "{bench}: too wide for the dense check; pick smaller sweep circuits"
    );
    let ua = Unitary::from_dag_over(&a, qs.clone(), &registry).unwrap();
    let ub = Unitary::from_dag_over(&b, qs, &registry).unwrap();
    let measured_phase_invariant = phase_invariant_distance(&ua, &ub);
    let measured_hs = hs_distance(&ua, &ub);

    assert!(
        measured_hs <= epsilon,
        "{bench}: hs_distance {measured_hs:.3e} exceeds epsilon {epsilon:.3e}"
    );
    // The slack exists because `hs <= claimed` is a theorem about exact arithmetic
    // and neither side is computed exactly. Its dominant term is not float noise: the
    // chain above charges every rewrite-rule application zero, but a rewrite is only
    // verified equivalent to within the loader's and matcher's 1e-9 tolerances, so a
    // run applying N rules can drift up to ~N * 1e-9 without a unit of charge —
    // order 1e-7 here, machine-epsilon in practice since real rules are identities.
    // 1e-6 sits above that and far below everything the assertion must catch: one
    // missed dust charge is ~7e-5, seventy times the slack.
    assert!(
        measured_hs <= claimed + 1e-6,
        "{bench}: the optimizer claimed {claimed:.3e} of error but the circuit is \
         {measured_hs:.3e} away in hs_distance — the claim under-reports reality"
    );

    Checked {
        outcome,
        claimed,
        measured_phase_invariant,
        measured_hs,
    }
}

/// One circuit, every assertion, in every build: an approximate backend's real error is
/// charged to the budget, the budget holds, and reality agrees with the claim.
#[test]
fn an_approximate_backend_stays_within_epsilon_claimed_and_measured() {
    if !python3_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let worker = approximate_worker(dir.path());
    let circuit = dusted("nam_rz/tof_3.qasm", dir.path());
    let backend = [
        "-resynth",
        "BQSKIT",
        "--bqskit-worker",
        worker.to_str().unwrap(),
    ];
    let checked = check_budget(&circuit, "NAM", "TOTAL", 1e-2, 40, &backend);

    let stats = checked.outcome.resynth.expect("a backend was configured");
    assert!(
        stats.accepted > 0,
        "the approximate backend was never accepted, so the budget was never spent: {stats:?}"
    );
    assert!(
        checked.claimed > 0.0,
        "acceptances happened but no error was charged — the accounting is asleep"
    );
}

/// The sweep: several small benchmarks, same assertions each, plus the sweep-level fact
/// that the runs really did spend budget somewhere.
#[test]
#[ignore = "sweeps several circuits through resynthesis; run in release"]
fn resynthesis_sweep_stays_within_epsilon_claimed_and_measured() {
    if !python3_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let worker = approximate_worker(dir.path());

    // Chosen by *used* wires (4-5 each), not by name or file size: every nam file
    // declares `qreg q[16]`, and some innocuous-looking ones touch most of it.
    let benches = [
        "nam_rz/ham3_102.qasm",
        "nam_rz/ex-1_166.qasm",
        "nam_rz/3_17_13.qasm",
        "nam_rz/miller_11.qasm",
        "nam_rz/4gt11_84.qasm",
        "nam_rz/decod24-v0_38.qasm",
    ];
    let backend = [
        "-resynth",
        "BQSKIT",
        "--bqskit-worker",
        worker.to_str().unwrap(),
    ];
    let mut spent = 0usize;
    for bench in benches {
        let circuit = dusted(bench, dir.path());
        let checked = check_budget(&circuit, "NAM", "TOTAL", 1e-2, 60, &backend);
        let stats = checked.outcome.resynth.expect("a backend was configured");
        if stats.accepted > 0 && checked.claimed > 0.0 {
            spent += 1;
        }
        println!(
            "{bench}: accepted={} claimed={:.3e} measured={:.3e} hs={:.3e}",
            stats.accepted, checked.claimed, checked.measured_phase_invariant, checked.measured_hs
        );
    }
    assert!(
        spent > 0,
        "no run in the sweep ever charged the budget, so the accounting went untested"
    );
}

/// The real BQSKit, end to end: `guoq` spawns `py/bqskit_worker.py`, BQSKit
/// synthesizes each partition's unitary directly into the NAM basis, and every accepted
/// result's measured error is charged against the budget. The input carries dust so a
/// genuinely approximate result has something to win; whether BQSKit's results come
/// back exact or approximate is its own business, which is why this test asserts the
/// budget's guarantees and non-vacuity but not `claimed > 0` — that determinism is the
/// stub's job.
#[test]
#[ignore = "runs the real BQSKit; slow, run in release"]
fn bqskit_end_to_end_stays_within_epsilon_claimed_and_measured() {
    if !bqskit_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let circuit = dusted("nam_rz/tof_3.qasm", dir.path());
    let worker = repo_root().join("py/bqskit_worker.py");
    let backend = [
        "-resynth",
        "BQSKIT",
        "--bqskit-worker",
        worker.to_str().unwrap(),
        // Level 1 keeps a call at a few seconds; the budget arithmetic under test is
        // the same at every level.
        "--bqskit-opt-level",
        "1",
    ];
    let checked = check_budget(&circuit, "NAM", "TOTAL", 1e-2, 8, &backend);

    let stats = checked.outcome.resynth.expect("a backend was configured");
    assert!(
        stats.accepted > 0,
        "the real BQSKit never produced an acceptable result: {stats:?}"
    );
}

/// The real Synthetiq, end to end: the target unitary is written in Synthetiq's on-disk
/// format and qubit-ordering convention, the C++ binary searches for Clifford+T
/// circuits, and the best result is read back, measured, and charged. The first run of
/// this test caught the port writing the target in the wrong qubit order — Synthetiq
/// reads it little-endian, as Qiskit does — so beyond the budget's guarantees it pins
/// the whole file-format contract with the real binary.
#[test]
#[ignore = "runs the real Synthetiq binary; run in release"]
fn synthetiq_end_to_end_stays_within_epsilon_claimed_and_measured() {
    let Some(binary) = synthetiq_binary() else {
        return;
    };
    let circuit = repo_root().join("benchmarks/nam_t_tdg/tof_3.qasm");
    let backend = [
        "-resynth",
        "SYNTHETIQ",
        "--synthetiq-binary",
        binary.to_str().unwrap(),
        // A handful of candidates per call keeps the search short; correctness of what
        // comes back is what is under test, not synthesis quality.
        "--synthetiq-num-circuits",
        "10",
        "--synthetiq-threads",
        "4",
    ];
    let checked = check_budget(&circuit, "CLIFFORDT", "FT", 1e-2, 8, &backend);

    let stats = checked.outcome.resynth.expect("a backend was configured");
    assert!(
        stats.accepted > 0,
        "the real Synthetiq never produced an acceptable result: {stats:?}"
    );
}
