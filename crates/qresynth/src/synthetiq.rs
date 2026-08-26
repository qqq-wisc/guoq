//! The Synthetiq backend, without Python.
//!
//! Synthetiq is a C++ program (`lib/synthetiq`, built to `bin/main`). The reference
//! reached it through `resynth.py`, which wrote the input file, ran the binary, read the
//! output directory, picked the best circuit, and translated gate sets — none of which
//! needs an interpreter. This module does the same work directly, so the Clifford+T path
//! runs with no Python process at all.
//!
//! The three things the Python was actually contributing are reproduced here: the
//! non-standard-gate fixups Synthetiq emits, the best-by-T-count selection from
//! `main_analysis`, and the `qubits[` to `q[` register rename.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use num_complex::Complex64;

use qcircuit::{qasm, Dag, GateRegistry};
use qsemantics::Unitary;

use crate::backend::{Backend, Request};

/// How to run Synthetiq.
#[derive(Debug, Clone)]
pub struct SynthetiqConfig {
    /// Path to the `main` binary.
    pub binary: PathBuf,
    /// Working directory; Synthetiq resolves `data/input` and `data/output` beneath it.
    pub working_dir: PathBuf,
    /// How many candidate circuits to ask for.
    pub num_circuits: u32,
    /// Threads to give it.
    pub threads: u32,
    /// How long to wait before giving up.
    pub timeout: std::time::Duration,
}

impl Default for SynthetiqConfig {
    fn default() -> Self {
        Self {
            binary: PathBuf::from("lib/synthetiq/bin/main"),
            working_dir: PathBuf::from("lib/synthetiq"),
            num_circuits: 100,
            threads: 8,
            timeout: std::time::Duration::from_secs(300),
        }
    }
}

/// Synthetiq, driven directly.
#[derive(Debug, Clone)]
pub struct Synthetiq {
    pub config: SynthetiqConfig,
}

impl Synthetiq {
    pub fn new(config: SynthetiqConfig) -> Self {
        Self { config }
    }

    /// `true` if the binary is present and executable.
    pub fn available(&self) -> bool {
        self.config.binary.is_file()
    }
}

impl Backend for Synthetiq {
    fn run(&self, request: &Request, registry: &GateRegistry) -> Result<Option<Dag>> {
        if !self.available() {
            bail!(
                "Synthetiq binary not found at {}. Build it with `make` in lib/synthetiq, \
                 or pass --synthetiq-binary.",
                self.config.binary.display()
            );
        }

        let n = request.circuit.num_qubits();
        if n == 0 {
            return Ok(None);
        }
        let unitary = target_unitary(&request.circuit, registry)
            .context("building the block's unitary for Synthetiq")?;

        // A unique name per call. The reference used `circ_{randint(0, 1000000)}`, which
        // collides in a long run and leaves the residue behind on any early exit.
        let job = unique_name();
        let input_dir = self.config.working_dir.join("data/input");
        let output_dir = self.config.working_dir.join("data/output").join(&job);
        std::fs::create_dir_all(&input_dir)
            .with_context(|| format!("creating {}", input_dir.display()))?;
        let input_file = input_dir.join(format!("{job}.txt"));

        let guard = Cleanup {
            files: vec![input_file.clone()],
            dirs: vec![output_dir.clone()],
        };

        std::fs::write(&input_file, encode_target(&job, unitary.matrix(), n))
            .with_context(|| format!("writing {}", input_file.display()))?;

        let status = Command::new(&self.config.binary)
            .current_dir(&self.config.working_dir)
            .arg(format!("{job}.txt"))
            .arg("-c")
            .arg(self.config.num_circuits.to_string())
            .arg("-eps")
            .arg(format!("{:e}", request.epsilon_hint.max(1e-15)))
            .arg("-h")
            .arg(self.config.threads.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .with_context(|| format!("running {}", self.config.binary.display()))?;

        if !status.success() {
            return Ok(None);
        }
        let best = read_best(&output_dir, registry)?;
        drop(guard);
        Ok(best)
    }

    fn name(&self) -> &str {
        "synthetiq"
    }
}

/// Removes Synthetiq's scratch files even if the call fails.
struct Cleanup {
    files: Vec<PathBuf>,
    dirs: Vec<PathBuf>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        for f in &self.files {
            let _ = std::fs::remove_file(f);
        }
        for d in &self.dirs {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}

/// The block's unitary in Synthetiq's qubit-ordering convention.
///
/// Synthetiq reads the target matrix little-endian — `qubits[0]` is the *least*
/// significant bit of a basis index, as in Qiskit — while [`Unitary`] puts the first
/// qubit of its ordering in the most significant position. Building over the reversed
/// ordering hands Synthetiq the matrix it expects, so the circuit it returns implements
/// the block under this codebase's convention too. The reference never faced this
/// because it wrote the matrix with Qiskit's `Operator`, which shares Synthetiq's
/// convention; encoding our matrix directly makes every result a bit-reversal of the
/// block, which the measurement then rejects as over budget.
fn target_unitary(circuit: &Dag, registry: &GateRegistry) -> Result<Unitary> {
    let mut order: Vec<qcircuit::QubitId> = circuit.qubits().to_vec();
    order.reverse();
    Ok(Unitary::from_dag_over(circuit, order, registry)?)
}

fn unique_name() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("guoq_{}_{n}", std::process::id())
}

/// Synthetiq's target-unitary file format.
///
/// Name, qubit count, then the matrix as `(re,im)` pairs, then a mask of the entries that
/// matter — all ones, since the whole unitary is being matched.
pub fn encode_target(name: &str, matrix: &ndarray::Array2<Complex64>, num_qubits: usize) -> String {
    let dim = 1usize << num_qubits;
    let mut out = String::with_capacity(dim * dim * 24);
    out.push_str(name);
    out.push('\n');
    out.push_str(&num_qubits.to_string());
    out.push('\n');
    for i in 0..dim {
        for j in 0..dim {
            let z = matrix[[i, j]];
            out.push_str(&format!("({},{}) ", z.re, z.im));
        }
        out.push('\n');
    }
    for _ in 0..dim {
        for _ in 0..dim {
            out.push_str("1 ");
        }
        out.push('\n');
    }
    out
}

/// Gates Synthetiq emits that are not standard QASM.
///
/// Lifted from the reference's `NON_STANDARD_GATES` table in `resynth.py`.
fn fix_non_standard(qasm_text: &str) -> String {
    qasm_text.replace("scz", "cz").replace(" U ", " cx ")
}

/// A candidate Synthetiq produced.
#[derive(Debug, Clone)]
struct Candidate {
    dag: Dag,
    t_count: usize,
    t_depth: usize,
    score: f64,
}

/// Read Synthetiq's output directory and pick the best circuit.
///
/// Selection is by T-count, then T-depth, then Synthetiq's own score — the same order as
/// `main_analysis` in the reference, which had the other two criteria commented out.
fn read_best(dir: &Path, registry: &GateRegistry) -> Result<Option<Dag>> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(None);
    };
    let mut best: Option<Candidate> = None;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        // Synthetiq names its output `<score>-<count>-...`.
        let score = path
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split('-').next())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(f64::MAX);

        let cleaned = fix_non_standard(&text).replace("qubits[", "q[");
        let Ok(dag) = qasm::parse(&cleaned) else {
            continue;
        };
        if dag
            .gate_indices()
            .iter()
            .any(|&i| registry.get(&dag.gate(i).gate).is_err())
        {
            continue;
        }

        let cand = Candidate {
            t_count: count_t(&dag),
            t_depth: t_depth(&dag),
            score,
            dag,
        };
        best = Some(match best {
            None => cand,
            Some(b) => {
                let better =
                    (cand.t_count, cand.t_depth, cand.score) < (b.t_count, b.t_depth, b.score);
                if better {
                    cand
                } else {
                    b
                }
            }
        });
    }
    Ok(best.map(|c| c.dag))
}

fn count_t(dag: &Dag) -> usize {
    dag.gate_indices()
        .into_iter()
        .filter(|&i| {
            let op = dag.gate(i);
            qcircuit::is_t_gate(&op.gate, &op.params)
        })
        .count()
}

/// Longest chain of T gates on any wire.
fn t_depth(dag: &Dag) -> usize {
    let mut depth = 0usize;
    for layer in dag.layers() {
        if layer.iter().any(|&i| {
            let op = dag.gate(i);
            qcircuit::is_t_gate(&op.gate, &op.params)
        }) {
            depth += 1;
        }
    }
    depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    #[test]
    fn encodes_a_target_unitary() {
        let dag = qasm::parse("h q[0];").unwrap();
        let u = Unitary::from_dag(&dag, &reg()).unwrap();
        let text = encode_target("job", u.matrix(), 1);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "job");
        assert_eq!(lines[1], "1");
        // Two matrix rows, then two mask rows.
        assert_eq!(lines.len(), 6);
        assert!(lines[2].starts_with('('));
        assert_eq!(lines[4].trim(), "1 1");
        assert_eq!(lines[5].trim(), "1 1");
    }

    #[test]
    fn encoding_is_parseable_back_to_the_same_numbers() {
        let dag = qasm::parse("h q[0]; t q[0];").unwrap();
        let u = Unitary::from_dag(&dag, &reg()).unwrap();
        let text = encode_target("j", u.matrix(), 1);
        let rows: Vec<&str> = text.lines().skip(2).take(2).collect();
        let mut got = Array2::<Complex64>::zeros((2, 2));
        for (i, row) in rows.iter().enumerate() {
            for (j, tok) in row.split_whitespace().enumerate() {
                let inner = tok.trim_start_matches('(').trim_end_matches(')');
                let (re, im) = inner.split_once(',').unwrap();
                got[[i, j]] = Complex64::new(re.parse().unwrap(), im.parse().unwrap());
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((got[[i, j]] - u.matrix()[[i, j]]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn encodes_multi_qubit_targets() {
        let dag = qasm::parse("cx q[0], q[1];").unwrap();
        let u = Unitary::from_dag(&dag, &reg()).unwrap();
        let text = encode_target("j", u.matrix(), 2);
        // 2 header lines, 4 matrix rows, 4 mask rows.
        assert_eq!(text.lines().count(), 10);
    }

    #[test]
    fn non_standard_gates_are_fixed_up() {
        assert_eq!(fix_non_standard("scz q[0], q[1];"), "cz q[0], q[1];");
        assert!(fix_non_standard("h q[0]; U q[0], q[1];").contains("cx"));
    }

    #[test]
    fn t_counting() {
        let d = qasm::parse("t q[0]; tdg q[0]; h q[0]; t q[1];").unwrap();
        assert_eq!(count_t(&d), 3);
        let d = qasm::parse("h q[0]; cx q[0], q[1];").unwrap();
        assert_eq!(count_t(&d), 0);
    }

    #[test]
    fn t_depth_counts_layers_containing_t() {
        // Two T gates on the same wire: depth 2.
        let d = qasm::parse("t q[0]; t q[0];").unwrap();
        assert_eq!(t_depth(&d), 2);
        // Two T gates on different wires: one layer, depth 1.
        let d = qasm::parse("t q[0]; t q[1];").unwrap();
        assert_eq!(t_depth(&d), 1);
        let d = qasm::parse("h q[0];").unwrap();
        assert_eq!(t_depth(&d), 0);
    }

    #[test]
    fn selects_the_lowest_t_count() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("0.5-1-a.qasm"), "t q[0]; t q[0]; t q[0];").unwrap();
        std::fs::write(dir.path().join("0.9-1-b.qasm"), "t q[0];").unwrap();
        std::fs::write(dir.path().join("0.1-1-c.qasm"), "t q[0]; t q[0];").unwrap();
        let best = read_best(dir.path(), &reg()).unwrap().unwrap();
        assert_eq!(count_t(&best), 1, "should pick the fewest T gates");
    }

    #[test]
    fn ties_on_t_count_are_broken_by_depth_then_score() {
        let dir = tempfile::tempdir().unwrap();
        // Both have two T gates; the first has them on one wire (depth 2), the second on
        // two wires (depth 1).
        std::fs::write(dir.path().join("0.1-1-deep.qasm"), "t q[0]; t q[0];").unwrap();
        std::fs::write(dir.path().join("0.9-1-flat.qasm"), "t q[0]; t q[1];").unwrap();
        let best = read_best(dir.path(), &reg()).unwrap().unwrap();
        assert_eq!(t_depth(&best), 1, "should prefer the shallower circuit");
    }

    #[test]
    fn register_names_are_translated() {
        let dir = tempfile::tempdir().unwrap();
        // Synthetiq writes its register as `qubits`.
        std::fs::write(dir.path().join("0.1-1-x.qasm"), "h qubits[0]; t qubits[1];").unwrap();
        let best = read_best(dir.path(), &reg()).unwrap().unwrap();
        for q in best.qubits() {
            assert!(q.starts_with("q["), "register not renamed: {q}");
        }
    }

    #[test]
    fn an_empty_or_missing_directory_yields_nothing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_best(dir.path(), &reg()).unwrap().is_none());
        assert!(read_best(Path::new("/nonexistent"), &reg())
            .unwrap()
            .is_none());
    }

    #[test]
    fn unparseable_candidates_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("0.1-1-bad.qasm"), "this is not qasm ((").unwrap();
        std::fs::write(dir.path().join("0.2-1-good.qasm"), "h q[0];").unwrap();
        let best = read_best(dir.path(), &reg()).unwrap().unwrap();
        assert_eq!(best.gate_count(), 1);
    }

    #[test]
    fn candidates_using_unknown_gates_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("0.1-1-bad.qasm"), "notagate q[0];").unwrap();
        assert!(read_best(dir.path(), &reg()).unwrap().is_none());
    }

    #[test]
    fn a_missing_binary_is_a_clear_error() {
        let s = Synthetiq::new(SynthetiqConfig {
            binary: PathBuf::from("/nonexistent/main"),
            ..SynthetiqConfig::default()
        });
        assert!(!s.available());
        let req = Request {
            circuit: qasm::parse("h q[0];").unwrap(),
            target_gate_set: "none".into(),
            epsilon_hint: 1e-8,
        };
        let err = s.run(&req, &reg()).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("lib/synthetiq") || msg.contains("not found"),
            "{msg}"
        );
    }

    /// The target handed to Synthetiq must be little-endian: `cx q[0], q[1]` (control
    /// `q[0]`) has its X block on the *odd* basis indices, where `q[0]` is the low bit.
    /// Encoding this codebase's own big-endian matrix instead makes Synthetiq return
    /// bit-reversed circuits, every one of which the measurement rejects — which is how
    /// the mismatch was found.
    #[test]
    fn the_target_is_encoded_in_synthetiqs_qubit_order() {
        let dag = qasm::parse("cx q[0], q[1];").unwrap();
        let u = target_unitary(&dag, &reg()).unwrap();
        let m = u.matrix();
        // |01> (index 1: q1=0, q0=1) maps to |11> (index 3), and back.
        for (i, j, want) in [
            (0, 0, 1.0),
            (2, 2, 1.0),
            (3, 1, 1.0),
            (1, 3, 1.0),
            (1, 1, 0.0),
            (3, 3, 0.0),
        ] {
            assert!(
                (m[[i, j]] - Complex64::new(want, 0.0)).norm() < 1e-12,
                "entry [{i},{j}] should be {want}"
            );
        }
    }

    /// Job names must not collide, which the reference's `randint(0, 1000000)` does in a
    /// long run.
    #[test]
    fn job_names_are_unique() {
        let names: std::collections::HashSet<String> = (0..1000).map(|_| unique_name()).collect();
        assert_eq!(names.len(), 1000);
    }

    #[test]
    fn scratch_files_are_removed_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("scratch.txt");
        let d = dir.path().join("outdir");
        std::fs::write(&f, "x").unwrap();
        std::fs::create_dir_all(&d).unwrap();
        {
            let _g = Cleanup {
                files: vec![f.clone()],
                dirs: vec![d.clone()],
            };
        }
        assert!(!f.exists());
        assert!(!d.exists());
    }
}
