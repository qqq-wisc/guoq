//! The BQSKit backend, as a worker process rather than a server.
//!
//! BQSKit is a large numerical synthesis stack; reimplementing it is not a porting task.
//! What *can* go is the way the reference reached it: `resynth.py` ran an HTTP server on
//! `localhost:8080` that the user started by hand before invoking GUOQ, kept running
//! across invocations, and stopped with a keyboard interrupt. `wisq` existed partly to
//! babysit that lifecycle.
//!
//! Here `guoq` spawns `py/bqskit_worker.py` itself, talks to it over stdin and stdout in
//! newline-delimited JSON, and kills it on drop. No port, no manual step, and the worker
//! only starts if `-resynth BQSKIT` is actually asked for. Everything that is not "call
//! BQSKit" — gate-set translation, choosing among results, measuring error — happens on
//! this side, which is why the Python is 60 lines rather than 400.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use qcircuit::{qasm, Dag, GateRegistry};

use crate::backend::{Backend, Request};

/// How to start the worker.
#[derive(Debug, Clone)]
pub struct BqskitConfig {
    /// Python interpreter to use.
    pub python: String,
    /// Path to `bqskit_worker.py`.
    pub worker: PathBuf,
    /// BQSKit optimization level.
    pub opt_level: u8,
    /// How long to wait for the worker to report that its imports finished.
    pub startup_timeout: std::time::Duration,
    /// How long to wait for the reply to one request.
    ///
    /// BQSKit's `Compiler` forks a runtime server; if that server dies (it did, when
    /// two instances collided on its fixed default port), the worker's `compile` call
    /// blocks forever and, without this bound, so did the whole optimization. Matches
    /// the Synthetiq default.
    pub request_timeout: std::time::Duration,
}

impl Default for BqskitConfig {
    fn default() -> Self {
        Self {
            python: "python3".into(),
            worker: PathBuf::from("py/bqskit_worker.py"),
            opt_level: 3,
            startup_timeout: std::time::Duration::from_secs(120),
            request_timeout: std::time::Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Serialize)]
struct WorkerRequest<'a> {
    circuit: &'a str,
    opt_level: u8,
    epsilon: f64,
    /// The gate set's `resynth_target` name; the worker turns known names into a
    /// BQSKit `MachineModel` so results land directly in the requested basis.
    target: &'a str,
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    #[serde(default)]
    ok: bool,
    #[serde(default)]
    circuit: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReadyLine {
    #[serde(default)]
    ready: bool,
    #[serde(default)]
    error: Option<String>,
}

/// A worker gave no reply within the configured deadline.
#[derive(Debug)]
struct RequestTimedOut(std::time::Duration);

impl std::fmt::Display for RequestTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the BQSKit worker gave no reply within {:.0?}", self.0)
    }
}

impl std::error::Error for RequestTimedOut {}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    /// Lines from the worker's stdout, read by a dedicated thread so a reply can be
    /// awaited with a deadline; a blocking `read_line` here is what let a hung worker
    /// hang the whole optimization. The thread exits when the worker's stdout closes.
    lines: std::sync::mpsc::Receiver<std::io::Result<String>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        // The worker is ours; it does not outlive us.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// BQSKit, reached through a worker process this type owns.
pub struct Bqskit {
    config: BqskitConfig,
    worker: Mutex<Option<Worker>>,
    /// The last message a failed request came back with, for diagnostics.
    last_error: Mutex<Option<String>>,
}

impl std::fmt::Debug for Bqskit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bqskit")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Bqskit {
    pub fn new(config: BqskitConfig) -> Self {
        Self {
            config,
            worker: Mutex::new(None),
            last_error: Mutex::new(None),
        }
    }

    /// Why the most recent request failed, if one did.
    ///
    /// A failed synthesis is not an error for the search — it just does not get a result
    /// this iteration — but silently discarding the reason makes a misconfigured backend
    /// impossible to diagnose. The reference had no equivalent: on any failure its HTTP
    /// client returned the *input* circuit, so a broken backend looked like a working one
    /// that happened not to improve anything.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|g| g.clone())
    }

    /// `true` if the worker script exists. Whether BQSKit imports is only known once the
    /// worker starts, and is reported then.
    pub fn worker_script_present(&self) -> bool {
        self.config.worker.is_file()
    }

    fn spawn(&self) -> Result<Worker> {
        if !self.worker_script_present() {
            bail!(
                "BQSKit worker script not found at {}. Pass --bqskit-worker.",
                self.config.worker.display()
            );
        }
        let mut child = Command::new(&self.config.python)
            .arg(&self.config.worker)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| {
                format!(
                    "starting `{} {}`",
                    self.config.python,
                    self.config.worker.display()
                )
            })?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let (tx, lines) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                let sent = match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => tx.send(Ok(line)),
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                };
                if sent.is_err() {
                    break;
                }
            }
        });

        let mut worker = Worker {
            child,
            stdin,
            lines,
        };

        // The worker announces itself once BQSKit has finished importing, which takes
        // several seconds. Without the handshake the first request would look like a hang.
        let line = Self::next_line(&mut worker, self.config.startup_timeout)
            .context("waiting for the BQSKit worker to start")?;
        let ready: ReadyLine = serde_json::from_str(line.trim())
            .with_context(|| format!("unexpected greeting from the BQSKit worker: {line:?}"))?;
        if !ready.ready {
            bail!(
                "the BQSKit worker could not start: {}",
                ready.error.unwrap_or_else(|| "unknown reason".into())
            );
        }

        Ok(worker)
    }

    /// The next stdout line, or an error if `timeout` passes first.
    fn next_line(worker: &mut Worker, timeout: std::time::Duration) -> Result<String> {
        use std::sync::mpsc::RecvTimeoutError;
        match worker.lines.recv_timeout(timeout) {
            Ok(Ok(line)) => Ok(line),
            Ok(Err(e)) => Err(e).context("reading from the BQSKit worker"),
            Err(RecvTimeoutError::Timeout) => Err(anyhow::Error::new(RequestTimedOut(timeout))),
            Err(RecvTimeoutError::Disconnected) => bail!("the BQSKit worker closed its output"),
        }
    }

    /// Send one request, restarting the worker once if it has died.
    ///
    /// A timeout is not retried: waiting out the deadline already cost the search that
    /// much wall clock, and resending the same request would cost it again. The hung
    /// worker is killed, so the *next* attempt starts from a fresh one.
    fn exchange(&self, request: &WorkerRequest<'_>) -> Result<WorkerResponse> {
        let mut guard = self.worker.lock().expect("worker mutex");
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some(self.spawn()?);
            }
            let worker = guard.as_mut().expect("worker present");
            match Self::send(worker, request, self.config.request_timeout) {
                Ok(response) => return Ok(response),
                Err(e) => {
                    // Dropping the worker kills the process, hung or dead alike.
                    *guard = None;
                    if attempt > 0 || e.downcast_ref::<RequestTimedOut>().is_some() {
                        return Err(e);
                    }
                }
            }
        }
        unreachable!("the loop returns on both branches")
    }

    fn send(
        worker: &mut Worker,
        request: &WorkerRequest<'_>,
        timeout: std::time::Duration,
    ) -> Result<WorkerResponse> {
        let line = serde_json::to_string(request)?;
        worker.stdin.write_all(line.as_bytes())?;
        worker.stdin.write_all(b"\n")?;
        worker.stdin.flush()?;

        let response = Self::next_line(worker, timeout)?;
        Ok(serde_json::from_str(response.trim())?)
    }
}

impl Backend for Bqskit {
    fn run(&self, request: &Request, _registry: &GateRegistry) -> Result<Option<Dag>> {
        if request.circuit.gate_count() == 0 {
            return Ok(None);
        }
        let text = qasm::to_qasm(&request.circuit);
        let response = self.exchange(&WorkerRequest {
            circuit: &text,
            opt_level: self.config.opt_level,
            epsilon: request.epsilon_hint.max(1e-15),
            target: &request.target_gate_set,
        })?;

        if let Some(message) = &response.error {
            if let Ok(mut slot) = self.last_error.lock() {
                *slot = Some(message.clone());
            }
        }
        if !response.ok {
            // A failed request is not fatal: the search simply does not get a result
            // this time. The reference returned the *input* circuit on any error
            // (`Bqskit.socket` and `Synthetiq.socket` both end `return circuit;`), which
            // the caller then spliced back in as though work had been done -- and, with
            // nothing measuring the result, could not tell the difference.
            return Ok(None);
        }
        let Some(circuit) = response.circuit else {
            return Ok(None);
        };
        Ok(qasm::parse(&circuit).ok())
    }

    fn name(&self) -> &str {
        "bqskit"
    }

    fn shutdown(&self) {
        if let Ok(mut guard) = self.worker.lock() {
            *guard = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg() -> GateRegistry {
        GateRegistry::with_builtins()
    }

    fn request(src: &str) -> Request {
        Request {
            circuit: qasm::parse(src).unwrap(),
            target_gate_set: "none".into(),
            epsilon_hint: 1e-8,
        }
    }

    #[test]
    fn a_missing_worker_script_is_a_clear_error() {
        let b = Bqskit::new(BqskitConfig {
            worker: PathBuf::from("/nonexistent/worker.py"),
            ..BqskitConfig::default()
        });
        assert!(!b.worker_script_present());
        let msg = format!("{:#}", b.run(&request("h q[0];"), &reg()).unwrap_err());
        assert!(msg.contains("worker"), "unhelpful message: {msg}");
    }

    #[test]
    fn an_empty_circuit_needs_no_worker() {
        let b = Bqskit::new(BqskitConfig {
            worker: PathBuf::from("/nonexistent/worker.py"),
            ..BqskitConfig::default()
        });
        // No gates, so nothing is asked of the backend and the missing script never
        // matters.
        assert!(b.run(&request("qreg q[2];"), &reg()).unwrap().is_none());
    }

    /// A worker that answers the handshake and then nothing. Without the request
    /// deadline this test never finishes — which is exactly what happened to a whole
    /// optimization when BQSKit's runtime server died underneath the worker.
    #[test]
    fn a_hung_worker_times_out_instead_of_hanging() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("hang.sh");
        std::fs::write(&script, "#!/bin/sh\necho '{\"ready\": true}'\nsleep 600\n").unwrap();
        let b = Bqskit::new(BqskitConfig {
            python: "/bin/sh".into(),
            worker: script,
            request_timeout: std::time::Duration::from_millis(200),
            ..BqskitConfig::default()
        });
        let started = std::time::Instant::now();
        let err = b.run(&request("h q[0];"), &reg()).unwrap_err();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(30),
            "the deadline was not enforced"
        );
        let msg = format!("{err:#}");
        assert!(msg.contains("no reply"), "unhelpful message: {msg}");
        // The hung worker was killed; nothing is left holding the slot.
        assert!(b.worker.lock().unwrap().is_none());
    }

    #[test]
    fn the_shipped_worker_script_exists() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let script = root.join("py/bqskit_worker.py");
        assert!(script.is_file(), "{} is missing", script.display());
        let text = std::fs::read_to_string(&script).unwrap();
        // The protocol this module speaks.
        assert!(text.contains("\"ready\""));
        assert!(text.contains("\"ok\""));
        assert!(text.contains("synthesis_epsilon"));
        // No server: check the imports, since the module docstring names what this
        // replaces.
        assert!(!text.contains("import socketserver"));
        assert!(!text.contains("from http.server"));
        assert!(!text.contains("serve_forever"));
        assert!(!text.contains(".bind("));
    }

    /// The worker script must be small: everything that does not need Python was moved
    /// to the Rust side.
    #[test]
    fn the_worker_script_is_small() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let text = std::fs::read_to_string(root.join("py/bqskit_worker.py")).unwrap();
        let code_lines = text
            .lines()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty() && !t.starts_with('#')
            })
            .count();
        assert!(
            code_lines < 120,
            "the worker has grown to {code_lines} lines; keep logic on the Rust side"
        );
    }

    #[test]
    fn request_serialisation_matches_the_protocol() {
        let json = serde_json::to_string(&WorkerRequest {
            circuit: "h q[0];",
            opt_level: 3,
            epsilon: 1e-8,
            target: "nam",
        })
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["circuit"], "h q[0];");
        assert_eq!(parsed["opt_level"], 3);
        assert!((parsed["epsilon"].as_f64().unwrap() - 1e-8).abs() < 1e-20);
        assert_eq!(parsed["target"], "nam");
    }

    #[test]
    fn response_parsing_handles_both_shapes() {
        let ok: WorkerResponse =
            serde_json::from_str(r#"{"ok": true, "circuit": "h q[0];"}"#).unwrap();
        assert!(ok.ok);
        assert_eq!(ok.circuit.as_deref(), Some("h q[0];"));

        let err: WorkerResponse =
            serde_json::from_str(r#"{"ok": false, "error": "boom"}"#).unwrap();
        assert!(!err.ok);
        assert_eq!(err.error.as_deref(), Some("boom"));

        // A truncated response must not panic.
        assert!(serde_json::from_str::<WorkerResponse>("{}").is_ok());
    }

    #[test]
    fn ready_line_parsing() {
        let r: ReadyLine = serde_json::from_str(r#"{"ready": true}"#).unwrap();
        assert!(r.ready);
        let r: ReadyLine =
            serde_json::from_str(r#"{"ready": false, "error": "import failed"}"#).unwrap();
        assert!(!r.ready);
        assert!(r.error.unwrap().contains("import"));
    }

    /// A stand-in worker exercises the protocol without needing BQSKit installed.
    #[test]
    fn the_protocol_round_trips_against_a_stub_worker() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("stub.py");
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
    # Return a fixed, genuinely equivalent circuit: h;h == identity, so an empty result.
    sys.stdout.write(json.dumps({"ok": True, "circuit": "OPENQASM 2.0;\nqreg q[1];\n"}) + "\n")
    sys.stdout.flush()
"#,
        )
        .unwrap();

        let b = Bqskit::new(BqskitConfig {
            worker: script,
            ..BqskitConfig::default()
        });
        if Command::new("python3").arg("--version").output().is_err() {
            return; // no interpreter here; the rest of the suite still covers the logic
        }
        let out = b.run(&request("h q[0]; h q[0];"), &reg()).unwrap();
        let out = out.expect("the stub always answers");
        assert_eq!(out.gate_count(), 0);

        // A second request reuses the same worker rather than starting another.
        let again = b.run(&request("h q[0]; h q[0];"), &reg()).unwrap();
        assert!(again.is_some());
        b.shutdown();
    }

    #[test]
    fn a_worker_reporting_failure_yields_no_result() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("failing.py");
        std::fs::write(
            &script,
            r#"
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\n")
sys.stdout.flush()
for line in sys.stdin:
    if line.strip():
        sys.stdout.write(json.dumps({"ok": False, "error": "synthesis failed"}) + "\n")
        sys.stdout.flush()
"#,
        )
        .unwrap();
        let b = Bqskit::new(BqskitConfig {
            worker: script,
            ..BqskitConfig::default()
        });
        if Command::new("python3").arg("--version").output().is_err() {
            return;
        }
        // A failure is reported as "no result", never as the input handed back unchanged.
        assert!(b.run(&request("h q[0];"), &reg()).unwrap().is_none());
        // ...and the reason is kept, so a misconfigured backend can be diagnosed.
        assert_eq!(b.last_error().as_deref(), Some("synthesis failed"));
    }

    #[test]
    fn a_worker_that_cannot_import_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("noimport.py");
        std::fs::write(
            &script,
            r#"
import json, sys
sys.stdout.write(json.dumps({"ready": False, "error": "import failed: no bqskit"}) + "\n")
sys.stdout.flush()
"#,
        )
        .unwrap();
        let b = Bqskit::new(BqskitConfig {
            worker: script,
            ..BqskitConfig::default()
        });
        if Command::new("python3").arg("--version").output().is_err() {
            return;
        }
        let msg = format!("{:#}", b.run(&request("h q[0];"), &reg()).unwrap_err());
        assert!(msg.contains("bqskit"), "unhelpful message: {msg}");
    }
}
