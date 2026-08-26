//! Command-line arguments, compatible with the reference implementation's spelling.
//!
//! The reference used argparse4j, which accepts single-dash multi-character flags such as
//! `-opt FIDELITY` and `-search BEAM`. `clap` reserves single-dash for one-character
//! flags, so [`normalize`] rewrites the reference's spellings to their long forms before
//! parsing. Keeping that mapping in one table means `wisq` and
//! `evaluation/run_guoq.py` keep working unchanged.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;

/// The reference's single-dash multi-character flags, and their long equivalents.
const LEGACY_FLAGS: &[(&str, &str)] = &[
    ("-g", "--gate-set"),
    ("-r", "--rules"),
    ("-sr", "--symb-rules"),
    ("-q", "--queue-size"),
    ("-out", "--output-dir"),
    ("-job", "--job-info"),
    ("-search", "--search-strategy"),
    ("-opt", "--opt-obj"),
    ("-resynth", "--resynth-alg"),
    ("-eps", "--epsilon"),
    ("-maxsynth", "--max-resynth-allowed"),
    ("-temp", "--temperature"),
    ("-prunetemp", "--prune-temperature"),
    ("-cool", "--cooling-rate"),
    ("-itersprune", "--iters-before-prune"),
    ("-secsprune", "--secs-before-prune"),
];

/// Rewrite legacy flag spellings and expand `@file` argument files.
///
/// An `@file` argument is replaced by the file's contents, one argument per line, which
/// is how `evaluation/run_guoq.py` passes its configuration.
pub fn normalize<I, S>(args: I) -> Result<Vec<String>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out = Vec::new();
    for arg in args {
        let arg = arg.as_ref();
        if let Some(path) = arg.strip_prefix('@') {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading argument file `{path}`"))?;
            for line in text.lines() {
                let line = line.trim();
                if !line.is_empty() {
                    out.push(rewrite_flag(line));
                }
            }
        } else {
            out.push(rewrite_flag(arg));
        }
    }
    Ok(out)
}

fn rewrite_flag(arg: &str) -> String {
    // Only exact matches are rewritten: `-g` must not capture `-guess`.
    if let Some((_, long)) = LEGACY_FLAGS.iter().find(|(short, _)| *short == arg) {
        return (*long).to_string();
    }
    // `-opt=FIDELITY` as well as `-opt FIDELITY`.
    if let Some((short, long)) = arg
        .split_once('=')
        .and_then(|(k, _)| LEGACY_FLAGS.iter().find(|(s, _)| *s == k))
    {
        let value = &arg[short.len() + 1..];
        return format!("{long}={value}");
    }
    arg.to_string()
}

/// Optimize a quantum circuit.
#[derive(Debug, Parser)]
#[command(
    name = "guoq",
    about = "Optimize a quantum circuit with rewrite rules and resynthesis",
    long_about = None,
    disable_help_flag = false
)]
pub struct Cli {
    /// Path to the circuit to optimize.
    pub circuit: PathBuf,

    /// Gate set of the input circuit, and of the output. Optional: when omitted, the
    /// smallest known gate set covering every gate the circuit uses is inferred, and if
    /// none covers it, rules are synthesized for the circuit's own gates (once — the
    /// file is cached under the rules directory). When passed, it is used as given,
    /// whatever gates the circuit holds.
    #[arg(long = "gate-set", value_name = "SET")]
    pub gate_set: Option<String>,

    /// Path to a non-symbolic rule file.
    #[arg(long = "rules", value_name = "FILE")]
    pub rules: Option<PathBuf>,

    /// Path to a symbolic rule file.
    #[arg(long = "symb-rules", value_name = "FILE")]
    pub symb_rules: Option<PathBuf>,

    /// Directory holding the default rule files.
    #[arg(long = "rules-dir", default_value = "rules", value_name = "DIR")]
    pub rules_dir: PathBuf,

    /// Additional gate-set definitions, as TOML.
    #[arg(long = "gate-sets", value_name = "FILE")]
    pub gate_sets: Option<PathBuf>,

    /// Maximum number of candidates kept in the search queue.
    #[arg(long = "queue-size", default_value_t = 1)]
    pub queue_size: usize,

    /// Directory for output circuits; created if absent.
    #[arg(long = "output-dir", default_value = ".", value_name = "DIR")]
    pub output_dir: PathBuf,

    /// Tag inserted into output file names.
    #[arg(long = "job-info", default_value = "")]
    pub job_info: String,

    /// Largest sub-circuit, in gates, a symbolic rule may span.
    #[arg(long = "max-symb-size", default_value_t = 10)]
    pub max_symb_size: usize,

    /// Largest number of qubits a symbolic rule's sub-circuit may touch.
    #[arg(long = "max-symb-qubits", default_value_t = 7)]
    pub max_symb_qubits: usize,

    /// Optimize the circuit in windows of this many gates instead of all at once.
    ///
    /// A single search over a very large circuit stops discriminating: the change one
    /// rewrite makes to the total cost becomes a vanishing fraction of it, so nearly
    /// every move is accepted and the walk stops steering. Windowing keeps the ratio in
    /// the range the search was tuned for. 0 optimizes the whole circuit at once;
    /// omitted, windowing turns on by itself above `--window-threshold` gates.
    ///
    /// The default of 192 comes from a sweep rather than from a guess, and the curve has
    /// an interior optimum: below about 96 the per-window setup swamps the work done
    /// inside, and above about 256 the acceptance rule stops discriminating. A 32-gate
    /// window does worse than a 2048-gate one.
    #[arg(long = "window")]
    pub window: Option<usize>,

    /// Circuit size, in gates, above which windowing turns on by itself.
    #[arg(long = "window-threshold", default_value_t = 2048)]
    pub window_threshold: usize,

    /// Optimize windows one at a time rather than concurrently.
    ///
    /// Concurrency does not change *what* a window produces: its random stream comes from
    /// its position, not from when it is visited, so the same set of completed windows
    /// yields the same circuit either way. Under a wall-clock budget the runs still
    /// differ, because the parallel one finishes more windows before time runs out. This
    /// flag is for measurement and for leaving cores alone, not for reproducibility.
    #[arg(long = "window-serial")]
    pub window_serial: bool,

    /// Cap the worker thread pool at this many threads.
    ///
    /// Applies to everything that runs in parallel: the windowed search's concurrent
    /// windows and the parallel verification of symbolic rules at load. Unset uses one
    /// thread per available core. `1` is equivalent to `--window-serial` for the search
    /// while also serialising rule loading; the per-window budget arithmetic sees the
    /// capped count, so budgets stay honest under it.
    #[arg(long = "threads", value_name = "N")]
    pub threads: Option<usize>,

    /// Passes over the circuit when windowing, each with boundaries shifted by half a
    /// window so that gates at a seam get an interior turn.
    ///
    /// Defaults to one. Extra rounds visit nothing new — they re-visit the circuit with
    /// the seams moved — and under a fixed budget each one divides the time the windows
    /// get. Measured, that trade loses: seam coverage is worth less than the time it
    /// costs, at 20 and at 60 seconds alike. Raise it if the budget is generous enough
    /// that windows converge with time to spare.
    #[arg(long = "window-rounds", default_value_t = 1)]
    pub window_rounds: usize,

    /// How far apart, in topological positions, a symbolic rule's two halves may sit.
    ///
    /// Bounds a search that is otherwise cubic in the circuit size. 0 means no limit.
    #[arg(long = "max-symb-span", default_value_t = 512)]
    pub max_symb_span: usize,

    /// Largest number of qubits a rewrite rule may act on; -1 for no limit.
    #[arg(long = "max-rule-qubits", default_value_t = -1, allow_negative_numbers = true)]
    pub max_rule_qubits: i64,

    /// Largest number of qubits a resynthesis partition may span.
    #[arg(long = "max-partition-qubits", default_value_t = 3)]
    pub max_partition_qubits: usize,

    /// Drop rewrite rules that leave the gate count unchanged.
    #[arg(long = "remove-size-preserving-rules")]
    pub remove_size_preserving_rules: bool,

    /// Use size-preserving symbolic rules.
    #[arg(long = "use-size-preserving-symb-rules")]
    pub use_size_preserving_symb_rules: bool,

    /// Also use size-preserving rules in reverse.
    #[arg(long = "use-size-preserve-reflection")]
    pub use_size_preserve_reflection: bool,

    /// Also use rules that increase the gate count.
    #[arg(long = "use-size-increasing-rules")]
    pub use_size_increasing_rules: bool,

    /// Preserve the input circuit's qubit connectivity.
    #[arg(long = "preserve-mapping")]
    pub preserve_mapping: bool,

    /// Reject rules whose search pattern contains a compound angle expression.
    ///
    /// Reproduces the reference's `+`/`-` filter, which discarded a large part of the ion
    /// rule set. Off by default; see docs/PORTING-NOTES.md.
    #[arg(long = "reject-compound-pattern-angles")]
    pub reject_compound_pattern_angles: bool,

    /// Search strategy: BEAM, MCMC, SIM_ANN, BEAM_MCMC, or REDUCE.
    ///
    /// REDUCE first drives the circuit to a deterministic fixpoint — every strictly
    /// cost-reducing rule applied exhaustively — then continues as BEAM_MCMC on
    /// whatever budget remains. On large, freshly synthesized circuits the fixpoint
    /// reaches in seconds what the stochastic search spends most of its budget
    /// rediscovering.
    #[arg(
        long = "search-strategy",
        default_value = "BEAM_MCMC",
        value_name = "S"
    )]
    pub search_strategy: String,

    /// Optimization objective.
    #[arg(long = "opt-obj", default_value = "FIDELITY", value_name = "O")]
    pub opt_obj: String,

    /// Resynthesis backend.
    #[arg(long = "resynth-alg", value_name = "ALG")]
    pub resynth_alg: Option<String>,

    /// Total error budget for resynthesis.
    #[arg(long = "epsilon", default_value_t = 1e-8)]
    pub epsilon: f64,

    /// Slices the error budget into this many equal per-call accuracy hints for the
    /// backend (the reference's epsilon split); -1 hints each call with the full
    /// remaining budget. Unlike the reference it does not cap how many resynthesis
    /// calls may run: resynthesis continues as long as budget remains.
    #[arg(
        long = "max-resynth-allowed",
        default_value_t = 100,
        allow_negative_numbers = true
    )]
    pub max_resynth_allowed: i64,

    /// Random seed. Omit for a time-derived one.
    #[arg(long = "seed")]
    pub seed: Option<u64>,

    /// Keep gates a rewrite has turned into the identity, rather than removing them.
    ///
    /// Rotation merging produces `rz(0)` routinely. The reference removed these as a side
    /// effect of re-parsing every rewritten circuit; here it is an explicit step, on by
    /// default.
    #[arg(long = "keep-identity-gates")]
    pub keep_identity_gates: bool,

    /// How a worse candidate's acceptance probability is computed.
    ///
    /// `ratio` is the reference's `exp(-beta * candidate / current)`, where
    /// `--temperature` is an inverse temperature and larger values mean a greedier
    /// search. `delta` is textbook Metropolis, `exp(-delta / temperature)`, which scales
    /// with how much worse a candidate is; the same `--temperature` denotes a far more
    /// exploratory search under it, so retune when switching.
    #[arg(long = "acceptance", value_name = "RULE", default_value = "ratio")]
    pub acceptance: String,

    /// Metropolis temperature, or the softmax temperature for queue selection.
    #[arg(long = "temperature", default_value_t = 10.0)]
    pub temperature: f64,

    /// Temperature for pruning the rule set; 0 prunes greedily.
    #[arg(long = "prune-temperature", default_value_t = 0.0)]
    pub prune_temperature: f64,

    /// Cooling rate; 0 for none.
    #[arg(long = "cooling-rate", default_value_t = 0.0)]
    pub cooling_rate: f64,

    /// Iterations before rule pruning begins; -1 to never wait.
    #[arg(long = "iters-before-prune", default_value_t = -1, allow_negative_numbers = true)]
    pub iters_before_prune: i64,

    /// Seconds before rule pruning begins; -1 to never wait.
    #[arg(long = "secs-before-prune", default_value_t = -1, allow_negative_numbers = true)]
    pub secs_before_prune: i64,

    /// Transformations to sample per iteration; -1 for all of them.
    #[arg(
        long = "num-transf-sample",
        default_value_t = 1,
        allow_negative_numbers = true
    )]
    pub num_transf_sample: i64,

    /// How many index slots resynthesis occupies when sampling.
    #[arg(long = "resynth-weight", default_value_t = 1)]
    pub resynth_weight: usize,

    /// Backend-specific resynthesis options, as JSON.
    #[arg(long = "resynth-args", value_name = "JSON")]
    pub resynth_args: Option<String>,

    /// Python interpreter used for the BQSKit worker.
    #[arg(long = "python", default_value = "python3", value_name = "EXE")]
    pub python: String,

    /// Path to the BQSKit worker script.
    #[arg(
        long = "bqskit-worker",
        default_value = "py/bqskit_worker.py",
        value_name = "FILE"
    )]
    pub bqskit_worker: PathBuf,

    /// Path to the Synthetiq `main` binary.
    #[arg(
        long = "synthetiq-binary",
        default_value = "lib/synthetiq/bin/main",
        value_name = "FILE"
    )]
    pub synthetiq_binary: PathBuf,

    /// BQSKit optimization level.
    #[arg(long = "bqskit-opt-level", default_value_t = 3)]
    pub bqskit_opt_level: u8,

    /// Seconds to wait for one BQSKit reply before killing and restarting the worker.
    #[arg(long = "bqskit-timeout", default_value_t = 300, value_name = "SECS")]
    pub bqskit_timeout: u64,

    /// Candidate circuits to ask Synthetiq for.
    #[arg(long = "synthetiq-num-circuits", default_value_t = 100)]
    pub synthetiq_num_circuits: u32,

    /// Threads to give Synthetiq.
    #[arg(long = "synthetiq-threads", default_value_t = 8)]
    pub synthetiq_threads: u32,

    /// Apply each rule at one site per iteration rather than at every disjoint site.
    #[arg(long = "apply-once")]
    pub apply_once: bool,

    /// Cost of a two-qubit gate in one-qubit-gate units.
    #[arg(long = "fidelity", default_value_t = 1)]
    pub fidelity: i64,

    /// One-qubit gate error rate.
    #[arg(long = "error-1q")]
    pub error_1q: Option<f64>,

    /// Two-qubit gate error rate.
    #[arg(long = "error-2q")]
    pub error_2q: Option<f64>,

    /// Stop after this many seconds.
    ///
    /// The reference relied entirely on an external `timeout(1)`, which remains supported;
    /// this is for when an internal limit is more convenient.
    #[arg(long = "timeout", value_name = "SECS")]
    pub timeout: Option<u64>,

    /// Stop after this many iterations.
    #[arg(long = "max-iters", value_name = "N")]
    pub max_iters: Option<u64>,

    /// Check the final circuit against the input end to end, where that is tractable.
    #[arg(long = "verify-final")]
    pub verify_final: bool,

    /// Re-verify each symbolic rewrite's span by simulation before applying it.
    ///
    /// Off by default: constraints are checked at rule load, and each match's region is
    /// checked against them before rewriting, so this re-proves an implication whose
    /// premises are already established. Turn it on as defense in depth against bugs in
    /// the matching machinery itself.
    #[arg(long = "verify-rewrites")]
    pub verify_rewrites: bool,

    /// Log verbosity: 0 quiet, 1 progress, 2 with config, 3 with rules applied.
    #[arg(long = "verbosity", default_value_t = 0)]
    pub verbosity: u8,
}

impl Cli {
    /// Parse from an argument list, applying legacy-flag normalization.
    pub fn parse_from_args<I, S>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let normalized = normalize(args)?;
        Ok(Cli::try_parse_from(normalized)?)
    }

    /// `-1` sentinels become `None`.
    pub fn optional_usize(value: i64) -> Option<usize> {
        (value >= 0).then_some(value as usize)
    }

    /// The rule file to use, falling back to the gate set's default.
    pub fn resolve_rule_file(
        &self,
        explicit: Option<&Path>,
        default_name: &str,
    ) -> Result<PathBuf> {
        if let Some(p) = explicit {
            if !p.exists() {
                bail!("rule file `{}` does not exist", p.display());
            }
            return Ok(p.to_path_buf());
        }
        if default_name.is_empty() {
            bail!("no rule file given and the gate set defines no default");
        }
        let path = self.rules_dir.join(default_name);
        if !path.exists() {
            bail!(
                "default rule file `{}` does not exist; pass --rules or --rules-dir",
                path.display()
            );
        }
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_flags_are_rewritten() {
        let got = normalize(["-opt", "FIDELITY", "-g", "IBMN", "circ.qasm"]).unwrap();
        assert_eq!(
            got,
            vec!["--opt-obj", "FIDELITY", "--gate-set", "IBMN", "circ.qasm"]
        );
    }

    #[test]
    fn legacy_flags_with_equals_are_rewritten() {
        let got = normalize(["-opt=TOTAL", "-q=8000"]).unwrap();
        assert_eq!(got, vec!["--opt-obj=TOTAL", "--queue-size=8000"]);
    }

    #[test]
    fn non_flag_arguments_pass_through() {
        let got = normalize(["circ.qasm", "-gremlin", "--already-long"]).unwrap();
        assert_eq!(got, vec!["circ.qasm", "-gremlin", "--already-long"]);
    }

    #[test]
    fn every_reference_flag_has_a_mapping() {
        // The flags `evaluation/run_guoq.py` and the README use.
        for flag in [
            "-g",
            "-r",
            "-sr",
            "-q",
            "-out",
            "-job",
            "-search",
            "-opt",
            "-resynth",
            "-eps",
            "-maxsynth",
            "-temp",
            "-prunetemp",
            "-cool",
            "-itersprune",
            "-secsprune",
        ] {
            assert!(
                LEGACY_FLAGS.iter().any(|(s, _)| *s == flag),
                "{flag} has no mapping"
            );
        }
    }

    #[test]
    fn argument_files_are_expanded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("args.txt");
        std::fs::write(&path, "-opt\nTOTAL\n-q\n8000\n\ncirc.qasm\n").unwrap();
        let got = normalize([format!("@{}", path.display())]).unwrap();
        assert_eq!(
            got,
            vec!["--opt-obj", "TOTAL", "--queue-size", "8000", "circ.qasm"]
        );
    }

    #[test]
    fn a_missing_argument_file_is_an_error() {
        assert!(normalize(["@/nonexistent/args.txt"]).is_err());
    }

    /// The README's flagship command must parse.
    #[test]
    fn readme_fidelity_command_parses() {
        let cli = Cli::parse_from_args([
            "guoq",
            "-g",
            "IBMN",
            "-opt",
            "FIDELITY",
            "--error-1q",
            "0.0003",
            "--error-2q",
            "0.0115",
            "benchmarks/ibmnew/tof_3.qasm",
        ])
        .unwrap();
        assert_eq!(cli.gate_set.as_deref(), Some("IBMN"));
        assert_eq!(cli.opt_obj, "FIDELITY");
        assert_eq!(cli.error_1q, Some(0.0003));
        assert_eq!(cli.error_2q, Some(0.0115));
        assert_eq!(cli.circuit, PathBuf::from("benchmarks/ibmnew/tof_3.qasm"));
    }

    /// The README's QUESO command.
    #[test]
    fn readme_queso_command_parses() {
        let cli = Cli::parse_from_args([
            "guoq",
            "-g",
            "IBMN",
            "-opt",
            "TOTAL",
            "-search",
            "BEAM",
            "-temp",
            "0",
            "-q",
            "8000",
            "-resynth",
            "NONE",
            "benchmarks/ibmnew/tof_3.qasm",
        ])
        .unwrap();
        assert_eq!(cli.search_strategy, "BEAM");
        assert_eq!(cli.temperature, 0.0);
        assert_eq!(cli.queue_size, 8000);
        assert_eq!(cli.resynth_alg.as_deref(), Some("NONE"));
    }

    #[test]
    fn threads_caps_the_pool_and_defaults_to_all_cores() {
        let cli = Cli::parse_from_args(["guoq", "--threads", "3", "c.qasm"]).unwrap();
        assert_eq!(cli.threads, Some(3));
        let cli = Cli::parse_from_args(["guoq", "c.qasm"]).unwrap();
        assert_eq!(cli.threads, None);
    }

    #[test]
    fn resynthesis_backend_paths_have_defaults() {
        let cli = Cli::parse_from_args(["guoq", "c.qasm"]).unwrap();
        assert_eq!(cli.python, "python3");
        assert_eq!(cli.bqskit_worker, PathBuf::from("py/bqskit_worker.py"));
        assert_eq!(
            cli.synthetiq_binary,
            PathBuf::from("lib/synthetiq/bin/main")
        );
        assert_eq!(cli.max_partition_qubits, 3);
    }

    #[test]
    fn defaults_match_the_reference() {
        let cli = Cli::parse_from_args(["guoq", "c.qasm"]).unwrap();
        assert_eq!(cli.search_strategy, "BEAM_MCMC");
        assert_eq!(cli.opt_obj, "FIDELITY");
        assert_eq!(cli.queue_size, 1);
        assert_eq!(cli.temperature, 10.0);
        assert_eq!(cli.epsilon, 1e-8);
        assert_eq!(cli.max_resynth_allowed, 100);
        assert_eq!(cli.max_symb_qubits, 7);
        assert_eq!(cli.max_symb_size, 10);
        assert_eq!(cli.max_rule_qubits, -1);
        assert_eq!(cli.verbosity, 0);
        assert_eq!(cli.rules_dir, PathBuf::from("rules"));
    }

    #[test]
    fn negative_sentinels_become_none() {
        assert_eq!(Cli::optional_usize(-1), None);
        assert_eq!(Cli::optional_usize(0), Some(0));
        assert_eq!(Cli::optional_usize(100), Some(100));
    }

    #[test]
    fn boolean_flags_default_off() {
        let cli = Cli::parse_from_args(["guoq", "c.qasm"]).unwrap();
        assert!(!cli.apply_once);
        assert!(!cli.preserve_mapping);
        assert!(!cli.remove_size_preserving_rules);
        assert!(!cli.reject_compound_pattern_angles);

        let cli =
            Cli::parse_from_args(["guoq", "--apply-once", "--preserve-mapping", "c.qasm"]).unwrap();
        assert!(cli.apply_once);
        assert!(cli.preserve_mapping);
    }

    #[test]
    fn a_missing_circuit_is_an_error() {
        assert!(Cli::parse_from_args(["guoq"]).is_err());
    }

    #[test]
    fn unknown_flags_are_an_error() {
        assert!(Cli::parse_from_args(["guoq", "--nonsense", "c.qasm"]).is_err());
    }

    #[test]
    fn rule_file_resolution() {
        let dir = tempfile::tempdir().unwrap();
        let rules = dir.path().join("rules");
        std::fs::create_dir_all(&rules).unwrap();
        let named = rules.join("rules_q3_s6_nam.txt");
        std::fs::write(&named, "").unwrap();

        let mut cli = Cli::parse_from_args(["guoq", "c.qasm"]).unwrap();
        cli.rules_dir = rules.clone();
        assert_eq!(
            cli.resolve_rule_file(None, "rules_q3_s6_nam.txt").unwrap(),
            named
        );
        assert!(cli.resolve_rule_file(None, "missing.txt").is_err());
        assert!(cli.resolve_rule_file(None, "").is_err());
        assert_eq!(cli.resolve_rule_file(Some(&named), "x").unwrap(), named);
        assert!(cli
            .resolve_rule_file(Some(Path::new("/nope.txt")), "x")
            .is_err());
    }
}
