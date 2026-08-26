//! Wiring: read a circuit, load rules, run the search, write the results.

use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use qcircuit::{qasm, GateRegistry, GateSet, GateSetLibrary};
use qopt::{
    Candidate, CostModel, NullResynthesizer, Objective, Resynthesizer, SearchConfig, SearchResult,
    Strategy, TransformationSet, VerifiedBackend,
};
use qresynth::{
    Backend, Bqskit, BqskitConfig, NoBackend, PartitionLimits, Synthetiq, SynthetiqConfig,
    VerifiedResynthesizer,
};
use qrules::legacy::{self, LoadOptions};
use qrules::SymbolicLimits;
use qsemantics::{phase_invariant_distance, Unitary};
use serde_json::json;

use crate::args::Cli;
use crate::log::Logger;

/// What a run produced.
#[derive(Debug)]
pub struct RunOutcome {
    pub result: SearchResult,
    pub original_gates: usize,
    /// Set when `--verify-final` ran; `None` when the circuit was too wide to check.
    pub verified_distance: Option<f64>,
    /// How resynthesis attempts went, when a backend was configured.
    pub resynth: Option<qopt::ResynthStats>,
    /// The gate set the run used — as passed, inferred from the circuit, or ad-hoc.
    pub gate_set: String,
}

/// Which resynthesis backend was asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    None,
    Bqskit,
    Synthetiq,
}

impl BackendChoice {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "NONE" => Some(BackendChoice::None),
            "BQSKIT" => Some(BackendChoice::Bqskit),
            "SYNTHETIQ" => Some(BackendChoice::Synthetiq),
            _ => None,
        }
    }

    /// The reference's default: BQSKit for continuous objectives, Synthetiq for
    /// fault-tolerant ones, nothing otherwise.
    pub fn default_for(objective: Objective) -> Self {
        match objective {
            Objective::Fidelity | Objective::TwoQ => BackendChoice::Bqskit,
            Objective::Ft | Objective::T => BackendChoice::Synthetiq,
            _ => BackendChoice::None,
        }
    }
}

/// Run the optimizer as configured.
pub fn optimize(cli: &Cli) -> Result<RunOutcome> {
    // Before anything parallel happens — rule loading verifies symbolic rules
    // concurrently, so the cap must land first. Rayon's global pool can only be sized
    // once per process; a second `optimize` call in the same process (tests do this)
    // keeps the first call's pool, so only complain if the size actually disagrees.
    if let Some(n) = cli.threads {
        let n = n.max(1);
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
        if rayon::current_num_threads() != n {
            eprintln!(
                "warning: thread pool already initialized with {} threads; --threads {n} ignored",
                rayon::current_num_threads()
            );
        }
    }
    let mut registry = GateRegistry::with_builtins();
    let mut gate_sets = GateSetLibrary::builtin();
    if let Some(path) = &cli.gate_sets {
        gate_sets
            .extend_from_file(path)
            .with_context(|| format!("loading gate sets from {}", path.display()))?;
    }
    let _ = &mut registry;

    let source = std::fs::read_to_string(&cli.circuit)
        .with_context(|| format!("reading circuit {}", cli.circuit.display()))?;
    let circuit = qasm::parse(&source)
        .with_context(|| format!("parsing circuit {}", cli.circuit.display()))?;
    let circuit_name = cli
        .circuit
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "circuit.qasm".into());

    let objective = Objective::parse(&cli.opt_obj)
        .with_context(|| format!("unknown optimization objective `{}`", cli.opt_obj))?;
    let strategy = Strategy::parse(&cli.search_strategy)
        .with_context(|| format!("unknown search strategy `{}`", cli.search_strategy))?;

    // Resolve the two-qubit gate weight. `--fidelity` wins if given; otherwise derive it
    // from the error rates, which is what the README's example does.
    let breakeven = resolve_breakeven(cli, objective)?;

    // Explicit `-g` wins outright, whatever gates the circuit holds; without it the
    // gate set is inferred from the circuit — the smallest shipped set covering every
    // gate it uses, or an ad-hoc set built from those gates when none does.
    let (gate_set, inferred_adhoc) = match &cli.gate_set {
        Some(name) => (gate_sets.get(name)?.clone(), false),
        None => {
            let mut used: Vec<String> = circuit
                .gate_indices()
                .iter()
                .map(|&i| circuit.gate(i).gate.to_string())
                .collect();
            used.sort();
            used.dedup();
            resolve_gate_set(&gate_sets, &used, &registry)?
        }
    };
    let gate_set = &gate_set;
    gate_set.validate(&registry)?;
    let target_gate_set = gate_set.resynth_target.clone();

    // An ad-hoc gate set has no shipped corpus; its rules are synthesized on first use
    // and cached under the rules directory, keyed by the gate combination. An explicit
    // `--rules` still wins, exactly as it does for a shipped set.
    if inferred_adhoc && cli.rules.is_none() {
        let cache = cli.rules_dir.join(&gate_set.default_rules);
        if !cache.exists() {
            eprintln!(
                "no gate set covers this circuit; synthesizing rules for `{}` into {} (cached for later runs)",
                gate_set.name,
                cache.display()
            );
            let count = crate::synth::synthesize_default_rules(gate_set, &registry, &cache)?;
            eprintln!("synthesized {count} rules");
        }
    }

    let load_opts = LoadOptions {
        drop_size_preserving: cli.remove_size_preserving_rules,
        add_size_preserving_reflection: cli.use_size_preserve_reflection,
        add_size_increasing: cli.use_size_increasing_rules,
        max_rule_qubits: Cli::optional_usize(cli.max_rule_qubits),
        preserve_mapping: cli.preserve_mapping,
        reject_compound_pattern_angles: cli.reject_compound_pattern_angles,
    };

    let plain_path =
        cli.resolve_rule_file(cli.rules.as_deref(), gate_set.default_rules.as_str())?;
    // An ad-hoc gate set has no symbolic corpus at all; that is a property of the set,
    // not a missing file to complain about.
    let symb_path = if cli.symb_rules.is_none() && gate_set.default_symb_rules.is_empty() {
        None
    } else {
        Some(cli.resolve_rule_file(
            cli.symb_rules.as_deref(),
            gate_set.default_symb_rules.as_str(),
        )?)
    };

    let plain_report = legacy::load_file(&plain_path, &load_opts, &registry)?;
    let symb_lines = match &symb_path {
        Some(path) => legacy::load_file(path, &load_opts, &registry)?.symbolic_lines,
        None => Vec::new(),
    };
    let (symbolic_rules, symb_rejected) = legacy::parse_symbolic(&symb_lines, &registry);

    let mut transformations =
        TransformationSet::new(plain_report.rules, symbolic_rules, cli.resynth_weight);
    if cli.resynth_weight == 1 {
        // The reference derived the weight from the rule count when it was left at its
        // default, so that resynthesis is not drowned out by thousands of rules.
        transformations.weight_from_fraction(0.015);
    }

    let logger = Logger::new(cli.verbosity, circuit_name.clone());
    logger.initial(&circuit, &registry);
    logger.config(&json!({
        "gate_set": gate_set.name,
        "objective": objective.as_flag(),
        "strategy": strategy.as_flag(),
        "fidelity_breakeven": breakeven,
        "rules": plain_path.display().to_string(),
        "symb_rules": symb_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        "num_rules": transformations.num_plain(),
        "num_symb_rules": transformations.num_symbolic(),
        "num_rules_rejected": plain_report.rejected.len(),
        "num_symb_rules_rejected": symb_rejected.len(),
        "resynth_weight": transformations.resynth_weight(),
        "seed": cli.seed,
    }));

    let config = SearchConfig {
        strategy,
        objective,
        fidelity_breakeven: breakeven,
        queue_size: cli.queue_size.max(1),
        temperature: cli.temperature,
        acceptance: qopt::AcceptanceRule::parse(&cli.acceptance)
            .ok_or_else(|| anyhow::anyhow!("unknown acceptance rule `{}`", cli.acceptance))?,
        cooling_rate: cli.cooling_rate,
        prune_temperature: cli.prune_temperature,
        num_transformations_sample: Cli::optional_usize(cli.num_transf_sample),
        iters_before_prune: Cli::optional_usize(cli.iters_before_prune),
        secs_before_prune: Cli::optional_usize(cli.secs_before_prune).map(|v| v as u64),
        apply_once: cli.apply_once,
        normalize_identities: !cli.keep_identity_gates,
        epsilon: cli.epsilon,
        seed: cli.seed.unwrap_or(0x5EED),
        time_limit: cli.timeout.map(Duration::from_secs),
        max_iters: cli.max_iters,
        output_dir: cli.output_dir.clone(),
        job_info: cli.job_info.clone(),
        verbosity: cli.verbosity,
    };

    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("creating {}", config.output_dir.display()))?;

    let model = CostModel::new(objective, breakeven, registry.clone());
    let original_gates = circuit.gate_count();
    let latest_path = config.latest_path(&circuit_name);
    write_circuit(&circuit, &latest_path)?;

    let limits = SymbolicLimits {
        max_qubits: cli.max_symb_qubits,
        max_gates: cli.max_symb_size,
        max_span: (cli.max_symb_span > 0).then_some(cli.max_symb_span),
        // The search owns the clock and fills this in once it starts.
        deadline: None,
        verify_rewrites: cli.verify_rewrites,
    };
    // Choosing a backend. Only the one actually requested is constructed, so nothing is
    // started -- no worker process, no subprocess -- unless it is going to be used.
    let choice = match &cli.resynth_alg {
        Some(name) => BackendChoice::parse(name)
            .with_context(|| format!("unknown resynthesis backend `{name}`"))?,
        None => BackendChoice::default_for(objective),
    };
    let backend: Box<dyn Backend> = match choice {
        BackendChoice::None => Box::new(NoBackend),
        BackendChoice::Bqskit => Box::new(Bqskit::new(BqskitConfig {
            python: cli.python.clone(),
            worker: cli.bqskit_worker.clone(),
            opt_level: cli.bqskit_opt_level,
            request_timeout: std::time::Duration::from_secs(cli.bqskit_timeout),
            ..BqskitConfig::default()
        })),
        BackendChoice::Synthetiq => Box::new(Synthetiq::new(SynthetiqConfig {
            binary: cli.synthetiq_binary.clone(),
            working_dir: cli
                .synthetiq_binary
                .parent()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| std::path::PathBuf::from("lib/synthetiq")),
            num_circuits: cli.synthetiq_num_circuits,
            threads: cli.synthetiq_threads,
            ..SynthetiqConfig::default()
        })),
    };

    let mut verified = VerifiedResynthesizer::new(backend.as_ref(), target_gate_set);
    verified.limits = PartitionLimits {
        max_qubits: cli.max_partition_qubits,
        ..PartitionLimits::default()
    };
    // The reference's epsilon / MAX_RESYNTH_ALLOWED split, kept as the per-call
    // accuracy hint so the spending spreads evenly across many calls — but only as a
    // hint: the number of calls is unbounded, and acceptance is the measurement
    // against each candidate's remaining budget. -1 hints the full remaining budget.
    verified.epsilon_hint =
        (cli.max_resynth_allowed > 0).then(|| cli.epsilon / cli.max_resynth_allowed as f64);
    let verified_backend = VerifiedBackend::new(verified);
    let null = NullResynthesizer;
    // `Sync` so windows can be optimized concurrently; both backends already are.
    let resynth: &(dyn Resynthesizer + Sync) = if choice == BackendChoice::None {
        &null
    } else {
        &verified_backend
    };

    let mut search = qopt::Search::new(
        &config,
        &registry,
        &transformations,
        resynth,
        limits.clone(),
    );

    // Windowing turns on by itself above a size where a single search stops
    // discriminating -- one rewrite's effect on the total cost becomes a vanishing
    // fraction of it -- unless the size is set explicitly.
    let window_gates = cli
        .window
        .unwrap_or(if circuit.gate_count() > cli.window_threshold {
            qopt::WindowConfig::default().gates
        } else {
            0
        });

    let result = if window_gates > 0 {
        let started = Instant::now();
        let mut dag = circuit.clone();
        // Mid-run visibility: a progress line as the run advances, and a valid
        // best-so-far on disk, so a long windowed run is distinguishable from a hung
        // one and can be interrupted without losing what it found. Serialized by the
        // splice lock, so plain file writes are safe here.
        let logger_ref = &logger;
        let latest = latest_path.clone();
        let on_window_progress = move |dag: &qcircuit::Dag, p: &qopt::WindowProgress| {
            logger_ref.window_progress(p, started.elapsed().as_secs_f64());
            let _ = write_circuit(dag, &latest);
        };
        let report = qopt::optimize_windowed(
            &mut dag,
            &qopt::WindowConfig {
                gates: window_gates,
                rounds: cli.window_rounds,
                deadline: config.time_limit.map(|l| started + l),
                parallel: !cli.window_serial,
            },
            &config,
            &registry,
            &transformations,
            resynth,
            limits,
            &model,
            &on_window_progress,
        );
        logger.windows(&report);
        let best = Candidate::new(dag, &model, 0);
        let _ = write_circuit(&best.dag, &latest_path);
        qopt::SearchResult {
            best,
            iterations: report.windows as u64,
            elapsed: started.elapsed(),
            time_to_best: started.elapsed().as_secs_f64(),
            applications: Vec::new(),
        }
    } else {
        let model_ref = &model;
        let logger_ref = &logger;
        let latest = latest_path.clone();
        let mut on_improvement = move |candidate: &Candidate, elapsed: f64| {
            logger_ref.improvement(candidate, model_ref, elapsed, elapsed);
            let _ = write_circuit(&candidate.dag, &latest);
        };
        search.run(circuit.clone(), &mut on_improvement)
    };

    backend.shutdown();
    if choice != BackendChoice::None {
        logger.resynth_stats(&verified_backend.stats());
    }
    logger.final_result(&result, &model);
    write_circuit(&result.best.dag, &config.final_path(&circuit_name))?;
    write_circuit(&result.best.dag, &latest_path)?;

    let verified_distance = if cli.verify_final {
        verify(&circuit, &result.best.dag, &registry)
    } else {
        None
    };
    if let Some(d) = verified_distance {
        if d > cli.epsilon.max(1e-9) {
            bail!(
                "final circuit differs from the input by {d:.3e}, over the {:.3e} budget",
                cli.epsilon
            );
        }
    }

    Ok(RunOutcome {
        result,
        original_gates,
        verified_distance,
        resynth: (choice != BackendChoice::None).then(|| verified_backend.stats()),
        gate_set: gate_set.name.clone(),
    })
}

/// The gate set for a circuit that named none: the smallest shipped (or `--gate-sets`)
/// set covering every gate the circuit uses — smallest, so a nam circuit lands on nam
/// rather than on some wider set that happens to contain it — or, when none covers it,
/// an ad-hoc set built from the circuit's own gates. Returns `(set, true)` for the
/// ad-hoc case, which is the caller's cue to synthesize rules for it.
fn resolve_gate_set(
    library: &GateSetLibrary,
    used: &[String],
    registry: &GateRegistry,
) -> Result<(GateSet, bool)> {
    if let Some(set) = library
        .iter()
        .filter(|s| used.iter().all(|g| s.contains_gate(g)))
        .min_by_key(|s| s.gates.len())
    {
        return Ok((set.clone(), false));
    }
    for gate in used {
        if !registry.contains(gate) {
            bail!(
                "no gate set covers this circuit, and `{gate}` is not a defined gate, so \
                 rules cannot be synthesized for it; pass -g to name a gate set explicitly"
            );
        }
    }
    // Built through the same TOML path as every other gate set, so there is exactly one
    // way a gate set comes to exist. The angle basis is the standard two-variable one;
    // it only matters if the circuit uses parametric gates.
    let name = format!("auto-{}", used.join("-"));
    let toml = format!(
        r#"
[[gateset]]
name = "{name}"
description = "inferred from the input circuit"
gates = [{gates}]
resynth_target = ""
default_rules = "auto/rules_q3_s3_{name}.txt"
default_symb_rules = ""
synth_angles = ["theta1", "theta2", "theta1+theta2"]
"#,
        gates = used
            .iter()
            .map(|g| format!("\"{g}\""))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let set = GateSetLibrary::from_toml(&toml)
        .map_err(|e| anyhow::anyhow!("building the inferred gate set `{name}`: {e}"))?
        .get(&name)?
        .clone();
    Ok((set, true))
}

/// The two-qubit gate weight, from `--fidelity` or from the measured error rates.
fn resolve_breakeven(cli: &Cli, objective: Objective) -> Result<i64> {
    if objective == Objective::Ft && cli.fidelity == 1 {
        // The reference hardcoded 50 for the fault-tolerant objective.
        return Ok(50);
    }
    if cli.fidelity != 1 {
        return Ok(cli.fidelity);
    }
    if objective != Objective::Fidelity {
        return Ok(cli.fidelity);
    }
    match (cli.error_1q, cli.error_2q) {
        (Some(e1), Some(e2)) => CostModel::breakeven_from_error_rates(e1, e2)
            .context("--error-1q and --error-2q do not give a usable two-qubit gate weight"),
        _ => bail!("the FIDELITY objective needs --fidelity, or both --error-1q and --error-2q"),
    }
}

/// Compare the optimized circuit against the input, where that is tractable.
fn verify(
    original: &qcircuit::Dag,
    optimized: &qcircuit::Dag,
    registry: &GateRegistry,
) -> Option<f64> {
    let mut qubits: Vec<qcircuit::QubitId> = original.qubits().to_vec();
    for q in optimized.qubits() {
        if !qubits.contains(q) {
            qubits.push(q.clone());
        }
    }
    qubits.sort();
    if qubits.len() > 12 {
        return None;
    }
    let a = Unitary::from_dag_over(original, qubits.clone(), registry).ok()?;
    let b = Unitary::from_dag_over(optimized, qubits, registry).ok()?;
    Some(phase_invariant_distance(&a, &b))
}

fn write_circuit(dag: &qcircuit::Dag, path: &std::path::Path) -> Result<()> {
    std::fs::write(path, qasm::to_qasm(dag))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
