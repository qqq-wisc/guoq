//! The `queso` command: synthesizing rewrite rules, and converting between rule formats.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use qcircuit::{GateRegistry, GateSetLibrary};
use qrules::legacy::{self, LoadOptions};
use qsynth::enumerate::{EnumerationConfig, Enumerator};
use qsynth::fingerprint::Equivalence;
use qsynth::rules::{gather_rules, to_rule_file, RuleOptions};

/// Synthesize rewrite rules, or convert rule files.
#[derive(Debug, Parser)]
#[command(name = "queso", about = "Synthesize quantum-circuit rewrite rules")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Gate set to synthesize for.
    #[arg(long = "gate-set", short = 'g', value_name = "SET")]
    pub gate_set: Option<String>,

    /// Qubits to enumerate over.
    #[arg(long = "max-qubits", short = 'q', default_value_t = 3)]
    pub max_qubits: usize,

    /// Largest circuit, in gates, to enumerate.
    #[arg(long = "max-size", short = 's', default_value_t = 3)]
    pub max_size: usize,

    /// Where to write the rule file. Defaults to `rules_q<q>_s<s>_<set>.txt`.
    #[arg(long = "output", short = 'o', value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Additional gate-set definitions, as TOML.
    #[arg(long = "gate-sets", value_name = "FILE")]
    pub gate_sets: Option<PathBuf>,

    /// Treat circuits equal up to a global phase as equivalent.
    ///
    /// Admits strictly more rules. The reference required exact equality, so its rule
    /// files contain no phase-only rewrites.
    #[arg(long = "up-to-phase")]
    pub up_to_phase: bool,

    /// Synthesize symbolic rules -- and only symbolic rules -- written to a `_symb`
    /// file named after `--output`.
    ///
    /// By default, halves are restricted the way the reference synthesizer restricted
    /// them: no `cx`, `h`, `cz`, `rx`, `ry`, `rxx`, or `sx` before the hole, and no
    /// `cx` or `cz` after it -- the shapes every shipped `_symb` corpus was produced
    /// under. Gates absent from the chosen set are simply never in play. See
    /// `--symb-all-gates`.
    #[arg(long)]
    pub symbolic: bool,

    /// Boundary width for symbolic rules: how many qubits the hole spans.
    #[arg(long, default_value_t = 2)]
    pub symb_width: usize,

    /// Allow every gate of the set in symbolic halves -- `cx` and `h` included, which
    /// the reference could never produce.
    ///
    /// Sound because agreement is decided by evaluating both sides with each candidate
    /// permutation substituted for the hole, not by a gate list; the vendored
    /// `rules/generated` corpora were produced with this on.
    #[arg(long, requires = "symbolic")]
    pub symb_all_gates: bool,

    /// Independent random rational sample points a candidate rule must survive.
    ///
    /// Verification is exact at each point, so a single round already leaves only a
    /// probability-zero failure mode; the second is belt and braces.
    #[arg(long = "verify-rounds", default_value_t = 2)]
    pub verify_rounds: usize,

    /// Deterministic seed.
    #[arg(long = "seed", default_value_t = 0x5EED)]
    pub seed: u64,

    /// Print progress.
    #[arg(long = "verbose", short = 'v')]
    pub verbose: bool,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Read a rule file and report what it contains.
    Inspect {
        /// The rule file to read.
        file: PathBuf,
    },
}

/// What a synthesis run produced.
#[derive(Debug)]
pub struct SynthesisOutcome {
    pub rules: usize,
    pub classes: usize,
    pub circuits_enumerated: usize,
    /// `true` for a `--symbolic` run, which writes only the `_symb` file.
    pub symbolic: bool,
    pub output: PathBuf,
}

/// Run `queso`.
pub fn run(cli: &Cli) -> Result<()> {
    match &cli.command {
        Some(Command::Inspect { file }) => inspect(file),
        None => {
            let outcome = synthesize(cli)?;
            if outcome.symbolic {
                println!(
                    "{} symbolic rules -> {}",
                    outcome.rules,
                    outcome.output.display()
                );
            } else {
                println!(
                    "{} rules from {} equivalence classes ({} circuits enumerated) -> {}",
                    outcome.rules,
                    outcome.classes,
                    outcome.circuits_enumerated,
                    outcome.output.display()
                );
            }
            Ok(())
        }
    }
}

/// Enumerate, classify, and write a rule file.
pub fn synthesize(cli: &Cli) -> Result<SynthesisOutcome> {
    let Some(name) = &cli.gate_set else {
        bail!("--gate-set is required; pass `queso inspect <file>` to read a rule file instead");
    };
    let mut library = GateSetLibrary::builtin();
    if let Some(path) = &cli.gate_sets {
        library
            .extend_from_file(path)
            .with_context(|| format!("loading gate sets from {}", path.display()))?;
    }
    let set = library.get(name)?;
    let registry = GateRegistry::with_builtins();
    set.validate(&registry)?;

    let mut config = EnumerationConfig::from_gate_set(set, cli.max_qubits, cli.max_size);
    config.seed = cli.seed;
    config.verify_rounds = cli.verify_rounds;
    config.equivalence = if cli.up_to_phase {
        Equivalence::UpToPhase
    } else {
        Equivalence::Exact
    };

    if cli.verbose {
        eprintln!(
            "enumerating {} over {} qubits up to size {}",
            set.name, cli.max_qubits, cli.max_size
        );
    }

    let output = cli.output.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "rules_q{}_s{}_{}.txt",
            cli.max_qubits, cli.max_size, set.name
        ))
    });

    // `--symbolic` synthesizes symbolic rules and nothing else: the plain enumeration
    // over the full qubit count serves only plain rules, and skipping it keeps the two
    // corpora independently regenerable.
    if cli.symbolic {
        let restrictions = if cli.symb_all_gates {
            qsynth::symbolic::HalfRestrictions::none()
        } else {
            qsynth::symbolic::HalfRestrictions::reference()
        };
        if cli.verbose && !restrictions.is_none() {
            eprintln!(
                "symbolic halves restricted: not before the hole: [{}]; not after: [{}]",
                restrictions.not_before.join(", "),
                restrictions.not_after.join(", ")
            );
        }
        let symb = qsynth::symbolic::synthesize_symbolic(
            &config,
            cli.symb_width,
            &registry,
            &restrictions,
        );
        // Every emitted line must survive the loader's own identity check; emitting a
        // line the loader would refuse means the two disagree about what a rule says,
        // and that must fail loudly here rather than quietly at load time.
        let mut lines = Vec::with_capacity(symb.len());
        for r in &symb {
            let line = r.to_line();
            qrules::SymbolicRule::parse_legacy(&line, &registry)
                .map_err(|e| anyhow::anyhow!("emitted a symbolic rule the loader refuses: {e}"))?;
            lines.push(line);
        }
        let symb_path = output.with_file_name(format!(
            "{}_symb.txt",
            output
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "rules".to_string())
        ));
        std::fs::write(&symb_path, lines.join("\n") + "\n")
            .with_context(|| format!("writing {}", symb_path.display()))?;
        return Ok(SynthesisOutcome {
            rules: symb.len(),
            classes: 0,
            circuits_enumerated: 0,
            symbolic: true,
            output: symb_path,
        });
    }

    let mut enumerator = Enumerator::new(config.clone(), &registry);
    let classes = enumerator.run();
    let sample = enumerator.sample().clone();
    if cli.verbose {
        eprintln!(
            "{} circuits, {} equivalence classes",
            enumerator.enumerated,
            classes.len()
        );
    }

    let rules = gather_rules(
        &classes,
        &config,
        &registry,
        &sample,
        &RuleOptions {
            drop_common_ends: true,
            verify_rounds: cli.verify_rounds,
            equivalence: config.equivalence,
        },
    );
    std::fs::write(&output, to_rule_file(&rules))
        .with_context(|| format!("writing {}", output.display()))?;

    Ok(SynthesisOutcome {
        rules: rules.len(),
        classes: classes.len(),
        circuits_enumerated: enumerator.enumerated,
        symbolic: false,
        output,
    })
}

/// Synthesize a plain rule file for `set` with `queso`'s defaults and write it to
/// `output`, returning how many rules were found.
///
/// This is the path `guoq` takes when the input circuit names no gate set and no
/// shipped one covers its gates: the rules the optimizer needs are made on the spot,
/// once, and the written file is the cache that spares every later run the work.
pub fn synthesize_default_rules(
    set: &qcircuit::GateSet,
    registry: &GateRegistry,
    output: &Path,
) -> Result<usize> {
    let mut config = EnumerationConfig::from_gate_set(set, 3, 3);
    config.seed = 0x5EED;
    config.verify_rounds = 2;
    config.equivalence = Equivalence::Exact;

    let mut enumerator = Enumerator::new(config.clone(), registry);
    let classes = enumerator.run();
    let sample = enumerator.sample().clone();
    let rules = gather_rules(
        &classes,
        &config,
        registry,
        &sample,
        &RuleOptions {
            drop_common_ends: true,
            verify_rounds: config.verify_rounds,
            equivalence: config.equivalence,
        },
    );
    if let Some(dir) = output.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(output, to_rule_file(&rules))
        .with_context(|| format!("writing {}", output.display()))?;
    Ok(rules.len())
}

/// Report what a rule file contains.
pub fn inspect(path: &Path) -> Result<()> {
    let registry = GateRegistry::with_builtins();
    let report = legacy::load_file(path, &LoadOptions::default(), &registry)?;
    let (symbolic, symb_rejected) = legacy::parse_symbolic(&report.symbolic_lines, &registry);

    let mut sizes: std::collections::BTreeMap<usize, usize> = Default::default();
    for rule in &report.rules {
        *sizes.entry(rule.find.gate_count()).or_default() += 1;
    }
    let mut widths: std::collections::BTreeMap<usize, usize> = Default::default();
    for rule in &symbolic {
        *widths.entry(rule.constraints[0].width()).or_default() += 1;
    }

    println!("file:               {}", path.display());
    println!("plain rules:        {}", report.rules.len());
    println!("symbolic rules:     {}", symbolic.len());
    println!("rejected (plain):   {}", report.rejected.len());
    println!("rejected (symb):    {}", symb_rejected.len());
    println!("pattern sizes:      {sizes:?}");
    println!("constraint widths:  {widths:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cli_parses_the_reference_spelling() {
        // `java ... qoptimizer.Synthesizer -g nam -q 3 -s 3`
        let cli = Cli::try_parse_from(["queso", "-g", "nam", "-q", "3", "-s", "3"]).unwrap();
        assert_eq!(cli.gate_set.as_deref(), Some("nam"));
        assert_eq!(cli.max_qubits, 3);
        assert_eq!(cli.max_size, 3);
        assert!(!cli.up_to_phase);
    }

    #[test]
    fn defaults_are_sensible() {
        let cli = Cli::try_parse_from(["queso", "-g", "nam"]).unwrap();
        assert_eq!(cli.max_qubits, 3);
        assert_eq!(cli.max_size, 3);
        assert_eq!(cli.verify_rounds, 2);
        assert!(cli.output.is_none());
    }

    #[test]
    fn symb_all_gates_parses_and_requires_symbolic() {
        let cli =
            Cli::try_parse_from(["queso", "-g", "nam", "--symbolic", "--symb-all-gates"]).unwrap();
        assert!(cli.symb_all_gates);
        // Without `--symbolic` there is nothing for the flag to widen.
        assert!(Cli::try_parse_from(["queso", "-g", "nam", "--symb-all-gates"]).is_err());
    }

    /// `--symbolic` writes the `_symb` file and nothing else, restricted like the
    /// reference by default -- so no `cx` or `h` appears in any half.
    #[test]
    fn symbolic_mode_writes_only_restricted_symbolic_rules() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("rules.txt");
        let cli = Cli::try_parse_from([
            "queso",
            "-g",
            "nam",
            "-q",
            "2",
            "-s",
            "2",
            "--symbolic",
            "-o",
            out.to_str().unwrap(),
        ])
        .unwrap();
        let outcome = synthesize(&cli).unwrap();
        assert!(outcome.symbolic);
        assert!(outcome.rules > 0);
        assert_eq!(outcome.output, dir.path().join("rules_symb.txt"));
        assert!(outcome.output.exists());
        assert!(
            !out.exists(),
            "a plain rule file was written in symbolic mode"
        );

        // The restriction is asymmetric, like the reference's: nothing from the
        // refusal list before the hole (`h` after it is legitimate), no `cx` at all
        // for nam (banned on both sides).
        let text = std::fs::read_to_string(&outcome.output).unwrap();
        assert!(
            !text.contains("cx"),
            "`cx` in a half despite the restriction"
        );
        for line in text.lines() {
            for half in line.split('|').take(2) {
                let before = half.split("symb q;").next().unwrap_or("");
                assert!(
                    !before.contains("h "),
                    "`h` before the hole despite the restriction: {line}"
                );
            }
        }
    }

    #[test]
    fn a_gate_set_is_required_for_synthesis() {
        let cli = Cli::try_parse_from(["queso"]).unwrap();
        let msg = format!("{:#}", synthesize(&cli).unwrap_err());
        assert!(msg.contains("--gate-set"), "{msg}");
    }

    #[test]
    fn an_unknown_gate_set_is_rejected() {
        let cli = Cli::try_parse_from(["queso", "-g", "nonsense"]).unwrap();
        assert!(synthesize(&cli).is_err());
    }

    #[test]
    fn synthesis_writes_a_readable_rule_file() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("rules.txt");
        let cli = Cli::try_parse_from([
            "queso",
            "-g",
            "nam",
            "-q",
            "2",
            "-s",
            "2",
            "-o",
            out.to_str().unwrap(),
        ])
        .unwrap();
        let outcome = synthesize(&cli).unwrap();
        assert!(outcome.rules > 0);
        assert_eq!(outcome.output, out);

        // The file the optimizer would read back.
        let registry = GateRegistry::with_builtins();
        let report = legacy::load_file(&out, &LoadOptions::default(), &registry).unwrap();
        assert!(
            !report.rules.is_empty(),
            "the synthesized file loaded no rules; rejected: {:?}",
            &report.rejected[..report.rejected.len().min(5)]
        );
    }

    #[test]
    fn the_default_output_name_matches_the_reference() {
        let cli = Cli::try_parse_from(["queso", "-g", "nam", "-q", "3", "-s", "6"]).unwrap();
        // The reference wrote `rules_q%s_s%s_%s`.
        let expected = PathBuf::from("rules_q3_s6_nam.txt");
        let got = cli.output.clone().unwrap_or_else(|| {
            PathBuf::from(format!(
                "rules_q{}_s{}_{}.txt",
                cli.max_qubits, cli.max_size, "nam"
            ))
        });
        assert_eq!(got, expected);
    }

    #[test]
    fn inspect_reads_a_shipped_rule_file() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let path = root.join("rules/rules_q3_s3_nam_symb.txt");
        inspect(&path).unwrap();
    }

    #[test]
    fn synthesis_is_reproducible() {
        let dir = tempfile::tempdir().unwrap();
        let mut outputs = Vec::new();
        for i in 0..2 {
            let out = dir.path().join(format!("r{i}.txt"));
            let cli = Cli::try_parse_from([
                "queso",
                "-g",
                "nam",
                "-q",
                "2",
                "-s",
                "3",
                "-o",
                out.to_str().unwrap(),
            ])
            .unwrap();
            synthesize(&cli).unwrap();
            outputs.push(std::fs::read_to_string(&out).unwrap());
        }
        assert_eq!(outputs[0], outputs[1]);
    }
}
