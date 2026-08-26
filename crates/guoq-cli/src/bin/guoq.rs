//! The `guoq` binary.

use std::process::ExitCode;

use guoq_cli::{optimize, Cli};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cli = match Cli::parse_from_args(args) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    match optimize(&cli) {
        Ok(outcome) => {
            if cli.verbosity == 0 {
                println!(
                    "{} -> {} gates",
                    outcome.original_gates,
                    outcome.result.best.dag.gate_count()
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
