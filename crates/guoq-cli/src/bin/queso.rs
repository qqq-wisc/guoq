//! The `queso` binary.

use std::process::ExitCode;

use clap::Parser;
use guoq_cli::synth;

fn main() -> ExitCode {
    let cli = match synth::Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return ExitCode::from(2);
        }
    };
    match synth::run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
