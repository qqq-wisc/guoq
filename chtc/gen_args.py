#!/usr/bin/env python3
"""
Generate the HTCondor `queue ... from` args file(s) for the GUOQ routed-depth /
depth-ft sweep.

Each output line is:  <circuit.qasm> <GUOQ Optimizer flags...>
and is consumed by guoq.sub via `queue Program, args from <file>`, where
HTCondor assigns the first whitespace token to $(Program) and the remainder to
$(args). runner.py forwards $(args) straight to qoptimizer.Optimizer.

Sweep (per circuit): {ROUTED_DEPTH, DEPTH_FT} x {SEEDS}, for one routing backend
(scir, scmr or fastls; pick with --backend, per-backend knobs below).
Gate set is Clifford+T and resynthesis is off, matching the nam_t_tdg benchmarks.

The two circuit sets are kept in SEPARATE args files (and should be submitted
into separate output areas) because they share benchmark file names -- e.g.
circuits_guoq/4_49_16.qasm and circuits_guoq_after_pyzx/4_49_16.qasm would
otherwise collide in transfer_input_files and in the results_<id> dir on the
submit node. `--all` writes both files in one go.

Paths baked into the docker image (see chtc/Dockerfile):
  --rules-dir /home/rules            GUOQ rewrite rules
  --qmr-dir   /opt/qmr               solver binaries at /opt/qmr/target/release/
                                     (run-scir, run-scmr, fastls -- GUOQ resolves
                                     the one for --qmr-backend under this dir)
"""

import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# --- experiment knobs -------------------------------------------------------
# Optimization objectives to sweep. Both route with the QMR backend (ROUTED_DEPTH
# is pure routed depth; DEPTH_FT combines FT cost with routed depth as a tiebreak).
OPT_OBJS = ["DEPTH_FT", "ROUTED_DEPTH"]
# Seeds per (circuit, objective). Only the scir backend takes a --qmr-seed; for
# scmr/fastls the seed loop just queues len(SEEDS) repeat jobs per config (GUOQ's
# own search is stochastic, so repeats still give spread).
SEEDS = [1]
GATE_SET = "CLIFFORDT"
RESYNTH = "SYNTHETIQ"
EPSILON = 1e-8

# Which routing solver GUOQ's cost objective calls (--qmr-backend):
#   scir   - run-scir NISQ SWAP routing (weighted routed depth)
#   scmr   - run-scmr surface-code lattice-surgery routing (parallel step count)
#   fastls - FastLS fast-lattice-surgery routing (routing-layer count)
# Overridable per invocation with --backend.
QMR_BACKEND = "scmr"

# Routing config used to score routed depth during search. Arch is a layout name
# auto-sized to the circuit; fastls auto-sizes with no named layout so it has no
# entry here.
QMR_ARCH = {
    "scir": "test-compact",   # compact | square-sparse | test-compact
    "scmr": "compact",        # compact | square_sparse
}
QMR_TRIALS = 3                # routing solves averaged per cost evaluation

# scir-only knobs.
QMR_CHUNKS = 1                # 1 = whole-circuit (quality ceiling, single thread)
QMR_RECONCILE = "reversal"    # reversal | reversal-compressed | maps | sabre

# scmr-only: onepass | parallel | joint-optimize-par (anytime search, best taken).
QMR_SCMR_MODE = "joint-optimize-par"

# fastls-only: optional path (inside the image) to a FastLS config .toml tuning
# its simulated-annealing params. None omits -c and uses FastLS's defaults.
QMR_FASTLS_CONFIG = None

# Baked-in image paths.
RULES_DIR = "/home/rules"
QMR_DIR = "/opt/qmr"

# Default circuit sets (dir name under the repo root -> output args file).
CIRCUIT_SETS = {
    "circuits_guoq": "guoq.args",
    "after_pyzx_pt": "guoq_after_pyzxpt.args",
}
# ---------------------------------------------------------------------------


def guoq_flags(opt_obj, seed, backend):
    flags = [
        "-g", GATE_SET,
        "-opt", opt_obj,
        "-resynth", RESYNTH,
        "--rules-dir", RULES_DIR,
        "--qmr-dir", QMR_DIR,
        "--qmr-backend", backend.upper(),  # Java enum names are uppercase
        "--qmr-trials", str(QMR_TRIALS),
    ]
    if backend in QMR_ARCH:
        flags += ["--qmr-arch", QMR_ARCH[backend]]
    if backend == "scir":
        flags += [
            "--qmr-chunks", str(QMR_CHUNKS),
            "--qmr-reconcile", QMR_RECONCILE,
            "--qmr-seed", str(seed),
        ]
    elif backend == "scmr":
        flags += ["--qmr-scmr-mode", QMR_SCMR_MODE]
    elif backend == "fastls":
        if QMR_FASTLS_CONFIG:
            flags += ["--qmr-fastls-config", QMR_FASTLS_CONFIG]
    flags += ["-eps", str(EPSILON)]
    return " ".join(flags)


def write_args(circuit_dir, out_file, backend):
    if not os.path.isdir(circuit_dir):
        raise SystemExit(f"error: circuit dir not found: {circuit_dir}")
    circuits = sorted(f for f in os.listdir(circuit_dir) if f.endswith(".qasm"))
    if not circuits:
        raise SystemExit(f"error: no .qasm circuits in {circuit_dir}")

    n = 0
    with open(out_file, "w") as out:
        for circuit in circuits:
            for opt_obj in OPT_OBJS:
                for seed in SEEDS:
                    out.write(f"{circuit} {guoq_flags(opt_obj, seed, backend)}\n")
                    n += 1
    print(f"{circuit_dir}: {len(circuits)} circuits x {len(OPT_OBJS)} objs "
          f"x {len(SEEDS)} seeds ({backend}) -> wrote {n} lines to {out_file}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit-dir", default=None,
                    help="circuit set to generate for (dir name under repo root, "
                         "or a path). default: use --all")
    ap.add_argument("--out", default=None,
                    help="output args file (default: derived from the dir name)")
    ap.add_argument("--all", action="store_true",
                    help=f"generate for both default sets: {list(CIRCUIT_SETS)}")
    ap.add_argument("--backend", default=QMR_BACKEND,
                    choices=["scir", "scmr", "fastls"],
                    help="routing solver GUOQ's cost objective calls")
    args = ap.parse_args()

    if args.all or args.circuit_dir is None:
        for name, out_file in CIRCUIT_SETS.items():
            write_args(os.path.join(REPO_ROOT, name), os.path.join(SCRIPT_DIR, out_file),
                       args.backend)
        return

    circuit_dir = args.circuit_dir
    if not os.path.isabs(circuit_dir) and not os.path.isdir(circuit_dir):
        circuit_dir = os.path.join(REPO_ROOT, circuit_dir)
    out_file = args.out or CIRCUIT_SETS.get(
        os.path.basename(circuit_dir.rstrip("/")),
        os.path.basename(circuit_dir.rstrip("/")) + ".args",
    )
    out_file = out_file if os.path.isabs(out_file) else os.path.join(SCRIPT_DIR, out_file)
    write_args(circuit_dir, out_file, args.backend)


if __name__ == "__main__":
    main()
