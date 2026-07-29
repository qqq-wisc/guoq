#!/usr/bin/env python3
"""
Collect the median circuit (across trials) for each benchmark from a guoq
results tree and accumulate them into a single benchmarks directory.

For every benchmark we gather all `latest_sol__<benchmark>.qasm` files found
anywhere under the root (one per trial), rank them, and copy the median trial
to <out_dir>/<benchmark>.qasm.

Ranking (chosen by the user):
  * primary   : T count            (ascending)
  * tiebreak  : two-qubit count    (ascending)
  * median    : upper-middle -> index len(trials)//2 after sorting
                (for 10 trials this is index 5, i.e. the 6th-best)

T count           = number of `t` and `tdg` gates.
Two-qubit count   = number of gate applications acting on exactly 2 qubits.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

DEFAULT_ROOT = (
    "/eval-5-27/rq4/"
    "guoq-6-13"
)

# Glob for the optimized solution files inside each results dir.
# Filenames look like: latest_sol_<jobid>_2_<circuit>.qasm
SOL_GLOB = "latest_sol_*.qasm"

# Benchmark result dirs are named results_<circuit>.
RESULTS_PREFIX = "results_"

# Lines that are not gate applications.
NON_GATE_KEYWORDS = {
    "OPENQASM", "include", "qreg", "creg", "gate", "opaque",
    "barrier", "measure", "reset", "if",
}

# name(optional params)  operands...
_GATE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s+(.*)$")


def parse_counts(path: Path):
    """Return (t_count, two_qubit_count) for a QASM file."""
    t_count = 0
    two_q_count = 0
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            line = line.split("//", 1)[0].strip()  # drop trailing comment
            if not line:
                continue
            if line.endswith(";"):
                line = line[:-1].strip()
            if not line:
                continue

            first = line.split()[0].split("(")[0]
            if first in NON_GATE_KEYWORDS:
                continue

            name = first.lower()
            if name in ("t", "tdg"):
                t_count += 1

            m = _GATE_RE.match(line)
            if m:
                operands = [o for o in m.group(3).split(",") if o.strip()]
                if len(operands) == 2:
                    two_q_count += 1

    return t_count, two_q_count


def benchmark_name(results_dir: Path) -> str:
    """`results_4_49_16` -> `4_49_16`."""
    name = results_dir.name
    assert name.startswith(RESULTS_PREFIX), name
    return name[len(RESULTS_PREFIX):]


def collect(root: Path):
    """Group solution files by benchmark (one results_<circuit> dir each)."""
    groups: dict[str, list[Path]] = {}
    for d in sorted(root.glob(f"{RESULTS_PREFIX}*")):
        if not d.is_dir():
            continue
        trials = sorted(p for p in d.glob(SOL_GLOB) if p.is_file())
        if trials:
            groups[benchmark_name(d)] = trials
    return groups


def pick_median(files: list[Path]):
    """Rank by (t, 2q) ascending and return (chosen_path, t, 2q)."""
    scored = []
    for f in files:
        t, q = parse_counts(f)
        scored.append((t, q, f))
    scored.sort(key=lambda x: (x[0], x[1]))
    idx = len(scored) // 2  # upper-middle
    t, q, f = scored[idx]
    return f, t, q, scored


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="root of the results tree (default: %(default)s)")
    ap.add_argument("--out", default="guoq_after_pyzx",
                    help="output benchmarks directory (default: %(default)s)")
    ap.add_argument("--expect-trials", type=int, default=10,
                    help="warn if a benchmark does not have this many trials")
    ap.add_argument("--dry-run", action="store_true",
                    help="report selections but do not copy files")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"error: root directory not readable: {root}\n"
                 f"(if this is a network volume, grant the terminal/app "
                 f"Full Disk Access in System Settings > Privacy & Security)")

    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    groups = collect(root)
    if not groups:
        sys.exit(f"error: no {RESULTS_PREFIX}*/{SOL_GLOB} files found under {root}")

    print(f"found {len(groups)} benchmark(s) under {root}\n")
    for bench in sorted(groups):
        files = groups[bench]
        chosen, t, q, scored = pick_median(files)
        note = ""
        if len(files) != args.expect_trials:
            note = f"  [WARN: {len(files)} trials, expected {args.expect_trials}]"

        dest = out_dir / f"{bench}.qasm"
        print(f"{bench}: {len(files)} trials -> median T={t} 2q={q}{note}")
        print(f"    from {chosen}")
        print(f"    ranked (T,2q): {[(a, b) for a, b, _ in scored]}")
        if not args.dry_run:
            shutil.copy2(chosen, dest)
            print(f"    -> {dest}")
        print()

    if args.dry_run:
        print("dry run: no files written.")
    else:
        print(f"done. wrote {len(groups)} circuit(s) to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
