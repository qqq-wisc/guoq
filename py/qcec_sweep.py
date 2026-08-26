"""Sweep a benchmark directory through the optimizer and verify every result with QCEC.

Usage: qcec_sweep.py <guoq-binary> <rules-dir> <bench-dir> <gate-set> <objective> [count]

Optimizes the `count` smallest (by file size) .qasm files in the benchmark directory
under each search strategy, and checks every optimized circuit against its original
with MQT QCEC. Rewrites preserve unitaries up to a global phase, so both `equivalent`
and `equivalent_up_to_global_phase` verdicts pass; anything else — including QCEC
giving up — is a failure, never a skip. A negative control appends an extra `x` gate
to one optimized circuit and requires QCEC to reject it, so a vacuously-passing
checker cannot go unnoticed.

Modeled on the sweep in qqq-wisc/tzap's test suite.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from mqt import qcec
from mqt.qcec.pyqcec import EquivalenceCriterion

ACCEPTED = (
    EquivalenceCriterion.equivalent,
    EquivalenceCriterion.equivalent_up_to_global_phase,
)
# Both search families: the default stochastic search and the deterministic-harvest
# strategy exercise different rewrite machinery (sampled application vs. the reducing
# fixpoint), and both apply symbolic rules from the gate set's default corpus.
STRATEGIES = ("BEAM_MCMC", "REDUCE")
# Bounded by iterations, not wall clock, so every machine optimizes the same amount
# and a slow CI runner cannot turn a real failure into a flake (or vice versa).
BUDGET = ("--max-iters", "400", "--seed", "7")


def main() -> None:
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    guoq, rules_dir = sys.argv[1], sys.argv[2]
    bench_dir = Path(sys.argv[3])
    gate_set, objective = sys.argv[4], sys.argv[5]
    count = int(sys.argv[6]) if len(sys.argv) > 6 else 6

    files = sorted(bench_dir.glob("*.qasm"), key=lambda p: p.stat().st_size)[:count]
    if not files:
        sys.exit(f"no .qasm files found in {bench_dir}")

    failures = []
    last_optimized = None
    with tempfile.TemporaryDirectory() as tmp:
        for original in files:
            for strategy in STRATEGIES:
                label = f"{original.name} ({gate_set}, {strategy})"
                out_dir = Path(tmp) / f"{strategy}_{original.stem}"
                out_dir.mkdir()
                proc = subprocess.run(
                    [
                        guoq,
                        "-g",
                        gate_set,
                        "-opt",
                        objective,
                        "-search",
                        strategy,
                        "-resynth",
                        "NONE",
                        *BUDGET,
                        "--rules-dir",
                        rules_dir,
                        "-out",
                        str(out_dir),
                        str(original),
                    ],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    print(f"FAIL {label}: guoq exited {proc.returncode}")
                    print(proc.stderr, file=sys.stderr)
                    failures.append(label)
                    continue
                optimized = out_dir / f"optimized__{original.name}"
                if not optimized.exists():
                    print(f"FAIL {label}: no optimized_ file written")
                    failures.append(label)
                    continue

                verdict = qcec.verify(str(original), str(optimized)).equivalence
                status = "PASS" if verdict in ACCEPTED else "FAIL"
                print(f"{status} {label}: {verdict}", flush=True)
                if verdict not in ACCEPTED:
                    failures.append(label)
                last_optimized = (original, optimized)

        if last_optimized is not None:
            negative_control(*last_optimized, Path(tmp))

    checks = len(files) * len(STRATEGIES)
    if failures:
        sys.exit(f"{len(failures)}/{checks} checks failed QCEC: {failures}")
    print(f"all {checks} optimized circuits verified equivalent")


def negative_control(original: Path, optimized: Path, tmp: Path) -> None:
    """An optimized circuit with one extra `x` must be reported as NOT equivalent."""
    text = optimized.read_text()
    reg = re.search(r"qreg\s+(\w+)\s*\[", text)
    if reg is None:
        sys.exit(f"negative control: no qreg found in {optimized}")
    tampered = tmp / f"tampered_{original.name}"
    tampered.write_text(text + f"x {reg.group(1)}[0];\n")
    verdict = qcec.verify(str(original), str(tampered)).equivalence
    if verdict in ACCEPTED:
        sys.exit(f"negative control FAILED: extra x reported as {verdict}")
    print(f"PASS negative control ({original.name} + extra x): {verdict}")


if __name__ == "__main__":
    main()
