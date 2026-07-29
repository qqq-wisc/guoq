#!/usr/bin/env python3
"""
Run tzap (-Osuper) on every circuit in a directory and collect all results
in one output directory:

  <out>/<circuit>.qasm    optimized circuit (flat, like after_pyzx/)
  <out>/results.json      per-circuit gate counts + compile time

Already-completed circuits are skipped on re-run (use --force to redo).
Failures are recorded in results.json and do not stop the batch.
"""

import argparse
import json
import subprocess
from pathlib import Path
from time import time_ns

from qiskit import QuantumCircuit


def counts(qasm_file: Path) -> dict:
    circuit = QuantumCircuit.from_qasm_file(str(qasm_file))
    ops = circuit.count_ops()
    return {
        "total": circuit.size(),
        "2q": circuit.num_nonlocal_gates(),
        "t": ops.get("t", 0) + ops.get("tdg", 0),
    }


def run_one(input_file: Path, output_file: Path) -> dict:
    result = {}
    for k, v in counts(input_file).items():
        result[f"original_{k}"] = v

    start = time_ns()
    proc = subprocess.run(
        ["tzap", "-Osuper", str(input_file), "-o", str(output_file)],
        capture_output=True,
        text=True,
    )
    end = time_ns()

    if proc.returncode != 0 or not output_file.exists():
        result["status"] = "error"
        result["error"] = (proc.stderr or proc.stdout).strip()[-2000:]
        return result

    for k, v in counts(output_file).items():
        result[f"optimized_{k}"] = v
    result["total_time"] = (end - start) / 1000000000
    result["status"] = "ok"
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--circuits", default="circuits_guoq",
                    help="input circuits directory (default: %(default)s)")
    ap.add_argument("--out", default="after_tzap",
                    help="output directory (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="redo circuits that already have results")
    args = ap.parse_args()

    circuits_dir = Path(args.circuits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / "results.json"
    results = {}
    if results_file.exists() and not args.force:
        results = json.loads(results_file.read_text())

    circuit_files = sorted(circuits_dir.glob("*.qasm"))
    if not circuit_files:
        raise SystemExit(f"no .qasm files found in {circuits_dir}")

    for i, input_file in enumerate(circuit_files, 1):
        circuit_id = input_file.stem
        output_file = out_dir / input_file.name

        if (
            not args.force
            and results.get(circuit_id, {}).get("status") == "ok"
            and output_file.exists()
        ):
            print(f"[{i}/{len(circuit_files)}] {circuit_id}: skipped (done)")
            continue

        try:
            result = run_one(input_file, output_file)
        except Exception as e:
            result = {"status": "error", "error": f"{type(e).__name__}: {e}"}

        result["circuit_id"] = circuit_id
        result["method"] = "tzap"
        results[circuit_id] = result

        if result["status"] == "ok":
            print(
                f"[{i}/{len(circuit_files)}] {circuit_id}: "
                f"{result['original_total']} -> {result['optimized_total']} gates, "
                f"t {result['original_t']} -> {result['optimized_t']}, "
                f"{result['total_time']:.2f}s"
            )
        else:
            print(f"[{i}/{len(circuit_files)}] {circuit_id}: ERROR")

        results_file.write_text(json.dumps(results, indent=4))

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    errors = [c for c, r in results.items() if r["status"] != "ok"]
    print(f"\ndone: {ok}/{len(circuit_files)} ok, results in {out_dir.resolve()}")
    if errors:
        print(f"errors ({len(errors)}): {', '.join(errors)}")


if __name__ == "__main__":
    main()
