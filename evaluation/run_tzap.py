from qiskit import QuantumCircuit
from time import time_ns
import argparse
import os
import json
import subprocess


def t_count_qiskit(circuit):
    count_ops = circuit.count_ops()
    return count_ops.get("t", 0) + count_ops.get("tdg", 0)


def run(args: dict):
    cluster = args["cluster"]
    process = args["process"]
    input_file = args["circuit_file"]

    circuit_id = input_file[input_file.rfind("/") + 1 :].replace(".qasm", "")

    out_dir = f"results_{circuit_id}"
    os.makedirs(out_dir, exist_ok=True)

    result = {}
    circuit = QuantumCircuit.from_qasm_file(input_file)
    result["original_total"] = circuit.size()
    result["original_2q"] = circuit.num_nonlocal_gates()
    result["original_t"] = t_count_qiskit(circuit)

    output_file = f"{out_dir}/optimized_{cluster}_{process}_{circuit_id}.qasm"

    start = time_ns()

    subprocess.run(
        ["tzap", "-Osuper", input_file, "-o", output_file],
        check=True,
    )

    end = time_ns()

    compiled = QuantumCircuit.from_qasm_file(output_file)
    result["optimized_total"] = compiled.size()
    result["optimized_2q"] = compiled.num_nonlocal_gates()
    result["optimized_t"] = t_count_qiskit(compiled)
    result["total_time"] = (end - start) / 1000000000

    result.update(args)
    result.update(
        {
            "circuit_id": circuit_id,
            "out_dir": out_dir,
            "method": "tzap",
        }
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TZap", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--cluster", type=str, default="none")
    parser.add_argument("--process", type=str, default="none")
    parser.add_argument("--circuit_file", type=str, required=True)

    args = parser.parse_args()

    result = run(vars(args))

    with open(
        f"{result['out_dir']}/results_{args.cluster}_{args.process}.json",
        "w",
    ) as f:
        json.dump(result, f, indent=4)
