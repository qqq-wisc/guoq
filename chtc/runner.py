#!/usr/bin/env python3
"""
Drive one GUOQ optimization job on CHTC.

This is a CHTC-oriented adaptation of ../evaluation/run_guoq.py. Differences:

  * GUOQ CLI flags are passed straight through on argv (via HTCondor's
    `arguments = "$(Program) $(args) --run_id ..."`) instead of being read from
    an encoded file. Anything this script does not recognize is forwarded to the
    Optimizer verbatim (parse_known_args), so gen_args.py fully controls the run.
  * verbosity is 10 (was 2).
  * the (potentially huge) GUOQ .out log is gzip-compressed before the job's
    output sandbox is transferred back, so the submit server does not run out of
    space. Parsing of the log for the results JSON happens first, on the
    uncompressed file.
  * the resynth log is optional (these runs use -resynth NONE, which starts no
    resynth server and writes no resynth log).
  * --skip-guoq runs ONLY the mapping-and-routing pass, on the input circuit
    (skipping the GUOQ optimization entirely), to record the baseline routed cost
    of the unoptimized circuit. Backend/arch/etc. are still read from the
    forwarded flags, so it routes with the same config the optimized runs use.

The jar, rules and the routing solver binaries (run-scir, run-scmr, fastls) are
baked into the docker image (amxu/guoq-scmr:*), so a job only needs its circuit
transferred in. The solvers are reached by GUOQ via `--qmr-dir /opt/qmr` (see
chtc/Dockerfile); which one runs is selected by --qmr-backend, and the rescoring
pass here dispatches on the same flag.

Output (mirrors run_guoq.py, dir name matches the results_<id> convention):
    results_<circuit_id>/results_<cluster>_<process>.json   (metadata + counts)
    results_<circuit_id>/guoq_log_<cluster>_<process>.out.gz (compressed log)
    results_<circuit_id>/latest_sol_<cluster>_<process>_<circuit_id>.qasm
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import time


def write_args_file(args_file, guoq_flags, circuit_file, job, out_dir, verbosity):
    """Write a GUOQ @-argfile: one token per line, then job/out/verbosity, then
    the circuit as the trailing positional."""
    with open(args_file, "w") as f:
        for token in guoq_flags:
            f.write(f"{token}\n")
        f.write(f"-job\n{job}\n")
        f.write(f"-out\n{out_dir}\n")
        f.write(f"--verbosity\n{verbosity}\n")
        f.write(f"{circuit_file}\n")


def flag_value(flags, name, default=None):
    """Return the value following `name` in the flat guoq_flags list."""
    if name in flags:
        i = flags.index(name)
        if i + 1 < len(flags):
            return flags[i + 1]
    return default


def run_scir_once(bin_path, circuit, arch, chunks, reconcile, seed, timeout, out_json):
    """One run-scir routed-depth solve, mirroring qoptimizer.Qmr.solveOnce.

    Writes the full mapping-and-routing solution to `out_json` (kept, so it can be
    compressed and transferred back). Returns (cost, timed_out). Progress is
    discarded (the cost we want is in the JSON), so stdout/stderr go to DEVNULL --
    this also avoids the pipe-buffer deadlock Qmr.java has to drain around."""
    cmd = [
        bin_path, circuit, arch, "chunked",
        "--chunks", str(chunks),
        "--reconcile", reconcile,
        "--depth",
        "--seed", str(seed),
        "--output", out_json,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, True
    if proc.returncode != 0 or not os.path.exists(out_json):
        return None, False
    with open(out_json) as f:
        obj = json.load(f)
    cost = obj.get("cost")
    return (float(cost) if cost is not None else None), False


def run_scmr_once(bin_path, circuit, arch, mode, timeout, out_stream):
    """One run-scmr lattice-surgery solve, mirroring qoptimizer.Qmr.solveScmr.

    run-scmr writes JSON solutions to stdout: exactly one object for
    --onepass/--parallel, or a newline-delimited stream of improving solutions
    for --joint-optimize-par. The cost is the minimum over all objects emitted.
    Non-JSON progress lines are skipped. The raw stdout is saved to `out_stream`
    (kept, so the solutions can be compressed and transferred back).
    Returns (cost, timed_out)."""
    cmd = [bin_path, circuit, arch, f"--{mode}"]
    print(cmd)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, True
    with open(out_stream, "wb") as f:
        f.write(proc.stdout)
    if proc.returncode != 0:
        return None, False
    best = None
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            cost = json.loads(line).get("cost")
        except json.JSONDecodeError:
            continue
        if cost is not None and (best is None or float(cost) < best):
            best = float(cost)
    return best, False


def run_fastls_once(bin_path, circuit, config, timeout, out_json):
    """One FastLS solve, mirroring qoptimizer.Qmr.solveFastls: the routed depth
    is the number of routing layers, i.e. len(steps) in the -o result JSON.
    (FastLS's stdout "DEPTH:" line is the *input* circuit depth, not the routing
    result, so the JSON is read instead.) Writes the full result to `out_json`
    (kept, so it can be compressed and transferred back).
    Returns (cost, timed_out)."""
    cmd = [bin_path]
    if config:
        cmd += ["-c", config]
    cmd += ["-s", "-o", out_json, circuit]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, True
    if proc.returncode != 0 or not os.path.exists(out_json):
        return None, False
    with open(out_json) as f:
        obj = json.load(f)
    steps = obj.get("steps")
    return (float(len(steps)) if steps is not None else None), False


QMR_BINARIES = {"SCIR": "run-scir", "SCMR": "run-scmr", "FASTLS": "fastls"}


def rescore_with_qmr(opt_circuit, guoq_flags, per_trial_timeout, out_dir, job):
    """Re-measure GUOQ's output circuit with the SAME routing backend and config
    GUOQ used for its cost objective (--qmr-backend / arch / mode / chunks /
    reconcile / seed / trials), and return the average cost. Mirrors
    qoptimizer.Qmr.depth: every trial uses the same config (and, for scir, the
    same QMR_SEED) and the results are averaged.

    Each trial's routing solution (a JSON file for scir/fastls, the raw JSON-lines
    stdout stream for scmr) is written into out_dir and gzip-compressed, so the
    (potentially large) solutions come back cheaply."""
    # Default mirrors Params.QMR_BACKEND, which GUOQ uses when the flag is absent.
    backend = flag_value(guoq_flags, "--qmr-backend", "SCMR").upper()
    if backend not in QMR_BINARIES:
        return {"skipped": f"unknown --qmr-backend {backend}"}
    arch = flag_value(guoq_flags, "--qmr-arch")
    if backend != "FASTLS" and arch is None:
        return {"skipped": "no --qmr-arch (objective does not route)"}

    qmr_dir = flag_value(guoq_flags, "--qmr-dir", "/opt/qmr")
    trials = int(flag_value(guoq_flags, "--qmr-trials", "5"))
    bin_name = QMR_BINARIES[backend]
    bin_path = os.path.join(qmr_dir, "target", "release", bin_name)

    info = {
        "circuit": os.path.basename(opt_circuit),
        "backend": backend, "bin": bin_path,
        "trials": trials, "per_trial_timeout": per_trial_timeout,
    }
    if backend == "SCIR":
        chunks = flag_value(guoq_flags, "--qmr-chunks", "1")
        reconcile = flag_value(guoq_flags, "--qmr-reconcile", "reversal")
        seed = int(flag_value(guoq_flags, "--qmr-seed", "1"))
        info.update({"arch": arch, "chunks": int(chunks),
                     "reconcile": reconcile, "seed": seed})

        def solve(sol_file):
            return run_scir_once(bin_path, opt_circuit, arch, chunks, reconcile,
                                 seed, per_trial_timeout, sol_file)
    elif backend == "SCMR":
        mode = flag_value(guoq_flags, "--qmr-scmr-mode", "joint-optimize-par")
        info.update({"arch": arch, "mode": mode})

        def solve(sol_file):
            return run_scmr_once(bin_path, opt_circuit, arch, mode,
                                 per_trial_timeout, sol_file)
    else:  # FASTLS
        config = flag_value(guoq_flags, "--qmr-fastls-config")
        info.update({"config": config})

        def solve(sol_file):
            return run_fastls_once(bin_path, opt_circuit, config,
                                   per_trial_timeout, sol_file)

    if not os.path.exists(opt_circuit):
        info["error"] = "no optimized circuit found (GUOQ produced no output)"
        return info

    costs, timed_out, solution_files = [], 0, []
    start = time.perf_counter()
    for i in range(trials):
        sol_json = os.path.join(out_dir, f"routing_sol_{job}_trial{i + 1}.json")
        cost, to = solve(sol_json)
        costs.append(cost)
        timed_out += int(to)
        # Compress the (potentially large) solution JSON before transfer-back.
        if os.path.exists(sol_json):
            solution_files.append(os.path.basename(gzip_in_place(sol_json)))
        print(f"  {bin_name} rescore trial {i + 1}/{trials}: "
              f"cost={cost}{' (timed out)' if to else ''}", flush=True)

    ok = [c for c in costs if c is not None]
    info.update({
        "costs": costs,
        "num_timed_out": timed_out,
        "num_failed": sum(1 for c in costs if c is None) - timed_out,
        "avg_cost": (sum(ok) / len(ok)) if ok else None,
        "total_time": time.perf_counter() - start,
        "solution_files": solution_files,
    })
    return info


def gzip_in_place(path, level=9):
    """gzip `path` -> `path`.gz (level 9 == gzip -9) and remove the original."""
    gz_path = path + ".gz"
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=level) as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)
    return gz_path


def run(args, guoq_flags):
    run_id = args.run_id
    # HTCondor run_id is "<cluster>-<procid>"; GUOQ's job tag (which becomes part
    # of the latest_sol_<job>_<circuit>.qasm name) uses underscores, matching the
    # existing results naming convention (e.g. latest_sol_277290_2_4_49_16.qasm).
    job = run_id.replace("-", "_")

    jar = args.jar
    circuit_file = args.circuit_file
    timeout = args.timeout

    circuit_id = os.path.basename(circuit_file).replace(".qasm", "")
    # out_root keeps the two circuit sets (which share benchmark file names) in
    # separate output trees so results_<id> dirs never collide on the submit node.
    out_dir = os.path.join(args.out_root, f"results_{circuit_id}")
    os.makedirs(out_dir, exist_ok=True)

    # GUOQ-produced fields; stay empty/None in --skip-guoq mode, where the
    # optimization pass (and thus its log) is never produced.
    results = {}
    original_counts = {}
    config = None
    errors = None
    resynth_errors = None
    command = None
    args_file = None
    out_name = None

    if args.skip_guoq:
        # Routing-only: skip the GUOQ optimization pass entirely and route the
        # INPUT circuit as-is. This gives the baseline routed cost of the
        # unoptimized circuit (useful as a control for the optimized runs). No
        # GUOQ log is produced, so there is nothing to parse or compress.
        circuit_to_route = circuit_file
    else:
        out_name = f"{out_dir}/guoq_log_{job}.out"
        err_name = out_name.replace(".out", ".err")

        args_file = f"args_{job}.txt"
        args_file_path = f"{out_dir}/{args_file}"
        write_args_file(
            args_file_path, guoq_flags, circuit_file, job, out_dir, args.verbosity
        )

        command = (
            f"timeout {timeout} java -ea -Xmx{args.xmx} -cp {jar} "
            f"qoptimizer.Optimizer @{args_file_path}"
        )
        print(command)
        with open(out_name, "w") as out, open(err_name, "w") as err:
            proc = subprocess.Popen(command.split(" "), stdout=out, stderr=err)
            proc.wait()

        # Parse the (still uncompressed) log for GUOQ's JSON records.
        with open(out_name, "r") as f:
            log = f.readlines()
        for line in log:
            if "job_info" in line:
                config = json.loads(line)
                break
        for line in log:
            if "original_total" in line:
                original_counts = json.loads(line)
                break
        for line in reversed(log):
            if "rules_applied_to_best" in line:
                results = json.loads(line)
                results.pop("rules_applied_to_best", None)
                results.pop("all_rules_applied", None)
                break

        with open(err_name, "r") as f:
            contents = f.read()
            if "Exception" in contents or "Error" in contents:
                errors = "error in ours"

        # -resynth NONE starts no resynth server, so this log usually does not exist.
        resynth_log = f"{out_dir}/resynth_log_{job}.txt"
        if os.path.exists(resynth_log):
            with open(resynth_log) as f:
                contents = f.read()
                if "Exception" in contents or "Error" in contents:
                    resynth_errors = "error in resynth"

        # GUOQ's output circuit, to be re-scored below.
        circuit_to_route = os.path.join(
            out_dir, f"latest_sol_{job}_{os.path.basename(circuit_file)}")

    # Score the circuit with the routing solver directly, using the same backend +
    # config selected on the command line (the same one GUOQ optimizes against),
    # and record the average cost. In --skip-guoq mode this scores the input
    # circuit; otherwise it re-scores GUOQ's optimized output.
    rescore = rescore_with_qmr(circuit_to_route, guoq_flags, args.rescore_timeout,
                               out_dir, job)
    print(f"qmr rescore ({rescore.get('backend')}) "
          f"avg_cost={rescore.get('avg_cost')}", flush=True)

    result = dict(vars(args))
    result.pop("circuit_file", None)
    result.update(results)
    result.update(original_counts)
    result.update(
        {
            "circuit_id": circuit_id,
            "circuit_file": circuit_file,
            "out_dir": out_dir,
            "command": command,
            "guoq_flags": guoq_flags,
            "args_file": args_file,
            "guoq_config": config,
            "error": errors,
            "resynth_errors": resynth_errors,
            "qmr_rescore": rescore,
            "qmr_avg_cost": rescore.get("avg_cost"),
        }
    )

    # Compress the verbosity-10 GUOQ log before transfer-back (keeps the submit
    # node from filling up). Skipped in --skip-guoq mode, where there is no log.
    if out_name is not None:
        gz = gzip_in_place(out_name)
        print(f"compressed log: {gz}", flush=True)

    return out_dir, job, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one GUOQ job on CHTC; forwards unknown flags to the Optimizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("circuit_file", help="the .qasm circuit (transferred into cwd)")
    parser.add_argument("--run_id", type=str, default="0-0",
                        help="unique job id, HTCondor '<cluster>-<procid>'")
    parser.add_argument("--jar", type=str,
                        default="/home/GUOQ-1.0-jar-with-dependencies.jar")
    parser.add_argument("--timeout", type=int, default=60 * 60,
                        help="external wall-clock timeout in seconds")
    parser.add_argument("--xmx", type=str, default="20g",
                        help="JVM max heap (-Xmx)")
    parser.add_argument("--verbosity", type=int, default=10)
    parser.add_argument("--out-root", dest="out_root", type=str, default=".",
                        help="directory to place results_<id>/ under")
    parser.add_argument("--rescore-timeout", dest="rescore_timeout", type=int,
                        default=60 * 60,
                        help="per-trial timeout (s) for the routing rescoring pass")
    parser.add_argument("--skip-guoq", dest="skip_guoq", action="store_true",
                        help="skip the GUOQ optimization pass and only run mapping "
                             "and routing on the input circuit (baseline routed cost)")
    args, guoq_flags = parser.parse_known_args()

    out_dir, job, result = run(args, guoq_flags)

    with open(f"{out_dir}/results_{job}.json", "w") as f:
        json.dump(result, f, indent=4)
    print(f"wrote {out_dir}/results_{job}.json", flush=True)
