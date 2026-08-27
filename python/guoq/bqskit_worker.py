"""BQSKit resynthesis worker.

Reads one JSON request per line on stdin, writes one JSON response per line on stdout.
Spawned and owned by the `guoq` process; there is no server, no port, and nothing for the
user to start or stop.

This replaces `resynth.py` from the reference implementation, which was a ~400-line
`socketserver.TCPServer` on port 8080 that the user had to launch by hand, keep running
across invocations, and shut down with a keyboard interrupt. Everything that is not
"call BQSKit" -- picking the best circuit, translating gate sets, measuring error --
now happens on the Rust side, so this file only has to do the one thing that genuinely
needs Python.

Protocol
--------
Request:  {"circuit": "<qasm>", "opt_level": 3, "epsilon": 1e-8, "target": "nam"}
Response: {"ok": true, "circuit": "<qasm>"} | {"ok": false, "error": "<message>"}

`target` is the gate set's `resynth_target` name. Known names become a BQSKit
`MachineModel`, so synthesis lands directly in the requested basis; the reference
instead compiled into BQSKit's default `u3`/`cx` basis and re-translated with Qiskit's
`BasisTranslator` afterwards, which is a second dependency doing a job the synthesizer
can do itself. Unknown or empty names fall back to BQSKit's default basis.

Results are polished before they are returned. BQSKit's convergence test is the
trace-identity distance, whose float64 noise floor is about 1e-8 (see the numerical
note in docs/PORTING-NOTES.md), so circuits it reports as converged scatter anywhere
from 1e-14 to 1e-7 in the elementwise metric the Rust side actually charges against
the budget. Re-instantiating the returned circuit's parameters and keeping whichever
version measures better elementwise costs a fraction of the compile and reliably lands
near machine precision -- without it, a tight `--epsilon` (1e-8, say) rejects every
BQSKit result even though the same circuit's parameters could have satisfied it.

A "ready" line is written once at startup so the parent knows imports have finished;
BQSKit takes several seconds to load, and the parent should not conclude it has hung.
"""

import json
import sys


def main() -> int:
    try:
        import numpy as np
        from bqskit import MachineModel, compile
        from bqskit.compiler import Compiler
        from bqskit.ir.gates import (
            CNOTGate,
            HGate,
            RXGate,
            RXXGate,
            RYGate,
            RZGate,
            SXGate,
            XGate,
        )
        from bqskit.ir.lang.qasm2 import OPENQASM2Language
    except Exception as exc:  # noqa: BLE001 - report any import failure to the parent
        sys.stdout.write(json.dumps({"ready": False, "error": f"import failed: {exc}"}) + "\n")
        sys.stdout.flush()
        return 1

    # Keyed by each gate set's `resynth_target` (crates/qcircuit/src/gatesets.toml).
    gate_sets = {
        "nam": {CNOTGate(), RZGate(), HGate(), XGate()},
        "ibm_new": {CNOTGate(), RZGate(), SXGate(), XGate()},
        "ion": {RXXGate(), RZGate(), RXGate(), RYGate()},
    }

    def polish(target, result):
        # Warm-started local refinement of the compiled circuit's parameters against
        # the elementwise phase-aligned residuals -- the metric the Rust side charges.
        # It has to be warm-started and elementwise: BQSKit's `instantiate` re-seeds
        # from scratch and lands in worse optima for restricted bases, and BQSKit's
        # own cost cannot resolve anything below its ~1e-8 noise floor. From a result
        # already at ~1e-7 this converges to machine precision in a handful of
        # Gauss-Newton steps. Best-effort by design: any failure keeps the original.
        if result.num_params == 0:
            return result
        t = np.asarray(target)

        def residuals(params):
            u = np.asarray(result.get_unitary(params))
            trace = np.trace(t.conj().T @ u)
            phase = trace / abs(trace) if abs(trace) > 0 else 1.0
            r = t - u * np.conjugate(phase)
            return np.concatenate([r.real.ravel(), r.imag.ravel()])

        try:
            from scipy.optimize import least_squares

            start = np.array(result.params)
            fit = least_squares(residuals, start, xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if np.linalg.norm(fit.fun) < np.linalg.norm(residuals(start)):
                result.set_params(fit.x)
        except Exception:  # noqa: BLE001
            pass
        return result

    # One compiler for the whole session: starting it is the expensive part, which is why
    # a persistent worker exists rather than a fresh interpreter per call.
    compiler = Compiler()
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                # BQSKit's own QASM 2 parser, so the worker needs nothing but bqskit --
                # the reference went through Qiskit here purely as a parsing detour.
                circuit = OPENQASM2Language().decode(request["circuit"])
                target = gate_sets.get(request.get("target", ""))
                model = (
                    MachineModel(circuit.num_qudits, gate_set=target) if target else None
                )
                unitary = circuit.get_unitary()
                result = compile(
                    unitary,
                    optimization_level=int(request.get("opt_level", 3)),
                    synthesis_epsilon=float(request.get("epsilon", 1e-8)),
                    compiler=compiler,
                    model=model,
                )
                result = polish(unitary, result)
                response = {"ok": True, "circuit": result.to("qasm")}
            except Exception as exc:  # noqa: BLE001 - one bad request must not kill the worker
                response = {"ok": False, "error": str(exc)}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    finally:
        compiler.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
