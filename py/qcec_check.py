#!/usr/bin/env python3
"""Check two QASM circuits for equivalence with MQT QCEC.

The optimizer's own end-to-end check (`--verify-final`) builds dense unitaries, which
caps it at about twelve qubits. QCEC checks equivalence symbolically, so it reaches the
circuits the native check cannot -- a 226k-gate, 16-qubit circuit verifies in seconds.

Usage: qcec_check.py A.qasm B.qasm
Prints one word on stdout and exits 0 if the circuits are equivalent (a global phase is
no difference; nothing observable depends on it, and the optimizer's own distances are
phase-invariant for the same reason), 1 if they are not, 2 if the check itself failed.
"""

import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: qcec_check.py A.qasm B.qasm", file=sys.stderr)
        return 2
    try:
        import mqt.qcec as qcec
    except ImportError:
        print("mqt.qcec is not installed: pip install mqt.qcec", file=sys.stderr)
        return 2
    try:
        result = qcec.verify(sys.argv[1], sys.argv[2])
    except Exception as e:  # noqa: BLE001 -- any checker failure is a verdict of "unknown"
        print(f"qcec failed: {e}", file=sys.stderr)
        return 2
    verdict = str(result.equivalence)
    print(verdict)
    if "no_information" in verdict or "probably" in verdict:
        # The checker gave up or fell back to sampling; treat as a failed check rather
        # than a pass, so a timeout can never masquerade as equivalence.
        return 2
    return 0 if "equivalent" in verdict and "not_equivalent" not in verdict else 1


if __name__ == "__main__":
    sys.exit(main())
