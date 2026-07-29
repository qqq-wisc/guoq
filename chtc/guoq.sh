#!/bin/bash
# Executable run by each HTCondor job. The GUOQ jar, rewrite rules and the
# run-scir binary are all prebuilt into the docker image, so this just drives
# one (circuit, objective) run via runner.py.
#
# Resynthesis (-resynth SYNTHETIQ | BQSKIT) works by GUOQ POSTing to a local
# Python server on :8080 (resynth.py). GUOQ does NOT launch that server, so when
# the run resynthesizes we start it here first and tear it down on exit. It runs
# in its own venv (/opt/resynth-venv, which has bqskit) so it doesn't disturb the
# base qiskit env runner.py uses. -resynth NONE (the default) starts nothing.
set -euo pipefail

echo "job args: $@"
echo "host: $(hostname)  cpus: $(nproc)"
java -version 2>&1 | head -1

RESYNTH_PY=/home/resynth.py                  # baked path -> LIB_DIR=/home/lib
RESYNTH_PYTHON=/opt/resynth-venv/bin/python  # isolated venv (see chtc/Dockerfile)
RESYNTH_PID=""

cleanup() {
    if [[ -n "$RESYNTH_PID" ]]; then
        echo "== stopping resynth server (pid $RESYNTH_PID) =="
        kill "$RESYNTH_PID" 2>/dev/null || true
        wait "$RESYNTH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# resynth alg = the token following -resynth / --resynth-alg (default NONE).
resynth_alg="NONE"
prev=""
for a in "$@"; do
    case "$prev" in
        -resynth|--resynth-alg) resynth_alg="$a" ;;
    esac
    prev="$a"
done

if [[ "$resynth_alg" != "NONE" ]]; then
    echo "== starting resynth server ($resynth_alg) on :8080 =="
    if [[ "$resynth_alg" == "BQSKIT" ]]; then
        "$RESYNTH_PYTHON" "$RESYNTH_PY" --bqskit --bqskit_auto_workers \
            > resynth_server.log 2>&1 &
    else
        "$RESYNTH_PYTHON" "$RESYNTH_PY" > resynth_server.log 2>&1 &
    fi
    RESYNTH_PID=$!

    # Wait for it to bind :8080 (importing bqskit / compiler init can be slow).
    ready=""
    for _ in $(seq 1 120); do
        if ! kill -0 "$RESYNTH_PID" 2>/dev/null; then
            echo "resynth server exited during startup; log:"
            cat resynth_server.log || true
            exit 1
        fi
        if (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null; then
            ready=1
            echo "resynth server is up"
            break
        fi
        sleep 1
    done
    if [[ -z "$ready" ]]; then
        echo "resynth server did not bind :8080 within 120s; log:"
        cat resynth_server.log || true
        exit 1
    fi
fi

echo "== running GUOQ =="
python3 runner.py "$@"
