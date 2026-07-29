# syntax=docker/dockerfile:1.7
#
# GUOQ + routing-solver image for the routed-depth / depth-ft CHTC sweep.
#
# Base amxu/scmr:2 already carries the compiled run-scir routing solver (at
# /usr/local/bin/run-scir) plus python3 + qiskit. On top of it we add a JDK 21 +
# Maven, build the GUOQ jar FROM SOURCE, and bake the jar, rewrite rules, and the
# routing binaries into the image so CHTC jobs transfer nothing but their circuit.
#
# The ROUTED_DEPTH / DEPTH_FT objectives can call three routing backends; this image
# bakes in all three (see Qmr.java / --qmr-backend). All three binaries are exposed
# under a SINGLE root so one `--qmr-dir /opt/qmr` works for every backend (GUOQ and
# runner.py append `target/release/<binary>`, so --qmr-dir is the root, not that dir):
#   SCIR   -> /opt/qmr/target/release/run-scir    [from base]
#   SCMR   -> /opt/qmr/target/release/run-scmr    [built below]
#   FASTLS -> /opt/qmr/target/release/fastls      [symlink to /opt/fastls; built below]
#
# Build context MUST be the repo root (the runtime stage COPYs src/, rules/, ...).
# FastLS source lives OUTSIDE the repo, so it is supplied as a named build context.
# Build with (or just: ./chtc/build_docker_image.sh 1):
#   docker buildx build --platform linux/amd64 -f chtc/Dockerfile \
#       --build-context fastls=research/FastLS \
#       --load -t amxu/guoq-scmr:1 .

# ============================================================================
# stage 1: build the Rust routing binaries (run-scmr from GitHub, fastls from
# the local `fastls` build context). Built on rust:bookworm so the resulting
# binaries link the same glibc as the bookworm-based runtime stage below. This
# whole stage is discarded -- only the two binaries are copied forward, so the
# Rust toolchain and build trees never reach the final image.
# ============================================================================
FROM rust:bookworm AS rustbuild

# run-scmr: sibling of run-scir; auto-sizes a named surface-code layout to the
# circuit and reports parallel routing depth. Cloned from GitHub.
RUN git clone --depth 1 https://github.com/qqq-wisc/qmr-compiler-generator /src/qmr \
    && cargo build --release --manifest-path /src/qmr/Cargo.toml --bin run-scmr \
    && test -x /src/qmr/target/release/run-scmr

# fastls: fast-lattice-surgery router (edition 2024, needs rust >= 1.85). Copy only
# the crate sources from the `fastls` named context -- NOT its ~400M local target/,
# which is a host build and useless here. No external path deps, so this is complete.
COPY --from=fastls Cargo.toml Cargo.lock build.rs example_config.toml fast_config.toml /src/fastls/
COPY --from=fastls src   /src/fastls/src
COPY --from=fastls tests /src/fastls/tests
RUN cargo build --release --manifest-path /src/fastls/Cargo.toml --bin fastls \
    && test -x /src/fastls/target/release/fastls

# ============================================================================
# stage 2: runtime image
# ============================================================================
FROM amxu/scmr:2

# --- toolchain: JDK 21 (GUOQ targets Java 21) + Maven -----------------------
# scmr:2 is debian bookworm based; bookworm ships openjdk-17, so pull Temurin 21
# from the Adoptium apt repo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates maven \
    && install -d /etc/apt/keyrings \
    && wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public \
        | gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb bookworm main" \
        > /etc/apt/sources.list.d/adoptium.list \
    && apt-get update && apt-get install -y --no-install-recommends temurin-21-jdk \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

# --- build the GUOQ jar from source -----------------------------------------
WORKDIR /build
COPY pom.xml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.m2 \
    mvn -q -Dmaven.test.skip package \
    && cp target/GUOQ-1.0-jar-with-dependencies.jar /home/GUOQ-1.0-jar-with-dependencies.jar

# --- runtime assets baked into the image ------------------------------------
# rules/ is referenced via `--rules-dir /home/rules`. benchmarks/ is NOT baked
# in: CHTC jobs get their circuit via transfer_input_files, and copying the
# ~400M benchmark tree would bloat both the build context and the image.
COPY rules /home/rules
COPY resynth.py /home/resynth.py
COPY chtc/runner.py /home/runner.py

# GUOQ's Qmr.java invokes `${--qmr-dir}/target/release/<binary>`, so expose each
# routing binary at the path it expects.
#
# SCIR + SCMR share --qmr-dir /opt/qmr: run-scir is the base image's prebuilt
# binary (symlinked in), run-scmr is the freshly built binary from the rustbuild
# stage, both under /opt/qmr/target/release.
RUN mkdir -p /opt/qmr/target/release \
    && ln -s "$(command -v run-scir)" /opt/qmr/target/release/run-scir \
    && /opt/qmr/target/release/run-scir --help >/dev/null 2>&1 || true
COPY --from=rustbuild /src/qmr/target/release/run-scmr /opt/qmr/target/release/run-scmr

# FastLS binary lives at /opt/fastls (its config tomls sit alongside it), but is also
# symlinked into /opt/qmr/target/release so `--qmr-dir /opt/qmr` reaches it like the
# others. By default GUOQ passes no -c, so FastLS uses its built-in defaults (Compact
# layout + default SA params). The config tomls are baked in for runs that want to
# override that: --qmr-fastls-config /opt/fastls/fast_config.toml for a faster
# near-greedy pass (note both tomls set layout = SquareSparse, which would override the
# Compact default).
COPY --from=rustbuild /src/fastls/target/release/fastls /opt/fastls/target/release/fastls
COPY --from=fastls example_config.toml fast_config.toml /opt/fastls/
RUN ln -s /opt/fastls/target/release/fastls /opt/qmr/target/release/fastls \
    && test -x /opt/qmr/target/release/run-scmr \
    && test -x /opt/qmr/target/release/fastls

# --- Synthetiq: resynthesis backend for GUOQ (`-resynth SYNTHETIQ`) ----------
# GUOQ's resynth.py expects Synthetiq at <resynth.py dir>/lib/synthetiq and runs
# `bin/main` with that directory as CWD, so build it at /home/lib/synthetiq. The
# GUOQ image (FROM this one) places resynth.py at /home and inherits this layer, so
# its LIB_DIR resolves to /home/lib. `main` is C++17 + OpenMP, compiled with the
# build-essential toolchain (g++/make) and git inherited from the amxu/scmr base;
# the gate sets under data/ ship with the repo and are read at run time.
RUN git clone --depth 1 https://github.com/eth-sri/synthetiq.git /home/lib/synthetiq
WORKDIR /home/lib/synthetiq
RUN make clean && make all && test -x bin/main

# --- Python deps for the resynth server -------------------------------------
# resynth.py (the localhost:8080 server GUOQ POSTs to) imports bqskit at load time,
# even for Synthetiq-only runs, so bqskit must be importable. Install it in an
# ISOLATED venv, NOT the base /app/.venv: that venv pins qiskit 2.5 on py3.13 for
# runner.py/SCMR, while BQSKit's qiskit bridge lags the latest qiskit -- co-installing
# would risk a qiskit downgrade that breaks the SCMR tooling. Synthetiq itself needs
# no Python (bin/main is native; its requirements.txt is only for the paper's
# post-processing notebooks). Launch the server with this interpreter, e.g.:
#   /opt/resynth-venv/bin/python /home/resynth.py                       # Synthetiq
#   /opt/resynth-venv/bin/python /home/resynth.py --bqskit --bqskit_auto_workers
#
# UV_PYTHON_INSTALL_DIR puts the managed CPython 3.11 in /opt, not root's $HOME:
# CHTC runs the image as a non-root user (via Apptainer), and a venv pointing at an
# interpreter under /root (mode 700) fails to exec with "Permission denied". The
# final chmod makes the interpreter + venv traversable/readable by that user.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.11 \
    && uv venv --python 3.11 /opt/resynth-venv \
    && uv pip install --python /opt/resynth-venv/bin/python bqskit qiskit \
    && chmod -R a+rX /opt/uv-python /opt/resynth-venv

WORKDIR /home
CMD ["bash", "-l"]
