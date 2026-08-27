# GUOQ

A Rust implementation of **GUOQ** (quantum-circuit optimization interleaving rewrite
rules with unitary resynthesis) and **QUESO** (rewrite-rule synthesis). This is the
`rust-port` branch of [qqq-wisc/guoq](https://github.com/qqq-wisc/guoq); the Java
reference implementation lives on `main`.

## Installing

```bash
pip install guoq
```

installs the `guoq` and `queso` binaries as console scripts, plus a small `guoq`
Python package for locating them from other tools (this is how
[wisq](https://github.com/qqq-wisc/wisq) invokes the optimizer):

```python
import guoq
guoq.find_guoq_bin()       # path to the optimizer binary
guoq.find_bqskit_worker()  # path to the bundled BQSKit worker script
```

## Building from source

```bash
cargo build --release            # binaries in target/release/{guoq,queso}
cargo test --workspace           # test suite
pip install maturin && maturin build --release   # the wheel
```

## Running

```bash
guoq -g NAM -opt TOTAL -search BEAM -temp 0 -resynth NONE circuit.qasm
```

The CLI is flag-compatible with the Java reference, including `@argsfile` expansion.
There is no resynthesis server: `guoq` owns its backends' lifecycles, spawning the
Synthetiq binary (`--synthetiq-binary`) or the BQSKit worker (`--bqskit-worker`,
bundled in the wheel) itself when resynthesis is requested.
