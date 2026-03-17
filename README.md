[![Build and Open wisq PR with new JAR](https://github.com/qqq-wisc/guoq/actions/workflows/deploy-to-wisq.yml/badge.svg)](https://github.com/qqq-wisc/guoq/actions/workflows/deploy-to-wisq.yml)

> [!NOTE]  
> We recommend using GUOQ via [wisq](https://github.com/qqq-wisc/wisq), which provides a more seamless experience. Some notable features include transpiling the input circuit to the target gate set (GUOQ expects the input to be in the target gate set) and an integrated fault-tolerant mapping/routing phase. The resynthesis server startup/shutdown is also handled for you in wisq. This repository should be used for extending or modifying GUOQ and QUESO or synthesizing rewrite rules via QUESO.

# GUOQ (and QUESO)

GUOQ [[1]](https://arxiv.org/abs/2411.04104) is a quantum-circuit optimizer built on top of [QUESO [2]](https://github.com/qqq-wisc/queso). 
It unifies rewrite rules and unitary synthesis, and uses a radically fast and simple random algorithm to leverage the complementary characteristics of both optimizations.
This branch represents the latest version of GUOQ and QUESO, and should be used for best results. QUESO can be run by instantiating GUOQ with specific parameters, [as described below](#optimizing-a-circuit-using-queso).
To reproduce the experimental conditions of the paper, see the [Zenodo artifact](https://doi.org/10.5281/zenodo.14055562), which contains a Docker image with everything set up.

## Dependencies
- Java 21
- Maven 3
- Python >= 3.8 and dependencies in `requirements.txt`
  - `pip install -r requirements.txt`
- gcc and make (for [Synthetiq](https://github.com/eth-sri/synthetiq))

## Installation

1. Clone this repo *including Synthetiq*
```bash
git clone https://github.com/qqq-wisc/guoq.git --recurse-submodules
```
If you forget the `--recurse-submodules` option the first time around, you can fetch the submodule later with
```bash
git submodule update --init --recursive
```

2. Build GUOQ
```bash
cd guoq
mvn package -Dmaven.test.skip
```

The JAR will be in `guoq/target`.

3. Build Synthetiq (optional &mdash; only needed if you'd like to use Synthetiq as a resynthesis algorithm for Clifford + T)

Follow the instructions in `lib/synthetiq/README.md`. Ignore the Python instructions.

## Usage
### GUOQ
### Optimize a Circuit Using GUOQ
#### General workflow
1. Start the local resynthesis Python server: `python resynth.py [options]`. Use `--bqskit --bqskit_auto_workers` if you intend to use BQSKit for resynthesis.
2. Invoke GUOQ with a timeout: `timeout 3600 java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer [options] <circuit_file_to_optimize>`. The best solution found so far will continuously be output to the following file: `<output_dir>/latest_sol_<job_info>_<input file name>` where the parameters `output_dir` is the current working directory and `job_info` is empty by default. This is why the timeout mechanism is external. This step can be repeated as much as desired with the same resynthesis server. 
3. When finished, shut down the resynthesis server by sending a keyboard interrupt (ctrl + c).

#### Example commands

Optimize the *fidelity* of a circuit in the IBM-EAGLE gate set (assuming 1q error rate of 0.0003 and 2q error rate of 0.0115).
This uses BQSKit by default:
```bash
timeout 3600 java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer -g IBMN -opt FIDELITY --error-1q 0.0003 --error-2q 0.0115 benchmarks/ibmnew/tof_3.qasm
```

Optimize the *two-qubit gate count* of a circuit in the Nam gate set:
```bash
timeout 3600 java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer -g NAM -opt TWO_Q benchmarks/nam_rz/tof_3.qasm
```

Optimize the *T and CX count* of a circuit in the Clifford + T gate set (assuming a T gate is as costly as 50 CX gates, which is tunable via `--fidelity`). This uses Synthetiq by default:
```bash
timeout 3600 java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer -g CLIFFORDT -opt FT benchmarks/nam_t_tdg/tof_3.qasm
```

Optimize the *T and CX count* of a circuit in the Clifford + T gate set *with rewrite rules only* (resynthesis server not needed):
```bash
timeout 3600 java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer -g CLIFFORDT -opt FT -resynth NONE benchmarks/nam_t_tdg/tof_3.qasm
```

Optimize the *T and CX count* of a circuit in the Clifford + T gate set *with resynthesis only*:
```bash
timeout 3600 java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer -g CLIFFORDT -opt FT -r rules/empty.txt -sr rules/empty.txt benchmarks/nam_t_tdg/tof_3.qasm
```

#### Other Options
GUOQ has many knobs to play with. To see options available:
```bash
java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer -h
```

### Supported Gate Sets
If the gate set you are looking for is not in this list, read [this section](#add-support-for-a-new-gate-set) to see how to add a new gate set. 
Additionally, feel free to open an issue requesting a new gate set. Pull requests adding new gate sets are welcome!

- IBMQ20 (IBMO): `u1, u2, u3, cx`
- IBM-EAGLE (IBMN): `rz, sx, x, cx`
- Ion trap: `rx, ry, rz, rxx`
- Nam [[3]](https://www.nature.com/articles/s41534-018-0072-4): `h, rz, x, cx`
- Clifford + T: `t, tdg, s, sdg, x, h, cx`
- Rigetti: `rx(pi/2), rx(-pi/2), rx(pi), rz, cz` (Currently not supported by resynthesis)

#### Add Support for a New Gate Set
1. [Extend QUESO](#defining-a-new-gate-set) and [synthesize rules](#synthesizing-rules) for the new gate set.
2. Modify `resynth.py` to have resynthesis return the result in desired gate set. You can see examples of this by searching for `target_gateset`.

### Supported Resynthesis Algorithms
- [BQSKit](https://github.com/BQSKit/bqskit)
- [Synthetiq](https://github.com/eth-sri/synthetiq)

#### Add Support for a New Resynthesis Algorithm
Extending GUOQ with a new resynthesis algorithm is straightforward. The API in `resynth.py` needs to be extended with a new endpoint and a 
new class that hits the endpoint from Java needs to be added (e.g. `src/main/java/qoptimizer/Bqskit.java`). Feel free to open an issue requesting a new resynthesis algorithm. Pull requests adding new resynthesis algorithm are welcome!

### QUESO
### Optimizing a Circuit Using QUESO
QUESO does not use resynthesis, so the server steps do not need to be run here. It does support using resynthesis though.

#### Example command
Optimize the *total gate count* of a circuit in the IBM-EAGLE gate set:
```bash
timeout 3600 java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Optimizer -g IBMN -opt TOTAL -search BEAM -temp 0 -q 8000 -resynth NONE benchmarks/ibmnew/tof_3.qasm
```

In general, use these flags for QUESO: `-opt TOTAL -search BEAM -temp 0 -q 8000 -resynth NONE`

We recommend using the following rule sizes, which we have pre-generated rule files for in the [rules](rules) directory. If you do not specify rule files, these will be used by default:
|           | Not Symbolic | Symbolic |
|-----------|--------------|----------|
| Nam       | 6            | 3        |
| IBM-EAGLE | 6            | 3        |
| IBMQ20    | 4            | 3        |
| Rigetti   | 5            | 3        |
| Ion trap  | 3            | 3        |
| Clifford + T  | 6            | 3        |

### Defining a New Gate Set
1. Derive path sum semantics for the relevant gates
2. Implement path sum semantics in a method in `PathSum.java` for each gate
3. Add each new gate to `applyGate(...)` in `Synthesizer.java` and specify how angles should be enumerated if this is a parameterized gate
4. Add the gate set at the bottom of `Synthesizer.java` with the desired angles

### Synthesizing Rules

To see options available:
```bash
java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Synthesizer -h
```

Example commands:
```bash
java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Synthesizer -g nam -q 3 -s 3
java -cp target/GUOQ-1.0-jar-with-dependencies.jar qoptimizer.Synthesizer -g nam -q 3 -s 6
```

## References
[1] Amanda Xu, Abtin Molavi, Swamit Tannu, Aws Albarghouthi. Optimizing Quantum Circuits, Fast and Slow. Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS '25). https://doi.org/10.1145/3669940.3707240

[2] Amanda Xu, Abtin Molavi, Lauren Pick, Swamit Tannu, Aws Albarghouthi. Synthesizing quantum-circuit optimizers. Proceedings of the ACM on Programming Languages. Volume 7, PLDI, 2023. https://doi.org/10.1145/3591254

[3] Yunseong Nam, Neil J Ross, Yuan Su, Andrew M Childs, and Dmitri Maslov. 2018. Automated optimization of large quantum circuits with continuous parameters. npj Quantum Information 4, 1 (2018), 1-12.
