package qoptimizer.config;

import org.apache.commons.math3.util.Pair;

import java.util.Map;
import java.util.Random;

public class Params {
    /**
     * max priority queue size
     */
    public static int QUEUE_SIZE = 1;
    /**
     * max number of qubits allowed for symbolic circuit
     */
    public static int MAX_SYMB_QUBITS = 7;
    /**
     * max size (number of gates) to consider for symbolic circuit
     */
    public static int MAX_SYMB_SIZE = 10;
    /**
     * max qubits allowed in rules. -1 for no limit
     */
    public static int MAX_RULE_QUBITS = -1;
    /**
     * remove size-preserving nonsymbolic rules
     */
    public static boolean REMOVE_SIZE_PRESERVING_RULES = false;
    /**
     * use size-preserving symbolic rules
     */
    public static boolean USE_SIZE_PRESERVING_SYMB_RULES = false;
    /**
     * use the reflection of size preserving rule as well. i.e. use both directions of rule
     */
    public static boolean USE_SIZE_PRESERVE_RULE_REFLECTION = false;
    /**
     * use size increasing nonsymb rules
     */
    public static boolean USE_SIZE_INCREASING_RULES = false;
    /**
     * preserve connectivity of input circuit (preserve mapping)
     */
    public static boolean PRESERVE_MAPPING = false;
    /**
     * relative path to directory for output circuit. creates directory if it does not exist
     */
    public static String OUTPUT_DIR = "";
    /**
     * job info to add to output file name
     */
    public static String JOB_INFO = "";
    /**
     * search algorithm to use for applying transformations
     */
    public static SearchStrategy SEARCH_STRATEGY = SearchStrategy.BEAM_MCMC;
    /**
     * optimization objective
     */
    public static OptObj OPTIMIZATION_OBJECTIVE = OptObj.FIDELITY;
    /**
     * Cost of a 2q gate in terms of 1q gates for FIDELITY opt obj. Or weight of T gate vs. 2q gate for FT opt obj.
     */
    public static int FIDELITY_BREAKEVEN = 1;
    /**
     * path to the routing-solver checkout for the selected QMR_BACKEND: the qmr-compiler-generator
     * repo for SCIR/SCMR, or the FastLS repo for FASTLS
     */
    public static String QMR_DIR = null;
    /**
     * which routing solver backend to call. SCIR uses QMR_DIR/target/release/run-scir; SCMR uses
     * the generated solver QMR_DIR/generated-solvers/scmr. see {@link QmrBackend}
     */
    public static QmrBackend QMR_BACKEND = QmrBackend.SCMR;
    /**
     * target architecture the ROUTED_DEPTH opt obj routes against, as a layout name auto-sized to
     * the circuit's qubit count. SCIR: compact, square-sparse, test-compact. SCMR: compact,
     * square_sparse. ignored by FASTLS, which auto-sizes its architecture with no name to select
     */
    public static String QMR_ARCH = null;
    /**
     * solve mode for the SCMR backend: onepass, parallel, or joint-optimize-par. onepass and parallel
     * each emit one solution; joint-optimize-par runs a parallel anytime search and we take its best.
     * ignored by the SCIR backend
     */
    public static String QMR_SCMR_MODE = "joint-optimize-par";
    /**
     * optional path to a FastLS config .toml (tunes its simulated-annealing params). null omits -c
     * and FastLS uses its built-in defaults. only used by the FASTLS backend
     */
    public static String QMR_FASTLS_CONFIG = null;
    /**
     * number of chunks the routing solver splits the circuit into. chunks are solved in parallel,
     * so this trades solution quality for latency. 1 is the quality ceiling and is single threaded
     */
    public static int QMR_CHUNKS = 1;
    /**
     * strategy the routing solver uses to stitch chunk boundaries back together. one of:
     * reversal, reversal-compressed, maps, sabre
     */
    public static String QMR_RECONCILE = "reversal";
    /**
     * number of independent routing solves averaged per cost evaluation. the solver is not
     * deterministic, so this reduces noise at proportional cost in time
     */
    public static int QMR_TRIALS = 5;
    /**
     * base seed passed to the routing solver's --seed. trial i of a cost evaluation uses
     * QMR_SEED + i, so a whole evaluation is reproducible from this value. note the solver is only
     * partially seeded (--seed fixes the initial qubit map, not the internal search), so equal
     * seeds do not guarantee equal depth
     */
    public static int QMR_SEED = 1;
    /**
     *
     */
    public static Double ERROR_1Q = null;
    /**
     *
     */
    public static Double ERROR_2Q = null;
    /**
     * resynthesis algorithm
     */
    public static Resynth RESYNTH_ALG = null;
    /**
     * additional parameters for resynthesis
     */
    public static ResynthArgs RESYNTH_ARGS = new ResynthArgs(3, 100, 8);
    /**
     * maximum number of calls to resynthesis allowed in final circuit. -1 if no limit
     */
    public static int MAX_RESYNTH_ALLOWED = 100;
    /**
     * error threshold for final circuit
     */
    public static double EPSILON = 1e-8;
    /**
     * seed
     */
    public static int SEED = new Random().nextInt();
    /**
     * temperature for simulated annealing or beta for mcmc. for beam search, temperature is for softmax. If 0 then polls priority queue.
     */
    public static double TEMPERATURE = 10.0;
    /**
     * cooling rate for simulated annealing. 0 for mcmc. for beam search, this is the rate to prune rules so 0 for no pruning.
     */
    public static double COOLING_RATE = 0.0;
    /**
     * temperature for pruning rules. 0 samples rules greedily
     */
    public static double PRUNE_TEMPERATURE = 0.0;
    /**
     * iterations to wait before starting to prune rules
     */
    public static int ITERS_BEFORE_PRUNE = -1;
    /**
     * seconds to wait before starting to prune rules
     */
    public static int SECS_BEFORE_PRUNE = -1;
    /**
     * number of transformations to sample per iteration
     */
    public static int NUM_TRANSFORMATIONS_SAMPLE = 1;
    /**
     * weight of resynthesis when sampling transformations randomly
     */
    public static int RESYNTH_WEIGHT = 1;
    public static double RESYNTH_PERCENTAGE = 0.015;
    /**
     * apply rewrite rule only once per iteration if true instead of all disjoint matches
     */
    public static boolean APPLY_ONCE = false;
    /**
     * 0: no logs
     * 1: log progress
     * 2: log progress and config info
     * 3: log progress, config info, and rules applied
     */
    public static int VERBOSITY = 0;
    /**
     * gate set
     */
    public static GateSet GATE_SET = null;
    /**
     * nonsymbolic rule file
     */
    public static String RULE_FILE = null;
    /**
     * symbolic rule file
     */
    public static String SYMB_RULE_FILE = null;

    public static String RULES_DIR = "rules";

    public static final Map<GateSet, Pair<String, String>> GATE_SET_RULES_MAP = Map.of(
            GateSet.NAM, new Pair<>("rules_q3_s6_nam.txt", "rules_q3_s3_nam_symb.txt"),
            GateSet.IBMO, new Pair<>("rules_q3_s4_ibm.txt", "rules_q3_s3_ibm_symb.txt"),
            GateSet.IBMN, new Pair<>("rules_q3_s6_ibmnew.txt", "rules_q3_s3_ibmnew_symb.txt"),
            GateSet.RIGETTI, new Pair<>("rules_q3_s5_rigetti.txt", "rules_q3_s3_rigetti_symb.txt"),
            GateSet.ION, new Pair<>("rules_q3_s3_ion.txt", "rules_q3_s3_ion_symb.txt"),
            GateSet.CLIFFORDT, new Pair<>("rules_q3_s6_cliffordt.txt", "rules_q3_s3_cliffordt_symb.txt")
    );

    public static final Map<GateSet, String> GATE_SET_RESYNTH_MAP = Map.of(
            GateSet.NAM, "nam",
            GateSet.IBMO, "none",
            GateSet.IBMN, "ibm_new",
            GateSet.RIGETTI, "",
            GateSet.ION, "ion",
            GateSet.CLIFFORDT, "none"
    );

    public static void setResynthWeight(int totalRules) {
        if (totalRules == 0) {
            return;
        }
        if (RESYNTH_WEIGHT == 1) {
            RESYNTH_WEIGHT = (int) (RESYNTH_PERCENTAGE * totalRules);
        }
    }
}
