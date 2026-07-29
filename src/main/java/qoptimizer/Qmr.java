package qoptimizer;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import qoptimizer.config.Params;
import qoptimizer.config.QmrBackend;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper around the external QMR mapping-and-routing solver (the {@code run-scir} binary of
 * pub-qmr-compiler-generator), used to score a circuit by the depth of its routed solution.
 *
 * <p>Each call writes the circuit to a temp QASM file, spawns the solver, and reads the reported
 * cost back out of the solver's JSON output. There is no caching: one call == one full solve.
 */
public class Qmr {

    /**
     * Mean routed depth over several independent solves. The solver is not deterministic, so a
     * single solve is a noisy estimate; averaging reduces noise.
     */
    public static int depth(String circuit) {
        long total = 0;
        for (int i = 0; i < Params.QMR_TRIALS; i++) {
            total += solveOnce(circuit, Params.QMR_SEED);
        }
        return (int) Math.round((double) total / Params.QMR_TRIALS);
    }

    private static int solveOnce(String circuit, int seed) {
        switch (Params.QMR_BACKEND) {
            case SCIR:
                return solveScir(circuit, seed);
            case SCMR:
                return solveScmr(circuit);
            case FASTLS:
                return solveFastls(circuit);
            default:
                throw new RuntimeException("Unsupported QMR backend: " + Params.QMR_BACKEND);
        }
    }

    /**
     * run-scir logs progress to stdout on the hot path, so its JSON result is directed to a file via
     * --output rather than being parsed off the stream. Under --depth the top-level "cost" field is
     * the weighted physical time of the parallel-compressed schedule.
     */
    private static int solveScir(String circuit, int seed) {
        Path qasm = null;
        Path json = null;
        try {
            qasm = Files.createTempFile("qmr", ".qasm");
            json = Files.createTempFile("qmr", ".json");
            Files.write(qasm, circuit.getBytes(StandardCharsets.UTF_8));

            List<String> commands = new ArrayList<>();
            commands.add(Params.QMR_DIR + "/target/release/run-scir");
            commands.add(qasm.toString());
            commands.add(Params.QMR_ARCH);
            commands.add("chunked");
            commands.add("--chunks");
            commands.add(String.valueOf(Params.QMR_CHUNKS));
            commands.add("--reconcile");
            commands.add(Params.QMR_RECONCILE);
            commands.add("--depth");
            commands.add("--seed");
            commands.add(String.valueOf(seed));
            commands.add("--output");
            commands.add(json.toString());

            ProcessBuilder pb = new ProcessBuilder(commands);
            pb.redirectErrorStream(true);
            Process pr = pb.start();

            // Progress logging is O(steps) and will fill the pipe buffer and deadlock the solver if
            // left unread, so drain it even though it is discarded (the result is in the json file).
            drain(pr.getInputStream());

            int exit = pr.waitFor();
            if (exit != 0) {
                throw new RuntimeException("run-scir exited with status " + exit);
            }

            try (FileReader reader = new FileReader(json.toFile())) {
                JsonObject obj = new Gson().fromJson(reader, JsonObject.class);
                if (obj == null || !obj.has("cost")) {
                    throw new RuntimeException("run-scir produced no cost field");
                }
                return (int) Math.round(obj.get("cost").getAsDouble());
            }
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("QMR routing solve failed", e);
        } finally {
            deleteQuietly(qasm);
            deleteQuietly(json);
        }
    }

    /**
     * run-scmr takes (circuit, arch-layout-name, --mode), auto-sizes the named surface-code layout
     * to the circuit (like run-scir), and writes its result as JSON to stdout: exactly one object for
     * --onepass/--parallel, or a newline-delimited stream of improving solutions for
     * --joint-optimize-par. The top-level "cost" field is the parallel routing depth (step count); we
     * take the minimum over all objects emitted, which is the single value for the one-shot modes and
     * the best-found value for --joint-optimize-par.
     */
    private static int solveScmr(String circuit) {
        Path qasm = null;
        try {
            qasm = Files.createTempFile("qmr", ".qasm");
            Files.write(qasm, circuit.getBytes(StandardCharsets.UTF_8));

            List<String> commands = new ArrayList<>();
            commands.add(Params.QMR_DIR + "/target/release/run-scmr");
            commands.add(qasm.toString());
            commands.add(Params.QMR_ARCH);
            commands.add("--" + Params.QMR_SCMR_MODE);

            ProcessBuilder pb = new ProcessBuilder(commands);
            pb.redirectErrorStream(true);
            Process pr = pb.start();

            // The depth is on stdout, so capture it rather than discarding. Progress/diagnostic lines
            // do not start with '{' and are skipped by the JSON parse below.
            Integer best = null;
            try (BufferedReader br = new BufferedReader(new InputStreamReader(pr.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    Integer cost = parseCost(line);
                    if (cost != null && (best == null || cost < best)) {
                        best = cost;
                    }
                }
            }

            int exit = pr.waitFor();
            if (exit != 0) {
                throw new RuntimeException("scmr exited with status " + exit);
            }
            if (best == null) {
                throw new RuntimeException("scmr produced no cost field");
            }
            return best;
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("QMR routing solve failed", e);
        } finally {
            deleteQuietly(qasm);
        }
    }

    /**
     * FastLS auto-sizes its architecture to the circuit and writes a JSON result (via -o) whose
     * "steps" array is the list of routing layers. Its routed depth is the number of layers, i.e.
     * steps.length. (Its stdout "DEPTH:" line is the *input* circuit depth, not the routing result,
     * so we deliberately read the JSON instead.)
     */
    private static int solveFastls(String circuit) {
        Path qasm = null;
        Path json = null;
        try {
            qasm = Files.createTempFile("qmr", ".qasm");
            json = Files.createTempFile("qmr", ".json");
            Files.write(qasm, circuit.getBytes(StandardCharsets.UTF_8));

            List<String> commands = new ArrayList<>();
            commands.add(Params.QMR_DIR + "/target/release/fastls");
            if (Params.QMR_FASTLS_CONFIG != null) {
                commands.add("-c");
                commands.add(Params.QMR_FASTLS_CONFIG);
            }
            commands.add("-s"); // silence progress lines; the result goes to the -o json file
            commands.add("-o");
            commands.add(json.toString());
            commands.add(qasm.toString()); // qasm is the positional arg

            ProcessBuilder pb = new ProcessBuilder(commands);
            pb.redirectErrorStream(true);
            Process pr = pb.start();

            // Result is in the json file; drain stdout so the pipe buffer can't deadlock the solver.
            drain(pr.getInputStream());

            int exit = pr.waitFor();
            if (exit != 0) {
                throw new RuntimeException("fastls exited with status " + exit);
            }

            try (FileReader reader = new FileReader(json.toFile())) {
                JsonObject obj = new Gson().fromJson(reader, JsonObject.class);
                if (obj == null || !obj.has("steps")) {
                    throw new RuntimeException("fastls produced no steps array");
                }
                return obj.getAsJsonArray("steps").size();
            }
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("QMR routing solve failed", e);
        } finally {
            deleteQuietly(qasm);
            deleteQuietly(json);
        }
    }

    /**
     * Parses the "cost" field out of one line of scmr stdout, or null if the line is not a JSON
     * object carrying a cost (progress text, blank lines, merged stderr).
     */
    private static Integer parseCost(String line) {
        String trimmed = line.trim();
        if (!trimmed.startsWith("{")) {
            return null;
        }
        try {
            JsonObject obj = new Gson().fromJson(trimmed, JsonObject.class);
            if (obj == null || !obj.has("cost")) {
                return null;
            }
            return (int) Math.round(obj.get("cost").getAsDouble());
        } catch (RuntimeException e) {
            return null;
        }
    }

    private static void drain(InputStream stream) throws IOException {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            while (br.readLine() != null) {
                // discarded
            }
        }
    }

    private static void deleteQuietly(Path p) {
        if (p != null) {
            new File(p.toString()).delete();
        }
    }
}
