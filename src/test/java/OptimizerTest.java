import org.junit.jupiter.api.Test;
import qoptimizer.Optimizer;
import qoptimizer.parser.CircuitParser;

import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Comparator;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OptimizerTest {

    private Optimizer applier = new Optimizer(new Random(), 5);
    private Random rand = new Random();

    @Test
    public void testRule1() {
        try {
            // FIXME: Make this test work!!
            String circuit = "h q1; h q2; h q2; x q2;";
            String find = "h q0; h q0;";
            String replace = "";
            var circuitDag = CircuitParser.qasmToDag(circuit);
            // circuitDag = applier.applyRule(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
            // circuitDag = applier.applyRuleParallel(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
            circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
            assertEquals("h q1;\nx q2;\n", CircuitParser.dagToQasm(circuitDag));
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void testRule2() {
        String circuit = "x q0; x q1; cx q0,q1; cx q2,q0; cx q2,q1;";
        String find = "cx q0,q1; cx q2,q0; cx q2,q1;";
        String replace = "cx q2,q0; cx q0,q1;";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        // circuitDag = applier.applyRule(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        // FIXME: Fix this later!
        circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("x q0;\nx q1;\ncx q2,q0;\ncx q0,q1;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule3() {
        String circuit = "t q2; cx q2,q1; cx q2,q0; cx q3,q1;";
        String find = "cx q2,q1; cx q2,q0;";
        String replace = "cx q2,q0; cx q2,q1;";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("t q2;\ncx q2,q0;\ncx q2,q1;\ncx q3,q1;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule4() {
        // testing not matched, not real rule
        String circuit = "x q0; x q1; cx q0,q1; cx q2,q0; cx q2,q1;";
        String find = "x q3; x q1; cx q3,q1; cx q2,q4; cx q2,q1;";
        String replace = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("x q0;\nx q1;\ncx q0,q1;\ncx q2,q0;\ncx q2,q1;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule5() {
        // testing not matched
        String circuit = "t q4; cx q2,q4; cx q2,q6; tdg q4; cx q3,q5; cx q3,q4;";
        String find = "t q1; cx q0,q1; tdg q1; cx q0,q1;";
        String replace = "cx q0,q1; tdg q1; cx q0,q1; t q1;";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("t q4;\ncx q3,q5;\ncx q2,q4;\ncx q2,q6;\ntdg q4;\ncx q3,q4;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule6() {
        // testing not matched, not real rule
        String circuit = "s q2; cx q1,q2; cx q2,q3; tdg q3; cx q1,q3;";
        String find = "s q0; cx q2,q0; cx q2,q1;";
        String replace = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("s q2;\ncx q1,q2;\ncx q2,q3;\ntdg q3;\ncx q1,q3;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule7() {
        // testing matched, not real rule
        String circuit = "x q1; x q0; cx q0,q1; cx q1,q0; x q1; x q0;";
        String find = "cx q0,q1; cx q1,q0;";
        String replace = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        // FIXME: Remove parallel later

        // circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        circuitDag = applier.applyRuleParallelNewTimingFull(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        // circuitDag = applier.applyRule(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("x q1;\nx q0;\nx q1;\nx q0;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule8() {
        // testing not matched, not real rule
        String circuit = "x q1; x q0; cx q0,q1; h q1; cx q1,q0; x q1; x q0;";
        String find = "cx q0,q1; cx q1,q0;";
        String replace = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applyRule(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("x q1;\nx q0;\ncx q0,q1;\nh q1;\ncx q1,q0;\nx q1;\nx q0;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule9() {
        String circuit = "h q0; cx q2,q0; h q0; cx q2,q3; cx q3,q1; cx q0,q1;";
        String find = "h q0; cx q2,q0; h q0; cx q0,q1;";
        String replace = "cx q0,q1; h q0; cx q2,q0; h q0;";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        // circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        // circuitDag = applier.applyRuleParallelNewTimingFull(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        circuitDag = applier.applyRule(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("h q0;\ncx q2,q0;\nh q0;\ncx q2,q3;\ncx q3,q1;\ncx q0,q1;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule10() {
        String circuit = "t q13; cx q13,q16; tdg q16; cx q14,q16; t q16; h q16; ccz q9,q16,q15; t q14; h q9; h q16;";
        String find = "t q13; cx q13,q16; tdg q16; cx q14,q16; t q16; h q16; ccz q9,q16,q15; t q14; h q9; h q16;";
        String replace = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applyRule(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule11() {
        // test because second match shouldn't be found (would be not valid after first match applied)
        String circuit = "cx q0,q3; cx q3,q4; t q4; tdg q3; cx q0,q4; cx q2,q3; cx q0,q3; cx q2,q4; cx q2,q1; tdg q4;";
        String find = "cx q2,q0; cx q1,q0";
        String replace = "cx q1,q0; cx q2,q0;";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("cx q0,q3;\ncx q3,q4;\nt q4;\ntdg q3;\ncx q2,q3;\ncx q2,q4;\ncx q0,q4;\ncx q2,q1;\ntdg q4;\ncx q0,q3;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testRule12() {
        // test because second match shouldn't be found (would be not valid after first match applied)
        String circuit = "rz(pi/2) q0; rz(-pi/2) q0;";
        String find = "rz(theta1) q0; rz(theta2) q0;";
        String replace = "rz(theta1+theta2) q0;";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applyRuleParallelNew(circuitDag, replace, CircuitParser.qasmToDag(find), false, rand);
        assertEquals("", CircuitParser.dagToQasm(circuitDag));
    }

    // Helper method to load circuit from file
    private String loadCircuitFromFile(String filepath) {
        try {
            return Files.readString(Path.of(filepath));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load circuit file: " + filepath, e);
        }
    }

    @Test
    public void testParallelPerfBench() {
        // Test that is more of a benchmark, for parallel vs not
        String circuit = loadCircuitFromFile("latest_sol__qft_N100_basis_rz_rx_ry_cx.qasm");
        // String find = "rz(theta1) q0;";
        String find = "cx q0, q1; cx q1, q0; cx q0, q1;";
        String replace = "cx q1, q0; cx q0, q1; cx q1, q0;";
        int warmupIterations = 10;
        int benchmarkIterations = 200;

        // used for warm-up and nothing else
        int[] threadPoolSizes = {16, 32, 64, 128, 256, 512};
    
        // Warmup sequential
        System.out.println("Warming up sequential...");
        for (int i = 0; i < warmupIterations; i++) {
            var circuitDag = CircuitParser.qasmToDag(circuit);
            var findDag = CircuitParser.qasmToDag(find);
            applier.applyRule(circuitDag, replace, findDag, false, rand);
            System.out.printf("  Warmup iteration %d complete%n", i + 1);
        }
        
        // Warmup parallel for each thread pool size
        System.out.println("\nWarming up parallel (all thread pool sizes)...");
        for (int threadPoolSize : threadPoolSizes) {
            System.out.printf("  Warming up threadPoolSize=%d...%n", threadPoolSize);
            for (int i = 0; i < warmupIterations; i++) {
                var circuitDag = CircuitParser.qasmToDag(circuit);
                var findDag = CircuitParser.qasmToDag(find);
                applier.applyRuleParallelNew( // no thread ver
                    circuitDag, replace, findDag, false, rand
                );
            }
        }
        
        System.out.println("\nBenchmarking applyRule (sequential)...");
        long sequentialTotal = 0;
        for (int i = 0; i < benchmarkIterations; i++) {
            var circuitDag = CircuitParser.qasmToDag(circuit);
            var findDag = CircuitParser.qasmToDag(find);
            
            long startTime = System.nanoTime();
            circuitDag = applier.applyRule(circuitDag, replace, findDag, false, rand);
            long endTime = System.nanoTime();
            
            long duration = endTime - startTime;
            sequentialTotal += duration;
            System.out.printf("  Iteration %d: %.3f ms%n", i + 1, duration / 1_000_000.0);
        }
        
        System.out.println("\nBenchmarking applyRuleParallelNew (parallel)...");
        long parallelTotal = 0;
        for (int i = 0; i < benchmarkIterations; i++) {
            var circuitDag = CircuitParser.qasmToDag(circuit);
            var findDag = CircuitParser.qasmToDag(find);
            
            long startTime = System.nanoTime();
            // circuitDag = applier.applyRuleParallelNewTiming(circuitDag, replace, findDag, false, rand);
            // circuitDag = applier.applyRuleParallelNewTimingFull(circuitDag, replace, findDag, false, rand);
            circuitDag = applier.applyRuleParallelNew(circuitDag, replace, findDag, false, rand);
            long endTime = System.nanoTime();
            
            long duration = endTime - startTime;
            parallelTotal += duration;
            System.out.printf("  Iteration %d: %.3f ms%n", i + 1, duration / 1_000_000.0);
        }
        
        double sequentialAvg = sequentialTotal / (double) benchmarkIterations / 1_000_000.0;
        double parallelAvg = parallelTotal / (double) benchmarkIterations / 1_000_000.0;
        double speedup = sequentialAvg / parallelAvg;
        
        System.out.println("\n    BENCHMARK RESULTS");
        System.out.printf("Sequential (applyRule) average: %.3f ms%n", sequentialAvg);
        System.out.printf("Parallel (applyRuleParallelNew) average: %.3f ms%n", parallelAvg);
        System.out.printf("Speedup: %.2fx%n", speedup);

        assertTrue(parallelAvg < sequentialAvg); // Parallel should be faster
    }

    @Test
    public void testThreadPoolSizeScaling() {
        String circuit = loadCircuitFromFile("latest_sol__qft_N100_basis_rz_rx_ry_cx.qasm");
        // String find = "rz(theta1) q0;";
        // String replace = "";
        String find = "cx q0, q1; cx q1, q0; cx q0, q1;";
        String replace = "cx q1, q0; cx q0, q1; cx q1, q0;";
        
        int[] threadPoolSizes = {16, 32, 64, 128, 256, 512};
        int warmupIterations = 3;
        int sequentialIterations = 20;
        int parallelIterations = 20;


        // Warmup sequential
        System.out.println("Warming up sequential...");
        for (int i = 0; i < warmupIterations; i++) {
            var circuitDag = CircuitParser.qasmToDag(circuit);
            var findDag = CircuitParser.qasmToDag(find);
            applier.applyRule(circuitDag, replace, findDag, false, rand);
            System.out.printf("  Warmup iteration %d complete%n", i + 1);
        }
        
        // Warmup parallel for each thread pool size
        System.out.println("\nWarming up parallel (all thread pool sizes)...");
        for (int threadPoolSize : threadPoolSizes) {
            System.out.printf("  Warming up threadPoolSize=%d...%n", threadPoolSize);
            for (int i = 0; i < warmupIterations; i++) {
                var circuitDag = CircuitParser.qasmToDag(circuit);
                var findDag = CircuitParser.qasmToDag(find);
                applier.applyRuleParallelNewTimingThread(
                    threadPoolSize, circuitDag, replace, findDag, false, rand
                );
            }
        }
        
        System.out.println("=== THREAD POOL SIZE SCALING TEST ===\n");
        System.out.println("Circuit: latest_sol__qft_N100_basis_rz_rx_ry_cx.qasm");
        System.out.println("Pattern: rz(theta1) q0;");
        System.out.println("Sequential iterations: " + sequentialIterations);
        System.out.println("Parallel iterations per pool size: " + parallelIterations);
        System.out.println("Available processors: " + Runtime.getRuntime().availableProcessors());
        System.out.println("\n" + "=".repeat(80) + "\n");
        
        // ===== SEQUENTIAL BASELINE =====
        System.out.println("BENCHMARKING SEQUENTIAL (applyRule)...\n");
        List<Long> sequentialTimes = new ArrayList<>();
        
        for (int i = 0; i < sequentialIterations; i++) {
            var circuitDag = CircuitParser.qasmToDag(circuit);
            var findDag = CircuitParser.qasmToDag(find);
            
            long startTime = System.nanoTime();
            circuitDag = applier.applyRule(circuitDag, replace, findDag, false, rand);
            long endTime = System.nanoTime();
            
            long duration = endTime - startTime;
            sequentialTimes.add(duration);
            System.out.printf("  Iteration %2d: %.3f ms%n", i + 1, duration / 1_000_000.0);
        }
        
        double sequentialAvg = sequentialTimes.stream().mapToLong(Long::longValue).average().orElse(0) / 1_000_000.0;
        double sequentialMedian = calculateMedian(sequentialTimes) / 1_000_000.0;
        double sequentialMin = sequentialTimes.stream().mapToLong(Long::longValue).min().orElse(0) / 1_000_000.0;
        double sequentialMax = sequentialTimes.stream().mapToLong(Long::longValue).max().orElse(0) / 1_000_000.0;
        
        System.out.println("\nSequential Statistics:");
        System.out.printf("  Average: %.3f ms%n", sequentialAvg);
        System.out.printf("  Median:  %.3f ms%n", sequentialMedian);
        System.out.printf("  Min:     %.3f ms%n", sequentialMin);
        System.out.printf("  Max:     %.3f ms%n", sequentialMax);
        System.out.println("\n" + "=".repeat(80) + "\n");
        
        // ===== PARALLEL WITH DIFFERENT THREAD POOL SIZES =====
        Map<Integer, ParallelResult> results = new LinkedHashMap<>();
        
        for (int threadPoolSize : threadPoolSizes) {
            System.out.printf("BENCHMARKING PARALLEL (threadPoolSize=%d)...\n\n", threadPoolSize);
            List<Long> parallelTimes = new ArrayList<>();
            
            for (int i = 0; i < parallelIterations; i++) {
                var circuitDag = CircuitParser.qasmToDag(circuit);
                var findDag = CircuitParser.qasmToDag(find);
                
                long startTime = System.nanoTime();
                circuitDag = applier.applyRuleParallelNewTimingThread(
                    threadPoolSize, circuitDag, replace, findDag, false, rand
                );
                long endTime = System.nanoTime();
                
                long duration = endTime - startTime;
                parallelTimes.add(duration);
                System.out.printf("  Iteration %2d: %.3f ms%n", i + 1, duration / 1_000_000.0);
            }
            
            double parallelAvg = parallelTimes.stream().mapToLong(Long::longValue).average().orElse(0) / 1_000_000.0;
            double parallelMedian = calculateMedian(parallelTimes) / 1_000_000.0;
            double parallelMin = parallelTimes.stream().mapToLong(Long::longValue).min().orElse(0) / 1_000_000.0;
            double parallelMax = parallelTimes.stream().mapToLong(Long::longValue).max().orElse(0) / 1_000_000.0;
            
            ParallelResult result = new ParallelResult();
            result.threadPoolSize = threadPoolSize;
            result.avgTime = parallelAvg;
            result.medianTime = parallelMedian;
            result.minTime = parallelMin;
            result.maxTime = parallelMax;
            result.speedupAvg = sequentialAvg / parallelAvg;
            result.speedupMedian = sequentialMedian / parallelMedian;
            
            results.put(threadPoolSize, result);
            
            System.out.printf("\nParallel Statistics (threadPoolSize=%d):%n", threadPoolSize);
            System.out.printf("  Average: %.3f ms%n", parallelAvg);
            System.out.printf("  Median:  %.3f ms%n", parallelMedian);
            System.out.printf("  Min:     %.3f ms%n", parallelMin);
            System.out.printf("  Max:     %.3f ms%n", parallelMax);
            System.out.printf("  Speedup (avg):    %.3fx%n", result.speedupAvg);
            System.out.printf("  Speedup (median): %.3fx%n", result.speedupMedian);
            System.out.println("\n" + "=".repeat(80) + "\n");
        }
        
        // ===== FINAL SUMMARY =====
        System.out.println("\n" + "=".repeat(80));
        System.out.println("FINAL SUMMARY");
        System.out.println("=".repeat(80) + "\n");
        
        System.out.printf("Sequential Baseline: %.3f ms (median: %.3f ms)%n%n", sequentialAvg, sequentialMedian);
        
        System.out.println("Thread Pool Size Comparison:");
        System.out.println("-".repeat(80));
        System.out.printf("%-15s | %-12s | %-12s | %-12s | %-12s%n", 
                         "Pool Size", "Avg Time", "Median Time", "Speedup (avg)", "Speedup (med)");
        System.out.println("-".repeat(80));
        
        for (ParallelResult result : results.values()) {
            System.out.printf("%-15d | %9.3f ms | %9.3f ms | %10.3fx | %10.3fx%n",
                             result.threadPoolSize,
                             result.avgTime,
                             result.medianTime,
                             result.speedupAvg,
                             result.speedupMedian);
        }
        System.out.println("-".repeat(80));
        
        // Find best configuration
        ParallelResult best = results.values().stream()
            .max(Comparator.comparingDouble(r -> r.speedupMedian))
            .orElse(null);
        
        if (best != null) {
            System.out.printf("\nBest Configuration: Thread Pool Size = %d%n", best.threadPoolSize);
            System.out.printf("  Best speedup: %.3fx (%.3f ms vs %.3f ms sequential)%n", 
                             best.speedupMedian, best.medianTime, sequentialMedian);
            
            if (best.speedupMedian < 1.0) {
                System.out.println("\n⚠ WARNING: Parallel implementation is slower than sequential!");
                System.out.println("  Consider:");
                System.out.println("  - Workload may be too small to benefit from parallelization");
                System.out.println("  - Thread pool overhead dominates execution time");
                System.out.println("  - Check if pattern actually exists in circuit");
            } else {
                System.out.printf("\n✓ Parallel implementation achieved %.1f%% speedup%n", 
                                 (best.speedupMedian - 1) * 100);
            }
        }
        
        // Scaling efficiency analysis
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SCALING EFFICIENCY ANALYSIS");
        System.out.println("=".repeat(80) + "\n");
        
        ParallelResult baseline = results.get(threadPoolSizes[0]);
        System.out.printf("Baseline (threads=%d): %.3f ms%n%n", baseline.threadPoolSize, baseline.medianTime);
        
        System.out.println("Relative Speedup vs Baseline:");
        System.out.println("-".repeat(60));
        System.out.printf("%-15s | %-15s | %-20s%n", "Pool Size", "Time", "Speedup vs Baseline");
        System.out.println("-".repeat(60));
        
        for (ParallelResult result : results.values()) {
            double relativeSpeedup = baseline.medianTime / result.medianTime;
            System.out.printf("%-15d | %12.3f ms | %17.3fx%n",
                             result.threadPoolSize,
                             result.medianTime,
                             relativeSpeedup);
        }
        System.out.println("-".repeat(60));
        
        // Check for diminishing returns
        System.out.println("\nDiminishing Returns Analysis:");
        List<Integer> sizes = new ArrayList<>(results.keySet());
        for (int i = 1; i < sizes.size(); i++) {
            int prevSize = sizes.get(i - 1);
            int currSize = sizes.get(i);
            double prevTime = results.get(prevSize).medianTime;
            double currTime = results.get(currSize).medianTime;
            double improvement = ((prevTime - currTime) / prevTime) * 100;
            double threadsIncrease = ((double)(currSize - prevSize) / prevSize) * 100;
            
            System.out.printf("  %d → %d threads (+%.0f%%): %.1f%% faster%n",
                             prevSize, currSize, threadsIncrease, improvement);
        }
    }

    // Helper class to store results
    private static class ParallelResult {
        int threadPoolSize;
        double avgTime;
        double medianTime;
        double minTime;
        double maxTime;
        double speedupAvg;
        double speedupMedian;
    }

    // Helper method for median calculation
    private double calculateMedian(List<Long> times) {
        List<Long> sorted = new ArrayList<>(times);
        Collections.sort(sorted);
        int size = sorted.size();
        if (size % 2 == 0) {
            return (sorted.get(size/2 - 1) + sorted.get(size/2)) / 2.0;
        } else {
            return sorted.get(size/2);
        }
    }

    @Test
    public void testSymbRule0() {
        String circuit = "rz(-pi/4) q0; cx q1,q0; cx q0,q1; rz(-pi/4) q1;";
        String constraints = "[{[false, false]=[true, false], [true, false]=[false, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[true, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[false, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[true, false], [true, true]=[false, true]}, {[false, false]=[false, false], [true, false]=[false, true], [false, true]=[true, false], [true, true]=[true, true]}]";

        String findBefore = "rz(theta1) q0;";
        String findAfter = "rz(theta2) q1;";
        String replaceBefore = "rz(theta1+theta2) q0;";
        String replaceAfter = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applySymbRule(circuitDag, findBefore, findAfter, replaceBefore, replaceAfter, applier.parseConstraints(constraints), 7, 10, false, rand);
        assertEquals("rz(-1.5707963267948966) q0;\ncx q1,q0;\ncx q0,q1;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testSymbRule1() {
        String circuit = "t q0; cx q1,q0; cx q0,q1; t q1;";
        String constraints = "[{[false, false]=[true, false], [true, false]=[false, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[true, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[false, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[true, false], [true, true]=[false, true]}, {[false, false]=[false, false], [true, false]=[false, true], [false, true]=[true, false], [true, true]=[true, true]}]";

        String findBefore = "t q0;";
        String findAfter = "t q1;";
        String replaceBefore = "s q0;";
        String replaceAfter = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applySymbRule(circuitDag, findBefore, findAfter, replaceBefore, replaceAfter, applier.parseConstraints(constraints), 7, 10, false, rand);
        assertEquals("s q0;\ncx q1,q0;\ncx q0,q1;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testSymbRule2() {
        String circuit = "t q2; cx q2,q1; cx q2,q3; t q1; cx q1,q2; h q3; cx q2,q3; cx q1,q2; t q2;";
        String constraints = "[{[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, true]}]";

        String findBefore = "t q0;";
        String findAfter = "t q0;";
        String replaceBefore = "s q0;";
        String replaceAfter = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applySymbRule(circuitDag, findBefore, findAfter, replaceBefore, replaceAfter, applier.parseConstraints(constraints), 7, 10, false, rand);
        assertEquals("s q2;\ncx q2,q1;\nt q1;\ncx q2,q3;\ncx q1,q2;\nh q3;\ncx q2,q3;\ncx q1,q2;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testSymbRule3() {
        String circuit = "t q4; h q4; cx q2,q4; t q2; h q4; cx q1,q4; t q1;";
        String constraints = "[{[false, false]=[true, false], [true, false]=[false, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[true, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[false, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[true, false], [true, true]=[false, true]}, {[false, false]=[false, false], [true, false]=[false, true], [false, true]=[true, false], [true, true]=[true, true]}]";

        String findBefore = "t q0;";
        String findAfter = "t q1;";
        String replaceBefore = "s q0;";
        String replaceAfter = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applySymbRule(circuitDag, findBefore, findAfter, replaceBefore, replaceAfter, applier.parseConstraints(constraints), 7, 10, false, rand);
        assertEquals("t q4;\nh q4;\ncx q2,q4;\nt q2;\nh q4;\ncx q1,q4;\nt q1;\n", CircuitParser.dagToQasm(circuitDag));
    }

    @Test
    public void testSymbRule4() {
        String circuit = "t q2; s q2; cx q2,q1; cx q2,q3; t q1; cx q1,q2; h q3; cx q2,q3; cx q1,q2; t q2; s q2";
        String constraints = "[{[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, true]}, {[false, false]=[false, false], [true, false]=[true, true], [false, true]=[false, false], [true, true]=[true, true]}]";

        String findBefore = "t q0; s q0;";
        String findAfter = "t q0; s q0;";
        String replaceBefore = "s q0;";
        String replaceAfter = "";
        var circuitDag = CircuitParser.qasmToDag(circuit);
        circuitDag = applier.applySymbRule(circuitDag, findBefore, findAfter, replaceBefore, replaceAfter, applier.parseConstraints(constraints), 7, 10, false, rand);
        assertEquals("s q2;\ncx q2,q1;\nt q1;\ncx q2,q3;\ncx q1,q2;\nh q3;\ncx q2,q3;\ncx q1,q2;\n", CircuitParser.dagToQasm(circuitDag));
    }
}
