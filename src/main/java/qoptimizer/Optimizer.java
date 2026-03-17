package qoptimizer;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.mxgraph.layout.mxCircleLayout;
import com.mxgraph.layout.mxIGraphLayout;
import com.mxgraph.util.mxCellRenderer;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.util.Pair;
import org.jgrapht.GraphTests;
import org.jgrapht.Graphs;
import org.jgrapht.alg.lca.NaiveLCAFinder;
import org.jgrapht.ext.JGraphXAdapter;
import org.jgrapht.graph.DirectedMultigraph;
import qoptimizer.ast.BinOp;
import qoptimizer.ast.Expr;
import qoptimizer.ast.Real;
import qoptimizer.ast.Symbol;
import qoptimizer.ast.UnOp;
import qoptimizer.ast.Var;
import qoptimizer.circuit.Circuit;
import qoptimizer.circuit.CircuitDAG;
import qoptimizer.circuit.Edge;
import qoptimizer.circuit.Node;
import qoptimizer.circuit.OptCircuit;
import qoptimizer.circuit.OptCircuitComparator;
import qoptimizer.circuit.PathSum;
import qoptimizer.circuit.Match;
import qoptimizer.config.GateSet;
import qoptimizer.config.OptObj;
import qoptimizer.config.Params;
import qoptimizer.config.Resynth;
import qoptimizer.config.ResynthArgs;
import qoptimizer.config.SearchStrategy;
import qoptimizer.parser.CircuitParser;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static qoptimizer.parser.NodeVisitor.eval;

public class Optimizer {

    private Verifier verifier;

    public Optimizer(Random rand, int maxQubits) {
        this.verifier = new Verifier(rand, maxQubits);
    }

    private String gateNodeToQasm(Node node) {
        if (node.isCX()) {
            if (node.getId().equals("rxx")) {
                return String.format("%s(%s) %s, %s", node.getId(), node.getAngles().get(0), node.getQubits().get(0), node.getQubits().get(1));
            }
            return String.format("%s %s, %s", node.getId(), node.getQubits().get(0), node.getQubits().get(1));
        } else if (node.isCCZ()) {
            return String.format("%s %s, %s, %s", node.getId(), node.getQubits().get(0), node.getQubits().get(1), node.getQubits().get(2));
        } else if (node.getAngles() != null) {
            if (node.getAngles().size() == 1) {
                return String.format("%s(%s) %s", node.getId(), node.getAngles().get(0), node.getQubits().get(0));
            } else if (node.getAngles().size() == 2) {
                return String.format("%s(%s,%s) %s", node.getId(), node.getAngles().get(0), node.getAngles().get(1), node.getQubits().get(0));
            } else if (node.getAngles().size() == 3) {
                return String.format("%s(%s,%s,%s) %s", node.getId(), node.getAngles().get(0), node.getAngles().get(1), node.getAngles().get(2), node.getQubits().get(0));
            }
            throw new RuntimeException("angles");
        } else {
            return String.format("%s %s", node.getId(), node.getQubits().get(0));
        }
    }

    public void saveGraphImage(String filename, DirectedMultigraph<Node, Edge> dag) throws IOException {
        File imgFile = new File(filename);
        imgFile.createNewFile();
        JGraphXAdapter<Node, Edge> graphAdapter = new JGraphXAdapter<>(dag);
        mxIGraphLayout layout = new mxCircleLayout(graphAdapter);
        layout.execute(graphAdapter.getDefaultParent());

        BufferedImage image = mxCellRenderer.createBufferedImage(graphAdapter, null, 2, Color.WHITE, true, null);
        ImageIO.write(image, "PNG", imgFile);
    }

    private boolean matchOutgoing(DirectedMultigraph<Node, Edge> circuit,
                                  DirectedMultigraph<Node, Edge> pattern,
                                  Node circuitNode,
                                  Node patternNode,
                                  Map<Node, Node> patternToCirc,
                                  Map<Edge, Edge> patternToCircEdges,
                                  Map<String, Expr> angleMap,
                                  List<Node> succsToVisit) {
        for (Edge pattE : pattern.outgoingEdgesOf(patternNode)) {
            if (pattern.getEdgeTarget(pattE).isSinkQubit()) {
                continue;
            }
            boolean foundMatch = false;
            if (circuit.outDegreeOf(circuitNode) != pattern.outDegreeOf(patternNode)) {
                return false;
            }

            for (Edge circE : circuit.outgoingEdgesOf(circuitNode)) {
                if (patternToCirc.containsKey(pattern.getEdgeTarget(pattE))) {
                    if (pattE.sameSourceTargetLabels(circE) && circuit.getEdgeTarget(circE) == patternToCirc.get(pattern.getEdgeTarget(pattE))) {
                        foundMatch = true;
                    }
                } else {
                    if (pattE.sameSourceTargetLabels(circE) && circuit.getEdgeTarget(circE).getId().equals(pattern.getEdgeTarget(pattE).getId())) {
                        if (pattern.getEdgeTarget(pattE).getAngles() != null) {
                            if (matchAngles(circuit.getEdgeTarget(circE), pattern.getEdgeTarget(pattE), angleMap)) {
                                patternToCirc.put(pattern.getEdgeTarget(pattE), circuit.getEdgeTarget(circE));
                                foundMatch = true;
                            }
                        } else {
                            patternToCirc.put(pattern.getEdgeTarget(pattE), circuit.getEdgeTarget(circE));
                            foundMatch = true;
                        }
                    }
                }
                if (foundMatch) {
                    patternToCircEdges.put(pattE, circE);
                    succsToVisit.add(pattern.getEdgeTarget(pattE));
                    break;
                }
            }

            if (!foundMatch) {
                return false;
            }
        }
        return true;
    }

    private List<Node> allCXAncs(DirectedMultigraph<Node, Edge> circuit, Node node) {
        List<Node> ancs = new ArrayList<>();

        List<Node> ancsToVisit = new ArrayList<>();
        ancsToVisit.addAll(Graphs.predecessorListOf(circuit, node));
        while (!ancsToVisit.isEmpty()) {
            Node anc = ancsToVisit.get(ancsToVisit.size() - 1);
            ancsToVisit.remove(ancsToVisit.size() - 1);
            if (!anc.isGate()) {
                continue;
            }

            if (anc.isCX()) {
                ancs.add(anc);
            }
            ancsToVisit.addAll(Graphs.predecessorListOf(circuit, anc));
        }

        return ancs;
    }

    private List<Node> allCXDecs(DirectedMultigraph<Node, Edge> circuit, Node node) {
        List<Node> decs = new ArrayList<>();

        List<Node> decsToVisit = new ArrayList<>();
        decsToVisit.addAll(Graphs.successorListOf(circuit, node));
        while (!decsToVisit.isEmpty()) {
            Node dec = decsToVisit.get(decsToVisit.size() - 1);
            decsToVisit.remove(decsToVisit.size() - 1);
            if (!dec.isGate()) {
                continue;
            }

            if (dec.isCX()) {
                decs.add(dec);
            }
            decsToVisit.addAll(Graphs.successorListOf(circuit, dec));
        }

        return decs;
    }

    private String getCommonQubit(Node n1, Node n2) {
        if (n1.getQubits().get(0).equals(n2.getQubits().get(0)) || n1.getQubits().get(0).equals(n2.getQubits().get(1))) {
            return n1.getQubits().get(0);
        } else if (n1.getQubits().get(1).equals(n2.getQubits().get(1)) || n1.getQubits().get(1).equals(n2.getQubits().get(0))) {
            return n1.getQubits().get(1);
        }
        return null;
    }

    private boolean sameLCA(DirectedMultigraph<Node, Edge> circuit,
                            DirectedMultigraph<Node, Edge> pattern,
                            Node circuitNode,
                            Node patternNode,
                            Node circuitCXAnc,
                            Node pattCXAnc) {
        NaiveLCAFinder<Node, Edge> lcaP = new NaiveLCAFinder<>(pattern);
        String commonPattQubit = getCommonQubit(pattCXAnc, patternNode);
        String commonCircQubit = getCommonQubit(circuitCXAnc, circuitNode);

        for (Edge e : pattern.incomingEdgesOf(patternNode)) {
            if (!e.getQubit().equals(commonPattQubit)) {
                if (pattCXAnc != lcaP.getLCA(pattCXAnc, pattern.getEdgeSource(e))) {
                    for (Edge e2 : circuit.incomingEdgesOf(circuitNode)) {
                        if (!e2.getQubit().equals(commonCircQubit)) {
                            NaiveLCAFinder<Node, Edge> lcaC = new NaiveLCAFinder<>(circuit);
                            if (circuitCXAnc != lcaC.getLCA(circuitCXAnc, circuit.getEdgeSource(e2))) {
                                return true;
                            }
                        }
                    }
                } else {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean checkLCA(DirectedMultigraph<Node, Edge> circuit,
                             DirectedMultigraph<Node, Edge> pattern,
                             Node circuitNode,
                             Node patternNode,
                             Map<Node, Node> patternToCirc) {
        if (patternNode.isCX()) {
            for (Node cxAnc : allCXAncs(pattern, patternNode)) {
                if (!sameQubits(cxAnc, patternNode)) {
                    if (patternToCirc.containsKey(cxAnc)) {
                        if (!sameLCA(circuit, pattern, circuitNode, patternNode, patternToCirc.get(cxAnc), cxAnc)) {
                            return false;
                        }
                    }
                }
            }
            for (Node cxDec : allCXDecs(pattern, patternNode)) {
                if (!sameQubits(cxDec, patternNode)) {
                    if (patternToCirc.containsKey(cxDec)) {
                        if (!sameLCA(circuit, pattern, patternToCirc.get(cxDec), cxDec, circuitNode, patternNode)) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    private boolean matchIncoming(DirectedMultigraph<Node, Edge> circuit,
                                  DirectedMultigraph<Node, Edge> pattern,
                                  Node circuitNode,
                                  Node patternNode,
                                  Map<Node, Node> patternToCirc,
                                  Map<Edge, Edge> patternToCircEdges,
                                  Map<String, Expr> angleMap,
                                  List<Node> ancsToVisit) {
        for (Edge pattE : pattern.incomingEdgesOf(patternNode)) {
            if (pattern.getEdgeSource(pattE).isSourceQubit()) {
                continue;
            }
            boolean foundMatch = false;
            if (circuit.inDegreeOf(circuitNode) != pattern.inDegreeOf(patternNode)) {
                return false;
            }

            for (Edge circE : circuit.incomingEdgesOf(circuitNode)) {
                if (patternToCirc.containsKey(pattern.getEdgeSource(pattE))) {
                    if (pattE.sameSourceTargetLabels(circE) && circuit.getEdgeSource(circE) == patternToCirc.get(pattern.getEdgeSource(pattE))) {
                        if (checkLCA(circuit, pattern, circuitNode, patternNode, patternToCirc)) {
                            foundMatch = true;
                        }
                    }
                } else {
                    if (pattE.sameSourceTargetLabels(circE) && circuit.getEdgeSource(circE).getId().equals(pattern.getEdgeSource(pattE).getId())) {
                        if (checkLCA(circuit, pattern, circuitNode, patternNode, patternToCirc)) {
                            if (pattern.getEdgeSource(pattE).getAngles() != null) {
                                if (matchAngles(circuit.getEdgeSource(circE), pattern.getEdgeSource(pattE), angleMap)) {
                                    patternToCirc.put(pattern.getEdgeSource(pattE), circuit.getEdgeSource(circE));
                                    foundMatch = true;
                                }
                            } else {
                                patternToCirc.put(pattern.getEdgeSource(pattE), circuit.getEdgeSource(circE));
                                foundMatch = true;
                            }
                        }
                    }
                }
                if (foundMatch) {
                    patternToCircEdges.put(pattE, circE);
                    ancsToVisit.add(pattern.getEdgeSource(pattE));
                    break;
                }
            }

            if (!foundMatch) {
                return false;
            }
        }
        return true;
    }

    private boolean sameQubits(Node n1, Node n2) {
        return n1.getQubits().equals(n2.getQubits()) || (n1.getQubits().get(0).equals(n2.getQubits().get(1)) && n1.getQubits().get(1).equals(n2.getQubits().get(0)));
    }

    private boolean sameAngle(Expr angle1, Expr angle2) {
        return (eval(angle1) % (4 * Math.PI)) == (eval(angle2) % (4 * Math.PI));
    }

    private String replaceAngles(String replace, Map<String, Expr> angleMap) {
        for (String angle : angleMap.keySet()) {
            replace = replace.replace(angle, eval(angleMap.get(angle)).toString());
        }

        return replace;
    }

    // Helper function that, given a circuit, pattern, and start node within the circuit,
    // returns a match if it exists, or null otherwise.
    // Essentially does everything done within the for circN in nodes construct
    private Match matchAtNode(CircuitDAG circuit, CircuitDAG pattern, Node startNode, Map<Node, Node> patternToCirc, Map<Edge, Edge> patternToCircEdges, Map<String, Expr> angleMap, Set<Node> matched, Set<Node> replaced, List<Map<Node, Node>> matches) {
        patternToCirc.clear();
        patternToCircEdges.clear();
        angleMap.clear();
        
        // NOTE: It seems like we don't use replaced at all here, but replaced
        // is stateful in find, findParallel, and findMatchesParallel

        if (matched.contains(startNode) || replaced.contains(startNode)) {
            return null;
        }

        Node patternStart = pattern.roots().get(0);
        if (!startNode.isGate() || !startNode.getId().equals(patternStart.getId())) return null; // inverted check to original find

        patternToCirc.put(patternStart, startNode);
        if (patternStart.getAngles() != null && !matchAngles(startNode, patternStart, angleMap)) {
            return null;
        }

        List<Node> succsToVisit = new ArrayList<>();
        List<Node> ancsToVisit = new ArrayList<>();
        Set<Node> seen = new HashSet<>();

        if (!matchOutgoing(circuit.getDag(), pattern.getDag(), startNode, patternStart, patternToCirc, patternToCircEdges, angleMap, succsToVisit)) return null;
        if (!matchIncoming(circuit.getDag(), pattern.getDag(), startNode, patternStart, patternToCirc, patternToCircEdges, angleMap, ancsToVisit)) return null;

        seen.add(patternStart);

        boolean match = true;
        while (!succsToVisit.isEmpty() || !ancsToVisit.isEmpty()) {
            while (!succsToVisit.isEmpty()) {
                Node succ = succsToVisit.remove(0);
                if (seen.contains(succ)) continue;

                if (matched.contains(patternToCirc.get(succ)) || replaced.contains(patternToCirc.get(succ))) {
                    match = false;
                    break;
                }

                Node circNode = patternToCirc.get(succ);
                if (!matchOutgoing(circuit.getDag(), pattern.getDag(), circNode, succ, patternToCirc, patternToCircEdges, angleMap, succsToVisit)
                        || !matchIncoming(circuit.getDag(), pattern.getDag(), circNode, succ, patternToCirc, patternToCircEdges, angleMap, ancsToVisit)) {
                    match = false;
                    break;
                }
                seen.add(succ);
            }
            if (!match) break;

            while (!ancsToVisit.isEmpty()) {
                Node anc = ancsToVisit.remove(0);
                if (seen.contains(anc)) continue;
                Node circNode = patternToCirc.get(anc);

                if (matched.contains(circNode) || replaced.contains(circNode)) {
                    match = false;
                    break;
                }
                if (!matchOutgoing(circuit.getDag(), pattern.getDag(), circNode, anc, patternToCirc, patternToCircEdges, angleMap, succsToVisit)
                        || !matchIncoming(circuit.getDag(), pattern.getDag(), circNode, anc, patternToCirc, patternToCircEdges, angleMap, ancsToVisit)) {
                    match = false;
                    break;
                }
                seen.add(anc);
            }
        }

        if (!match || patternToCirc.size() != pattern.totalGateCount()) return null;
        matched.addAll(patternToCirc.values());
        matches.add(new HashMap<>(patternToCirc));

        int startDepth = patternToCirc.values().stream().mapToInt(Node::getDepth).min().orElse(startNode.getDepth());
        int endDepth = patternToCirc.values().stream().mapToInt(Node::getDepth).max().orElse(startNode.getDepth()) + 1;

        // NOTE: Introducing deep copies for patternToCirc, angleMap
        // UPDATE: Also introducing deep copies for matched, replaced
        return new Match(startNode, new HashMap<>(patternToCirc), startDepth, endDepth, new HashMap<>(angleMap), new HashSet<>(matched), new HashSet<>(replaced));
    }

    // Actually changes the CircuitDAG by applying the match from `matchAtNode`, does this in-place
    private CircuitDAG applyMatch(
            CircuitDAG circuit,
            CircuitDAG copy,
            CircuitDAG pattern,
            String replace,
            Match match,
            boolean applyOnce) {
        Map<Node, Node> patternToCirc = match.patternToCircMap;
        Map<String, String> patternToCircuitQubit = patternToCircuitQubit(patternToCirc);

        if (new HashSet<>(patternToCircuitQubit.values()).size() != patternToCircuitQubit.values().size()) {
            return null;
        }

        if (copy == null) {
            copy = new CircuitDAG(circuit);
        }

        String[] searchList = new String[patternToCircuitQubit.size() * 2];
        String[] replaceList = new String[patternToCircuitQubit.size() * 2];
        int i = 0;
        for (String key : patternToCircuitQubit.keySet()) {
            searchList[i] = key + ",";
            replaceList[i] = patternToCircuitQubit.get(key) + ",";
            i++;
            searchList[i] = key + ";";
            replaceList[i] = patternToCircuitQubit.get(key) + ";";
            i++;
        }

        String replaceAfterSubst = StringUtils.replaceEach(replace, searchList, replaceList);
        replaceAfterSubst = replaceAngles(replaceAfterSubst, match.angleMap);

        CircuitDAG replaceDag = CircuitParser.qasmToDag(replaceAfterSubst);
        match.replaced.addAll(replaceDag.nodes());

        // Apply replacement
        replace(copy.getDag(), pattern, replaceDag, patternToCirc, patternToCircuitQubit);

        // Optionally stop after one match
        if (applyOnce) {
            return copy;
        }

        // return updated circuit for continued search
        return copy;
    }

    public CircuitDAG find(CircuitDAG circuit, CircuitDAG pattern, String replace, boolean applyOnce, Random rand) {
        Map<Node, Node> patternToCirc = new HashMap<>();
        Map<Edge, Edge> patternToCircEdges = new HashMap<>();
        Map<String, Expr> angleMap = new HashMap<>();
        Set<Node> matched = new HashSet<>();
        Set<Node> replaced = new HashSet<>();
        List<Map<Node, Node>> matches = new ArrayList<>();

        CircuitDAG copy = null;
        List<Node> nodes = new ArrayList<>(circuit.nodes());
        Collections.shuffle(nodes, rand);

        for (Node circN : nodes) {
            Match match = matchAtNode(circuit, pattern, circN, patternToCirc, patternToCircEdges, angleMap, matched, replaced, matches);

            if (match != null) {
                CircuitDAG result = applyMatch(circuit, copy, pattern, replace, match, applyOnce);
                if (result != null) {
                    copy = result;
                    if (applyOnce) {
                        return copy;
                    }
                    circuit = copy;
                }
            }
        }
        return copy;
    }

    // Checks if the [startDepth, endDepth] interval overlaps with
    // any currently claimed intervals
    private boolean overlaps(Pair<Integer, Integer>[] claimedIntervals,
                             int startDepth, int endDepth) {
        for (Pair<Integer, Integer> interval : claimedIntervals) {
            if (interval == null) continue;
            int s = interval.getFirst();
            int e = interval.getSecond();

            // overlap test: [startDepth, endDepth) intersects [s, e)
            if (startDepth < e && endDepth > s) {
                return true;
            }
        }
        return false;
    }


    public ArrayList<Match> findMatchesParallel(CircuitDAG circuit, CircuitDAG pattern, String replace, int startDepth, int endDepth, int windowIdx, Pair<Integer, Integer>[] claimedIntervals, ReentrantReadWriteLock rwl, boolean applyOnce, Random rand) {
        ArrayList<Match> foundMatches = new ArrayList<>();
        Map<Node, Node> patternToCirc = new HashMap<>();
        int patternDepth = pattern.getDepth();
        Map<Edge, Edge> patternToCircEdges = new HashMap<>();
        Map<String, Expr> angleMap = new HashMap<>();
        Set<Node> matched = new HashSet<>();
        Set<Node> replaced = new HashSet<>();
        List<Map<Node, Node>> matches = new ArrayList<>();

        CircuitDAG copy = null;
        List<Node> nodesInWindow = circuit.nodes().stream()
            .filter(n -> n.getDepth() >= startDepth && n.getDepth() <= endDepth)
            .collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(nodesInWindow, rand);

        for (Node circN : nodesInWindow) {
            Match match = matchAtNode(circuit, pattern, circN, patternToCirc, patternToCircEdges, angleMap, matched, replaced, matches);
            // System.out.println("found a match find matches parallel!");

            if (match != null) {

                if (match.startDepth < startDepth ||
                    match.startDepth >= (windowIdx+1)*patternDepth) {
                    continue;
                } // else

                foundMatches.add(match);
            }
        }
        return foundMatches;
    }

    public ArrayList<Match> findMatchesParallelUpdated(CircuitDAG circuit, CircuitDAG pattern, String replace, int startDepth, int endDepth, int windowIdx, boolean applyOnce, Random rand) {
        ArrayList<Match> foundMatches = new ArrayList<>();
        Map<Node, Node> patternToCirc = new HashMap<>();
        int patternDepth = pattern.getDepth();
        Map<Edge, Edge> patternToCircEdges = new HashMap<>();
        Map<String, Expr> angleMap = new HashMap<>();
        Set<Node> matched = new HashSet<>();
        Set<Node> replaced = new HashSet<>();
        List<Map<Node, Node>> matches = new ArrayList<>();

        CircuitDAG copy = null;
        List<Node> nodesInWindow = circuit.nodes().stream()
            .filter(n -> n.getDepth() >= startDepth && n.getDepth() <= endDepth)
            .collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(nodesInWindow, rand);

        for (Node circN : nodesInWindow) {
            Match match = matchAtNode(circuit, pattern, circN, patternToCirc, patternToCircEdges, angleMap, matched, replaced, matches);
            // System.out.println("found a match find matches parallel!");

            if (match != null) {

                if (match.startDepth < startDepth ||
                    match.startDepth >= (windowIdx+1)*patternDepth) {
                    continue;
                } // else

                foundMatches.add(match);
            }
        }
        return foundMatches;
    }

    public CircuitDAG findParallel(CircuitDAG circuit, CircuitDAG pattern, String replace, int startDepth, int endDepth, int windowIdx, Pair<Integer, Integer>[] claimedIntervals, ReentrantReadWriteLock rwl, boolean applyOnce, Random rand) {
        Map<Node, Node> patternToCirc = new HashMap<>();
        Map<Edge, Edge> patternToCircEdges = new HashMap<>();
        Map<String, Expr> angleMap = new HashMap<>();
        Set<Node> matched = new HashSet<>();
        Set<Node> replaced = new HashSet<>();
        List<Map<Node, Node>> matches = new ArrayList<>();

        CircuitDAG copy = null;
        List<Node> nodesInWindow = circuit.nodes().stream()
            .filter(n -> n.getDepth() >= startDepth && n.getDepth() <= endDepth)
            .collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(nodesInWindow, rand);

        for (Node circN : nodesInWindow) {
            Match match = matchAtNode(circuit, pattern, circN, patternToCirc, patternToCircEdges, angleMap, matched, replaced, matches);
            // System.out.println("found a match!");

            if (match != null) {
                rwl.readLock().lock();
                boolean conflict = overlaps(claimedIntervals, startDepth, endDepth);
                rwl.readLock().unlock();

                if (conflict) continue; // don't add this match to the circuit, exists overlap

                // We’re about to mutate the shared DAG, so take the write lock
                rwl.writeLock().lock();
                try {
                    // System.out.println("applying a match!");
                    // System.out.println(Arrays.stream(claimedIntervals)
                    //     .map(p -> p == null ? "[unclaimed]" : "[" + p.getFirst() + "," + p.getSecond() + "]")
                    //     .collect(Collectors.joining(", ", "claimedIntervals before: ", "")));
            
                    CircuitDAG result = applyMatch(circuit, circuit, pattern, replace, match, applyOnce);
                    if (result != null) {
                        copy = result;
                        claimedIntervals[windowIdx] = new Pair<>(startDepth, endDepth);
            
                        // System.out.println(Arrays.stream(claimedIntervals)
                        //     .map(p -> p == null ? "[unclaimed]" : "[" + p.getFirst() + "," + p.getSecond() + "]")
                        //     .collect(Collectors.joining(", ", "claimedIntervals after: ", "")));
            
                        if (applyOnce) return copy;
                        circuit = copy;
                    }
                } finally {
                    rwl.writeLock().unlock();
                }

                // // NOTE: applyMatch works in-place with this call (we don't use copy at all)
                // // Can we modify the regular find similarly?
                // System.out.println("applying a match!");
                // // print the interval of the match before we apply it, or print claimedIntervals or something
                // System.out.println("claimed intervals: \n");
                // System.out.println(Arrays.stream(claimedIntervals)
                //     .map(p -> p == null ? "[unclaimed]" : "[" + p.getFirst() + "," + p.getSecond() + "]")
                //     .collect(Collectors.joining(", ", "claimedIntervals: ", "")));
                // CircuitDAG result = applyMatch(circuit, circuit, pattern, replace, match, replaced, angleMap, applyOnce);
                // if (result != null) {
                //     copy = result;
                //
                //     // update claimedIntervals
                //     rwl.writeLock().lock();
                //     claimedIntervals[windowIdx] = new Pair<>(startDepth, endDepth);
                //     System.out.println("claimed intervals: \n");
                //     System.out.println(Arrays.stream(claimedIntervals)
                //         .map(p -> p == null ? "[unclaimed]" : "[" + p.getFirst() + "," + p.getSecond() + "]")
                //         .collect(Collectors.joining(", ", "claimedIntervals: ", "")));
                //     rwl.writeLock().unlock();
                //
                //     if (applyOnce) {
                //         return copy;
                //     }
                //     circuit = copy;
                // }
            }
        }
        
        return copy;
    }

    public Map<String, String> patternToCircuitQubit(Map<Node, Node> patternToCirc) {
        Map<String, String> result = new HashMap<>();
        for (Node patternNode : patternToCirc.keySet()) {
            result.put(patternNode.getQubits().get(0), patternToCirc.get(patternNode).getQubits().get(0));
            if (patternNode.isCX()) {
                result.put(patternNode.getQubits().get(1), patternToCirc.get(patternNode).getQubits().get(1));
            } else if (patternNode.isCCZ()) {
                // TODO improve
                result.put(patternNode.getQubits().get(1), patternToCirc.get(patternNode).getQubits().get(1));
                result.put(patternNode.getQubits().get(2), patternToCirc.get(patternNode).getQubits().get(2));
            }
        }
        return result;
    }

    public void replace(DirectedMultigraph<Node, Edge> circuit,
                        CircuitDAG pattern,
                        CircuitDAG replace,
                        Map<Node, Node> patternToCirc,
                        Map<String, String> patternToCircuitQubit) {
        Map<String, Node> patternRoots = pattern.rootsMap();
        Map<String, Node> patternLeaves = pattern.leavesMap();

        Map<String, Node> replaceRoots = replace.rootsMap();
        Map<String, Node> replaceLeaves = replace.leavesMap();

        Map<String, Node> ancPatternRoots = new HashMap<>();
        for (String qubit : patternRoots.keySet()) {
            String circQubit = patternToCircuitQubit.getOrDefault(qubit, qubit);
            Node match = patternToCirc.getOrDefault(patternRoots.get(qubit), patternRoots.get(qubit));
            for (Edge e : circuit.incomingEdgesOf(match)) {
                if (e.getQubit().equals(circQubit)) {
                    ancPatternRoots.put(circQubit, circuit.getEdgeSource(e));
                }

            }
        }

        Map<String, Node> decPatternLeaves = new HashMap<>();
        for (String qubit : patternLeaves.keySet()) {
            String circQubit = patternToCircuitQubit.getOrDefault(qubit, qubit);
            Node match = patternToCirc.getOrDefault(patternLeaves.get(qubit), patternLeaves.get(qubit));
            for (Edge e : circuit.outgoingEdgesOf(match)) {
                if (e.getQubit().equals(circQubit)) {
                    decPatternLeaves.put(circQubit, circuit.getEdgeTarget(e));
                }
            }
        }

        Set<Node> toRemove = new HashSet<>();
        for (Node n : replace.nodes()) {
            if (n.isQubit()) {
                toRemove.add(n);
            }
        }
        replace.getDag().removeAllVertices(toRemove);
        Graphs.addGraph(circuit, replace.getDag());

        for (Node n : pattern.nodes()) {
            circuit.removeVertex(patternToCirc.getOrDefault(n, n));
        }

        for (String qubit : ancPatternRoots.keySet()) {
            if (replaceRoots.containsKey(qubit)) {
                circuit.addEdge(ancPatternRoots.get(qubit), replaceRoots.get(qubit), getEdge(ancPatternRoots.get(qubit), replaceRoots.get(qubit), qubit));
            } else {
                circuit.addEdge(ancPatternRoots.get(qubit), decPatternLeaves.get(qubit), getEdge(ancPatternRoots.get(qubit), decPatternLeaves.get(qubit), qubit));
            }
        }

        for (String qubit : replaceLeaves.keySet()) {
            // qubits not in replaceLeaves should not have been in replaceRoots and therefore were connected already to decPatternLeaves
            circuit.addEdge(replaceLeaves.get(qubit), decPatternLeaves.get(qubit), getEdge(replaceLeaves.get(qubit), decPatternLeaves.get(qubit), qubit));
        }
    }

    private Edge getEdge(Node source, Node target, String qubit) {
        Edge.Label sourceLabel = Edge.Label.NONE;
        Edge.Label targetLabel = Edge.Label.NONE;
        if (source.isCX()) {
            if (qubit.equals(source.getQubits().get(0))) {
                sourceLabel = Edge.Label.CONTROL;
            } else {
                sourceLabel = Edge.Label.TARGET;
            }
        } else if (source.isCCZ()) {
            if (qubit.equals(source.getQubits().get(0))) {
                sourceLabel = Edge.Label.CONTROL;
            } else if (qubit.equals(source.getQubits().get(1))) {
                sourceLabel = Edge.Label.CONTROL2;
            } else {
                sourceLabel = Edge.Label.TARGET;
            }
        }
        if (target.isCX()) {
            if (qubit.equals(target.getQubits().get(0))) {
                targetLabel = Edge.Label.CONTROL;
            } else {
                targetLabel = Edge.Label.TARGET;
            }
        } else if (target.isCCZ()) {
            if (qubit.equals(target.getQubits().get(0))) {
                targetLabel = Edge.Label.CONTROL;
            } else if (qubit.equals(target.getQubits().get(1))) {
                targetLabel = Edge.Label.CONTROL2;
            } else {
                targetLabel = Edge.Label.TARGET;
            }
        }

        return new Edge(sourceLabel, targetLabel, qubit);
    }

    public CircuitDAG applyRule(CircuitDAG circuit, String lhs, CircuitDAG rhs, boolean applyOnce, Random rand) {
        // FIXME: remove after testing done
        // System.out.println("IN APPLY RULE REGULAR");
        CircuitDAG pattern = rhs;
        var result = find(circuit, pattern, lhs, applyOnce, rand);
        if (result == null) {
            // System.out.println("FAILED TO APPLY RULE REGULAR");
            return circuit;
        }
        return result;
    }


    public CircuitDAG applyRuleParallelNew(CircuitDAG circuit, String replace, CircuitDAG pattern, boolean applyOnce, Random rand) {
        // FIXME: Remove this after testing done
        // System.out.println("IN APPLY RULE PARALLEL NEW");
        // Call this just in case we forgot, for each rule

        // TODO: Figure out of we can hoist this!
        circuit.assignDepthToNodes();
        pattern.assignDepthToNodes();
        int patternDepth = pattern.getDepth();
        int circuitDepth = circuit.getDepth();
        int numWindows = (int) Math.ceil((double) circuitDepth/patternDepth);
        // System.out.println("numWindows = " + numWindows);
        int cores = Runtime.getRuntime().availableProcessors();
        
        // attempt to multithread :D

        // NOTE: claimedIntervals is allowed to use this structure because we ASSUME applyOnce = true
        // For future, if/when we allow the ability to apply multiple times in a window, we can use
        // claimedIntervals and only store the earliest match for a given window, because that is what will
        // cause a conflict with the patternDepth overlap

        // FIXME: We no longer use claimedIntervals or RWL to do any checking!
        // Thus they are commented out here
        // Pair<Integer, Integer>[] claimedIntervals = new Pair[numWindows];
        // ReentrantReadWriteLock rwl = new ReentrantReadWriteLock(); // for claimedIntervals
        ExecutorService threadPool = Executors.newFixedThreadPool(Math.min(cores, numWindows));
        List<Future<ArrayList<Match>>> futures = new ArrayList<>();
        // Collect results: each thread returns its transformed sub-DAG or null
        for (int i = 0; i < numWindows; i++) {
            final int windowIndex = i;
            int startDepth = i * patternDepth;
            int endDepth = Math.min(circuitDepth, (i + 1) * patternDepth + patternDepth); // lookahead
            futures.add(threadPool.submit(() ->
                // findMatchesParallel(circuit, pattern, replace, startDepth, endDepth, windowIndex, claimedIntervals, rwl, applyOnce, rand)
                findMatchesParallelUpdated(circuit, pattern, replace, startDepth, endDepth, windowIndex, applyOnce, rand)
            ));
        }

        // Step 1. collect all matches from all threads
        List<Match> allMatches = new ArrayList<>();
        for (Future<ArrayList<Match>> f : futures) {
            try {
                allMatches.addAll(f.get());
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        threadPool.shutdown();

        // Step 2. Select non-overlapping matches using node-level conflict detection.
    // More precise than depth-interval overlap: two matches only conflict if they
    // actually share circuit nodes, not merely overlap in depth range.
    List<Match> selected = new ArrayList<>();
    Set<Node> globalReplaced = new HashSet<>();
    for (Match m : allMatches) {
        if (!m.conflictsWith(globalReplaced)) {
            selected.add(m);
            globalReplaced.addAll(m.getCircuitNodes());
        }
    }

    // Step 3. Apply
    for (Match m : selected) {
        CircuitDAG updated = applyMatch(
            circuit,
            circuit,
            pattern,
            replace,
            m,
            applyOnce
        );
        if (updated != null && applyOnce) break;
    }

    return circuit;
}

public CircuitDAG applyRuleParallelNewTiming(CircuitDAG circuit, String replace, CircuitDAG pattern, boolean applyOnce, Random rand) {
    circuit.assignDepthToNodes();
    pattern.assignDepthToNodes();
    int patternDepth = pattern.getDepth();
    int circuitDepth = circuit.getDepth();
    int numWindows = (int) Math.ceil((double) circuitDepth/patternDepth);
    int cores = Runtime.getRuntime().availableProcessors();
    
    long poolCreateStart = System.nanoTime();
    ExecutorService threadPool = Executors.newFixedThreadPool(Math.min(cores, numWindows));
    long poolCreateEnd = System.nanoTime();
    System.out.printf("Thread pool creation: %.3f ms%n", (poolCreateEnd - poolCreateStart) / 1_000_000.0);
    
    List<Future<ArrayList<Match>>> futures = new ArrayList<>();
    
    long submitStart = System.nanoTime();
    for (int i = 0; i < numWindows; i++) {
        final int windowIndex = i;
        int startDepth = i * patternDepth;
        int endDepth = Math.min(circuitDepth, (i + 1) * patternDepth + patternDepth);
        futures.add(threadPool.submit(() ->
            findMatchesParallelUpdated(circuit, pattern, replace, startDepth, endDepth, windowIndex, applyOnce, rand)
        ));
    }
    long submitEnd = System.nanoTime();
    System.out.printf("Task submission: %.3f ms%n", (submitEnd - submitStart) / 1_000_000.0);
    
    // Step 1. collect all matches from all threads
    long workStart = System.nanoTime();
    List<Match> allMatches = new ArrayList<>();
    for (Future<ArrayList<Match>> f : futures) {
        try {
            allMatches.addAll(f.get());
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }
    long workEnd = System.nanoTime();
    System.out.printf("Step 1 - Collect matches: %.3f ms%n", (workEnd - workStart) / 1_000_000.0);
    
    long shutdownStart = System.nanoTime();
    threadPool.shutdown();
    long shutdownEnd = System.nanoTime();
    System.out.printf("Thread pool shutdown: %.3f ms%n", (shutdownEnd - shutdownStart) / 1_000_000.0);
    
    long overheadTotal = (poolCreateEnd - poolCreateStart) + (submitEnd - submitStart) + (shutdownEnd - shutdownStart);
    long totalTime = (workEnd - workStart) + overheadTotal;
    System.out.printf("Total thread pool overhead: %.3f ms (%.1f%% of total time)%n", 
                     overheadTotal / 1_000_000.0, 
                     100.0 * overheadTotal / totalTime);
    
    // Step 2. Select non-overlapping matches (node-level conflict detection, no sort needed)
    long selectStart = System.nanoTime();
    List<Match> selected = new ArrayList<>();
    Set<Node> globalReplaced = new HashSet<>();
    for (Match m : allMatches) {
        if (!m.conflictsWith(globalReplaced)) {
            selected.add(m);
            globalReplaced.addAll(m.getCircuitNodes());
        }
    }
    long selectEnd = System.nanoTime();
    System.out.printf("Step 2 - Select non-overlapping: %.3f ms (%d selected from %d total)%n", 
                     (selectEnd - selectStart) / 1_000_000.0, selected.size(), allMatches.size());
    
    // Step 3. Apply
    long applyStart = System.nanoTime();
    for (Match m : selected) {
        CircuitDAG updated = applyMatch(circuit, circuit, pattern, replace, m, applyOnce);
        if (updated != null && applyOnce) break;
    }
    long applyEnd = System.nanoTime();
    System.out.printf("Step 3 - Apply matches: %.3f ms%n", (applyEnd - applyStart) / 1_000_000.0);
    
    long grandTotal = (applyEnd - poolCreateStart);
    System.out.printf("TOTAL TIME: %.3f ms%n", grandTotal / 1_000_000.0);
    
    return circuit;
}

public CircuitDAG applyRuleParallelNewTimingFull(CircuitDAG circuit, String replace, CircuitDAG pattern, boolean applyOnce, Random rand) {
    long methodStart = System.nanoTime();
    
    long depthStart = System.nanoTime();
    circuit.assignDepthToNodes();
    pattern.assignDepthToNodes();
    int patternDepth = pattern.getDepth();
    int circuitDepth = circuit.getDepth();
    int numWindows = (int) Math.ceil((double) circuitDepth/patternDepth);
    int cores = Runtime.getRuntime().availableProcessors();
    long depthEnd = System.nanoTime();
    System.out.printf("Depth assignment & setup: %.3f ms%n", (depthEnd - depthStart) / 1_000_000.0);

    System.out.println("\n=== THREADING INFO ===");
    System.out.printf("Available processors: %d%n", cores);
    System.out.printf("Number of windows: %d%n", numWindows);
    System.out.printf("Thread pool size: %d%n", Math.min(cores, numWindows));
    System.out.printf("Pattern depth: %d, Circuit depth: %d%n", patternDepth, circuitDepth);
    
    long poolCreateStart = System.nanoTime();
    ExecutorService threadPool = Executors.newFixedThreadPool(Math.min(cores, numWindows));
    long poolCreateEnd = System.nanoTime();
    System.out.printf("Thread pool creation: %.3f ms%n", (poolCreateEnd - poolCreateStart) / 1_000_000.0);
    
    List<Future<ArrayList<Match>>> futures = new ArrayList<>();
    
    long submitStart = System.nanoTime();
    for (int i = 0; i < numWindows; i++) {
        final int windowIndex = i;
        int startDepth = i * patternDepth;
        int endDepth = Math.min(circuitDepth, (i + 1) * patternDepth + patternDepth);
        futures.add(threadPool.submit(() ->
            findMatchesParallelUpdated(circuit, pattern, replace, startDepth, endDepth, windowIndex, applyOnce, rand)
        ));
    }
    long submitEnd = System.nanoTime();
    System.out.printf("Task submission: %.3f ms%n", (submitEnd - submitStart) / 1_000_000.0);
    
    // Step 1. Collect all matches from all threads
    long workStart = System.nanoTime();
    List<Match> allMatches = new ArrayList<>();
    for (Future<ArrayList<Match>> f : futures) {
        try {
            allMatches.addAll(f.get());
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }
    long workEnd = System.nanoTime();
    System.out.printf("Step 1 - Collect matches: %.3f ms%n", (workEnd - workStart) / 1_000_000.0);
    
    long shutdownStart = System.nanoTime();
    threadPool.shutdown();
    long shutdownEnd = System.nanoTime();
    System.out.printf("Thread pool shutdown: %.3f ms%n", (shutdownEnd - shutdownStart) / 1_000_000.0);
    
    long overheadTotal = (poolCreateEnd - poolCreateStart) + (submitEnd - submitStart) + (shutdownEnd - shutdownStart);
    System.out.printf("Thread pool overhead: %.3f ms%n", overheadTotal / 1_000_000.0);
    
    // Step 2. Select non-overlapping matches (node-level conflict detection, no sort needed)
    long selectStart = System.nanoTime();
    List<Match> selected = new ArrayList<>();
    Set<Node> globalReplaced = new HashSet<>();
    for (Match m : allMatches) {
        if (!m.conflictsWith(globalReplaced)) {
            selected.add(m);
            globalReplaced.addAll(m.getCircuitNodes());
        }
    }
    long selectEnd = System.nanoTime();
    System.out.printf("Step 2 - Select non-overlapping: %.3f ms (%d selected from %d total)%n", 
                     (selectEnd - selectStart) / 1_000_000.0, selected.size(), allMatches.size());
    
    // Step 3. Apply
    long applyStart = System.nanoTime();
    for (Match m : selected) {
        CircuitDAG updated = applyMatch(circuit, circuit, pattern, replace, m, applyOnce);
        if (updated != null && applyOnce) break;
    }
    long applyEnd = System.nanoTime();
    System.out.printf("Step 3 - Apply matches: %.3f ms%n", (applyEnd - applyStart) / 1_000_000.0);
    
    long methodEnd = System.nanoTime();
    long methodTotal = methodEnd - methodStart;
    
    System.out.println("\n=== TIMING BREAKDOWN ===");
    System.out.printf("Depth assignment:     %.3f ms%n", (depthEnd - depthStart) / 1_000_000.0);
    System.out.printf("Thread pool overhead: %.3f ms%n", overheadTotal / 1_000_000.0);
    System.out.printf("Parallel work:        %.3f ms%n", (workEnd - workStart) / 1_000_000.0);
    System.out.printf("Select matches:       %.3f ms%n", (selectEnd - selectStart) / 1_000_000.0);
    System.out.printf("Apply matches:        %.3f ms%n", (applyEnd - applyStart) / 1_000_000.0);
    
    long accountedTime = (depthEnd - depthStart) + overheadTotal + (workEnd - workStart) +
                         (selectEnd - selectStart) + (applyEnd - applyStart);
    long unaccountedTime = methodTotal - accountedTime;
    
    System.out.printf("-----------------------------%n");
    System.out.printf("Accounted time:       %.3f ms%n", accountedTime / 1_000_000.0);
    System.out.printf("Unaccounted time:     %.3f ms%n", unaccountedTime / 1_000_000.0);
    System.out.printf("TOTAL METHOD TIME:    %.3f ms%n", methodTotal / 1_000_000.0);
    
    return circuit;
}

public CircuitDAG applyRuleParallelNewTimingThread(int threadPoolSize, CircuitDAG circuit, String replace, CircuitDAG pattern, boolean applyOnce, Random rand) {
    long methodStart = System.nanoTime();
    
    long depthStart = System.nanoTime();
    circuit.assignDepthToNodes();
    pattern.assignDepthToNodes();
    int patternDepth = pattern.getDepth();
    int circuitDepth = circuit.getDepth();
    int numWindows = (int) Math.ceil((double) circuitDepth/patternDepth);
    int cores = Runtime.getRuntime().availableProcessors();
    long depthEnd = System.nanoTime();
    System.out.printf("Depth assignment & setup: %.3f ms%n", (depthEnd - depthStart) / 1_000_000.0);

    long poolCreateStart = System.nanoTime();
    ExecutorService threadPool = Executors.newFixedThreadPool(threadPoolSize);
    long poolCreateEnd = System.nanoTime();
    System.out.printf("Thread pool creation: %.3f ms%n", (poolCreateEnd - poolCreateStart) / 1_000_000.0);
    
    List<Future<ArrayList<Match>>> futures = new ArrayList<>();
    
    long submitStart = System.nanoTime();
    for (int i = 0; i < numWindows; i++) {
        final int windowIndex = i;
        int startDepth = i * patternDepth;
        int endDepth = Math.min(circuitDepth, (i + 1) * patternDepth + patternDepth);
        futures.add(threadPool.submit(() ->
            findMatchesParallelUpdated(circuit, pattern, replace, startDepth, endDepth, windowIndex, applyOnce, rand)
        ));
    }
    long submitEnd = System.nanoTime();
    System.out.printf("Task submission: %.3f ms%n", (submitEnd - submitStart) / 1_000_000.0);
    
    // Step 1. Collect all matches from all threads
    long workStart = System.nanoTime();
    List<Match> allMatches = new ArrayList<>();
    for (Future<ArrayList<Match>> f : futures) {
        try {
            allMatches.addAll(f.get());
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }
    long workEnd = System.nanoTime();
    System.out.printf("Step 1 - Collect matches: %.3f ms%n", (workEnd - workStart) / 1_000_000.0);
    
    long shutdownStart = System.nanoTime();
    threadPool.shutdown();
    long shutdownEnd = System.nanoTime();
    System.out.printf("Thread pool shutdown: %.3f ms%n", (shutdownEnd - shutdownStart) / 1_000_000.0);
    
    long overheadTotal = (poolCreateEnd - poolCreateStart) + (submitEnd - submitStart) + (shutdownEnd - shutdownStart);
    System.out.printf("Thread pool overhead: %.3f ms%n", overheadTotal / 1_000_000.0);
    
    // Step 2. Select non-overlapping matches (node-level conflict detection, no sort needed)
    long selectStart = System.nanoTime();
    List<Match> selected = new ArrayList<>();
    Set<Node> globalReplaced = new HashSet<>();
    for (Match m : allMatches) {
        if (!m.conflictsWith(globalReplaced)) {
            selected.add(m);
            globalReplaced.addAll(m.getCircuitNodes());
        }
    }
    long selectEnd = System.nanoTime();
    System.out.printf("Step 2 - Select non-overlapping: %.3f ms (%d selected from %d total)%n", 
                     (selectEnd - selectStart) / 1_000_000.0, selected.size(), allMatches.size());
    
    // Step 3. Apply
    long applyStart = System.nanoTime();
    for (Match m : selected) {
        CircuitDAG updated = applyMatch(circuit, circuit, pattern, replace, m, applyOnce);
        if (updated != null && applyOnce) break;
    }
    long applyEnd = System.nanoTime();
    System.out.printf("Step 3 - Apply matches: %.3f ms%n", (applyEnd - applyStart) / 1_000_000.0);
    
    long methodEnd = System.nanoTime();
    long methodTotal = methodEnd - methodStart;
    
    System.out.println("\n=== TIMING BREAKDOWN ===");
    System.out.printf("Depth assignment:     %.3f ms%n", (depthEnd - depthStart) / 1_000_000.0);
    System.out.printf("Thread pool overhead: %.3f ms%n", overheadTotal / 1_000_000.0);
    System.out.printf("Parallel work:        %.3f ms%n", (workEnd - workStart) / 1_000_000.0);
    System.out.printf("Select matches:       %.3f ms%n", (selectEnd - selectStart) / 1_000_000.0);
    System.out.printf("Apply matches:        %.3f ms%n", (applyEnd - applyStart) / 1_000_000.0);
    
    long accountedTime = (depthEnd - depthStart) + overheadTotal + (workEnd - workStart) +
                         (selectEnd - selectStart) + (applyEnd - applyStart);
    long unaccountedTime = methodTotal - accountedTime;
    
    System.out.printf("-----------------------------%n");
    System.out.printf("Accounted time:       %.3f ms%n", accountedTime / 1_000_000.0);
    System.out.printf("Unaccounted time:     %.3f ms%n", unaccountedTime / 1_000_000.0);
    System.out.printf("TOTAL METHOD TIME:    %.3f ms%n", methodTotal / 1_000_000.0);
    
    return circuit;
}


    // Returns the updated CircuitDAG given some input pattern to apply, does rule matching and application in parallel
    // We use a form of depth parallelism by dividing the circuit into subcircuits, and finding matches in each subcircuits
    // via pointer analysis
    // NOTE: applyOnce defaults to true for now, at call-time

    // Overall plan of attack:
    // 1. (Correctly) create threads for each subcircuit
    // 2. Call a modified find on each subcircuit that takes in a circuit window and returns
    //    an interval (?) or some information specifying the starting node for where the pattern
    //    should be applied/replaced
    // 3. Collect all of those instances and sequentially apply them?
    // 
    // Alternate idea:
    // 1. (Correctly) create threads for each subcircuit
    // 2. Call find on each subcircuit, in parallel apply the change to the same circuit object
    //    we can check this via the claimedIntervals structure
    // 3. Don't need to collect stuff later
    public CircuitDAG applyRuleParallel(CircuitDAG circuit, String replace, CircuitDAG pattern, boolean applyOnce, Random rand) {
        // FIXME: Remove this after testing done
        // System.out.println("IN APPLY RULE PARALLEL");
        // Call this just in case we forgot, for each rule
        circuit.assignDepthToNodes();
        pattern.assignDepthToNodes();
        int patternDepth = pattern.getDepth();
        int circuitDepth = circuit.getDepth();
        int numWindows = (int) Math.ceil((double) circuitDepth/patternDepth);
        // System.out.println("numWindows = " + numWindows);
        int cores = Runtime.getRuntime().availableProcessors();
        
        // attempt to multithread :D

        // NOTE: claimedIntervals is allowed to use this structure because we ASSUME applyOnce = true
        // For future, if/when we allow the ability to apply multiple times in a window, we can use
        // claimedIntervals and only store the earliest match for a given window, because that is what will
        // cause a conflict with the patternDepth overlap
        Pair<Integer, Integer>[] claimedIntervals = new Pair[numWindows];
        ReentrantReadWriteLock rwl = new ReentrantReadWriteLock(); // for claimedIntervals
        ExecutorService threadPool = Executors.newFixedThreadPool(Math.min(cores, numWindows));
        List<Future<CircuitDAG>> futures = new ArrayList<>();
        // Collect results: each thread returns its transformed sub-DAG or null
        for (int i = 0; i < numWindows; i++) {
            final int windowIndex = i;
            int startDepth = i * patternDepth;
            int endDepth = Math.min(circuitDepth, (i + 1) * patternDepth + patternDepth); // lookahead
            futures.add(threadPool.submit(() ->
                findParallel(circuit, pattern, replace, startDepth, endDepth, windowIndex, claimedIntervals, rwl, applyOnce, rand)
            ));
        }

        // wait for all threads to finish
        for (Future<CircuitDAG> f : futures) {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        threadPool.shutdown();
        return circuit; // updated in-place via findParallel

    }

    public CircuitDAG applySymbRule(CircuitDAG circuit,
                                    String findBefore, String findAfter,
                                    String replaceBefore, String replaceAfter,
                                    List<Map<boolean[], boolean[]>> constraints,
                                    int maxSymbQubits,
                                    int maxSymbSize,
                                    boolean applyOnce,
                                    Random rand) {
        DirectedMultigraph<Node, Edge> patternBefore = CircuitParser.qasmToDag(findBefore).getDag();
        DirectedMultigraph<Node, Edge> patternAfter = CircuitParser.qasmToDag(findAfter).getDag();

        Map<String, Expr> angleMap = new HashMap<>();
        List<Node> symb = findSymb(circuit.getDag(), patternBefore, patternAfter, constraints, angleMap, maxSymbQubits, maxSymbSize);
        if (symb != null) {
            Map<String, String> qubitMap = getMap(symb, findBefore, findAfter);
            String[] searchList = new String[qubitMap.size() * 2];
            String[] replaceList = new String[qubitMap.size() * 2];
            int i = 0;
            for (String key : qubitMap.keySet()) {
                replaceList[i] = key + ",";
                searchList[i] = qubitMap.get(key) + ",";
                i++;
                replaceList[i] = key + ";";
                searchList[i] = qubitMap.get(key) + ";";
                i++;
            }

            replaceBefore = StringUtils.replaceEach(replaceBefore, searchList, replaceList);
            replaceAfter = StringUtils.replaceEach(replaceAfter, searchList, replaceList);

            String newLHS = replaceBefore + " " + getQasm(symb.subList(StringUtils.countMatches(findBefore, ";"), symb.size() - StringUtils.countMatches(findAfter, ";"))) + " " + replaceAfter;
            newLHS = newLHS.replace("  ", " ");
            newLHS = replaceAngles(newLHS, angleMap);

            String newRHS = getQasm(symb);
//            System.out.println(newLHS.trim() + " | " + newRHS.trim());
            return applyRule(circuit, newLHS.trim(), CircuitParser.qasmToDag(newRHS.trim()), applyOnce, rand); // TODO: do the replace here instead of finding again
        }
        return circuit;
    }

    public void replaceResynth(DirectedMultigraph<Node, Edge> circuit,
                               CircuitDAG pattern,
                               CircuitDAG replace) {
        replace(circuit, pattern, replace, new HashMap<>(), new HashMap<>());
    }

    public Pair<CircuitDAG, String> applyResynth(Resynth resynthAlg, CircuitDAG circuit, Random rand, double epsilon, ResynthArgs resynthArgs) throws IOException, InterruptedException {
        RandomPartition randomPartition = new RandomPartition(rand, IPartitionPicker.MAX_PARTITION_QUBITS);
        CircuitDAG partition = randomPartition.getPartition(circuit);
        String partitionQasm = partition.toQASM();
        boolean sentWhole = false;
        // if (resynthAlg == Resynth.PYZX) {
        //     if (rand.nextInt() % 2 == 0) {
        //         System.out.println("Sending whole circ to PyZX");
        //         partitionQasm = circuit.toQASM();
        //         sentWhole = true;
        //     }
        // }

        String resynth;
        if (resynthAlg == Resynth.BQSKIT) {
            resynth = Bqskit.socket(partitionQasm, resynthArgs.getBqskitOptLevel(), epsilon, Params.GATE_SET_RESYNTH_MAP.get(Params.GATE_SET));
        } else if (resynthAlg == Resynth.SYNTHETIQ) {
            resynth = Synthetiq.socket(partitionQasm, resynthArgs.getSynthetiqNumCircuits(), epsilon, resynthArgs.getSynthetiqThreads(), Params.GATE_SET_RESYNTH_MAP.get(Params.GATE_SET));
        // } else if (resynthAlg == Resynth.PYZX) {
        //     resynth = Pyzx.socket(partitionQasm);
        //     if (sentWhole) {
        //         return new Pair<>(CircuitParser.qasmToDag(resynth), "resynthesized whole circuit");
        //     }
        } else {
            throw new RuntimeException("unsupported resynth alg: " + resynthAlg);
        }

        CircuitDAG resynthDAG = CircuitParser.qasmToDag(resynth, partition.getQubitRenameMap());

        CircuitDAG copy = new CircuitDAG(circuit);
        replaceResynth(copy.getDag(), partition, resynthDAG);
        return new Pair<>(copy, partitionQasm + "\nresynthesized to\n" + resynth + "\nusing\n" + partition.getQubitRenameMap());
    }

    private Map<String, String> getMap(List<Node> symb, String findBefore, String findAfter) {
        Map<String, String> map = new HashMap<>();
        Node b = symb.get(0);
        Node a = symb.get(symb.size() - 1);

        map.put(b.getQubits().get(0), findBefore.substring(findBefore.indexOf("q"), findBefore.indexOf(";")));
        map.put(a.getQubits().get(0), findAfter.substring(findAfter.indexOf("q"), findAfter.indexOf(";")));
        return map;
    }

    private String getQasm(List<Node> symb) {
        String qasm = "";
        for (Node n : symb) {
            qasm = qasm + gateNodeToQasm(n) + "; ";
        }
        return qasm.trim();
    }

    private Circuit opsToCircuit(List<Node> ops) {
        Expr phi = new Real(1);
        TreeMap<String, Expr> f = new TreeMap<>();
        PathSum s = new PathSum(phi, f);
        ArrayList<PathSum> pathSum = new ArrayList<>(Arrays.asList(s));
        Circuit c = new Circuit(new ArrayList<>(), pathSum, new ArrayList<>());

        for (Node op : ops) {
            switch (op.getId()) {
                case "h": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.h(c, op.getQubits().get(0));
                    break;
                }
                case "sx": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.sx(c, op.getQubits().get(0));
                    break;
                }
                case "t": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.rz(c, op.getQubits().get(0), new BinOp(Expr.Op.DIV, new Symbol("pi"), new Real(4)));
                    break;
                }
                case "tdg": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.rz(c, op.getQubits().get(0), new UnOp(Expr.Op.MINUS, new BinOp(Expr.Op.DIV, new Symbol("pi"), new Real(4))));
                    break;
                }
                case "s": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.rz(c, op.getQubits().get(0), new BinOp(Expr.Op.DIV, new Symbol("pi"), new Real(2)));
                    break;
                }
                case "sdg": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.rz(c, op.getQubits().get(0), new UnOp(Expr.Op.MINUS, new BinOp(Expr.Op.DIV, new Symbol("pi"), new Real(2))));
                    break;
                }
                case "rz": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.rz(c, op.getQubits().get(0), op.getAngles().get(0));
                    break;
                }
                case "rx": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.rx(c, op.getQubits().get(0), op.getAngles().get(0));
                    break;
                }
                case "ry": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.ry(c, op.getQubits().get(0), op.getAngles().get(0));
                    break;
                }
                case "rxx": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    if (!c.hasQubit(op.getQubits().get(1))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(1), new Var(op.getQubits().get(1)));
                        }
                    }
                    PathSum.rxx(c, op.getQubits().get(0), op.getQubits().get(1), op.getAngles().get(0));
                    break;
                }
                case "u1": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.u1(c, op.getQubits().get(0), op.getAngles().get(0));
                    break;
                }
                case "u2": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.u2(c, op.getQubits().get(0), op.getAngles().get(0), op.getAngles().get(1));
                    break;
                }
                case "u3": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.u3(c, op.getQubits().get(0), op.getAngles().get(0), op.getAngles().get(1), op.getAngles().get(2));
                    break;
                }
                case "x": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.x(c, op.getQubits().get(0));
                    break;
                }
                case "z": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    PathSum.rz(c, op.getQubits().get(0), new Symbol("pi"));
                    break;
                }
                case "cx": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    if (!c.hasQubit(op.getQubits().get(1))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(1), new Var(op.getQubits().get(1)));
                        }
                    }
                    PathSum.cx(c, op.getQubits().get(0), op.getQubits().get(1));
                    break;
                }
                case "cz": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    if (!c.hasQubit(op.getQubits().get(1))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(1), new Var(op.getQubits().get(1)));
                        }
                    }
                    PathSum.cz(c, op.getQubits().get(0), op.getQubits().get(1));
                    break;
                }
                case "ccz": {
                    if (!c.hasQubit(op.getQubits().get(0))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(0), new Var(op.getQubits().get(0)));
                        }
                    }
                    if (!c.hasQubit(op.getQubits().get(1))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(1), new Var(op.getQubits().get(1)));
                        }
                    }
                    if (!c.hasQubit(op.getQubits().get(2))) {
                        for (PathSum symb : c.getPathSum()) {
                            symb.getF().put(op.getQubits().get(2), new Var(op.getQubits().get(2)));
                        }
                    }
                    PathSum.cx(c, op.getQubits().get(1), op.getQubits().get(2));
                    PathSum.rz(c, op.getQubits().get(2), new BinOp(Expr.Op.DIV, new BinOp(Expr.Op.MULT, new Real(7), new Symbol("pi")), new Real(4)));
                    PathSum.cx(c, op.getQubits().get(0), op.getQubits().get(2));
                    PathSum.rz(c, op.getQubits().get(2), new BinOp(Expr.Op.DIV, new Symbol("pi"), new Real(4)));
                    PathSum.cx(c, op.getQubits().get(1), op.getQubits().get(2));
                    PathSum.rz(c, op.getQubits().get(2), new BinOp(Expr.Op.DIV, new BinOp(Expr.Op.MULT, new Real(7), new Symbol("pi")), new Real(4)));
                    PathSum.cx(c, op.getQubits().get(0), op.getQubits().get(2));
                    PathSum.cx(c, op.getQubits().get(0), op.getQubits().get(1));
                    PathSum.rz(c, op.getQubits().get(1), new BinOp(Expr.Op.DIV, new BinOp(Expr.Op.MULT, new Real(7), new Symbol("pi")), new Real(4)));
                    PathSum.cx(c, op.getQubits().get(0), op.getQubits().get(1));
                    PathSum.rz(c, op.getQubits().get(0), new BinOp(Expr.Op.DIV, new Symbol("pi"), new Real(4)));
                    PathSum.rz(c, op.getQubits().get(1), new BinOp(Expr.Op.DIV, new Symbol("pi"), new Real(4)));
                    PathSum.rz(c, op.getQubits().get(2), new BinOp(Expr.Op.DIV, new Symbol("pi"), new Real(4)));
                    break;
                }
                default:
                    throw new RuntimeException(String.format("unimplemented gate: %s", op.getId()));
            }
        }

        return c;
    }

    public List<List<Node>> topoSort(DirectedMultigraph<Node, Edge> circuit) {
//        dagToQasm(circuit); // useful for sanity check when debugging
        List<List<Node>> layers = new ArrayList<>();
        Set<Node> added = new HashSet<>();
        Set<Node> vertices = new HashSet<>(circuit.vertexSet());

        while (added.size() != circuit.vertexSet().size()) {
            List<Node> verticesInLayer = new ArrayList<>();
            for (Node n : vertices) {
                List<Node> preds = Graphs.predecessorListOf(circuit, n);
                if (added.containsAll(preds)) {
                    verticesInLayer.add(n);
                }
            }
            vertices.removeAll(verticesInLayer);
            layers.add(verticesInLayer);
            added.addAll(verticesInLayer);
        }

        return layers;
    }

    private boolean matchAngles(Node circN, Node patternN, Map<String, Expr> angleMap) {
        Map<String, Expr> tempAngleMap = new HashMap<>();
        tempAngleMap.putAll(angleMap);
        boolean matchAngles = true;
        int i = 0;
        for (Expr angle : patternN.getAngles()) {
            String key = angle.toString();
            if (key.contains("theta")) {
                if (tempAngleMap.containsKey(key)) {
                    if (!sameAngle(tempAngleMap.get(key), circN.getAngles().get(i))) {
                        matchAngles = false;
                        break;
                    }
                } else {
                    tempAngleMap.put(key, circN.getAngles().get(i));
                }
            } else {
                if (!sameAngle(angle, circN.getAngles().get(i))) {
                    matchAngles = false;
                    break;
                }
            }
            i++;
        }
        if (matchAngles) {
            angleMap.clear();
            angleMap.putAll(tempAngleMap);
            return true;
        } else {
            return false;
        }
    }

    // TODO: deprecate once findSymb refactored
    public List<Node> getCircuitRoots(DirectedMultigraph<Node, Edge> dag) {
        List<Node> roots = new ArrayList<>();
        for (Node n : dag.vertexSet()) {
            if (n.isSourceQubit()) {
                roots.addAll(Graphs.successorListOf(dag, n));
            }
        }
        return roots;
    }

    public List<Node> findSymb(DirectedMultigraph<Node, Edge> circuit,
                               DirectedMultigraph<Node, Edge> patternBefore,
                               DirectedMultigraph<Node, Edge> patternAfter,
                               List<Map<boolean[], boolean[]>> constraints,
                               Map<String, Expr> angleMap,
                               int maxSymbQubits,
                               int maxSymbSize) {
        List<Node> beforeRoots = getCircuitRoots(patternBefore);
        Node patternBeforeStart = beforeRoots.get(0);
        List<Node> afterRoots = getCircuitRoots(patternAfter);
        Node patternAfterStart = afterRoots.get(0);
        boolean sameQubit = patternBeforeStart.getQubits().equals(patternAfterStart.getQubits());

        List<List<Node>> circuitTopo = topoSort(circuit);

        for (int i = 0; i < circuitTopo.size(); i++) {
            for (Node circN : circuitTopo.get(i)) {
                angleMap.clear();
                if (circN.isGate() && circN.getId().equals(patternBeforeStart.getId())) {
                    if (patternBeforeStart.getAngles() != null) {
                        if (!matchAngles(circN, patternBeforeStart, angleMap)) {
                            continue;
                        }
                    }
                    Node next = Graphs.successorListOf(patternBefore, patternBeforeStart).get(0);
                    int s = 1;
                    boolean foundMatchOuter = true;
                    while (!next.isSinkQubit()) {
                        boolean foundMatchInner = false;
                        for (Node circN3 : circuitTopo.get(i + s)) {
                            if (circN3.isGate() && circN3.getId().equals(next.getId()) && circN.getQubits().equals(circN3.getQubits())) {
                                if (next.getAngles() != null) {
                                    if (matchAngles(circN3, next, angleMap)) {
                                        foundMatchInner = true;
                                        break;
                                    }
                                } else {
                                    foundMatchInner = true;
                                    break;
                                }
                            }
                        }
                        if (!foundMatchInner) {
                            foundMatchOuter = false;
                            break;
                        }
                        next = Graphs.successorListOf(patternBefore, next).get(0);
                        s++;
                    }
                    if (!foundMatchOuter) {
                        continue;
                    }

                    Set<String> blockedQubits = new HashSet<>();
                    Set<String> trackedQubits = new HashSet<>();
                    List<Node> symb = new ArrayList<>();
                    List<Node> symbToReplace = new ArrayList<>();
                    symbToReplace.add(circN);
                    next = Graphs.successorListOf(patternBefore, patternBeforeStart).get(0);
                    Node circNext = Graphs.successorListOf(circuit, circN).get(0);
                    while (!next.isSinkQubit()) {
                        symbToReplace.add(circNext);
                        circNext = Graphs.successorListOf(circuit, circNext).get(0);
                        next = Graphs.successorListOf(patternBefore, next).get(0);
                    }
                    trackedQubits.add(circN.getQubits().get(0));
                    for (int j = i + s; j < circuitTopo.size(); j++) {
                        if (trackedQubits.size() > maxSymbQubits || symb.size() > maxSymbSize) {
                            break;
                        }
                        for (Node circN2 : circuitTopo.get(j)) {
                            if (!circN2.isGate()) {
                                continue;
                            }
                            Set<String> trackedIntersection = new HashSet<>(trackedQubits); // use the copy constructor
                            trackedIntersection.retainAll(circN2.getQubits());
                            if (!trackedIntersection.isEmpty()) {
                                trackedQubits.addAll(circN2.getQubits());

                                if (trackedQubits.size() > maxSymbQubits || symb.size() > maxSymbSize) {
                                    break;
                                }

                                boolean match = false;
                                if (sameQubit) {
                                    if (!blockedQubits.contains(circN2.getQubits().get(0)) && circN.getQubits().equals(circN2.getQubits()) && circN2.getId().equals(patternAfterStart.getId())) {
                                        if (patternAfterStart.getAngles() != null) {
                                            if (matchAngles(circN2, patternAfterStart, angleMap)) {
                                                Node nextA = Graphs.successorListOf(patternAfter, patternAfterStart).get(0);
                                                int t = 1;
                                                boolean foundMatchOuterA = true;
                                                while (!nextA.isSinkQubit()) {
                                                    boolean foundMatchInnerA = false;
                                                    for (Node circN4 : circuitTopo.get(j + t)) {
                                                        if (circN4.isGate() && circN4.getId().equals(nextA.getId()) && circN4.getQubits().equals(circN2.getQubits())) {
                                                            if (nextA.getAngles() != null) {
                                                                if (matchAngles(circN4, nextA, angleMap)) {
                                                                    foundMatchInnerA = true;
                                                                    break;
                                                                }
                                                            } else {
                                                                foundMatchInnerA = true;
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    if (!foundMatchInnerA) {
                                                        foundMatchOuterA = false;
                                                        break;
                                                    }
                                                    nextA = Graphs.successorListOf(patternAfter, nextA).get(0);
                                                    t++;
                                                }
                                                if (foundMatchOuterA) {
                                                    match = true;
                                                }
                                            }
                                        } else {
                                            Node nextA = Graphs.successorListOf(patternAfter, patternAfterStart).get(0);
                                            int t = 1;
                                            boolean foundMatchOuterA = true;
                                            while (!nextA.isSinkQubit()) {
                                                boolean foundMatchInnerA = false;
                                                for (Node circN4 : circuitTopo.get(j + t)) {
                                                    if (circN4.isGate() && circN4.getId().equals(nextA.getId()) && circN4.getQubits().equals(circN2.getQubits())) {
                                                        if (nextA.getAngles() != null) {
                                                            if (matchAngles(circN4, nextA, angleMap)) {
                                                                foundMatchInnerA = true;
                                                                break;
                                                            }
                                                        } else {
                                                            foundMatchInnerA = true;
                                                            break;
                                                        }
                                                    }
                                                }
                                                if (!foundMatchInnerA) {
                                                    foundMatchOuterA = false;
                                                    break;
                                                }
                                                nextA = Graphs.successorListOf(patternAfter, nextA).get(0);
                                                t++;
                                            }
                                            if (foundMatchOuterA) {
                                                match = true;
                                            }
                                        }
                                    }
                                } else {
                                    if (!blockedQubits.contains(circN2.getQubits().get(0)) && !circN.getQubits().equals(circN2.getQubits()) && circN2.getId().equals(patternAfterStart.getId())) {
                                        if (patternAfterStart.getAngles() != null) {
                                            if (matchAngles(circN2, patternAfterStart, angleMap)) {
                                                Node nextA = Graphs.successorListOf(patternAfter, patternAfterStart).get(0);
                                                int t = 1;
                                                boolean foundMatchOuterA = true;
                                                while (!nextA.isSinkQubit()) {
                                                    boolean foundMatchInnerA = false;
                                                    for (Node circN4 : circuitTopo.get(j + t)) {
                                                        if (circN4.isGate() && circN4.getId().equals(nextA.getId()) && circN4.getQubits().equals(circN2.getQubits())) {
                                                            if (nextA.getAngles() != null) {
                                                                if (matchAngles(circN4, nextA, angleMap)) {
                                                                    foundMatchInnerA = true;
                                                                    break;
                                                                }
                                                            } else {
                                                                foundMatchInnerA = true;
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    if (!foundMatchInnerA) {
                                                        foundMatchOuterA = false;
                                                        break;
                                                    }
                                                    nextA = Graphs.successorListOf(patternAfter, nextA).get(0);
                                                    t++;
                                                }
                                                if (foundMatchOuterA) {
                                                    match = true;
                                                }
                                            }
                                        } else {
                                            Node nextA = Graphs.successorListOf(patternAfter, patternAfterStart).get(0);
                                            int t = 1;
                                            boolean foundMatchOuterA = true;
                                            while (!nextA.isSinkQubit()) {
                                                boolean foundMatchInnerA = false;
                                                for (Node circN4 : circuitTopo.get(j + t)) {
                                                    if (circN4.isGate() && circN4.getId().equals(nextA.getId()) && circN4.getQubits().equals(circN2.getQubits())) {
                                                        if (nextA.getAngles() != null) {
                                                            if (matchAngles(circN4, nextA, angleMap)) {
                                                                foundMatchInnerA = true;
                                                                break;
                                                            }
                                                        } else {
                                                            foundMatchInnerA = true;
                                                            break;
                                                        }
                                                    }
                                                }
                                                if (!foundMatchInnerA) {
                                                    foundMatchOuterA = false;
                                                    break;
                                                }
                                                nextA = Graphs.successorListOf(patternAfter, nextA).get(0);
                                                t++;
                                            }
                                            if (foundMatchOuterA) {
                                                match = true;
                                            }
                                        }
                                    }
                                }

                                boolean goToElse = false;
                                if (match) {
                                    Circuit symbCirc = opsToCircuit(symb);
                                    if (symbCirc.getUsedQubits().size() <= maxSymbQubits) {
                                        for (Map<boolean[], boolean[]> constraint : constraints) {
                                            boolean satisfiesConstraint = true;
                                            for (Map.Entry<boolean[], boolean[]> e : constraint.entrySet()) {
                                                Map<String, Integer> qubitMap = new HashMap<>();
                                                Map<String, Boolean> expectedMap = new HashMap<>();
                                                if (patternBeforeStart.getQubits().get(0).equals("q0")) {
                                                    qubitMap.put(circN.getQubits().get(0), e.getKey()[0] ? 1 : 0);
                                                    expectedMap.put(circN.getQubits().get(0), e.getValue()[0]);
                                                }
                                                if (patternAfterStart.getQubits().get(0).equals("q0")) {
                                                    qubitMap.put(circN2.getQubits().get(0), e.getKey()[0] ? 1 : 0);
                                                    expectedMap.put(circN2.getQubits().get(0), e.getValue()[0]);
                                                }
                                                if (patternBeforeStart.getQubits().get(0).equals("q1")) {
                                                    qubitMap.put(circN.getQubits().get(0), e.getKey()[1] ? 1 : 0);
                                                    expectedMap.put(circN.getQubits().get(0), e.getValue()[1]);
                                                }
                                                if (patternAfterStart.getQubits().get(0).equals("q1")) {
                                                    qubitMap.put(circN2.getQubits().get(0), e.getKey()[1] ? 1 : 0);
                                                    expectedMap.put(circN2.getQubits().get(0), e.getValue()[1]);
                                                }
                                                if (!verifier.verify(symbCirc, qubitMap, expectedMap)) {
                                                    satisfiesConstraint = false;
                                                    break;
                                                }
                                            }

                                            if (satisfiesConstraint) {
                                                symbToReplace.add(circN2);
                                                Node nextA = Graphs.successorListOf(patternAfter, patternAfterStart).get(0);
                                                Node circNextA = Graphs.successorListOf(circuit, circN2).get(0);
                                                while (!nextA.isSinkQubit()) {
                                                    symbToReplace.add(circNextA);
                                                    circNextA = Graphs.successorListOf(circuit, circNextA).get(0);
                                                    nextA = Graphs.successorListOf(patternAfter, nextA).get(0);
                                                }
                                                return symbToReplace;
                                            }
                                        }
                                    }
                                    goToElse = true;
                                } else {
                                    goToElse = true;
                                }

                                if (goToElse) {
                                    if (circN2.getId().equals("h")) {
                                        blockedQubits.add(circN2.getQubits().get(0));
//                                        removeSuccs(circuit, symb, circN2);
                                        symbToReplace.add(circN2);
                                    } else {
                                        Set<String> blockedIntersection = new HashSet<>(blockedQubits);
                                        blockedIntersection.retainAll(circN2.getQubits());
                                        if (!symbToReplace.contains(circN2) && blockedIntersection.isEmpty()) {
                                            symb.add(circN2);
                                            symbToReplace.add(circN2);
                                        } else {
                                            if (circN2.isCCZ()) {
                                                if (!blockedQubits.contains(circN2.getQubits().get(2))) {
                                                    // target in border, exclude qubit and successors
                                                    blockedQubits.add(circN2.getQubits().get(2));
                                                    symbToReplace.add(circN2);
//                                                removeSuccs(circuit, symb, circN2);
                                                } else if (blockedQubits.contains(circN2.getQubits().get(2))) {
                                                    // control in border, ignore
                                                    symbToReplace.add(circN2);
                                                }
                                            } else if (circN2.isCX()) {
                                                if (blockedQubits.contains(circN2.getQubits().get(0))) {
                                                    // target in border, exclude qubit and successors
                                                    blockedQubits.add(circN2.getQubits().get(1));
                                                    symbToReplace.add(circN2);
//                                                removeSuccs(circuit, symb, circN2);
                                                } else if (blockedQubits.contains(circN2.getQubits().get(1))) {
                                                    // control in border, ignore
                                                    symbToReplace.add(circN2);
                                                }
                                            } else {
                                                symbToReplace.add(circN2);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return null;
    }

//    private void removeSuccs(DirectedMultigraph<qoptimizer.circuit.Node, qoptimizer.circuit.Edge> circuit, List<qoptimizer.circuit.Node> symb, qoptimizer.circuit.Node succ) {
//        TopologicalOrderIterator<qoptimizer.circuit.Node, qoptimizer.circuit.Edge> circIter = new TopologicalOrderIterator<>(circuit);
//
//        boolean remove = false;
//        while (circIter.hasNext()) {
//            qoptimizer.circuit.Node next = circIter.next();
//            if (!next.isGate()) {
//                continue;
//            }
//            if (next == succ) {
//                remove = true;
//            }
//            if (remove) {
//                String blockedQubit = succ.getQubits().get(0);
//                if (blockedQubit.equals(next.getQubits().get(0)) || blockedQubit.equals(next.getQubits().get(1))) {
//                    symb.remove(next);
//                }
//                if (succ.isCX()) {
//                    String blockedQubit2 = succ.getQubits().get(1);
//                    if (blockedQubit2.equals(next.getQubits().get(0)) || blockedQubit2.equals(next.getQubits().get(1))) {
//                        symb.remove(next);
//                    }
//                }
//            }
//        }
//    }

    public List<Map<boolean[], boolean[]>> parseConstraints(String stringConstraints) {
        List<Map<boolean[], boolean[]>> constraints = new ArrayList<>();
        String[] splitOuter = stringConstraints.split("},");
        for (String out : splitOuter) {
            String[] splitInner = out.split("],");
            Map<boolean[], boolean[]> constraint = new HashMap<>();
            for (String in : splitInner) {
                in = in.replace("]=[", ", ");
                in = in.replace("[", "");
                in = in.replace("]", "");
                in = in.replace("{", "");
                in = in.replace("}", "");
                in = in.trim();

                boolean[] key = new boolean[2];
                boolean[] val = new boolean[2];
                String[] bools = in.split(", ");

                key[0] = Boolean.parseBoolean(bools[0]);
                key[1] = Boolean.parseBoolean(bools[1]);
                val[0] = Boolean.parseBoolean(bools[2]);
                val[1] = Boolean.parseBoolean(bools[3]);

                constraint.put(key, val);
            }
            constraints.add(constraint);
        }

        return constraints;
    }


    private boolean validRule(String pattern,
                              String replace,
                              CircuitDAG patternDag,
                              CircuitDAG replaceDag,
                              boolean removeSizePreservingRules,
                              int maxRuleQubits,
                              boolean preserveMapping) {
        if (removeSizePreservingRules) {
            if (StringUtils.countMatches(pattern, ";") == StringUtils.countMatches(replace, ";")) {
                return false;
            }
        }

        if (pattern.contains("+") || pattern.contains("-")) {
            return false;
        }

        Set<String> patternQubits = patternDag.getQubits();

        if (maxRuleQubits != -1 && patternQubits.size() > maxRuleQubits) {
            return false;
        }

        if (replace.contains("theta1") && !pattern.contains("theta1")) {
            return false;
        }
        if (replace.contains("theta2") && !pattern.contains("theta2")) {
            return false;
        }
        if (replace.contains("theta3") && !pattern.contains("theta3")) {
            return false;
        }
        if (replace.contains("theta4") && !pattern.contains("theta4")) {
            return false;
        }

        if (!GraphTests.isConnected(patternDag.getDag())) {
            return false;
        }

        Set<String> replaceQubits = replaceDag.getQubits();
        if (!patternQubits.containsAll(replaceQubits)) {
            return false;
        }

        if (patternDag.getDagHash() == replaceDag.getDagHash()) {
            return false;
        }

        if (preserveMapping) {
            Set<String> patternConnectivity = patternDag.getConnectivity();
            Set<String> replaceConnectivity = replaceDag.getConnectivity();
            if (!patternConnectivity.containsAll(replaceConnectivity)) {
                return false;
            }
        }

        return true;
    }

    private boolean validSymbRule(String patternBeforeSymb,
                                  String patternAfterSymb,
                                  String replaceBeforeSymb,
                                  String replaceAfterSymb,
                                  boolean useSizePreservingSymbRules,
                                  int maxRuleQubits,
                                  boolean preserveMapping) {
        if (patternBeforeSymb.isBlank() || patternAfterSymb.isBlank()) {
            return false;
        }

        String pattern = patternBeforeSymb + patternAfterSymb;
        String replace = replaceBeforeSymb + replaceAfterSymb;

        if (!useSizePreservingSymbRules) {
            if (StringUtils.countMatches(pattern, ";") == StringUtils.countMatches(replace, ";")) {
                return false;
            }
        }

        if (pattern.contains("+") || pattern.contains("-")) {
            return false;
        }

        if (replace.contains("theta1") && !pattern.contains("theta1")) {
            return false;
        }
        if (replace.contains("theta2") && !pattern.contains("theta2")) {
            return false;
        }
        if (replace.contains("theta3") && !pattern.contains("theta3")) {
            return false;
        }
        if (replace.contains("theta4") && !pattern.contains("theta4")) {
            return false;
        }

        CircuitDAG patternBeforeSymbDag = CircuitParser.qasmToDag(patternBeforeSymb);
        CircuitDAG patternAfterSymbDag = CircuitParser.qasmToDag(patternAfterSymb);
        CircuitDAG replaceBeforeSymbDag = CircuitParser.qasmToDag(replaceBeforeSymb);
        CircuitDAG replaceAfterSymbDag = CircuitParser.qasmToDag(replaceAfterSymb);
        Set<String> patternQubits = patternBeforeSymbDag.getQubits();
        patternQubits.addAll(patternAfterSymbDag.getQubits());

        if (maxRuleQubits != -1 && patternQubits.size() > maxRuleQubits) {
            return false;
        }

        Set<String> replaceQubits = replaceBeforeSymbDag.getQubits();
        replaceQubits.addAll(replaceAfterSymbDag.getQubits());
        if (!patternQubits.containsAll(replaceQubits)) {
            return false;
        }

        if (!GraphTests.isConnected(patternBeforeSymbDag.getDag())) {
            return false;
        }
        if (!GraphTests.isConnected(patternAfterSymbDag.getDag())) {
            return false;
        }

        if (preserveMapping) {
            Set<String> patternConnectivity = patternBeforeSymbDag.getConnectivity();
            patternConnectivity.addAll(patternAfterSymbDag.getConnectivity());
            Set<String> replaceConnectivity = replaceBeforeSymbDag.getConnectivity();
            replaceConnectivity.addAll(replaceAfterSymbDag.getConnectivity());
            if (!patternConnectivity.containsAll(replaceConnectivity)) {
                return false;
            }
        }

        return true;
    }

    private List<Pair<CircuitDAG, String>> getRules(String filename, boolean removeSizePreservingRules, int maxRuleQubits, boolean preserveMapping, boolean addSizePreserveReflection, boolean addSizeIncrease, OptObj optObj) throws IOException {
        java.util.List<String> rules = Files.readAllLines(Path.of(filename));
        List<Pair<CircuitDAG, String>> result = new ArrayList<>();

        for (String rule : rules) {
            String[] splitRule = rule.split(" \\| ");
            CircuitDAG lhs = CircuitParser.qasmToDag(splitRule[0]);
            CircuitDAG rhs = CircuitParser.qasmToDag(splitRule[1]);

            if (optObj == OptObj.TOTAL_IGNORE_RZ) {
                if (StringUtils.countMatches(splitRule[0], ";") == StringUtils.countMatches(splitRule[1], ";")) {
                    if (validRule(splitRule[0], splitRule[1], lhs, rhs, removeSizePreservingRules, maxRuleQubits, preserveMapping) && StringUtils.countMatches(splitRule[0], "rz") < StringUtils.countMatches(splitRule[1], "rz")) {
                        result.add(new Pair<>(lhs, splitRule[1] + " | " + splitRule[0]));
                    } else {
                        if (validRule(splitRule[1], splitRule[0], rhs, lhs, removeSizePreservingRules, maxRuleQubits, preserveMapping)) {
                            result.add(new Pair<>(rhs, rule));
                        }
                    }
                } else {
                    if (validRule(splitRule[1], splitRule[0], rhs, lhs, removeSizePreservingRules, maxRuleQubits, preserveMapping)) {
                        result.add(new Pair<>(rhs, rule));
                    }
                }
            } else {
                if (validRule(splitRule[1], splitRule[0], rhs, lhs, removeSizePreservingRules, maxRuleQubits, preserveMapping)) {
                    result.add(new Pair<>(rhs, rule));
                }
            }

            if (addSizePreserveReflection) {
                if (StringUtils.countMatches(splitRule[0], ";") == StringUtils.countMatches(splitRule[1], ";")) {
                    if (validRule(splitRule[0], splitRule[1], lhs, rhs, removeSizePreservingRules, maxRuleQubits, preserveMapping)) {
                        result.add(new Pair<>(lhs, splitRule[1] + " | " + splitRule[0]));
                    }
                }
            }

            if (addSizeIncrease) {
                if (StringUtils.countMatches(splitRule[0], ";") < StringUtils.countMatches(splitRule[1], ";")) {
                    if (validRule(splitRule[0], splitRule[1], lhs, rhs, removeSizePreservingRules, maxRuleQubits, preserveMapping)) {
                        result.add(new Pair<>(lhs, splitRule[1] + " | " + splitRule[0]));
                    }
                }
            }
        }

        return result;
    }

    private List<String> getSymbRules(String pathSymb, boolean useSizePreservingSymbRules, int maxRuleQubits, boolean preserveMapping, boolean addSizePreserveReflection) throws IOException {
        List<String> rulesSymb = Files.readAllLines(Path.of(pathSymb));
        List<String> pruned = new ArrayList<>();

        for (String rule : rulesSymb) {
            String[] splitRule = rule.split(" \\| ");

            String find = splitRule[1];
            String replace = splitRule[0];

            int findSymbIndex = find.indexOf("symb");
            String findBeforeSymb = StringUtils.stripStart(find.substring(0, findSymbIndex).trim(), ";");
            String findAfterSymb = StringUtils.stripStart(find.substring(find.indexOf(";", findSymbIndex)).trim(), ";").trim();
            int replaceSymbIndex = replace.indexOf("symb");
            String replaceBeforeSymb = StringUtils.stripStart(replace.substring(0, replaceSymbIndex).trim(), ";");
            String replaceAfterSymb = StringUtils.stripStart(replace.substring(replace.indexOf(";", replaceSymbIndex)).trim(), ";").trim();

            if (validSymbRule(findBeforeSymb, findAfterSymb, replaceBeforeSymb, replaceAfterSymb, useSizePreservingSymbRules, maxRuleQubits, preserveMapping)) {
                pruned.add(rule);
            }

            if (useSizePreservingSymbRules && addSizePreserveReflection) {
                if (StringUtils.countMatches(find, ";") == StringUtils.countMatches(replace, ";")) {
                    if (validSymbRule(replaceBeforeSymb, replaceAfterSymb, findBeforeSymb, findAfterSymb, useSizePreservingSymbRules, maxRuleQubits, preserveMapping)) {
                        pruned.add(splitRule[1] + " | " + splitRule[0] + " | " + splitRule[2]);
                    }
                }
            }
        }
        return pruned;
    }

    public OptCircuit optimizeStochastic(OptCircuit circuit,
                                         String filename,
                                         String ruleFileName,
                                         String symbRuleFileName,
                                         Path output) throws IOException, InterruptedException, ExecutionException {
        List<Pair<CircuitDAG, String>> rules = getRules(ruleFileName, Params.REMOVE_SIZE_PRESERVING_RULES, Params.MAX_RULE_QUBITS, Params.PRESERVE_MAPPING, Params.USE_SIZE_PRESERVE_RULE_REFLECTION, Params.USE_SIZE_INCREASING_RULES, Params.OPTIMIZATION_OBJECTIVE);
        List<String> rulesSymb = getSymbRules(symbRuleFileName, Params.USE_SIZE_PRESERVING_SYMB_RULES, Params.MAX_RULE_QUBITS, Params.PRESERVE_MAPPING, Params.USE_SIZE_PRESERVE_RULE_REFLECTION);
        Params.setResynthWeight(rules.size() + rulesSymb.size());

        if (Params.VERBOSITY >= 2) {
            System.out.println("Total rules: %s, normal: %s, symb: %s".formatted(rules.size() + rulesSymb.size(), rules.size(), rulesSymb.size()));
        }

        OptCircuit currentCircuit = circuit;
        OptCircuit bestCircuit = circuit;
        long timeToBest = 0;
        long timeStart = System.currentTimeMillis();
        Map<String, Integer> ruleCount = new HashMap<>();

        int min = 0;
        int max = rules.size() + rulesSymb.size() + 1;
        Random rand = new Random(new Random(Params.SEED).nextInt());

        Future<Pair<CircuitDAG, String>> resynthThread = null;
        double acceptP = 1;

        while ((Params.TEMPERATURE > 1) || (Params.COOLING_RATE == 0)) {
            int ruleToUse = rand.nextInt(max - min) + min;

            OptCircuit candidate = currentCircuit;

            if (resynthThread != null && resynthThread.isDone()) {
                Pair<CircuitDAG, String> resynthResult = resynthThread.get();
                var rulesApplied = new ArrayList<>(currentCircuit.getRulesApplied());
                candidate = new OptCircuit(resynthResult.getFirst(), rulesApplied, System.currentTimeMillis(), timeToBest);
//                rulesApplied.add(new Pair(resynthResult.getSecond(), candidate.getCircuit().totalGateCount()));
                resynthThread = null;
            } else {
                if (ruleToUse < rules.size()) {
                    Pair<CircuitDAG, String> rule = rules.get(ruleToUse);
                    String[] splitRule = rule.getSecond().split(" \\| ");
                    var rulesApplied = new ArrayList<>(currentCircuit.getRulesApplied());
                    candidate = new OptCircuit(applyRule(currentCircuit.getCircuit(), splitRule[0], rule.getFirst(), Params.APPLY_ONCE, rand), rulesApplied, System.currentTimeMillis(), timeToBest);
//                    rulesApplied.add(new Pair(rule.getSecond(), candidate.getCircuit().totalGateCount()));
                } else if (ruleToUse < rules.size() + rulesSymb.size()) {
                    String ruleSymb = rulesSymb.get(ruleToUse - rules.size());
                    String[] splitRule = ruleSymb.split(" \\| ");

                    String find = splitRule[1];
                    String replace = splitRule[0];

                    int findSymbIndex = find.indexOf("symb");
                    String findBeforeSymb = StringUtils.stripStart(find.substring(0, findSymbIndex).trim(), ";");
                    String findAfterSymb = StringUtils.stripStart(find.substring(find.indexOf(";", findSymbIndex)).trim(), ";").trim();
                    int replaceSymbIndex = replace.indexOf("symb");
                    String replaceBeforeSymb = StringUtils.stripStart(replace.substring(0, replaceSymbIndex).trim(), ";");
                    String replaceAfterSymb = StringUtils.stripStart(replace.substring(replace.indexOf(";", replaceSymbIndex)).trim(), ";").trim();

                    var rulesApplied = new ArrayList<>(currentCircuit.getRulesApplied());
                    candidate = new OptCircuit(applySymbRule(currentCircuit.getCircuit(), findBeforeSymb, findAfterSymb, replaceBeforeSymb, replaceAfterSymb, parseConstraints(splitRule[2]), Params.MAX_SYMB_QUBITS, Params.MAX_SYMB_SIZE, Params.APPLY_ONCE, rand), rulesApplied, System.currentTimeMillis(), timeToBest);
//                    rulesApplied.add(new Pair(ruleSymb, candidate.getCircuit().totalGateCount()));
                } else {
                    if (Params.RESYNTH_ALG == Resynth.NONE) {
                        continue;
                    }
                    if (resynthThread == null) {
                        if (currentCircuit.countResynthApplications() < Params.MAX_RESYNTH_ALLOWED || Params.MAX_RESYNTH_ALLOWED == -1) {
                            CircuitDAG input = currentCircuit.getCircuit();
                            resynthThread = ThreadUtils.submit(() -> applyResynth(Params.RESYNTH_ALG, input, rand, Params.EPSILON / Params.MAX_RESYNTH_ALLOWED, Params.RESYNTH_ARGS));
                            continue;
                        }
                    }
                }
            }

            int candidateSize = candidate.getCircuit().cost(Params.OPTIMIZATION_OBJECTIVE);
            int currentSize = currentCircuit.getCircuit().cost(Params.OPTIMIZATION_OBJECTIVE);

            if (Params.COOLING_RATE == 0) { // MCMC
                acceptP = Math.min(1, Math.exp(-Params.TEMPERATURE * (candidateSize / currentSize))); // integer division here is truncated
            } else { // Simulated Annealing
                acceptP = saAcceptanceProbability(currentSize, candidateSize, Params.TEMPERATURE);
            }

            if (rand.nextDouble() <= acceptP) {
                currentCircuit = candidate;
            }

            if (lessThanCurrBest(currentCircuit.getCircuit(), bestCircuit.getCircuit(), Params.OPTIMIZATION_OBJECTIVE)) {
                bestCircuit = currentCircuit;
                timeToBest = ((System.currentTimeMillis() - timeStart) / 1000);
                bestCircuit.setTimeToBest(timeToBest);

                String qasm = CircuitParser.dagToQasm(bestCircuit.getCircuit());

                logIntermediateInfo(bestCircuit, ruleCount, (System.currentTimeMillis() - timeStart) / 1000.0, false);
                writeCircuitToFile(bestCircuit.getCircuit(), output + String.format("/latest_sol_%s_%s", Params.JOB_INFO, filename));
            }

            Params.TEMPERATURE *= 1 - Params.COOLING_RATE;
        }

        return bestCircuit;
    }

    public static double saAcceptanceProbability(int currentDistance, int newDistance, double temperature) {
        if (newDistance < currentDistance) {
            return 1.0;
        }

        return Math.exp((currentDistance - newDistance) / temperature);
    }

    private double[] softmax(List<Integer> weights, double temperature) {
        double[] probs = new double[weights.size()];
        double sum = 0;
        int max = weights.stream().max(Integer::compare).get();
        for (int i = 0; i < weights.size(); i++) {
            probs[i] = Math.exp((weights.get(i) - max) / temperature);
            sum += probs[i];
        }

        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        return probs;
    }

    private int sampleIndex(double[] distribution, Random random) {
        double rand = random.nextDouble();
        double cumulativeProb = 0;
        for (int i = 0; i < distribution.length; i++) {
            cumulativeProb += distribution[i];
            if (rand <= cumulativeProb) {
                return i;
            }
        }

        return distribution.length - 1;
    }

    private int sampleSoftmax(List<Integer> weights, double temperature, Random random) {
        double[] probs = softmax(weights, temperature);
        return sampleIndex(probs, random);
    }

    private OptCircuit dequeueCircuit(PriorityQueue<OptCircuit> q, double temperature, OptObj optObj, Random random) {
        if (temperature == 0) {
            return q.poll();
        }

        List<OptCircuit> qList = new ArrayList<>(q);
        List<Integer> weights = qList.stream().map(c -> -c.getCircuit().cost(optObj)).collect(Collectors.toList());
        int index = sampleSoftmax(weights, temperature, random);
        q.remove(qList.get(index));
        return qList.get(index);
    }

    private Pair<CircuitDAG, String> sampleRule(List<Pair<CircuitDAG, String>> rules, double temperature, OptCircuit bestCircuit, HashMap<String, Integer> rulesCount, Random random) {
        List<Integer> weights = rules.stream().map(r -> scoreRule(r.getSecond(), bestCircuit, rulesCount)).collect(Collectors.toList());
        if (temperature == 0) {
            int maxWeight = weights.stream().max(Integer::compare).get();
            return rules.get(weights.indexOf(maxWeight));
        }
        int index = sampleSoftmax(weights, temperature, random);
        return rules.get(index);
    }

    private void sampleRules(List<Pair<CircuitDAG, String>> rules, List<Pair<CircuitDAG, String>> rulesToUse, long numRulesToUse, double temperature, OptCircuit bestCircuit, HashMap<String, Integer> rulesCount, Random random) {
        var tempRules = new ArrayList<>(rules);
        for (int i = 0; i < numRulesToUse; i++) {
            Pair<CircuitDAG, String> rule = sampleRule(tempRules, temperature, bestCircuit, rulesCount, random);
            rulesToUse.add(rule);
            tempRules.remove(rule);
        }
    }

    private String getRule(int index, List<Pair<CircuitDAG, String>> rulesNormal, List<String> rulesSymb) {
        if (index < rulesNormal.size()) {
            return rulesNormal.get(index).getSecond();
        } else if (index < rulesSymb.size() + rulesSymb.size()) {
            return rulesSymb.get(index - rulesSymb.size());
        } else {
            return "resynth";
        }
    }

    private void sampleRulesInt(List<Integer> rules, List<Integer> rulesToUse, long numRulesToUse, double temperature, OptCircuit bestCircuit, HashMap<String, Integer> rulesCount, Random random, List<Pair<CircuitDAG, String>> rulesNormal, List<String> rulesSymb) {
        var tempRules = new ArrayList<>(rules);
        for (int i = 0; i < numRulesToUse; i++) {
            int rule = 0;
            List<Integer> weights = rules.stream().map(r -> scoreRule(getRule(r, rulesNormal, rulesSymb), bestCircuit, rulesCount)).collect(Collectors.toList());
            if (temperature == 0) {
                int maxWeight = weights.stream().max(Integer::compare).get();
                rule = rules.get(weights.indexOf(maxWeight));
                tempRules.remove(weights.indexOf(maxWeight));
            } else {
                int index = sampleSoftmax(weights, temperature, random);
                rule = rules.get(index);
                tempRules.remove(index);
            }
            rulesToUse.add(rule);
        }
    }

    private int scoreRule(String rule, OptCircuit bestCircuit, HashMap<String, Integer> rulesApplied) {
        int numAppliedTotal = rulesApplied.getOrDefault(rule, 0);
        int numAppliedBest = bestCircuit.countRulesApplications(rule);

        return numAppliedTotal + (2 * numAppliedBest);
    }

    public OptCircuit optimizeBeam(OptCircuit circuit,
                                   HashMap<String, Integer> ruleCount,
                                   boolean onlySymb,
                                   String filename,
                                   String ruleFileName,
                                   String symbRuleFileName,
                                   Path output) throws IOException, ExecutionException, InterruptedException {
        List<Pair<CircuitDAG, String>> rules = null;
        if (!onlySymb) {
            rules = getRules(ruleFileName, Params.REMOVE_SIZE_PRESERVING_RULES, Params.MAX_RULE_QUBITS, Params.PRESERVE_MAPPING, Params.USE_SIZE_PRESERVE_RULE_REFLECTION, Params.USE_SIZE_INCREASING_RULES, Params.OPTIMIZATION_OBJECTIVE);
        }
        List<String> rulesSymb = getSymbRules(symbRuleFileName, Params.USE_SIZE_PRESERVING_SYMB_RULES, Params.MAX_RULE_QUBITS, Params.PRESERVE_MAPPING, Params.USE_SIZE_PRESERVE_RULE_REFLECTION);
        Params.setResynthWeight(rules.size() + rulesSymb.size());

        if (Params.VERBOSITY >= 2) {
            System.out.println("Total rules: %s, normal: %s, symb: %s".formatted(rules.size() + rulesSymb.size(), rules.size(), rulesSymb.size()));
        }

        OptCircuit bestCircuit = circuit;
        Set<Integer> seen = new HashSet<>();
        seen.add(circuit.hashCode());
        PriorityQueue<OptCircuit> q = new PriorityQueue<>(new OptCircuitComparator(Params.OPTIMIZATION_OBJECTIVE));
        q.add(circuit);
        long timeStart = System.currentTimeMillis();

        Future<Pair<CircuitDAG, String>> resynthThread = null;

        int iters = 0;
        double numRulesToUse = rules.size();
        double numSymbRulesToUse = rulesSymb.size();
        Random rand = new Random(new Random(Params.SEED).nextInt());

        while (!q.isEmpty()) {
            iters++;

            // update best circuit if necessary
            OptCircuit c = q.peek();
            if (lessThanCurrBest(c.getCircuit(), bestCircuit.getCircuit(), Params.OPTIMIZATION_OBJECTIVE)) {
                bestCircuit = c;
                bestCircuit.setTimeToBest(((System.currentTimeMillis() - timeStart) / 1000));

                String qasm = CircuitParser.dagToQasm(bestCircuit.getCircuit());

                logIntermediateInfo(bestCircuit, ruleCount, (System.currentTimeMillis() - timeStart) / 1000.0, true);
                writeCircuitToFile(bestCircuit.getCircuit(), output + String.format("/latest_sol_%s_%s", Params.JOB_INFO, filename));
            }

            c = dequeueCircuit(q, Params.TEMPERATURE, Params.OPTIMIZATION_OBJECTIVE, rand);

            // Process rules (potentially prune during search)
            List<Pair<CircuitDAG, String>> rulesToUse = new ArrayList<>();
            List<String> rulesToUseSymb = new ArrayList<>();

            if (Params.COOLING_RATE == 0 || iters < Params.ITERS_BEFORE_PRUNE || (System.currentTimeMillis() - timeStart) / 1000 < Params.SECS_BEFORE_PRUNE) {
                rulesToUse = rules;
                rulesToUseSymb = rulesSymb;
            } else {
                sampleRules(rules, rulesToUse, Math.round(numRulesToUse), Params.PRUNE_TEMPERATURE, bestCircuit, ruleCount, rand);

                var tempSymbRules = rulesSymb.stream().map(r -> new Pair<CircuitDAG, String>(null, r)).collect(Collectors.toList());
                List<Pair<CircuitDAG, String>> sampledSymbRules = new ArrayList<>();
                sampleRules(tempSymbRules, sampledSymbRules, Math.round(numSymbRulesToUse), Params.PRUNE_TEMPERATURE, bestCircuit, ruleCount, rand);
                rulesToUseSymb = sampledSymbRules.stream().map(Pair::getSecond).collect(Collectors.toList());

                numRulesToUse = numRulesToUse * (1 - Params.COOLING_RATE);
                numSymbRulesToUse = numSymbRulesToUse * (1 - Params.COOLING_RATE);
            }

            // prune queue if too large
            if (q.size() > (Params.QUEUE_SIZE + 1000)) {
                PriorityQueue<OptCircuit> pruned = new PriorityQueue<>(new OptCircuitComparator(Params.OPTIMIZATION_OBJECTIVE));
                while (pruned.size() != Params.QUEUE_SIZE) {
                    pruned.add(q.poll());
                }
                q = pruned;
            }

            logBeamSearchIterInfo(filename, q.size(), seen.size(), bestCircuit, c, (System.currentTimeMillis() - timeStart) / 1000, rulesToUse.size(), rulesToUseSymb.size());

            // qoptimizer.config.Resynth "rule"
            if (Params.RESYNTH_ALG != Resynth.NONE) {
                if (resynthThread == null && c.countResynthApplications() < Params.MAX_RESYNTH_ALLOWED || Params.MAX_RESYNTH_ALLOWED == -1) {
                    CircuitDAG input = c.getCircuit();
                    resynthThread = ThreadUtils.submit(() -> applyResynth(Params.RESYNTH_ALG, input, rand, Params.EPSILON / Params.MAX_RESYNTH_ALLOWED, Params.RESYNTH_ARGS));
                } else {
                    if (resynthThread.isDone()) {
                        Pair<CircuitDAG, String> resynthResult = resynthThread.get();
                        CircuitDAG resynthCirc = resynthResult.getFirst();
                        int hashb = resynthCirc.getDagHash();
                        if (leqCurrBest(resynthCirc, bestCircuit.getCircuit(), Params.OPTIMIZATION_OBJECTIVE) && !seen.contains(hashb)) {
                            OptCircuit newOptCirc = new OptCircuit(resynthCirc, c, resynthResult.getSecond(), System.nanoTime());
                            q.add(newOptCirc);
                        }
                        seen.add(hashb);
                        resynthThread = null;
                    }
                }
            }

            // Normal rules
            if (!onlySymb) {
                for (Pair<CircuitDAG, String> rule : rulesToUse) {
                    String[] splitRule = rule.getSecond().split(" \\| ");
                    CircuitDAG result = applyRule(c.getCircuit(), splitRule[0], rule.getFirst(), Params.APPLY_ONCE, rand);

                    if (result == c.getCircuit()) {
                        continue;
                    }

                    int hash = result.getDagHash();
                    if (leqCurrBest(result, bestCircuit.getCircuit(), Params.OPTIMIZATION_OBJECTIVE) && !seen.contains(hash)) {
                        if (ruleCount.containsKey(rule.getSecond())) {
                            ruleCount.put(rule.getSecond(), ruleCount.get(rule.getSecond()) + 1);
                        } else {
                            ruleCount.put(rule.getSecond(), 1);
                        }
                        OptCircuit newOptCirc = new OptCircuit(result, c, rule.getSecond(), System.nanoTime());
                        q.add(newOptCirc);
                    }
                    seen.add(hash);
                }
            }

            // Symbolic rules
            for (String ruleSymb : rulesToUseSymb) {
                String[] splitRule = ruleSymb.split(" \\| ");

                String find = splitRule[1];
                String replace = splitRule[0];

                int findSymbIndex = find.indexOf("symb");
                String findBeforeSymb = StringUtils.stripStart(find.substring(0, findSymbIndex).trim(), ";");
                String findAfterSymb = StringUtils.stripStart(find.substring(find.indexOf(";", findSymbIndex)).trim(), ";").trim();
                int replaceSymbIndex = replace.indexOf("symb");
                String replaceBeforeSymb = StringUtils.stripStart(replace.substring(0, replaceSymbIndex).trim(), ";");
                String replaceAfterSymb = StringUtils.stripStart(replace.substring(replace.indexOf(";", replaceSymbIndex)).trim(), ";").trim();

                CircuitDAG result = applySymbRule(c.getCircuit(), findBeforeSymb, findAfterSymb, replaceBeforeSymb, replaceAfterSymb, parseConstraints(splitRule[2]), Params.MAX_SYMB_QUBITS, Params.MAX_SYMB_SIZE, Params.APPLY_ONCE, rand);

                if (result == c.getCircuit()) {
                    continue;
                }

                int hash = result.getDagHash();
                if (leqCurrBest(result, bestCircuit.getCircuit(), Params.OPTIMIZATION_OBJECTIVE) && !seen.contains(hash)) {
                    if (ruleCount.containsKey(ruleSymb)) {
                        ruleCount.put(ruleSymb, ruleCount.get(ruleSymb) + 1);
                    } else {
                        ruleCount.put(ruleSymb, 1);
                    }
                    OptCircuit newOptCirc = new OptCircuit(result, c, ruleSymb, System.nanoTime());
                    q.add(newOptCirc);
                }
                seen.add(hash);
            }
        }

        return bestCircuit;
    }

    public OptCircuit optimizeBeamMCMC(OptCircuit circuit,
                                       HashMap<String, Integer> ruleCount,
                                       boolean onlySymb,
                                       String filename,
                                       String ruleFileName,
                                       String symbRuleFileName,
                                       Path output) throws IOException, ExecutionException, InterruptedException {
        List<Pair<CircuitDAG, String>> rules = null;
        if (!onlySymb) {
            rules = getRules(ruleFileName, Params.REMOVE_SIZE_PRESERVING_RULES, Params.MAX_RULE_QUBITS, Params.PRESERVE_MAPPING, Params.USE_SIZE_PRESERVE_RULE_REFLECTION, Params.USE_SIZE_INCREASING_RULES, Params.OPTIMIZATION_OBJECTIVE);
        }
        List<String> rulesSymb = getSymbRules(symbRuleFileName, Params.USE_SIZE_PRESERVING_SYMB_RULES, Params.MAX_RULE_QUBITS, Params.PRESERVE_MAPPING, Params.USE_SIZE_PRESERVE_RULE_REFLECTION);
        Params.setResynthWeight(rules.size() + rulesSymb.size());

        if (Params.VERBOSITY >= 2) {
            System.out.println("Total rules: %s, normal: %s, symb: %s".formatted(rules.size() + rulesSymb.size(), rules.size(), rulesSymb.size()));
        }

        OptCircuit bestCircuit = circuit;
        Set<Integer> seen = new HashSet<>();
        seen.add(circuit.hashCode());
        PriorityQueue<OptCircuit> q = new PriorityQueue<>(new OptCircuitComparator(Params.OPTIMIZATION_OBJECTIVE));
        q.add(circuit);
        long timeStart = System.currentTimeMillis();

        Future<Pair<CircuitDAG, String>> resynthThread = null;

        int iters = 0;
        Random rand = new Random(new Random(Params.SEED).nextInt());
        int numRulesApplied = ruleCount.size();

        while (!q.isEmpty()) {
            iters++;

            // update best circuit if necessary
            OptCircuit c = q.peek();
            if (bestSol(c.getCircuit(), bestCircuit.getCircuit(), Params.OPTIMIZATION_OBJECTIVE)) {
                bestCircuit = c;
                bestCircuit.setTimeToBest(((System.currentTimeMillis() - timeStart) / 1000));

                String qasm = CircuitParser.dagToQasm(bestCircuit.getCircuit());

                logIntermediateInfo(bestCircuit, ruleCount, (System.currentTimeMillis() - timeStart) / 1000.0, Params.VERBOSITY >= 3);
                writeCircuitToFile(bestCircuit.getCircuit(), output + String.format("/latest_sol_%s_%s", Params.JOB_INFO, filename));
            }

            c = dequeueCircuit(q, Params.TEMPERATURE, Params.OPTIMIZATION_OBJECTIVE, rand);

            // prune queue if too large
            if (q.size() > (Params.QUEUE_SIZE)) {
                PriorityQueue<OptCircuit> pruned = new PriorityQueue<>(new OptCircuitComparator(Params.OPTIMIZATION_OBJECTIVE));
                while (pruned.size() != Params.QUEUE_SIZE) {
                    pruned.add(q.poll());
                }
                q = pruned;
            }

            // Process rules (potentially prune during search)
            int min = 0;
            int max = rules.size() + rulesSymb.size() + Params.RESYNTH_WEIGHT;

            List<Integer> rulesToUse = new ArrayList<>();

            if (Params.PRUNE_TEMPERATURE == 0) {
                if (Params.NUM_TRANSFORMATIONS_SAMPLE != -1) {
                    while (rulesToUse.size() != Params.NUM_TRANSFORMATIONS_SAMPLE) {
                        int randRule = rand.nextInt(max - min) + min;
                        if (!rulesToUse.contains(randRule)) {
                            rulesToUse.add(randRule);
                        }
                    }
                } else {
                    IntStream.range(0, rules.size() + rulesSymb.size() + 1).forEach(x -> rulesToUse.add(x));
                }
            } else {
                List<Integer> allRules = new ArrayList<>();
                IntStream.range(0, rules.size() + rulesSymb.size() + 1).forEach(x -> allRules.add(x));

                sampleRulesInt(allRules, rulesToUse, Params.NUM_TRANSFORMATIONS_SAMPLE, Params.PRUNE_TEMPERATURE, bestCircuit, ruleCount, rand, rules, rulesSymb);
                Params.PRUNE_TEMPERATURE *= 1 - Params.COOLING_RATE;
            }

            for (Integer ruleToUse : rulesToUse) {
                OptCircuit candidate = c;
                if (resynthThread != null && resynthThread.isDone()) {
                    Pair<CircuitDAG, String> resynthResult = resynthThread.get();
                    var rulesApplied = new ArrayList<>(c.getRulesApplied());
                    candidate = new OptCircuit(resynthResult.getFirst(), rulesApplied, System.currentTimeMillis(), (System.currentTimeMillis() - timeStart) / 1000);
                    rulesApplied.add(new Pair("resynth", candidate.getCircuit().totalGateCount()));
                    rulesApplied.add(new Pair(resynthResult.getSecond(), candidate.getCircuit().totalGateCount()));

                    if (ruleCount.containsKey("resynth")) {
                        ruleCount.put("resynth", ruleCount.get("resynth") + Params.RESYNTH_WEIGHT);
                    } else {
                        ruleCount.put("resynth", Params.RESYNTH_WEIGHT);
                    }

                    resynthThread = null;
                } else {
                    if (ruleToUse < rules.size()) {
                        Pair<CircuitDAG, String> rule = rules.get(ruleToUse);
                        String[] splitRule = rule.getSecond().split(" \\| ");
                        var rulesApplied = new ArrayList<>(c.getRulesApplied());
                        CircuitDAG cPrime = applyRule(c.getCircuit(), splitRule[0], rule.getFirst(), Params.APPLY_ONCE, rand);
                        candidate = new OptCircuit(cPrime, rulesApplied, System.currentTimeMillis(), (System.currentTimeMillis() - timeStart) / 1000);
                        if (cPrime != c.getCircuit()) {
                            rulesApplied.add(new Pair(rule.getSecond(), candidate.getCircuit().totalGateCount()));

                            if (ruleCount.containsKey(rule.getSecond())) {
                                ruleCount.put(rule.getSecond(), ruleCount.get(rule.getSecond()) + 1);
                            } else {
                                ruleCount.put(rule.getSecond(), 1);
                            }
                        }
                    } else if (ruleToUse < rules.size() + rulesSymb.size()) {
                        String ruleSymb = rulesSymb.get(ruleToUse - rules.size());
                        String[] splitRule = ruleSymb.split(" \\| ");

                        String find = splitRule[1];
                        String replace = splitRule[0];

                        int findSymbIndex = find.indexOf("symb");
                        String findBeforeSymb = StringUtils.stripStart(find.substring(0, findSymbIndex).trim(), ";");
                        String findAfterSymb = StringUtils.stripStart(find.substring(find.indexOf(";", findSymbIndex)).trim(), ";").trim();
                        int replaceSymbIndex = replace.indexOf("symb");
                        String replaceBeforeSymb = StringUtils.stripStart(replace.substring(0, replaceSymbIndex).trim(), ";");
                        String replaceAfterSymb = StringUtils.stripStart(replace.substring(replace.indexOf(";", replaceSymbIndex)).trim(), ";").trim();

                        CircuitDAG cPrime = applySymbRule(c.getCircuit(), findBeforeSymb, findAfterSymb, replaceBeforeSymb, replaceAfterSymb, parseConstraints(splitRule[2]), Params.MAX_SYMB_QUBITS, Params.MAX_SYMB_SIZE, Params.APPLY_ONCE, rand);
                        var rulesApplied = new ArrayList<>(c.getRulesApplied());
                        candidate = new OptCircuit(cPrime, rulesApplied, System.currentTimeMillis(), (System.currentTimeMillis() - timeStart) / 1000);
                        if (cPrime != c.getCircuit()) {
                            rulesApplied.add(new Pair(ruleSymb, candidate.getCircuit().totalGateCount()));

                            if (ruleCount.containsKey(ruleSymb)) {
                                ruleCount.put(ruleSymb, ruleCount.get(ruleSymb) + 1);
                            } else {
                                ruleCount.put(ruleSymb, 1);
                            }
                        }
                    } else {
                        if (Params.RESYNTH_ALG == Resynth.NONE) {
                            q.add(candidate);
                            continue;
                        }
                        if (resynthThread == null) {
                            if (c.countResynthApplications() < Params.MAX_RESYNTH_ALLOWED || Params.MAX_RESYNTH_ALLOWED == -1) {
                                CircuitDAG input = c.getCircuit();
                                resynthThread = ThreadUtils.submit(() -> applyResynth(Params.RESYNTH_ALG, input, rand, Params.EPSILON / Params.MAX_RESYNTH_ALLOWED, Params.RESYNTH_ARGS));
                                q.add(candidate);
                                continue;
                            }
                        }
                    }
                }

                int candidateSize = candidate.getCircuit().cost(Params.OPTIMIZATION_OBJECTIVE);
                int currentSize = c.getCircuit().cost(Params.OPTIMIZATION_OBJECTIVE);
                if (candidateSize <= currentSize) {
                    q.add(candidate);
                } else {
                    double acceptP = Math.min(1, Math.exp(-Params.TEMPERATURE * ((double) candidateSize / currentSize)));
                    if (rand.nextDouble() <= acceptP) {
                        q.add(candidate);
                    } else {
                        q.add(c);
                    }
                }
            }
        }

        return bestCircuit;
    }

    // TODO: Implement this with applyRuleParallel, findParallel, etc!
    // (And the same functions for symb rules!)

    // FIXME: use applyRuleParallelNew (AFTER TESTING)
    public OptCircuit optimizeBeamMCMCParallel(OptCircuit circuit,
                                       HashMap<String, Integer> ruleCount,
                                       boolean onlySymb,
                                       String filename,
                                       String ruleFileName,
                                       String symbRuleFileName,
                                       Path output) throws IOException, ExecutionException, InterruptedException {
        List<Pair<CircuitDAG, String>> rules = null;
        if (!onlySymb) {
            rules = getRules(ruleFileName, Params.REMOVE_SIZE_PRESERVING_RULES, Params.MAX_RULE_QUBITS, Params.PRESERVE_MAPPING, Params.USE_SIZE_PRESERVE_RULE_REFLECTION, Params.USE_SIZE_INCREASING_RULES, Params.OPTIMIZATION_OBJECTIVE);
        }
        List<String> rulesSymb = getSymbRules(symbRuleFileName, Params.USE_SIZE_PRESERVING_SYMB_RULES, Params.MAX_RULE_QUBITS, Params.PRESERVE_MAPPING, Params.USE_SIZE_PRESERVE_RULE_REFLECTION);
        Params.setResynthWeight(rules.size() + rulesSymb.size());

        if (Params.VERBOSITY >= 2) {
            System.out.println("Total rules: %s, normal: %s, symb: %s".formatted(rules.size() + rulesSymb.size(), rules.size(), rulesSymb.size()));
        }

        OptCircuit bestCircuit = circuit;
        Set<Integer> seen = new HashSet<>();
        seen.add(circuit.hashCode());
        PriorityQueue<OptCircuit> q = new PriorityQueue<>(new OptCircuitComparator(Params.OPTIMIZATION_OBJECTIVE));
        q.add(circuit);
        long timeStart = System.currentTimeMillis();

        Future<Pair<CircuitDAG, String>> resynthThread = null;

        int iters = 0;
        Random rand = new Random(new Random(Params.SEED).nextInt());
        int numRulesApplied = ruleCount.size();

        while (!q.isEmpty()) {
            iters++;

            // update best circuit if necessary
            OptCircuit c = q.peek();
            if (bestSol(c.getCircuit(), bestCircuit.getCircuit(), Params.OPTIMIZATION_OBJECTIVE)) {
                bestCircuit = c;
                bestCircuit.setTimeToBest(((System.currentTimeMillis() - timeStart) / 1000));

                String qasm = CircuitParser.dagToQasm(bestCircuit.getCircuit());

                logIntermediateInfo(bestCircuit, ruleCount, (System.currentTimeMillis() - timeStart) / 1000.0, Params.VERBOSITY >= 3);
                writeCircuitToFile(bestCircuit.getCircuit(), output + String.format("/latest_sol_%s_%s", Params.JOB_INFO, filename));
            }

            c = dequeueCircuit(q, Params.TEMPERATURE, Params.OPTIMIZATION_OBJECTIVE, rand);

            // prune queue if too large
            if (q.size() > (Params.QUEUE_SIZE)) {
                PriorityQueue<OptCircuit> pruned = new PriorityQueue<>(new OptCircuitComparator(Params.OPTIMIZATION_OBJECTIVE));
                while (pruned.size() != Params.QUEUE_SIZE) {
                    pruned.add(q.poll());
                }
                q = pruned;
            }

            // Process rules (potentially prune during search)
            int min = 0;
            int max = rules.size() + rulesSymb.size() + Params.RESYNTH_WEIGHT;

            List<Integer> rulesToUse = new ArrayList<>();

            if (Params.PRUNE_TEMPERATURE == 0) {
                if (Params.NUM_TRANSFORMATIONS_SAMPLE != -1) {
                    while (rulesToUse.size() != Params.NUM_TRANSFORMATIONS_SAMPLE) {
                        int randRule = rand.nextInt(max - min) + min;
                        if (!rulesToUse.contains(randRule)) {
                            rulesToUse.add(randRule);
                        }
                    }
                } else {
                    IntStream.range(0, rules.size() + rulesSymb.size() + 1).forEach(x -> rulesToUse.add(x));
                }
            } else {
                List<Integer> allRules = new ArrayList<>();
                IntStream.range(0, rules.size() + rulesSymb.size() + 1).forEach(x -> allRules.add(x));

                sampleRulesInt(allRules, rulesToUse, Params.NUM_TRANSFORMATIONS_SAMPLE, Params.PRUNE_TEMPERATURE, bestCircuit, ruleCount, rand, rules, rulesSymb);
                Params.PRUNE_TEMPERATURE *= 1 - Params.COOLING_RATE;
            }

            // TODO: We can potentially parallelize this! Given N cores (?),
            // select the N first rules we want to test, do our parallel application for each of them,
            // and then do some sort of "scoring" pass to evaluate relative cost improvement for each
            // of the N rules. However, there can also be thread contention in this case (because of
            // nested parallelism within find and applyRule)
            for (Integer ruleToUse : rulesToUse) {
                OptCircuit candidate = c;
                if (resynthThread != null && resynthThread.isDone()) {
                    Pair<CircuitDAG, String> resynthResult = resynthThread.get();
                    var rulesApplied = new ArrayList<>(c.getRulesApplied());
                    candidate = new OptCircuit(resynthResult.getFirst(), rulesApplied, System.currentTimeMillis(), (System.currentTimeMillis() - timeStart) / 1000);
                    rulesApplied.add(new Pair("resynth", candidate.getCircuit().totalGateCount()));
                    rulesApplied.add(new Pair(resynthResult.getSecond(), candidate.getCircuit().totalGateCount()));

                    if (ruleCount.containsKey("resynth")) {
                        ruleCount.put("resynth", ruleCount.get("resynth") + Params.RESYNTH_WEIGHT);
                    } else {
                        ruleCount.put("resynth", Params.RESYNTH_WEIGHT);
                    }

                    resynthThread = null;
                } else {
                    if (ruleToUse < rules.size()) {
                        Pair<CircuitDAG, String> rule = rules.get(ruleToUse);
                        String[] splitRule = rule.getSecond().split(" \\| ");
                        var rulesApplied = new ArrayList<>(c.getRulesApplied());

                        // TODO: can we calculate circ depth, pat depth here?
                        CircuitDAG cPrime = applyRuleParallelNew(c.getCircuit(), splitRule[0], rule.getFirst(), Params.APPLY_ONCE, rand);
                        candidate = new OptCircuit(cPrime, rulesApplied, System.currentTimeMillis(), (System.currentTimeMillis() - timeStart) / 1000);
                        if (cPrime != c.getCircuit()) {
                            rulesApplied.add(new Pair(rule.getSecond(), candidate.getCircuit().totalGateCount()));

                            if (ruleCount.containsKey(rule.getSecond())) {
                                ruleCount.put(rule.getSecond(), ruleCount.get(rule.getSecond()) + 1);
                            } else {
                                ruleCount.put(rule.getSecond(), 1);
                            }
                        }
                    } else if (ruleToUse < rules.size() + rulesSymb.size()) {
                        String ruleSymb = rulesSymb.get(ruleToUse - rules.size());
                        String[] splitRule = ruleSymb.split(" \\| ");

                        String find = splitRule[1];
                        String replace = splitRule[0];

                        int findSymbIndex = find.indexOf("symb");
                        String findBeforeSymb = StringUtils.stripStart(find.substring(0, findSymbIndex).trim(), ";");
                        String findAfterSymb = StringUtils.stripStart(find.substring(find.indexOf(";", findSymbIndex)).trim(), ";").trim();
                        int replaceSymbIndex = replace.indexOf("symb");
                        String replaceBeforeSymb = StringUtils.stripStart(replace.substring(0, replaceSymbIndex).trim(), ";");
                        String replaceAfterSymb = StringUtils.stripStart(replace.substring(replace.indexOf(";", replaceSymbIndex)).trim(), ";").trim();

                        // FIXME: Apply symb rule also parallelize this!
                        // We currently default to applyOnce = true, for simplicity
                        CircuitDAG cPrime = applySymbRule(c.getCircuit(), findBeforeSymb, findAfterSymb, replaceBeforeSymb, replaceAfterSymb, parseConstraints(splitRule[2]), Params.MAX_SYMB_QUBITS, Params.MAX_SYMB_SIZE, Params.APPLY_ONCE, rand);
                        var rulesApplied = new ArrayList<>(c.getRulesApplied());
                        candidate = new OptCircuit(cPrime, rulesApplied, System.currentTimeMillis(), (System.currentTimeMillis() - timeStart) / 1000);
                        if (cPrime != c.getCircuit()) {
                            rulesApplied.add(new Pair(ruleSymb, candidate.getCircuit().totalGateCount()));

                            if (ruleCount.containsKey(ruleSymb)) {
                                ruleCount.put(ruleSymb, ruleCount.get(ruleSymb) + 1);
                            } else {
                                ruleCount.put(ruleSymb, 1);
                            }
                        }
                    } else {
                        if (Params.RESYNTH_ALG == Resynth.NONE) {
                            q.add(candidate);
                            continue;
                        }
                        if (resynthThread == null) {
                            if (c.countResynthApplications() < Params.MAX_RESYNTH_ALLOWED || Params.MAX_RESYNTH_ALLOWED == -1) {
                                CircuitDAG input = c.getCircuit();
                                resynthThread = ThreadUtils.submit(() -> applyResynth(Params.RESYNTH_ALG, input, rand, Params.EPSILON / Params.MAX_RESYNTH_ALLOWED, Params.RESYNTH_ARGS));
                                q.add(candidate);
                                continue;
                            }
                        }
                    }
                }

                int candidateSize = candidate.getCircuit().cost(Params.OPTIMIZATION_OBJECTIVE);
                int currentSize = c.getCircuit().cost(Params.OPTIMIZATION_OBJECTIVE);
                if (candidateSize <= currentSize) {
                    q.add(candidate);
                } else {
                    double acceptP = Math.min(1, Math.exp(-Params.TEMPERATURE * ((double) candidateSize / currentSize)));
                    if (rand.nextDouble() <= acceptP) {
                        q.add(candidate);
                    } else {
                        q.add(c);
                    }
                }
            }
        }

        return bestCircuit;
    }

    private boolean lessThanCurrBest(CircuitDAG c, CircuitDAG currentBest, OptObj optObj) {
        return c.cost(optObj) < currentBest.cost(optObj);
    }

    private boolean leqCurrBest(CircuitDAG c, CircuitDAG currentBest, OptObj optObj) {
        return c.cost(optObj) <= currentBest.cost(optObj);
    }

    private boolean bestSol(CircuitDAG c, CircuitDAG currentBest, OptObj optObj) {
        if (c.cost(optObj) < currentBest.cost(optObj)) {
            return true;
        }
        if (c.cost(optObj) == currentBest.cost(optObj)) {
            if (optObj == OptObj.TWO_Q) {
                return c.totalGateCount() < currentBest.totalGateCount();
            } else if (optObj == OptObj.T) {
                if (c.twoQGateCount() < currentBest.twoQGateCount()) {
                    return true;
                }
                if (c.twoQGateCount() == currentBest.twoQGateCount()) {
                    return c.totalGateCount() < currentBest.totalGateCount();
                }
            }
        }
        return false;
    }

    private void addCommonLogInfo(HashMap<String, String> log, OptCircuit bestCircuit, double secondsElapsed) {
        log.put("best_circuit_size", String.valueOf(bestCircuit.getCircuit().totalGateCount()));
        log.put("best_size_2q", String.valueOf(bestCircuit.getCircuit().twoQGateCount()));
        log.put("best_size_t", String.valueOf(bestCircuit.getCircuit().tGateCount()));
        log.put("time_to_best", String.valueOf(bestCircuit.getTimeToBest()));
        log.put("seconds_elapsed", String.valueOf(secondsElapsed));
    }

    private void logBeamSearchIterInfo(String filename, int queueSize, int seenSetSize, OptCircuit bestCircuit, OptCircuit currentCircuit, double secondsElapsed, int numRulesToUse, int numSymbRulesToUse) {
        if (Params.VERBOSITY < 1) {
            return;
        }
        Gson gson = new GsonBuilder().disableHtmlEscaping().create();
        HashMap<String, String> log = new HashMap<>();
        log.put("filename", filename);
        log.put("queue_size", String.valueOf(queueSize));
        log.put("seen_set_size", String.valueOf(seenSetSize));
        addCommonLogInfo(log, bestCircuit, secondsElapsed);
        log.put("num_rules_using", String.valueOf(numRulesToUse));
        log.put("num_symb_rules_using", String.valueOf(numSymbRulesToUse));
        log.put("seen_set_size", String.valueOf(seenSetSize));
        log.put("current_circuit_size", String.valueOf(currentCircuit.getCircuit().totalGateCount()));
        String json = gson.toJson(log);
        System.out.println(json);
    }

    private void logIntermediateInfo(OptCircuit bestCircuit, Map<String, Integer> allRulesApplied, double secondsElapsed, boolean verbose) {
        if (Params.VERBOSITY < 1) {
            return;
        }
        Gson gson = new GsonBuilder().disableHtmlEscaping().create();
        HashMap<String, String> log = new HashMap<>();
        addCommonLogInfo(log, bestCircuit, secondsElapsed);
        if (verbose) {
            log.put("rules_applied_to_best", gson.toJson(bestCircuit.getRulesApplied()));
            log.put("all_rules_applied", gson.toJson(allRulesApplied));
        } else {
            log.put("rules_applied_to_best", gson.toJson(new HashMap<>()));
            log.put("all_rules_applied", gson.toJson(new HashMap<>()));
        }
        String json = gson.toJson(log);
        System.out.println(json);
    }

    private void logFinalInfo(OptCircuit optimized, Map<String, Integer> allRulesApplied, long totalTime) {
        if (Params.VERBOSITY < 1) {
            return;
        }
        Gson gson = new GsonBuilder().disableHtmlEscaping().create();
        HashMap<String, String> log = new HashMap<>();
        addCommonLogInfo(log, optimized, totalTime);
        log.put("total_time", String.valueOf(totalTime));
        log.put("rules_applied_to_best", gson.toJson(optimized.getRulesApplied()));
        log.put("all_rules_applied", gson.toJson(allRulesApplied));
        String json = gson.toJson(log);
        System.out.println(json);
    }

    private void logInitialInfo(OptCircuit initialCirc) {
        if (Params.VERBOSITY < 1) {
            return;
        }
        Gson gson = new GsonBuilder().disableHtmlEscaping().create();
        HashMap<String, String> log = new HashMap<>();
        log.put("original_total", String.valueOf(initialCirc.getCircuit().totalGateCount()));
        log.put("original_2q", String.valueOf(initialCirc.getCircuit().twoQGateCount()));
        log.put("original_t", String.valueOf(initialCirc.getCircuit().tGateCount()));
        String json = gson.toJson(log);
        System.out.println(json);
    }

    public void optimizeOne(File file,
                            String ruleFileName,
                            String symbRuleFileName,
                            Path output
    ) throws IOException, InterruptedException, ExecutionException {
        if (Params.VERBOSITY >= 2) {
            System.out.println(file.getName());
        }
        String content = Files.readString(file.toPath());

        long time1 = System.currentTimeMillis();
        CircuitDAG circuitDAG = CircuitParser.qasmToDag(content);
        OptCircuit initialCircuit = new OptCircuit(circuitDAG, new ArrayList<>(), System.nanoTime(), 0);

        logInitialInfo(initialCircuit);
        writeCircuitToFile(circuitDAG, output + String.format("/latest_sol_%s_%s", Params.JOB_INFO, file.getName()));

        HashMap<String, Integer> allRulesApplied = new HashMap<>();
        OptCircuit optimized;
        if (Params.SEARCH_STRATEGY == SearchStrategy.BEAM) {
            optimized = optimizeBeam(initialCircuit, allRulesApplied, false, file.getName(), ruleFileName, symbRuleFileName, output);
        } else if (Params.SEARCH_STRATEGY == SearchStrategy.MCMC) {
            optimized = optimizeStochastic(initialCircuit, file.getName(), ruleFileName, symbRuleFileName, output);
        } else if (Params.SEARCH_STRATEGY == SearchStrategy.SIM_ANN) {
            optimized = optimizeStochastic(initialCircuit, file.getName(), ruleFileName, symbRuleFileName, output);
        } else if (Params.SEARCH_STRATEGY == SearchStrategy.BEAM_MCMC) {
            optimized = optimizeBeamMCMC(initialCircuit, allRulesApplied, false, file.getName(), ruleFileName, symbRuleFileName, output);
        } else if (Params.SEARCH_STRATEGY == SearchStrategy.BEAM_MCMC_PARALLEL) {
            optimized = optimizeBeamMCMCParallel(initialCircuit, allRulesApplied, false, file.getName(), ruleFileName, symbRuleFileName, output);
        } else {
            throw new RuntimeException("Unsupported search strategy: " + Params.SEARCH_STRATEGY);
        }

        long time2 = System.currentTimeMillis();
        logFinalInfo(optimized, allRulesApplied, ((time2 - time1) / 1000));
        writeCircuitToFile(optimized.getCircuit(), output + String.format("/optimized_%s_%s", Params.JOB_INFO, file.getName()));
    }

    public void writeCircuitToFile(CircuitDAG circuit, String path) throws IOException {
        String qasm = CircuitParser.dagToQasm(circuit);
        Files.write(Path.of(path), Collections.singleton(qasm), StandardCharsets.UTF_8);
    }

    public static void main(String[] args) throws IOException {
        ArgumentParser parser = ArgumentParsers.newFor("GUOQ and QUESO").fromFilePrefix("@").build().defaultHelp(true);

        parser.addArgument("circuit")
                .metavar("path-to-circuit")
                .type(String.class)
                .help("Path to the circuit .qasm to compile");
        parser.addArgument("-g", "--gate-set")
                .type(GateSet.class)
                .help("gate set of input circuit (and output)");
        parser.addArgument("-r", "--rules")
                .type(String.class)
                .help("non-symbolic rules path");
        parser.addArgument("-sr", "--symb-rules")
                .type(String.class)
                .help("symbolic rules path");
        parser.addArgument("--rules-dir")
                .type(String.class)
                .setDefault(Params.RULES_DIR)
                .help("directory for rules files. not used if --rules and --symb-rules are specified.");
        parser.addArgument("-q", "--queue-size")
                .type(Integer.class)
                .setDefault(Params.QUEUE_SIZE)
                .help("max priority queue size");
        parser.addArgument("-out", "--output-dir")
                .type(String.class)
                .setDefault(Params.OUTPUT_DIR)
                .help("relative path to directory for output circuit. creates directory if it does not exist");
        parser.addArgument("-job", "--job-info")
                .type(String.class)
                .setDefault(Params.JOB_INFO)
                .help("job info to add to output file name");
        parser.addArgument("--max-symb-size")
                .type(Integer.class)
                .setDefault(Params.MAX_SYMB_SIZE)
                .help("max size (number of gates) to consider for symbolic circuit");
        parser.addArgument("--max-symb-qubits")
                .type(Integer.class)
                .setDefault(Params.MAX_SYMB_QUBITS)
                .help("max number of qubits allowed for symbolic circuit");
        parser.addArgument("--max-rule-qubits")
                .type(Integer.class)
                .setDefault(Params.MAX_RULE_QUBITS)
                .help("max qubits allowed in rules. -1 for no limit");
        parser.addArgument("--remove-size-preserving-rules")
                .type(Arguments.booleanType())
                .setDefault(Params.REMOVE_SIZE_PRESERVING_RULES)
                .action(Arguments.storeTrue())
                .help("remove size-preserving nonsymbolic rules");
        parser.addArgument("--use-size-preserving-symb-rules")
                .type(Arguments.booleanType())
                .setDefault(Params.USE_SIZE_PRESERVING_SYMB_RULES)
                .action(Arguments.storeTrue())
                .help("use size-preserving symbolic rules");
        parser.addArgument("--use-size-preserve-reflection")
                .type(Arguments.booleanType())
                .setDefault(Params.USE_SIZE_PRESERVE_RULE_REFLECTION)
                .action(Arguments.storeTrue())
                .help("use the reflection of size preserving rule as well. i.e. use both directions of rule");
        parser.addArgument("--use-size-increasing-rules")
                .type(Arguments.booleanType())
                .setDefault(Params.USE_SIZE_INCREASING_RULES)
                .action(Arguments.storeTrue())
                .help("use size increasing nonsymb rules");
        parser.addArgument("--preserve-mapping")
                .type(Arguments.booleanType())
                .setDefault(Params.PRESERVE_MAPPING)
                .action(Arguments.storeTrue())
                .help("preserve connectivity of input circuit (preserve mapping)");
        parser.addArgument("-search", "--search-strategy")
                .type(SearchStrategy.class)
                .setDefault(Params.SEARCH_STRATEGY)
                .help("search alg to use");
        parser.addArgument("-opt", "--opt-obj")
                .type(OptObj.class)
                .setDefault(Params.OPTIMIZATION_OBJECTIVE)
                .help("optimization objective");
        parser.addArgument("-resynth", "--resynth-alg")
                .type(Resynth.class)
                .setDefault(Params.RESYNTH_ALG)
                .help("resynth alg to use");
        parser.addArgument("-eps", "--epsilon")
                .type(Double.class)
                .setDefault(Params.EPSILON)
                .help("error threshold for final circuit");
        parser.addArgument("-maxsynth", "--max-resynth-allowed")
                .type(Integer.class)
                .setDefault(Params.MAX_RESYNTH_ALLOWED)
                .help("maximum number of calls to resynthesis allowed in final circuit. -1 if no limit");
        parser.addArgument("--seed")
                .type(Integer.class)
                .setDefault(Params.SEED)
                .help("seed for random partition for resynthesis");
        parser.addArgument("-temp", "--temperature")
                .type(Double.class)
                .setDefault(Params.TEMPERATURE)
                .help("temperature for simulated annealing or beta for mcmc. for beam search, temperature is for softmax. If 0 then polls priority queue.");
        parser.addArgument("-prunetemp", "--prune-temperature")
                .type(Double.class)
                .setDefault(Params.PRUNE_TEMPERATURE)
                .help("temperature for pruning rules. 0 samples rules greedily");
        parser.addArgument("-cool", "--cooling-rate")
                .type(Double.class)
                .setDefault(Params.COOLING_RATE)
                .help("cooling rate for simulated annealing. 0 for mcmc. for beam search, this is the rate to prune rules so 0 for no pruning.");
        parser.addArgument("-itersprune", "--iters-before-prune")
                .type(Integer.class)
                .setDefault(Params.ITERS_BEFORE_PRUNE)
                .help("iterations to wait before starting to prune rules");
        parser.addArgument("-secsprune", "--secs-before-prune")
                .type(Integer.class)
                .setDefault(Params.SECS_BEFORE_PRUNE)
                .help("seconds to wait before starting to prune rules");
        parser.addArgument("--resynth-args")
                .type(new ResynthArgs())
                .setDefault(Params.RESYNTH_ARGS)
                .help("json of args to use for the chosen resynth tool");
        parser.addArgument("--num-transf-sample")
                .type(Integer.class)
                .setDefault(Params.NUM_TRANSFORMATIONS_SAMPLE)
                .help("number of transformations to sample per iteration");
        parser.addArgument("--resynth-weight")
                .type(Integer.class)
                .setDefault(Params.RESYNTH_WEIGHT)
                .help("weight of resynthesis when sampling transformations randomly");
        parser.addArgument("--apply-once")
                .type(Arguments.booleanType())
                .setDefault(Params.APPLY_ONCE)
                .action(Arguments.storeTrue())
                .help("apply rewrite rule only once per iteration if true instead of all disjoint matches");
        parser.addArgument("--fidelity")
                .type(Integer.class)
                .setDefault(Params.FIDELITY_BREAKEVEN)
                .help("Cost of a 2q gate in terms of 1q gates for FIDELITY opt obj. Or weight of T gate vs. 2q gate for FT opt obj.");
        parser.addArgument("--error-1q")
                .type(Double.class)
                .setDefault(Params.ERROR_1Q)
                .help("lowest error rate for non-virtual 1q gate");
        parser.addArgument("--error-2q")
                .type(Double.class)
                .setDefault(Params.ERROR_2Q)
                .help("error rate for 2q gate");
        parser.addArgument("--verbosity")
                .type(Integer.class)
                .setDefault(Params.VERBOSITY)
                .help("log verbosity level (higher more verbose)");

        try {
            Namespace parsed = parser.parseArgs(args);
            Gson gson = new GsonBuilder().disableHtmlEscaping().create();

            Optimizer applier = new Optimizer(new Random(), parsed.getInt("max_symb_qubits"));

            Files.createDirectories(Paths.get(parsed.getString("output_dir")));

            Params.GATE_SET = parsed.get("gate_set");
            Params.QUEUE_SIZE = parsed.getInt("queue_size");
            Params.MAX_SYMB_QUBITS = parsed.getInt("max_symb_qubits");
            Params.MAX_SYMB_SIZE = parsed.getInt("max_symb_size");
            Params.MAX_RULE_QUBITS = parsed.getInt("max_rule_qubits");
            Params.REMOVE_SIZE_PRESERVING_RULES = parsed.getBoolean("remove_size_preserving_rules");
            Params.USE_SIZE_PRESERVING_SYMB_RULES = parsed.getBoolean("use_size_preserving_symb_rules");
            Params.USE_SIZE_PRESERVE_RULE_REFLECTION = parsed.getBoolean("use_size_preserve_reflection");
            Params.USE_SIZE_INCREASING_RULES = parsed.getBoolean("use_size_increasing_rules");
            Params.PRESERVE_MAPPING = parsed.getBoolean("preserve_mapping");
            Params.OUTPUT_DIR = parsed.getString("output_dir");
            Params.JOB_INFO = parsed.getString("job_info");
            Params.SEARCH_STRATEGY = parsed.get("search_strategy");
            Params.OPTIMIZATION_OBJECTIVE = parsed.get("opt_obj");
            Params.FIDELITY_BREAKEVEN = parsed.get("fidelity");
            Params.ERROR_1Q = parsed.getDouble("error_1q");
            Params.ERROR_2Q = parsed.getDouble("error_2q");
            Params.RESYNTH_ALG = parsed.get("resynth_alg");
            Params.RESYNTH_ARGS = parsed.get("resynth_args");
            Params.MAX_RESYNTH_ALLOWED = parsed.get("max_resynth_allowed");
            Params.EPSILON = parsed.getDouble("epsilon");
            Params.SEED = parsed.getInt("seed");
            Params.TEMPERATURE = parsed.getDouble("temperature");
            Params.COOLING_RATE = parsed.getDouble("cooling_rate");
            Params.PRUNE_TEMPERATURE = parsed.getDouble("prune_temperature");
            Params.ITERS_BEFORE_PRUNE = parsed.getInt("iters_before_prune");
            Params.SECS_BEFORE_PRUNE = parsed.getInt("secs_before_prune");
            Params.NUM_TRANSFORMATIONS_SAMPLE = parsed.getInt("num_transf_sample");
            Params.RESYNTH_WEIGHT = parsed.getInt("resynth_weight");
            Params.APPLY_ONCE = parsed.getBoolean("apply_once");
            Params.VERBOSITY = parsed.getInt("verbosity");
            Params.RULE_FILE = parsed.getString("rules");
            Params.SYMB_RULE_FILE = parsed.getString("symb_rules");
            Params.RULES_DIR = parsed.getString("rules_dir");

            if (Params.VERBOSITY >= 2) {
                System.out.println(gson.toJson(parsed.getAttrs()));
            }

            if (Params.RESYNTH_ALG == null) {
                if (Params.OPTIMIZATION_OBJECTIVE == OptObj.FIDELITY || Params.OPTIMIZATION_OBJECTIVE == OptObj.TWO_Q) {
                    Params.RESYNTH_ALG = Resynth.BQSKIT;
                } else if (Params.OPTIMIZATION_OBJECTIVE == OptObj.FT || Params.OPTIMIZATION_OBJECTIVE == OptObj.T) {
                    Params.RESYNTH_ALG = Resynth.SYNTHETIQ;
                } else {
                    Params.RESYNTH_ALG = Resynth.NONE;
                }
            }

            if (Params.OPTIMIZATION_OBJECTIVE == OptObj.FT) {
                Params.FIDELITY_BREAKEVEN = 50;
            }

            if (Params.OPTIMIZATION_OBJECTIVE == OptObj.FIDELITY && Params.FIDELITY_BREAKEVEN == 1) {
                if ((Params.ERROR_1Q == null || Params.ERROR_2Q == null)) {
                    throw new RuntimeException("--error-1q and --error-2q required for FIDELITY optimization objective");
                }
                Params.FIDELITY_BREAKEVEN = (int) (Math.log(1 - Params.ERROR_2Q) / Math.log(1 - Params.ERROR_1Q));
            }

            if (Params.RULE_FILE == null) {
                Params.RULE_FILE = Params.RULES_DIR + "/" + Params.GATE_SET_RULES_MAP.get(Params.GATE_SET).getKey();
            }
            if (Params.SYMB_RULE_FILE == null) {
                Params.SYMB_RULE_FILE = Params.RULES_DIR + "/" + Params.GATE_SET_RULES_MAP.get(Params.GATE_SET).getValue();
            }

            applier.optimizeOne(
                    new File(String.valueOf(FileSystems.getDefault().getPath(new String()).resolve(parsed.getString("circuit")))),
                    Params.RULE_FILE,
                    Params.SYMB_RULE_FILE,
                    FileSystems.getDefault().getPath(new String()).resolve(Params.OUTPUT_DIR).toAbsolutePath()
            );
        } catch (ArgumentParserException e) {
            parser.handleError(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        } finally {
            ThreadUtils.shutdown();
        }
    }
}
