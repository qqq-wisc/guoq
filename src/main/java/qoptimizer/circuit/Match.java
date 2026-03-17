package qoptimizer.circuit;

import java.util.Map;
import java.util.HashSet;
import java.util.Set;
import qoptimizer.circuit.Node;
import qoptimizer.ast.BinOp;
import qoptimizer.ast.Expr;
import qoptimizer.ast.Real;
import qoptimizer.ast.Symbol;
import qoptimizer.ast.UnOp;
import qoptimizer.ast.Var;

import lombok.Getter;
import lombok.Setter;

/* 
    private CircuitDAG applyMatch(
            CircuitDAG circuit,
            CircuitDAG copy,
            CircuitDAG pattern,
            String replace,
            Match match,
            Set<Node> replaced,
            Map<String, Expr> angleMap,
            boolean applyOnce) {
 *
 * */

// TODO: Add correct information to Match class so we
// can basically just pass it in and not have to think about
// other confusing data transfer stuff

// NOTE: Update the printer, and update:
// 1. The Match object that matchAtNode returns
// 2. The applyMatch function signature

@Getter
public class Match {
    public final Node startNode;
    public int startDepth;
    public int endDepth;

    public final Map<Node, Node> patternToCircMap;
    public final Map<String, Expr> angleMap;
    public final Set<Node> matched;
    public final Set<Node> replaced;



    public Match(Node startNode, Map<Node, Node> patternToCircMap, int startDepth, int endDepth, Map<String, Expr> angleMap,
            Set<Node> matched, Set<Node> replaced) {
        this.startNode = startNode;
        this.patternToCircMap = patternToCircMap;
        this.startDepth = startDepth;
        this.endDepth = endDepth;
        this.angleMap = angleMap;
        this.matched = matched;
        this.replaced = replaced;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Match(");
        sb.append("startNode=").append(startNode);
        sb.append(", startDepth=").append(startDepth);
        sb.append(", endDepth=").append(endDepth);

        sb.append(", patternToCircMap={");
        boolean first = true;
        for (Map.Entry<Node, Node> e : patternToCircMap.entrySet()) {
            if (!first) sb.append(", ");
            Node patternNode = e.getKey();
            Node circuitNode = e.getValue();
            sb.append(patternNode)
              .append(" -> ")
              .append(circuitNode)
              .append(" (depth=")
              .append(circuitNode.getDepth())
              .append(")");
            first = false;
        }
        sb.append("}");

        sb.append(", angleMap={");
        first = true;
        for (Map.Entry<String, Expr> e : angleMap.entrySet()) {
            if (!first) sb.append(", ");
            sb.append(e.getKey()).append("=").append(e.getValue());
            first = false;
        }
        sb.append("}");

        sb.append(", matched=");
        sb.append(summarizeSet(matched));

        sb.append(", replaced=");
        sb.append(summarizeSet(replaced));

        sb.append(")");
        return sb.toString();
    }

    private String summarizeSet(Set<Node> set) {
        if (set == null) return "null";
        int size = set.size();

        StringBuilder sb = new StringBuilder();
        sb.append("[size=").append(size);

        for (Node n : set) {
            sb.append(", ").append(n.toString());
        }

        if (size > 3) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }

    /**
     * Returns the set of circuit nodes involved in this match.
     * Used for conflict detection: two matches conflict iff their circuit node sets intersect.
     */
    public Set<Node> getCircuitNodes() {
        return new HashSet<>(patternToCircMap.values());
    }

    /**
     * Checks whether this match conflicts with another match,
     * i.e., they share at least one circuit node.
     */
    public boolean conflictsWith(Match other) {
        for (Node n : this.getCircuitNodes()) {
            if (other.getCircuitNodes().contains(n)) return true;
        }
        return false;
    }

    /**
     * Checks whether this match conflicts with a set of already-claimed circuit nodes.
     * More efficient than conflictsWith(Match) when building a globalReplaced set incrementally.
     */
    public boolean conflictsWith(Set<Node> claimed) {
        for (Node n : getCircuitNodes()) {
            if (claimed.contains(n)) return true;
        }
        return false;
    }

}

