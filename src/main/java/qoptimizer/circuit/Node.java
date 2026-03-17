package qoptimizer.circuit;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import lombok.Getter;
import lombok.Setter;
import qoptimizer.ast.Expr;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static qoptimizer.parser.NodeVisitor.eval;

public class Node {

    public enum Type {
        // if this changes then total gate count in CircuitDAG should be updated
        GATE,
        QUBIT_SOURCE,
        QUBIT_SINK
    }

    @Getter
    private String id;
    private Type type;
    @Getter
    @Setter
    private List<String> qubits;
    @Getter
    private List<Expr> angles;
    @Getter
    @Setter
    private Integer depth;
    @Getter
    private List<String> qregNames;

    public Node(String id, Type type, List<String> qubits, List<Expr> angles) {
        this.id = id;
        this.type = type;
        this.qubits = qubits;
        this.angles = angles;
        // instantiated in the CircuitDAG constructor
        this.depth = 0;
    }

    public Node(String id, Type type, List<String> qubits, List<Expr> angles, List<String> qregNames) {
        this.id = id;
        this.type = type;
        this.qubits = qubits;
        this.angles = angles;
        this.qregNames = qregNames;
        // instantiated in the CircuitDAG constructor
        this.depth = 0;
    }

    public boolean isGate() {
        return this.type.equals(Type.GATE);
    }

    public boolean isQubit() {
        return this.type.equals(Type.QUBIT_SOURCE) || this.type.equals(Type.QUBIT_SINK);
    }

    public boolean isSourceQubit() {
        return this.type.equals(Type.QUBIT_SOURCE);
    }

    public boolean isSinkQubit() {
        return this.type.equals(Type.QUBIT_SINK);
    }

    public boolean is2QGate() {
        return this.isGate() && this.qubits.size() == 2;
    }

    public boolean isCX() {
        return this.id.equals("cx") || this.id.equals("cz") || this.id.equals("rxx");
    }

    public boolean isTGate() {
        return this.isGate() && (this.id.equals("t") || this.id.equals("tdg") || (this.id.equals("rz") && Math.abs((eval(this.getAngles().get(0)) / Math.PI) % 0.5) == 0.25));
    }

    public boolean isCCZ() {
        return this.id.equals("ccz");
    }

    public int hash() {
        int result = Objects.hash(id, type, angles);
        result = 31 * result + (qubits != null ? qubits.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        return toStringHelper(HashBiMap.create());
    }

    public String toString(BiMap<String, String> qubitRenameMap) {
        return toStringHelper(qubitRenameMap);
    }

    private String toStringHelper(BiMap<String, String> qubitRenameMap) {
        String result = id;

        if (isQubit()) {
            return id + " " + type;
        }

        if (angles != null && !angles.isEmpty()) {
            result = result.concat("(");
            result = result.concat(String.join(",", angles.stream().map(Expr::toString).collect(Collectors.toList())));
            result = result.concat(")");
        }

        result = result.concat(" ");
        result = result.concat(String.join(",", qubits.stream().map(q -> qubitRenameMap.inverse().getOrDefault(q, q)).collect(Collectors.toList())));

        return result;
    }
}
