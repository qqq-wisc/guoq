package qoptimizer.circuit;

import qoptimizer.config.OptObj;

import java.util.Comparator;

public class OptCircuitComparator implements Comparator<OptCircuit> {

    private OptObj optObj;

    public OptCircuitComparator(OptObj optObj) {
        this.optObj = optObj;
    }

    @Override
    public int compare(OptCircuit o1, OptCircuit o2) {
        switch (this.optObj) {
            case TOTAL: { // total gate count -> create time
                return compareTotalGateCount(o1, o2);
            }
            case T: { // tcount -> total gate count -> create time
                return compareTGateCount(o1, o2);
            }
            case TWO_Q: { // 2q count -> total gate count -> create time
                return compare2qGateCount(o1, o2);
            }
            case TOTAL_IGNORE_RZ: { // total gate count ignoring rz -> create time
                return compareTotalGateCountIgnoreRz(o1, o2);
            }
            case ROUTED_DEPTH: { // depth of the routed circuit -> total gate count -> create time
                return compareRoutedDepth(o1, o2);
            }
            case DEPTH_FT: { // blended ft + routed depth -> total gate count -> create time
                return compareDepthFt(o1, o2);
            }
            default:
                throw new RuntimeException("Unsupported optObj: " + this.optObj);
        }
    }

    private int compareCreateTime(OptCircuit o1, OptCircuit o2) {
        if (o1.getCreateTime() < o2.getCreateTime()) {
            return -1;
        } else if (o1.getCreateTime() > o2.getCreateTime()) {
            return 1;
        } else {
            return 0;
        }
    }

    private int compareNumSizePreserve(OptCircuit o1, OptCircuit o2) {
        if (o1.getMostRecentNumSizePreserveRules() < o2.getMostRecentNumSizePreserveRules()) {
            return -1;
        } else if (o1.getMostRecentNumSizePreserveRules() > o2.getMostRecentNumSizePreserveRules()) {
            return 1;
        } else {
            return compareCreateTime(o1, o2);
        }
    }

    private int compareTotalGateCount(OptCircuit o1, OptCircuit o2) {
        return breakTiesCreateTime(o1, o2, o1.getCircuit().totalGateCount(), o2.getCircuit().totalGateCount());
    }

    private int compareTGateCount(OptCircuit o1, OptCircuit o2) {
        return breakTiesTotalGateCount(o1, o2, o1.getCircuit().tGateCount(), o2.getCircuit().tGateCount());
    }

    private int compare2qGateCount(OptCircuit o1, OptCircuit o2) {
        return breakTiesTotalGateCount(o1, o2, o1.getCircuit().twoQGateCount(), o2.getCircuit().twoQGateCount());
    }

    private int compareTotalGateCountIgnoreRz(OptCircuit o1, OptCircuit o2) {
        return breakTiesCreateTime(o1, o2, o1.getCircuit().totalGateCountIgnoreRz(), o2.getCircuit().totalGateCountIgnoreRz());
    }

    private int compareRoutedDepth(OptCircuit o1, OptCircuit o2) {
        return breakTiesTotalGateCount(o1, o2, o1.getCircuit().routedDepth(), o2.getCircuit().routedDepth());
    }

    // ft cost -> routed depth -> total gate count -> create time. routedDepth() runs the external
    // solver, so it is only reached on an ft-cost tie.
    private int compareDepthFt(OptCircuit o1, OptCircuit o2) {
        int byFt = Integer.compare(o1.getCircuit().ftCost(), o2.getCircuit().ftCost());
        if (byFt != 0) {
            return byFt;
        }
        int byDepth = Integer.compare(o1.getCircuit().routedDepth(), o2.getCircuit().routedDepth());
        if (byDepth != 0) {
            return byDepth;
        }
        return breakTiesCreateTime(o1, o2, o1.getCircuit().totalGateCount(), o2.getCircuit().totalGateCount());
    }

    private int breakTiesTotalGateCount(OptCircuit o1, OptCircuit o2, int o1Size, int o2Size) {
        if (o1Size < o2Size) {
            return -1;
        } else if (o1Size > o2Size) {
            return 1;
        } else {
            return compareTotalGateCount(o1, o2);
        }
    }

    private int breakTiesCreateTime(OptCircuit o1, OptCircuit o2, int o1Size, int o2Size) {
        if (o1Size < o2Size) {
            return -1;
        } else if (o1Size > o2Size) {
            return 1;
        } else {
            return compareCreateTime(o1, o2);
        }
    }
}
