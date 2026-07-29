package qoptimizer.config;

/**
 * Which external mapping-and-routing solver the ROUTED_DEPTH / DEPTH_FT objectives call.
 *
 * <p>SCIR is the run-scir binary (NISQ-style weighted-depth SWAP routing over a named architecture).
 * SCMR is the run-scmr binary (surface-code lattice-surgery routing over a named layout auto-sized
 * to the circuit, optimizing parallel step count). FASTLS is the FastLS fast-lattice-surgery router (its own
 * repo; architecture is auto-sized to the circuit and its routed depth is the number of routing
 * layers). The three take different command lines and report their depth differently, so
 * {@link qoptimizer.Qmr} dispatches on this.
 */
public enum QmrBackend {
    SCIR,
    SCMR,
    FASTLS
}
