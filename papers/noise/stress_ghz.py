from __future__ import annotations
import argparse, sys, os, time, tracemalloc
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..")) # clue is here

from clue.linalg import SparseRowMatrix as Circuit, SparseVector as State, NumericalSubspace, find_smallest_common_subspace
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator, DensityVector

from circuits import DURATIONS, Gate, embed1, embed2, H, CX, cx, h, idle_round, noisy_circuit, prepare
from results import append_row

CSV_HEADER = [
    "qubits", "state_dim", "epsilon", "idle_ns", "delta", "time_lumping", "memory_mb",
    "ambient_dim", "reduced_dim", "red_ratio", "noise_model", "gate_count", "prep_gate_count",
]

NOISE_MODEL = "depolarizing-per-gate-ibm"

## --------------------------------------------------------------------------
## GHZ preparation
## --------------------------------------------------------------------------
def ghz_state(n: int) -> State:
    r'''
        The ideal GHZ state. Kept as the reference that `ghz_gates` is checked against; the stress
        test itself starts from `\ket{0\ldots0}` and prepares this *noisily*.
    '''
    v = State(2**n, CC)
    v[0] = 1
    V = embed1(H, 0, n)
    for i in range(n-1):
        V = embed2(CX, i, n) * V
    return v.apply_matrix(V)

def ghz_gates(n: int) -> list[Gate]:
    r'''
        GHZ preparation in the IBM basis: `H` on qubit 0, then a `CX` ladder down the register.

        This is the whole circuit. Previously this file prepared an *exact* GHZ state and lumped a
        bit-flip channel acting on it, so the gates implementing GHZ carried no noise at all; now the
        preparation is the noisy circuit, matching the other two stress tests.
    '''
    return h(0) + [cx(i, i+1) for i in range(n-1)]

def idle_gates(n: int, idle_ns: float) -> list[Gate]:
    r'''
        The operator the lumping iterates: one time step in which every qubit idles, so `A^k` is
        decoherence over `k` steps.

        GHZ preparation is one-shot -- there is no loop in it to repeat -- and this file has no
        non-noisy counterpart in `papers/quantum/` to borrow a repeated unit from. So the repeated
        unit has to be chosen, and this is the depolarising analogue of the bit-flip round this test
        used to lump (which it reached from an *exactly* prepared GHZ state). The alternative would be
        to reinstate `bitflip_channel`, still available in `circuits.py`.
    '''
    return idle_round(n, idle_ns)

## --------------------------------------------------------------------------
## Stress test loop
## --------------------------------------------------------------------------
def run_size(n: int, epsilon: float, idle_ns: float, delta: float) -> dict:
    N = 2**n
    preparation = ghz_gates(n)
    gates = idle_gates(n, idle_ns)
    rho = prepare(preparation, n, epsilon) # noisy GHZ preparation, once -- a mixed state
    R = noisy_circuit(gates, n, epsilon)   # the repeated operator: one idle step

    tracemalloc.start()
    start = time.perf_counter()
    nqb = find_smallest_common_subspace(matrices=(R,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta)
    elapsed = time.perf_counter() - start
    memory = tracemalloc.get_traced_memory()[1] / (2**20)
    tracemalloc.stop()

    return {
        "qubits": n, "state_dim": N, "ambient_dim": nqb.ambient_dimension(),
        "reduced_dim": nqb.dim(), "time": elapsed, "memory": memory,
        "gate_count": len(gates), "prep_gate_count": len(preparation),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-qubits", type=int, default=2, help="smallest register size to try (default: 2)")
    parser.add_argument("--max-qubits", type=int, default=12, help="largest register size to try (default: 12)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="depolarizing parameter of one SX-length gate; each gate gets epsilon*duration/duration_sx, so RZ is noiseless and CX is ~11x SX (default: 1e-3)")
    parser.add_argument("--idle-ns", type=float, default=DURATIONS["cx"], help=f"length of one idle step in ns, which sets how much decoherence each iteration applies (default: the CX duration, {DURATIONS['cx']:g})")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="stop the sweep after a size takes longer than this (checked between sizes, not mid-computation; default: 30s)")
    parser.add_argument("--csv", type=str, default=None, help="append results to this CSV file (created with a header if new)")
    args = parser.parse_args()

    print("Stress test: noisy quantum bisimulation on GHZ state preparation")
    print("noise: depolarizing, per gate, IBM basis {CX, RZ, SX, X}")
    print("structure: noisy GHZ preparation once from |0...0>, then an idle step is iterated")
    print(f"epsilon={args.epsilon:g}  idle-ns={args.idle_ns:g}  delta={args.delta:g}  max-seconds/size={args.max_seconds:g}")
    print(f"{'qubits':>6} {'|states>':>9} {'A gates':>7} {'time (s)':>10} {'lumped':>8} {'ambient':>9} {'ratio':>8} {'factor':>10}")
    sys.stdout.flush()

    for n in range(args.min_qubits, args.max_qubits + 1):
        result = run_size(n, args.epsilon, args.idle_ns, args.delta)
        ratio = result["reduced_dim"] / result["ambient_dim"]
        factor = result["ambient_dim"] / result["reduced_dim"]
        print(
            f"{result['qubits']:>6} {result['state_dim']:>9} {result['gate_count']:>6} "
            f"{result['time']:>10.4f} {result['reduced_dim']:>8} {result['ambient_dim']:>9} "
            f"{ratio:>8.5f} {factor:>9.1f}x"
        )
        sys.stdout.flush()

        if args.csv:
            append_row(args.csv, CSV_HEADER, [
                result["qubits"], result["state_dim"], args.epsilon, args.idle_ns, args.delta,
                result["time"], result["memory"], result["ambient_dim"], result["reduced_dim"], ratio,
                NOISE_MODEL, result["gate_count"],
            ])

        if result["time"] > args.max_seconds:
            print(f"-- stopping: size n={n} took {result['time']:.1f}s, over the {args.max_seconds:g}s budget --")
            break

if __name__ == "__main__":
    main()
