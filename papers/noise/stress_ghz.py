from __future__ import annotations
import argparse, sys, os, time, tracemalloc
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..")) # clue is here

from clue.linalg import SparseRowMatrix as Circuit, SparseVector as State, NumericalSubspace, find_smallest_common_subspace
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator, DensityVector

from circuits import embed1, embed2, H, CX, bitflip_channel
from results import append_row

CSV_HEADER = ["qubits", "state_dim", "p", "delta", "time_lumping", "memory_mb", "ambient_dim", "reduced_dim", "red_ratio"]

## --------------------------------------------------------------------------
## GHZ preparation
## --------------------------------------------------------------------------
def ghz_state(n: int) -> State:
    v = State(2**n, CC)
    v[0] = 1
    V = embed1(H, 0, n)
    for i in range(n-1):
        V = embed2(CX, i, n) * V
    return v.apply_matrix(V)

## --------------------------------------------------------------------------
## Stress test loop
## --------------------------------------------------------------------------
def run_size(n: int, p: float, delta: float) -> dict:
    N = 2**n
    rho = DensityVector.from_tensor(ghz_state(n))
    R = bitflip_channel(n, p)

    tracemalloc.start()
    start = time.perf_counter()
    nqb = find_smallest_common_subspace(matrices=(R,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta)
    elapsed = time.perf_counter() - start
    memory = tracemalloc.get_traced_memory()[1] / (2**20)
    tracemalloc.stop()

    return {
        "qubits": n, "state_dim": N, "ambient_dim": nqb.ambient_dimension(),
        "reduced_dim": nqb.dim(), "time": elapsed, "memory": memory,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-qubits", type=int, default=2, help="smallest register size to try (default: 2)")
    parser.add_argument("--max-qubits", type=int, default=12, help="largest register size to try (default: 12)")
    parser.add_argument("--p", type=float, default=0.05, help="per-qubit bit-flip probability per round (default: 0.05)")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="stop the sweep after a size takes longer than this (checked between sizes, not mid-computation; default: 30s)")
    parser.add_argument("--csv", type=str, default=None, help="append results to this CSV file (created with a header if new)")
    args = parser.parse_args()

    print("Stress test: noisy quantum bisimulation on GHZ-state decoherence")
    print(f"p={args.p:g}  delta={args.delta:g}  max-seconds/size={args.max_seconds:g}")
    print(f"{'qubits':>6} {'|states>':>9} {'time (s)':>10} {'lumped':>8} {'ambient':>9} {'ratio':>8} {'factor':>10}")
    sys.stdout.flush()

    for n in range(args.min_qubits, args.max_qubits + 1):
        result = run_size(n, args.p, args.delta)
        ratio = result["reduced_dim"] / result["ambient_dim"]
        factor = result["ambient_dim"] / result["reduced_dim"]
        print(
            f"{result['qubits']:>6} {result['state_dim']:>9} "
            f"{result['time']:>10.4f} {result['reduced_dim']:>8} {result['ambient_dim']:>9} "
            f"{ratio:>8.5f} {factor:>9.1f}x"
        )
        sys.stdout.flush()

        if args.csv:
            append_row(args.csv, CSV_HEADER, [
                result["qubits"], result["state_dim"], args.p, args.delta,
                result["time"], result["memory"], result["ambient_dim"], result["reduced_dim"], ratio,
            ])

        if result["time"] > args.max_seconds:
            print(f"-- stopping: size n={n} took {result['time']:.1f}s, over the {args.max_seconds:g}s budget --")
            break

if __name__ == "__main__":
    main()
