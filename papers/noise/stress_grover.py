from __future__ import annotations
import argparse, sys, os, time, tracemalloc
from math import sqrt, pi, floor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..")) # clue is here

from clue.linalg import SparseRowMatrix as Circuit, SparseVector as State, NumericalSubspace, find_smallest_common_subspace
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator, DensityVector

from circuits import kron_pow, H, gate_failure_channel
from results import append_row

CSV_HEADER = [
    "qubits", "state_dim", "iterations", "target", "epsilon", "delta",
    "time_lumping", "memory_mb", "ambient_dim", "reduced_dim", "red_ratio", "gate",
]

## --------------------------------------------------------------------------
## Grover circuit
## --------------------------------------------------------------------------
def oracle(n: int, target: int) -> Circuit:
    r'''Diagonal phase-flip oracle: flips the sign of the ``target`` basis state.'''
    N = 2**n
    U = Circuit.eye(N, CC)
    U.increment(target, target, -2) # 1 -> -1, everything else stays 1
    return U

def diffuser(n: int) -> Circuit:
    r'''Grover diffuser `2|s><s| - I = H^{\otimes n} (2|0><0| - I) H^{\otimes n}`.'''
    N = 2**n
    Hn = kron_pow(H, n)
    Z0 = Circuit.eye(N, CC)
    for i in range(1, N):
        Z0.increment(i, i, -2) # flips the sign of every basis state except |0...0>
    return Hn * Z0 * Hn

def initial_state(n: int) -> State:
    v = State(2**n, CC)
    v[0] = 1
    return v.apply_matrix(kron_pow(H, n))

def noisy_grover(n: int, iterations: int, epsilon: float, target: int) -> DensityOperator:
    O = gate_failure_channel(oracle(n, target), epsilon)
    D = gate_failure_channel(diffuser(n), epsilon)
    return DensityOperator(operators=iterations*[O, D])

def optimal_iterations(n: int) -> int:
    r'''Standard Grover iteration count `\lfloor (\pi/4)\sqrt{N}\rfloor` (at least 1).'''
    return max(1, floor(pi/4*sqrt(2**n)))

## --------------------------------------------------------------------------
## Stress test loop
## --------------------------------------------------------------------------
def run_size(n: int, epsilon: float, delta: float, iterations: int | None) -> dict:
    N = 2**n
    target = N - 1
    iters = optimal_iterations(n) if iterations is None else iterations

    G = noisy_grover(n, iters, epsilon, target)
    rho = DensityVector.from_tensor(initial_state(n))

    tracemalloc.start()
    start = time.perf_counter()
    nqb = find_smallest_common_subspace(matrices=(G,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta)
    elapsed = time.perf_counter() - start
    memory = tracemalloc.get_traced_memory()[1] / (2**20)
    tracemalloc.stop()

    return {
        "qubits": n, "state_dim": N, "ambient_dim": nqb.ambient_dimension(),
        "reduced_dim": nqb.dim(), "iterations": iters, "target": target, "time": elapsed, "memory": memory,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-qubits", type=int, default=2, help="smallest register size to try (default: 2)")
    parser.add_argument("--max-qubits", type=int, default=10, help="largest register size to try (default: 10)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="gate-failure probability per layer (default: 1e-3)")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--iterations", type=int, default=None, help="fixed number of Grover iterations (default: optimal per size)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="stop the sweep after a size takes longer than this (checked between sizes, not mid-computation; default: 30s)")
    parser.add_argument("--csv", type=str, default=None, help="append results to this CSV file (created with a header if new)")
    args = parser.parse_args()

    print("Stress test: noisy quantum bisimulation on Grover's algorithm")
    print(f"epsilon={args.epsilon:g}  delta={args.delta:g}  max-seconds/size={args.max_seconds:g}")
    print(f"{'qubits':>6} {'|states>':>9} {'iters':>6} {'time (s)':>10} {'lumped':>8} {'ambient':>9} {'ratio':>8} {'factor':>10}")
    sys.stdout.flush()

    for n in range(args.min_qubits, args.max_qubits + 1):
        result = run_size(n, args.epsilon, args.delta, args.iterations)
        ratio = result["reduced_dim"] / result["ambient_dim"]
        factor = result["ambient_dim"] / result["reduced_dim"]
        print(
            f"{result['qubits']:>6} {result['state_dim']:>9} {result['iterations']:>6} "
            f"{result['time']:>10.4f} {result['reduced_dim']:>8} {result['ambient_dim']:>9} "
            f"{ratio:>8.5f} {factor:>9.1f}x"
        )
        sys.stdout.flush()

        if args.csv:
            append_row(args.csv, CSV_HEADER, [
                result["qubits"], result["state_dim"], result["iterations"], result["target"], args.epsilon,
                args.delta, result["time"], result["memory"], result["ambient_dim"], result["reduced_dim"],
                ratio, f"target={result['target']}",
            ])

        if result["time"] > args.max_seconds:
            print(f"-- stopping: size n={n} took {result['time']:.1f}s, over the {args.max_seconds:g}s budget --")
            break

if __name__ == "__main__":
    main()
