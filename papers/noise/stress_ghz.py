from __future__ import annotations
import argparse, sys, os, time
from math import sqrt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..")) # clue is here

from numpy import kron
from clue.linalg import SparseRowMatrix as Circuit, SparseVector as State, NumericalSubspace, find_smallest_common_subspace
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator, DensityVector

## --------------------------------------------------------------------------
## Small gate-building toolkit (dense kron via numpy, mirroring papers/noise/script.py)
## --------------------------------------------------------------------------
def kronecker(A: Circuit, B: Circuit) -> Circuit:
    return Circuit.from_list(kron(A.to_numpy(), B.to_numpy()), CC)

def kron_pow(A: Circuit, n: int) -> Circuit:
    result = A
    for _ in range(1, n):
        result = kronecker(result, A)
    return result

def embed1(gate: Circuit, i: int, n: int) -> Circuit:
    r'''Embed a 1-qubit gate acting on qubit ``i`` into an ``n``-qubit register.'''
    result = gate
    if n-i-1 > 0:
        result = kronecker(result, kron_pow(I, n-i-1))
    if i > 0:
        result = kronecker(kron_pow(I, i), result)
    return result

def embed2(gate: Circuit, i: int, n: int) -> Circuit:
    r'''Embed a 2-qubit gate acting on qubits ``(i, i+1)`` into an ``n``-qubit register.'''
    result = gate
    if n-i-2 > 0:
        result = kronecker(result, kron_pow(I, n-i-2))
    if i > 0:
        result = kronecker(kron_pow(I, i), result)
    return result

I = Circuit.eye(2, CC)
X = Circuit(2, CC); X.increment(0,1,1); X.increment(1,0,1)
H = Circuit(2, CC)
H.increment(0,0,1/sqrt(2)); H.increment(0,1,1/sqrt(2))
H.increment(1,0,1/sqrt(2)); H.increment(1,1,-1/sqrt(2))
CX = Circuit(4, CC)
CX.increment(0,0,1); CX.increment(1,1,1); CX.increment(2,3,1); CX.increment(3,2,1)

## --------------------------------------------------------------------------
## GHZ preparation and per-qubit bit-flip noise
## --------------------------------------------------------------------------
def ghz_state(n: int) -> State:
    v = State(2**n, CC)
    v[0] = 1
    V = embed1(H, 0, n)
    for i in range(n-1):
        V = embed2(CX, i, n) * V
    return v.apply_matrix(V)

def bitflip_round(n: int, p: float) -> DensityOperator:
    r'''One round of independent bit-flip noise (probability ``p``) on every qubit.'''
    N = 2**n
    identity = Circuit.eye(N, CC)
    channels = [DensityOperator(circuits=[identity, embed1(X, i, n)], probabilities=[1-p, p]) for i in range(n)]
    return DensityOperator(operators=channels)

## --------------------------------------------------------------------------
## Stress test loop
## --------------------------------------------------------------------------
def run_size(n: int, p: float, delta: float) -> dict:
    N = 2**n
    rho = DensityVector.from_tensor(ghz_state(n))
    R = bitflip_round(n, p)

    start = time.perf_counter()
    nqb = find_smallest_common_subspace(matrices=(R,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta)
    elapsed = time.perf_counter() - start

    return {"qubits": n, "state_dim": N, "ambient_dim": nqb.ambient_dimension(), "reduced_dim": nqb.dim(), "time": elapsed}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-qubits", type=int, default=2, help="smallest register size to try (default: 2)")
    parser.add_argument("--max-qubits", type=int, default=12, help="largest register size to try (default: 12)")
    parser.add_argument("--p", type=float, default=0.05, help="per-qubit bit-flip probability per round (default: 0.05)")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="stop the sweep after a size takes longer than this (checked between sizes, not mid-computation; default: 30s)")
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

        if result["time"] > args.max_seconds:
            print(f"-- stopping: size n={n} took {result['time']:.1f}s, over the {args.max_seconds:g}s budget --")
            break

if __name__ == "__main__":
    main()
