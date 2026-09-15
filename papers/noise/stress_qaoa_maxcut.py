from __future__ import annotations
import argparse, sys, os, time
from itertools import combinations
from math import sqrt, cos, sin
from random import Random

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

H = Circuit(2, CC)
H.increment(0,0,1/sqrt(2)); H.increment(0,1,1/sqrt(2))
H.increment(1,0,1/sqrt(2)); H.increment(1,1,-1/sqrt(2))

## --------------------------------------------------------------------------
## MaxCut graph and QAOA circuit
## --------------------------------------------------------------------------
def random_graph(n: int, density: float, seed: int) -> list[tuple[int,int]]:
    r'''Erdos-Renyi graph on ``n`` vertices; ``density=1.0`` gives the complete graph.'''
    rng = Random(seed)
    edges = [e for e in combinations(range(n), 2) if rng.random() < density]
    return edges or [(0, 1 % n)] if n > 1 else []

def cut_cost(x: int, edges: list[tuple[int,int]]) -> int:
    r'''Number of edges cut by the bipartition encoded by the bits of ``x``.'''
    return sum(1 for (u,v) in edges if ((x >> u) & 1) != ((x >> v) & 1))

def cost_unitary(n: int, edges: list[tuple[int,int]], gamma: float) -> Circuit:
    N = 2**n
    U = Circuit(N, CC)
    for x in range(N):
        c = cut_cost(x, edges)
        U.increment(x, x, CC(complex(cos(gamma*c), -sin(gamma*c))))
    return U

def mixer1(beta: float) -> Circuit:
    r'''Single-qubit mixer `e^{-i\beta X}`.'''
    M = Circuit(2, CC)
    M.increment(0,0, CC(complex(cos(beta), 0)));   M.increment(0,1, CC(complex(0, -sin(beta))))
    M.increment(1,0, CC(complex(0, -sin(beta))));  M.increment(1,1, CC(complex(cos(beta), 0)))
    return M

def mixer_unitary(n: int, beta: float) -> Circuit:
    return kron_pow(mixer1(beta), n)

def initial_state(n: int) -> State:
    v = State(2**n, CC)
    v[0] = 1
    return v.apply_matrix(kron_pow(H, n))

def noisy_layer(circuit: Circuit, epsilon: float) -> DensityOperator:
    r'''With probability ``1-epsilon`` apply the ideal gate; with probability ``epsilon`` it fails (identity).'''
    N = circuit.nrows
    return DensityOperator(circuits=[circuit, Circuit.eye(N, CC)], probabilities=[1-epsilon, epsilon])

def noisy_qaoa(n: int, edges: list[tuple[int,int]], layers: int, gamma: float, beta: float, epsilon: float) -> DensityOperator:
    Uc = noisy_layer(cost_unitary(n, edges, gamma), epsilon)
    Ub = noisy_layer(mixer_unitary(n, beta), epsilon)
    return DensityOperator(operators=layers*[Uc, Ub])

## --------------------------------------------------------------------------
## Stress test loop
## --------------------------------------------------------------------------
def run_size(n: int, density: float, seed: int, layers: int, gamma: float, beta: float, epsilon: float, delta: float) -> dict:
    N = 2**n
    edges = random_graph(n, density, seed)

    G = noisy_qaoa(n, edges, layers, gamma, beta, epsilon)
    rho = DensityVector.from_tensor(initial_state(n))

    start = time.perf_counter()
    nqb = find_smallest_common_subspace(matrices=(G,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta)
    elapsed = time.perf_counter() - start

    return {
        "vertices": n, "state_dim": N, "ambient_dim": nqb.ambient_dimension(),
        "reduced_dim": nqb.dim(), "edges": len(edges), "time": elapsed,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-vertices", type=int, default=2, help="smallest graph size to try (default: 2)")
    parser.add_argument("--max-vertices", type=int, default=4, help="largest graph size to try (default: 4; unlike Grover, this circuit shows no lumping, so cost grows fast -- raise with care)")
    parser.add_argument("--density", type=float, default=1.0, help="edge probability, 1.0 = complete graph (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the graph (default: 42)")
    parser.add_argument("--layers", type=int, default=1, help="number of QAOA (cost+mixer) layers p (default: 1)")
    parser.add_argument("--gamma", type=float, default=0.7, help="cost-unitary angle, shared by all layers (default: 0.7)")
    parser.add_argument("--beta", type=float, default=0.4, help="mixer angle, shared by all layers (default: 0.4)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="gate-failure probability per layer (default: 1e-3)")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="stop the sweep after a size takes longer than this (checked between sizes, not mid-computation; default: 30s)")
    args = parser.parse_args()

    print("Stress test: noisy quantum bisimulation on QAOA MaxCut")
    print(
        f"density={args.density:g}  layers={args.layers}  gamma={args.gamma:g}  beta={args.beta:g}  "
        f"epsilon={args.epsilon:g}  delta={args.delta:g}  max-seconds/size={args.max_seconds:g}"
    )
    print(f"{'vertices':>8} {'|states>':>9} {'edges':>6} {'time (s)':>10} {'lumped':>8} {'ambient':>9} {'ratio':>8} {'factor':>10}")
    sys.stdout.flush()

    for n in range(args.min_vertices, args.max_vertices + 1):
        result = run_size(n, args.density, args.seed, args.layers, args.gamma, args.beta, args.epsilon, args.delta)
        ratio = result["reduced_dim"] / result["ambient_dim"]
        factor = result["ambient_dim"] / result["reduced_dim"]
        print(
            f"{result['vertices']:>8} {result['state_dim']:>9} {result['edges']:>6} "
            f"{result['time']:>10.4f} {result['reduced_dim']:>8} {result['ambient_dim']:>9} "
            f"{ratio:>8.5f} {factor:>9.1f}x"
        )
        sys.stdout.flush()

        if result["time"] > args.max_seconds:
            print(f"-- stopping: size n={n} took {result['time']:.1f}s, over the {args.max_seconds:g}s budget --")
            break

if __name__ == "__main__":
    main()
