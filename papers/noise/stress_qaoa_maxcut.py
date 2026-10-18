r'''
    Stress test: noisy quantum bisimulation on the MaxCut cost layer.

    The gate-level noisy counterpart of the noise-free `papers/quantum/q_maxcut.py`, which lumps
    `cut_matrix` as the generator of `exp(-i*eigenvalue)`; the operator iterated here is that
    exponential, i.e. one `RZZ(-gamma)` per edge, with no QAOA mixer. Noise-free, its reduced
    dimension is the number of distinct cut *differences* (`cut_difference_count`), not the
    distinct cut values of the predecessor.
'''
from __future__ import annotations
import argparse, sys, os, time, tracemalloc
from itertools import combinations
from math import cos, sin
from random import Random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..")) # clue is here

from clue.linalg import SparseRowMatrix as Circuit, SparseVector as State, NumericalSubspace, find_smallest_common_subspace
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator, DensityVector

from circuits import Gate, kron_pow, H, hadamard_layer, noisy_circuit, prepare, rzz
from results import append_row

CSV_HEADER = [
    "vertices", "state_dim", "edges", "gamma", "epsilon", "delta",
    "time_lumping", "memory_mb", "ambient_dim", "reduced_dim", "red_ratio", "graph",
    "noise_model", "gate_count", "prep_gate_count", "predicted_noisefree_dim",
]

NOISE_MODEL = "depolarizing-per-gate-ibm"

## --------------------------------------------------------------------------
## MaxCut graph and the cut matrix
## --------------------------------------------------------------------------
def random_graph(n: int, density: float, seed: int) -> list[tuple[int,int]]:
    r'''Erdos-Renyi graph on ``n`` vertices; ``density=1.0`` gives the complete graph.'''
    rng = Random(seed)
    edges = [e for e in combinations(range(n), 2) if rng.random() < density]
    return edges or [(0, 1 % n)] if n > 1 else []

def cut_cost(state: int, edges: list[tuple[int,int]]) -> int:
    r'''Number of edges cut by the bipartition encoded by the bits of ``state``.'''
    return sum(1 for (u,v) in edges if ((state >> u) & 1) != ((state >> v) & 1))

def cut_matrix(n: int, edges: list[tuple[int,int]]) -> Circuit:
    r'''The diagonal matrix of cut counts; the reference ``cost_gates`` is checked against.'''
    N = 2**n
    M = Circuit(N, CC)
    for state in range(N):
        M.increment(state, state, CC(complex(cut_cost(state, edges), 0)))
    return M

def cut_values(n: int, edges: list[tuple[int,int]]) -> set[int]:
    r'''The distinct cut counts: the noise-free predecessor's expected reduced size.'''
    return {cut_cost(state, edges) for state in range(2**n)}

def cut_difference_count(n: int, edges: list[tuple[int,int]]) -> int:
    r'''The number of distinct `c_i - c_j`: the predicted reduced dimension in density space.'''
    values = cut_values(n, edges)
    return len({a - b for a in values for b in values})

def cost_unitary(n: int, edges: list[tuple[int,int]], gamma: float) -> Circuit:
    r'''`exp(-i\gamma\,\text{cut})` as one collapsed matrix; the reference for ``cost_gates``.'''
    N = 2**n
    U = Circuit(N, CC)
    for state in range(N):
        c = cut_cost(state, edges)
        U.increment(state, state, CC(complex(cos(gamma*c), -sin(gamma*c))))
    return U

def initial_state(n: int) -> State:
    r'''Reference for the noisy ``hadamard_layer``: the uniform superposition.'''
    v = State(2**n, CC)
    v[0] = 1
    return v.apply_matrix(kron_pow(H, n))

## --------------------------------------------------------------------------
## The circuit
## --------------------------------------------------------------------------
## `cost_unitary` above is kept only so the gate decomposition below can be checked against it.
def cost_gates(n: int, edges: list[tuple[int,int]], gamma: float) -> list[Gate]:
    r'''
        `e^{-i\gamma\,\text{cut}(x)}` in the IBM basis: one `RZZ(-\gamma)` per edge, exactly, up to
        a global phase. Qubit indices are flipped because `cut_cost` reads bit `u` of the state
        index while the gate layer counts qubits from the most significant bit.
    '''
    return [gate for (u,v) in edges for gate in rzz(-gamma, n-1-u, n-1-v)]

def prep_gates(n: int) -> list[Gate]:
    r'''State preparation, run **once**: the uniform superposition.'''
    return hadamard_layer(n)

## --------------------------------------------------------------------------
## Stress test loop
## --------------------------------------------------------------------------
def run_size(n: int, density: float, seed: int, gamma: float, epsilon: float, delta: float) -> dict:
    N = 2**n
    edges = random_graph(n, density, seed)

    preparation = prep_gates(n)
    gates = cost_gates(n, edges, gamma)
    rho = prepare(preparation, n, epsilon) # noisy, once -- generally a mixed state
    C = noisy_circuit(gates, n, epsilon)   # the repeated operator: exp(-i gamma cut)

    tracemalloc.start()
    start = time.perf_counter()
    nqb = find_smallest_common_subspace(matrices=(C,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta)
    elapsed = time.perf_counter() - start
    memory = tracemalloc.get_traced_memory()[1] / (2**20)
    tracemalloc.stop()

    return {
        "vertices": n, "state_dim": N, "ambient_dim": nqb.ambient_dimension(),
        "reduced_dim": nqb.dim(), "edges": edges, "time": elapsed, "memory": memory,
        "gate_count": len(gates), "prep_gate_count": len(preparation),
        "cut_values": len(cut_values(n, edges)), "predicted": cut_difference_count(n, edges),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-vertices", type=int, default=2, help="smallest graph size to try (default: 2)")
    parser.add_argument("--max-vertices", type=int, default=4, help="largest graph size to try (default: 4)")
    parser.add_argument("--density", type=float, default=1.0, help="edge probability, 1.0 = complete graph (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the graph (default: 42)")
    parser.add_argument("--gamma", type=float, default=1.0, help="cost-layer angle (default: 1.0, matching exp(-i*eigenvalue) in the noise-free cut_split)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="depolarizing parameter of one SX-length gate; each gate gets epsilon*duration/duration_sx, so RZ is noiseless and CX is ~11x SX (default: 1e-3; pass 0 for the noise-free control)")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="stop the sweep after a size takes longer than this (checked between sizes, not mid-computation; default: 30s)")
    parser.add_argument("--csv", type=str, default=None, help="append results to this CSV file (created with a header if new)")
    args = parser.parse_args()

    print("Stress test: noisy quantum bisimulation on the MaxCut cost layer")
    print("noise: depolarizing, per gate, IBM basis {CX, RZ, SX, X}")
    print("structure: noisy preparation once from |0...0>, then exp(-i*gamma*cut) is iterated")
    print("           (the circuit form of the cut_matrix the noise-free test lumps; no QAOA mixer)")
    print("predicted: distinct cut differences |{c_i - c_j}| in density space, against the")
    print("           predecessor's distinct cut values |{c_i}| in state space")
    print(
        f"density={args.density:g}  gamma={args.gamma:g}  "
        f"epsilon={args.epsilon:g}  delta={args.delta:g}  max-seconds/size={args.max_seconds:g}"
    )
    print(f"{'vertices':>8} {'|states>':>9} {'edges':>6} {'A gates':>7} {'time (s)':>10} {'lumped':>8} {'predict':>8} {'cuts':>5} {'ambient':>9} {'factor':>10}")
    sys.stdout.flush()

    for n in range(args.min_vertices, args.max_vertices + 1):
        result = run_size(n, args.density, args.seed, args.gamma, args.epsilon, args.delta)
        ratio = result["reduced_dim"] / result["ambient_dim"]
        factor = result["ambient_dim"] / result["reduced_dim"]
        print(
            f"{result['vertices']:>8} {result['state_dim']:>9} {len(result['edges']):>6} "
            f"{result['gate_count']:>7} "
            f"{result['time']:>10.4f} {result['reduced_dim']:>8} {result['predicted']:>8} "
            f"{result['cut_values']:>5} {result['ambient_dim']:>9} {factor:>9.1f}x"
        )
        sys.stdout.flush()

        if args.csv:
            append_row(args.csv, CSV_HEADER, [
                result["vertices"], result["state_dim"], len(result["edges"]), args.gamma,
                args.epsilon, args.delta, result["time"], result["memory"], result["ambient_dim"],
                result["reduced_dim"], ratio, repr(result["edges"]), NOISE_MODEL, result["gate_count"],
                result["prep_gate_count"], result["predicted"],
            ])

        if result["time"] > args.max_seconds:
            print(f"-- stopping: size n={n} took {result['time']:.1f}s, over the {args.max_seconds:g}s budget --")
            break

if __name__ == "__main__":
    main()
