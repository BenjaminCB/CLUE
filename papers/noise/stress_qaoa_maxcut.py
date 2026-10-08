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

from circuits import Gate, kron_pow, H, hadamard_layer, noisy_circuit, prepare, rx, rzz
from results import append_row

CSV_HEADER = [
    "vertices", "state_dim", "edges", "layers", "gamma", "beta", "epsilon", "delta",
    "time_lumping", "memory_mb", "ambient_dim", "reduced_dim", "red_ratio", "graph",
    "noise_model", "gate_count", "prep_gate_count",
]

NOISE_MODEL = "depolarizing-per-gate-ibm"

## --------------------------------------------------------------------------
## MaxCut graph and QAOA circuit
## --------------------------------------------------------------------------
def random_graph(n: int, density: float, seed: int) -> list[tuple[int,int]]:
    r'''Erdos-Renyi graph on ``n`` vertices; ``density=1.0`` gives the complete graph.'''
    rng = Random(seed)
    edges = [e for e in combinations(range(n), 2) if rng.random() < density]
    return edges or [(0, 1 % n)] if n > 1 else []

def cut_cost(state: int, edges: list[tuple[int,int]]) -> int:
    r'''Number of edges cut by the bipartition encoded by the bits of ``state``.'''
    return sum(1 for (u,v) in edges if ((state >> u) & 1) != ((state >> v) & 1))

def cost_unitary(n: int, edges: list[tuple[int,int]], gamma: float) -> Circuit:
    N = 2**n
    U = Circuit(N, CC)
    for state in range(N):
        c = cut_cost(state, edges)
        U.increment(state, state, CC(complex(cos(gamma*c), -sin(gamma*c))))
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
    r'''Reference for the noisy `hadamard_layer`; the stress test starts from `zero_state`.'''
    v = State(2**n, CC)
    v[0] = 1
    return v.apply_matrix(kron_pow(H, n))

## `cost_unitary` and `mixer_unitary` above build whole layers as single matrices; they are kept only
## so the gate decompositions below can be cross-checked against them. Note that the gate-level cost
## layer is *cheaper* than the matrix it replaces: 3 gates per edge instead of enumerating all 2^n
## diagonal entries.
def cost_gates(n: int, edges: list[tuple[int,int]], gamma: float) -> list[Gate]:
    r'''
        `e^{-i\gamma\,\text{cut}(x)}` in the IBM basis. Writing
        `\text{cut} = \sum_{(u,v) \in E} (1 - z_u z_v)/2` with `z_j` the `Z` eigenvalue, this is
        `\prod_{(u,v)} e^{i\gamma Z_u Z_v/2}` up to the global phase `e^{-i\gamma|E|/2}`, i.e. one
        `RZZ(-\gamma)` per edge -- exactly, with no synthesis error.

        Qubit indices are flipped because `cut_cost` reads bit `u` of the state index while the gate
        layer counts qubits from the most significant bit (see `embedN`).
    '''
    return [gate for (u,v) in edges for gate in rzz(-gamma, n-1-u, n-1-v)]

def mixer_gates(n: int, beta: float) -> list[Gate]:
    r'''
        The mixer layer in the IBM basis. ``mixer1(beta)`` is `e^{-i\beta X}`, which is `RX(2\beta)`,
        so this is `RX(2\beta)` on every qubit -- 3 `RZ` + 2 `SX` each, again exact.
    '''
    return [gate for q in range(n) for gate in rx(2*beta, q)]

def prep_gates(n: int) -> list[Gate]:
    r'''State preparation, run **once**: the uniform superposition.'''
    return hadamard_layer(n)

def layer_gates(n: int, edges: list[tuple[int,int]], gamma: float, beta: float) -> list[Gate]:
    r'''
        One QAOA layer (cost then mixer) -- the operator the lumping iterates. Since `gamma` and
        `beta` are shared across layers, iterating this `p` times *is* the `p`-layer circuit, so the
        layer count does not belong in the operator.

        Caveat worth recording: this is NOT what the noise-free predecessor repeats. Its CLUE path
        lumps `cut_matrix` (`papers/quantum/q_maxcut.py:106`) -- the diagonal matrix of cut *counts*,
        not a unitary -- and its expected reduced size is the number of distinct cut values. This is
        the circuit analogue, which is a different experiment.
    '''
    return cost_gates(n, edges, gamma) + mixer_gates(n, beta)

## --------------------------------------------------------------------------
## Stress test loop
## --------------------------------------------------------------------------
def run_size(n: int, density: float, seed: int, gamma: float, beta: float, epsilon: float, delta: float) -> dict:
    N = 2**n
    edges = random_graph(n, density, seed)

    preparation = prep_gates(n)
    gates = layer_gates(n, edges, gamma, beta)
    rho = prepare(preparation, n, epsilon) # noisy, once -- generally a mixed state
    G = noisy_circuit(gates, n, epsilon)   # the repeated operator

    tracemalloc.start()
    start = time.perf_counter()
    nqb = find_smallest_common_subspace(matrices=(G,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta)
    elapsed = time.perf_counter() - start
    memory = tracemalloc.get_traced_memory()[1] / (2**20)
    tracemalloc.stop()

    return {
        "vertices": n, "state_dim": N, "ambient_dim": nqb.ambient_dimension(),
        "reduced_dim": nqb.dim(), "edges": edges, "time": elapsed, "memory": memory,
        "gate_count": len(gates), "prep_gate_count": len(preparation),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-vertices", type=int, default=2, help="smallest graph size to try (default: 2)")
    parser.add_argument("--max-vertices", type=int, default=4, help="largest graph size to try (default: 4; unlike Grover, this circuit shows no lumping, so cost grows fast -- raise with care)")
    parser.add_argument("--density", type=float, default=1.0, help="edge probability, 1.0 = complete graph (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the graph (default: 42)")
    parser.add_argument("--gamma", type=float, default=0.7, help="cost-unitary angle, shared by all layers (default: 0.7)")
    parser.add_argument("--beta", type=float, default=0.4, help="mixer angle, shared by all layers (default: 0.4)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="depolarizing parameter of one SX-length gate; each gate gets epsilon*duration/duration_sx, so RZ is noiseless and CX is ~11x SX (default: 1e-3)")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="stop the sweep after a size takes longer than this (checked between sizes, not mid-computation; default: 30s)")
    parser.add_argument("--csv", type=str, default=None, help="append results to this CSV file (created with a header if new)")
    args = parser.parse_args()

    print("Stress test: noisy quantum bisimulation on QAOA MaxCut")
    print("noise: depolarizing, per gate, IBM basis {CX, RZ, SX, X}")
    print("structure: noisy preparation once from |0...0>, then one cost+mixer layer is iterated")
    print(
        f"density={args.density:g}  gamma={args.gamma:g}  beta={args.beta:g}  "
        f"epsilon={args.epsilon:g}  delta={args.delta:g}  max-seconds/size={args.max_seconds:g}"
    )
    print(f"{'vertices':>8} {'|states>':>9} {'edges':>6} {'A gates':>7} {'time (s)':>10} {'lumped':>8} {'ambient':>9} {'ratio':>8} {'factor':>10}")
    sys.stdout.flush()

    for n in range(args.min_vertices, args.max_vertices + 1):
        result = run_size(n, args.density, args.seed, args.gamma, args.beta, args.epsilon, args.delta)
        ratio = result["reduced_dim"] / result["ambient_dim"]
        factor = result["ambient_dim"] / result["reduced_dim"]
        print(
            f"{result['vertices']:>8} {result['state_dim']:>9} {len(result['edges']):>6} "
            f"{result['gate_count']:>7} "
            f"{result['time']:>10.4f} {result['reduced_dim']:>8} {result['ambient_dim']:>9} "
            f"{ratio:>8.5f} {factor:>9.1f}x"
        )
        sys.stdout.flush()

        if args.csv:
            append_row(args.csv, CSV_HEADER, [
                result["vertices"], result["state_dim"], len(result["edges"]), 1, args.gamma, args.beta,
                args.epsilon, args.delta, result["time"], result["memory"], result["ambient_dim"],
                result["reduced_dim"], ratio, repr(result["edges"]), NOISE_MODEL, result["gate_count"],
                result["prep_gate_count"],
            ])

        if result["time"] > args.max_seconds:
            print(f"-- stopping: size n={n} took {result['time']:.1f}s, over the {args.max_seconds:g}s budget --")
            break

if __name__ == "__main__":
    main()
