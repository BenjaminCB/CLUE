from __future__ import annotations
import argparse, sys, os, time, tracemalloc
from math import sqrt, pi, floor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..")) # clue is here

from clue.linalg import SparseRowMatrix as Circuit, SparseVector as State, NumericalSubspace, find_smallest_common_subspace
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator, DensityVector

from circuits import Gate, kron_pow, H, h, hadamard_layer, mcz, noisy_circuit, prepare, x
from results import append_row

CSV_HEADER = [
    "qubits", "state_dim", "iterations", "target", "epsilon", "delta",
    "time_lumping", "memory_mb", "ambient_dim", "reduced_dim", "red_ratio", "gate",
    "noise_model", "gate_count", "prep_gate_count",
]

NOISE_MODEL = "depolarizing-per-gate-ibm" # recorded per row: `epsilon` means something different
                                         # here than in the older block-level results

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
    r'''The uniform superposition. Kept as the reference the noisy `hadamard_layer` is checked
    against; the stress test itself starts from `zero_state` and prepares this *noisily*.'''
    v = State(2**n, CC)
    v[0] = 1
    return v.apply_matrix(kron_pow(H, n))

## The two functions above build the oracle and diffuser as single collapsed matrices. They are kept
## because `tests/` cross-checks the gate decompositions below against them, but the stress test no
## longer uses them: noise has to act on the gates that *implement* these blocks, not on the blocks.
def oracle_gates(n: int, target: int) -> list[Gate]:
    r'''
        Phase-flip oracle, in the IBM basis. `MCZ` flips the sign of `\ket{1\ldots1}`, so conjugating
        with an `X` on every qubit where ``target`` has a 0 bit moves the flip onto ``target``.
    '''
    flips = [x(q) for q in range(n) if not ((target >> (n-1-q)) & 1)]
    return flips + mcz(range(n)) + flips

def diffuser_gates(n: int) -> list[Gate]:
    r'''
        `2\ket{s}\bra{s} - I = H^{\otimes n}(2\ket{0}\bra{0} - I)H^{\otimes n}` in the IBM basis.
        `2\ket{0}\bra{0} - I` is, up to a global sign, the phase flip on `\ket{0\ldots0}`, i.e.
        `X^{\otimes n}\, MCZ\, X^{\otimes n}`.
    '''
    layer_h = [gate for q in range(n) for gate in h(q)]
    layer_x = [x(q) for q in range(n)]
    return layer_h + layer_x + mcz(range(n)) + layer_x + layer_h

def prep_gates(n: int) -> list[Gate]:
    r'''State preparation, run **once**: the uniform superposition over all `2^n` basis states.'''
    return hadamard_layer(n)

def iteration_gates(n: int, target: int) -> list[Gate]:
    r'''
        One Grover iteration -- the oracle/diffuser loop -- which is the operator the lumping
        iterates.

        This matches the noise-free predecessor, whose CLUE path lumps a *single* Grover operator
        (`papers/quantum/q_search.py:40`); that is the operator its Theorem 4.2 `d = 2` describes.
        The iteration count deliberately does not enter here: feeding in `G^m` would measure the
        invariant subspace of `G^m` rather than of `G`, and how far the dynamics runs is what the
        Krylov termination discovers by itself.
    '''
    return oracle_gates(n, target) + diffuser_gates(n)

def optimal_iterations(n: int) -> int:
    r'''Standard Grover iteration count `\lfloor (\pi/4)\sqrt{N}\rfloor` (at least 1).'''
    return max(1, floor(pi/4*sqrt(2**n)))

## --------------------------------------------------------------------------
## Stress test loop
## --------------------------------------------------------------------------
def run_size(n: int, epsilon: float, delta: float) -> dict:
    N = 2**n
    target = N - 1

    preparation = prep_gates(n)
    gates = iteration_gates(n, target)
    rho = prepare(preparation, n, epsilon) # noisy, once -- generally a mixed state
    G = noisy_circuit(gates, n, epsilon)   # the repeated operator

    tracemalloc.start()
    start = time.perf_counter()
    nqb = find_smallest_common_subspace(matrices=(G,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta)
    elapsed = time.perf_counter() - start
    memory = tracemalloc.get_traced_memory()[1] / (2**20)
    tracemalloc.stop()

    return {
        "qubits": n, "state_dim": N, "ambient_dim": nqb.ambient_dimension(),
        "reduced_dim": nqb.dim(), "iterations": optimal_iterations(n), "target": target,
        "time": elapsed, "memory": memory,
        "gate_count": len(gates), "prep_gate_count": len(preparation),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-qubits", type=int, default=2, help="smallest register size to try (default: 2)")
    parser.add_argument("--max-qubits", type=int, default=6, help="largest register size to try (default: 6; the repeated operator is one Grover iteration, whose ancilla-free MCZ is Theta(n^2) gates)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="depolarizing parameter of one SX-length gate; each gate gets epsilon*duration/duration_sx, so RZ is noiseless and CZ is ~3.5x SX (default: 1e-3)")
    parser.add_argument("--delta", type=float, default=1e-6, help="absorption threshold for NumericalSubspace (default: 1e-6)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="stop the sweep after a size takes longer than this (checked between sizes, not mid-computation; default: 30s)")
    parser.add_argument("--csv", type=str, default=None, help="append results to this CSV file (created with a header if new)")
    args = parser.parse_args()

    print("Stress test: noisy quantum bisimulation on Grover's algorithm")
    print("noise: depolarizing, per gate, IBM Heron basis {CZ, RZ, SX, X}")
    print("structure: noisy preparation once from |0...0>, then the oracle/diffuser loop is iterated")
    print(f"epsilon={args.epsilon:g}  delta={args.delta:g}  max-seconds/size={args.max_seconds:g}")
    print(f"{'qubits':>6} {'|states>':>9} {'iters':>6} {'A gates':>8} {'time (s)':>10} {'lumped':>8} {'ambient':>9} {'ratio':>8} {'factor':>10}")
    sys.stdout.flush()

    for n in range(args.min_qubits, args.max_qubits + 1):
        result = run_size(n, args.epsilon, args.delta)
        ratio = result["reduced_dim"] / result["ambient_dim"]
        factor = result["ambient_dim"] / result["reduced_dim"]
        print(
            f"{result['qubits']:>6} {result['state_dim']:>9} {result['iterations']:>6} "
            f"{result['gate_count']:>8} "
            f"{result['time']:>10.4f} {result['reduced_dim']:>8} {result['ambient_dim']:>9} "
            f"{ratio:>8.5f} {factor:>9.1f}x"
        )
        sys.stdout.flush()

        if args.csv:
            append_row(args.csv, CSV_HEADER, [
                result["qubits"], result["state_dim"], result["iterations"], result["target"], args.epsilon,
                args.delta, result["time"], result["memory"], result["ambient_dim"], result["reduced_dim"],
                ratio, f"target={result['target']}", NOISE_MODEL, result["gate_count"],
                result["prep_gate_count"],
            ])

        if result["time"] > args.max_seconds:
            print(f"-- stopping: size n={n} took {result['time']:.1f}s, over the {args.max_seconds:g}s budget --")
            break

if __name__ == "__main__":
    main()
