import sys
from pathlib import Path

# pyright: reportArgumentType=false

# clue is here
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from clue.linalg import SparseRowMatrix as Circuit, SparseVector as State, NumericalSubspace, find_smallest_common_subspace
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator, DensityVector

from math import sqrt, log10, floor

from circuits import kronecker, kron_pow, I, X, Y, H, CX, gate_failure_channel

plus = State(2, CC)
plus[0], plus[1] = 1/sqrt(2), 1/sqrt(2)

minus = State(2, CC)
minus[0], minus[1] = 1/sqrt(2), -1/sqrt(2)

zero = State(8,CC)
zero[0] = 1

# Matrix for the composition of each layer in GHZ
U_1 = kronecker(kronecker(H,I),I) # Goes from 4x4 (the first kronecker) to 8x8 (the second)
U_2 = kronecker(CX,I) # 4x4 -> 8x8
U_3 = kronecker(I,CX) # 2x2 -> 8x8

# Identity 8
I8 = Circuit.eye(8, CC)

def CnNOT(n : int) -> Circuit:
    N = 2**n
    output = Circuit.eye(N,CC)
    output.increment(N-1, N-1, -1)
    output.increment(N-1, N-2, 1)
    
    output.increment(N-2, N-1, 1)
    output.increment(N-2, N-2, -1)

    return output

def not_CnNOT(n: int) -> Circuit:
    N = 2**n
    output = Circuit.eye(N,CC)
    output.increment(0, 0, -1)
    output.increment(0, 1, 1)
    
    output.increment(1, 0, 1)
    output.increment(1, 1, -1)

    return output

def G(n: int, epsilon: float) -> DensityOperator:
    O = CnNOT(n+1)
    P = [kronecker(kron_pow(H, n), I), kronecker(kron_pow(I, n), X), not_CnNOT(n+1), kronecker(kron_pow(H, n), I)]

    operators = [gate_failure_channel(circ, epsilon) for circ in [O] + P]
    return DensityOperator(operators=operators)

def G_input(n: int) -> DensityVector:
    v = State(2**(n+1), CC)
    v[1] = 1

    return DensityVector.from_tensor(v.apply_matrix(kron_pow(H, n+1)))

def run(G: DensityOperator, v: DensityVector):
    return find_smallest_common_subspace(
        (G,),
        (v,),
        subspace_class=NumericalSubspace
    )

def evolution(U, v, starting:float, finishing:float, increase="log") -> tuple[tuple[float,int]]:
    epsilon = starting
    result = []
    while epsilon < finishing:
        print(f"Computing the reduction with noise={epsilon:.04f}", flush=True, end="\r")
        S = run(U(epsilon), v(epsilon))
        result.append((epsilon,S.dim()))

        if increase == "log":
            epsilon = epsilon + 10**floor(log10(epsilon))
        elif increase == "linear":
            epsilon += starting

    return tuple(result)

import matplotlib.pyplot as plt

def plot(result: tuple[tuple[float, int]], scale: str = "linear"):
    xvalues, yvalues = list(zip(*result))
    plt.plot(xvalues, yvalues, 'o', linestyle="-")
    plt.xscale(scale)
    plt.show()
