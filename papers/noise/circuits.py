from __future__ import annotations
from math import sqrt
from numpy import kron

from clue.linalg import SparseRowMatrix as Circuit
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator

## --------------------------------------------------------------------------
## Dense-kron circuit-building toolkit
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

## --------------------------------------------------------------------------
## Elementary gates
## --------------------------------------------------------------------------
I = Circuit.eye(2, CC)

X = Circuit(2, CC)
X.increment(0,1,1); X.increment(1,0,1)

Y = Circuit(2, CC)
Y.increment(1,0,CC(1j)); Y.increment(0,1,CC(-1j))

H = Circuit(2, CC)
H.increment(0,0,1/sqrt(2)); H.increment(0,1,1/sqrt(2))
H.increment(1,0,1/sqrt(2)); H.increment(1,1,-1/sqrt(2))

CX = Circuit(4, CC)
CX.increment(0,0,1); CX.increment(1,1,1); CX.increment(2,3,1); CX.increment(3,2,1)

## --------------------------------------------------------------------------
## Noise channels
##
## Each constructor below takes an ideal circuit (or register size) and returns a
## `DensityOperator` super-operator implementing a specific noise model. Add new noise
## channels here so they can be shared between `script.py` and the `stress_*.py` benchmarks.
## --------------------------------------------------------------------------
def gate_failure_channel(circuit: Circuit, epsilon: float) -> DensityOperator:
    r'''
        "Gate failure" noise: with probability ``1-epsilon`` apply the ideal ``circuit``, with
        probability ``epsilon`` it fails and the identity is applied instead.
    '''
    N = circuit.nrows
    return DensityOperator(circuits=[circuit, Circuit.eye(N, CC)], probabilities=[1-epsilon, epsilon])

def bitflip_channel(n: int, p: float) -> DensityOperator:
    r'''
        One round of independent bit-flip noise: every qubit of an ``n``-qubit register flips
        (an ``X`` is applied) independently with probability ``p``.
    '''
    N = 2**n
    identity = Circuit.eye(N, CC)
    channels = [DensityOperator(circuits=[identity, embed1(X, i, n)], probabilities=[1-p, p]) for i in range(n)]
    return DensityOperator(operators=channels)
