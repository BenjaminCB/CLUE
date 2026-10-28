from __future__ import annotations
from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations
from math import sqrt, cos, sin
from typing import NamedTuple
from numpy import kron
from qiskit import QuantumCircuit, transpile

from clue.linalg import SparseRowMatrix as Circuit, SparseVector as State
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityOperator, DensityVector

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

## --------------------------------------------------------------------------
## General embedding
## --------------------------------------------------------------------------
def embedN(gate: Circuit, qubits: Sequence[int], n: int) -> Circuit:
    r'''
        Embed a ``k``-qubit ``gate`` acting on ``qubits`` into an ``n``-qubit register.

        Unlike :func:`embed2`, the qubits need be neither adjacent nor ordered, which is what
        ``CX`` on an arbitrary graph edge needs. Qubit ``q`` is bit ``n-1-q`` of a basis-state
        index, i.e. qubit 0 is the most significant bit -- the convention of :func:`embed1` /
        :func:`embed2` (the gate goes leftmost in the Kronecker product) and of ``CX`` above
        (control = qubit 0).
    '''
    k = len(qubits)
    if gate.nrows != 2**k:
        raise ValueError(f"A {gate.nrows}x{gate.nrows} gate cannot act on {k} qubits")
    if len(set(qubits)) != k:
        raise ValueError(f"Repeated qubit in {qubits}")

    shifts = [n-1-q for q in qubits]
    mask = 0
    for s in shifts:
        mask |= 1 << s

    def scatter(index: int) -> int:
        r'''Place the ``k`` bits of ``index`` at the register positions of ``qubits``.'''
        out = 0
        for position, shift in enumerate(shifts):
            if (index >> (k-1-position)) & 1:
                out |= 1 << shift
        return out

    untouched = [b for b in range(2**n) if not (b & mask)] # the 2^(n-k) spectator configurations
    result = Circuit(2**n, CC)
    for i in sorted(gate.nonzero):
        row, bi = gate[i], scatter(i)
        for j in sorted(row.nonzero):
            value, bj = row[j], scatter(j)
            for b in untouched:
                result.increment(b | bi, b | bj, value)
    return result

## --------------------------------------------------------------------------
## The IBM Heron basis {CZ, RZ(theta), SX, X}
## --------------------------------------------------------------------------
Z = Circuit(2, CC)
Z.increment(0,0,1); Z.increment(1,1,-1)

CZ = Circuit(4, CC) # diag(1,1,1,-1), symmetric in its two qubits
CZ.increment(0,0,1); CZ.increment(1,1,1); CZ.increment(2,2,1); CZ.increment(3,3,-1)

SX = Circuit(2, CC) # sqrt(X) = (1/2)[[1+i, 1-i], [1-i, 1+i]]; SX*SX = X
SX.increment(0,0,CC(complex( 0.5, 0.5))); SX.increment(0,1,CC(complex( 0.5,-0.5)))
SX.increment(1,0,CC(complex( 0.5,-0.5))); SX.increment(1,1,CC(complex( 0.5, 0.5)))

def RZ(theta: float) -> Circuit:
    r'''`RZ(\theta) = e^{-i\theta Z/2} = \text{diag}(e^{-i\theta/2}, e^{i\theta/2})`.'''
    M = Circuit(2, CC)
    M.increment(0,0, CC(complex(cos(theta/2), -sin(theta/2))))
    M.increment(1,1, CC(complex(cos(theta/2),  sin(theta/2))))
    return M

## Device-typical gate durations, in nanoseconds. `rz` is exactly 0 because a virtual gate cannot
## fail; the others are the whole reason the 1-qubit/2-qubit noise ratio is a documented device
## parameter here rather than a second free knob.
##
## Medians of `gate_length` over all qubits / couplings in the calibration snapshots shipped with
## qiskit-ibm-runtime 0.51.0 (`fake_provider/backends/<name>/props_<name>.json`, 2025-02-26).
DURATIONS = {"rz": 0.0, "sx": 24.0, "x": 24.0, "cz": 84.0} # ibm_fez (Heron r2)
# DURATIONS = {"rz": 0.0, "sx": 32.0, "x": 32.0, "cz": 68.0} # ibm_torino (Heron r1)
# DURATIONS = {"rz": 0.0, "sx": 36.0, "x": 36.0, "cz": 68.0} # ibm_marrakesh (Heron r2)

class Gate(NamedTuple):
    r'''
        One gate of a circuit, kept *un-embedded*: ``matrix`` is `2^k \times 2^k` for the ``k``
        qubits in ``qubits``, so the noise model can see which qubits the gate acts on (a
        depolarising twirl has to be placed on exactly those). ``duration`` is in nanoseconds and
        is what sets the noise strength.

        Gate lists are in **time order**: the first element is applied first, matching
        ``DensityOperator(operators=...)``, which :func:`DensityVector.apply_matrix` walks in list
        order.
    '''
    matrix: Circuit
    qubits: tuple[int, ...]
    duration: float
    name: str

def rz(theta: float, q: int) -> Gate: return Gate(RZ(theta), (q,), DURATIONS["rz"], "rz")
def sx(q: int) -> Gate:               return Gate(SX, (q,), DURATIONS["sx"], "sx")
def x(q: int) -> Gate:                return Gate(X, (q,), DURATIONS["x"], "x")
def cz(c: int, t: int) -> Gate:       return Gate(CZ, (c, t), DURATIONS["cz"], "cz")

## The basis gates `from_qiskit` can turn into `Gate`s, by qiskit name. Adding a gate to a basis
## means adding it here and giving it a duration in `DURATIONS`.
BASIS_GATES = {"rz": rz, "sx": sx, "x": x, "cz": cz}

## --------------------------------------------------------------------------
## Circuits from qiskit
##
## Circuits are written as qiskit `QuantumCircuit`s and transpiled by qiskit, so any gate qiskit
## knows can be used and any subset of `BASIS_GATES` can be the target basis. Only the *gate list*
## comes from qiskit: matrices and durations are ours, so the noise model stays in this file.
## --------------------------------------------------------------------------
def from_qiskit(circuit: QuantumCircuit, basis: Sequence[str] = ("cz", "rz", "sx", "x")) -> list[Gate]:
    r'''
        Transpile ``circuit`` to ``basis`` and convert it to a time-ordered list of `Gate`.

        Transpilation runs at ``optimization_level=0`` with no coupling map, so qiskit only
        decomposes: no gates are merged or cancelled, no SWAPs are routed, and no qubits are
        relabelled. The global phase it introduces is dropped, as it cancels in `\rho \mapsto
        U\rho U^\dagger`.

        qiskit numbers qubits little-endian (qubit ``q`` is bit ``q`` of a basis-state index) while
        :func:`embedN` takes qubit 0 as the most significant bit, so qiskit qubit ``q`` becomes our
        qubit ``n-1-q``. Basis-state indices therefore mean the same thing on both sides, and the
        result has the same matrix as ``qiskit.quantum_info.Operator(circuit)``.
    '''
    unknown = [name for name in basis if name not in BASIS_GATES]
    if unknown:
        raise ValueError(f"No matrix/duration for basis gate(s) {unknown}; add them to BASIS_GATES")

    transpiled = transpile(circuit, basis_gates=list(basis), optimization_level=0)
    n = transpiled.num_qubits
    gates = []
    for instruction in transpiled.data:
        name = instruction.operation.name
        if name == "barrier":
            continue
        if name not in basis:
            raise ValueError(f"Cannot convert '{name}': only unitary gates of {list(basis)} are supported")
        qubits = [n-1-transpiled.find_bit(q).index for q in instruction.qubits]
        params = [float(p) for p in instruction.operation.params]
        gates.append(BASIS_GATES[name](*params, *qubits))
    return gates

## --------------------------------------------------------------------------
## Gate-level depolarising noise
## --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _pauli_basis(k: int) -> tuple[Circuit, ...]:
    r'''The `4^k` tensor products of `(I, X, Y, Z)`, identity first.'''
    single = (I, X, Y, Z)
    result = single
    for _ in range(1, k):
        result = tuple(kronecker(P, Q) for P in result for Q in single)
    return result

_EMBEDDED_PAULIS: dict[tuple[tuple[int, ...], int], tuple[Circuit, ...]] = {}

def _embedded_paulis(qubits: tuple[int, ...], n: int) -> tuple[Circuit, ...]:
    key = (qubits, n)
    if key not in _EMBEDDED_PAULIS:
        _EMBEDDED_PAULIS[key] = tuple(embedN(P, qubits, n) for P in _pauli_basis(len(qubits)))
    return _EMBEDDED_PAULIS[key]

def depolarizing_gate(gate: Gate, n: int, epsilon: float, *, reference: float | None = None) -> DensityOperator:
    r'''
        Depolarising noise on a single ``gate`` of an ``n``-qubit register.

        The depolarising parameter comes from the gate's duration,

        The channel follows **qiskit's** convention, i.e. that of ``depolarizing_error(lam, k)``:

        .. MATH::

            \rho \mapsto (1-\lambda)\, U\rho U^\dagger + \lambda\, \text{tr}(\rho)\, I/2^k,

        which as a mixture over the `4^k` Pauli strings on the gate's own qubits puts weight
        `1-\lambda+\lambda/4^k` on the identity and `\lambda/4^k` on each of the rest. 
    '''
    U = embedN(gate.matrix, gate.qubits, n)
    t_reference = DURATIONS["sx"] if reference is None else reference
    lam = epsilon * gate.duration / t_reference
    if lam == 0: # virtual gate, or no noise at all: a single unitary branch
        return DensityOperator(circuits=[U], probabilities=[1.0])

    weight = lam / 4**len(gate.qubits)
    paulis = _embedded_paulis(gate.qubits, n)
    probabilities = [1 - lam + weight] + (len(paulis) - 1)*[weight]
    return DensityOperator(circuits=[P * U for P in paulis], probabilities=probabilities)

def noisy_circuit(gates: Sequence[Gate], n: int, epsilon: float, *, reference: float | None = None) -> DensityOperator:
    r'''
        Compose a time-ordered gate list into one noisy super-operator, one depolarising channel per
        gate.

        The composition is never materialised: `G` gates with `4^k` branches each would be a mixture
        of `\prod_g 4^{k_g}` unitaries, but :func:`DensityVector.apply_matrix` walks the chain one
        ensemble at a time, so the cost is linear in ``len(gates)``.
    '''
    if len(gates) == 0:
        return DensityOperator.eye(4**n)
    return DensityOperator(operators=[depolarizing_gate(g, n, epsilon, reference=reference) for g in gates])

## --------------------------------------------------------------------------
## Common structure for the stress tests
##
## Every stress test has the same shape: start from |0...0> and run one noisy circuit. State
## preparation is part of that circuit, so its gates carry noise like any others -- which is also the
## form the NQB draft assumes, `rho = sum_k p_k U_k |0><0| U_k^dagger`, "because circuits are
## initialized with |0>".
## --------------------------------------------------------------------------
def zero_state(n: int) -> State:
    r'''The computational-basis input `\ket{0\ldots0}` of an ``n``-qubit register.'''
    v = State(2**n, CC)
    v[0] = 1
    return v

def hadamard_layer(n: int) -> QuantumCircuit:
    r'''`H^{\otimes n}`: the uniform-superposition preparation of Grover and QAOA.'''
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    return circuit

def prepare(gates: Sequence[Gate], n: int, epsilon: float, *, reference: float | None = None) -> DensityVector:
    r'''
        Run a preparation circuit **once**, noisily, on `\ket{0\ldots0}`.

        The lumping iterates whatever operator it is given, so preparation must not be part of that
        operator -- `A^2` would prepare the state a second time on top of the evolved one. Instead the
        preparation channel is applied once here, and the (generally *mixed*) density matrix it
        produces becomes the vector the invariant subspace has to contain.
    '''
    rho = DensityVector.from_tensor(zero_state(n))
    if len(gates) == 0:
        return rho
    return rho.apply_matrix(noisy_circuit(gates, n, epsilon, reference=reference))
