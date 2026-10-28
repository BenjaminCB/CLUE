from __future__ import annotations
from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations
from math import sqrt, pi, cos, sin
from typing import NamedTuple
from numpy import kron

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

## --------------------------------------------------------------------------
## Derived gates, as IBM-basis gate lists
## --------------------------------------------------------------------------
def h(q: int) -> list[Gate]:
    r'''`H = e^{i\pi/4} RZ(\pi/2)\, SX\, RZ(\pi/2)`.'''
    return [rz(pi/2, q), sx(q), rz(pi/2, q)]

def cx(c: int, t: int) -> list[Gate]:
    r'''`CX = H_t\, CZ\, H_t`: conjugating by `H` on the target turns its `Z` into an `X`.'''
    return h(t) + [cz(c, t)] + h(t)

def rx(theta: float, q: int) -> list[Gate]:
    r'''`RX(\theta) = H\, RZ(\theta)\, H`, which collapses to the standard ZXZXZ form.'''
    return [rz(pi/2, q), sx(q), rz(theta + pi, q), sx(q), rz(pi/2, q)]

def rzz(theta: float, c: int, t: int) -> list[Gate]:
    r'''`RZZ(\theta) = e^{-i\theta Z_c Z_t/2} = CX;\, RZ(\theta)_t;\, CX`.'''
    return cx(c, t) + [rz(theta, t)] + cx(c, t)

def cp(phi: float, p: int, q: int) -> list[Gate]:
    r'''
        Controlled phase `\text{diag}(1,1,1,e^{i\phi})`, symmetric in ``p`` and ``q``.

        From `\ket{11}\bra{11} = (I-Z_p)(I-Z_q)/4`, all three factors commuting:
        `CP(\phi) = e^{i\phi/4} RZ(\phi/2)_p\, RZ(\phi/2)_q\, RZZ(-\phi/2)`.
    '''
    return [rz(phi/2, p), rz(phi/2, q)] + rzz(-phi/2, p, q)

def ccz(a: int, b: int, c: int) -> list[Gate]:
    r'''
        Doubly-controlled `Z`, via Barenco et al. Lemma 6.1 with `V = P(\pi/2)` (so `V^2 = Z`):
        `C(V)_{b,c};\, CX_{a,b};\, C(V^\dagger)_{b,c};\, CX_{a,b};\, C(V)_{a,c}`.

        The control/target labels are immaterial -- the result is diagonal and symmetric.
    '''
    return (cp(pi/2, b, c) + cx(a, b) + cp(-pi/2, b, c) + cx(a, b) + cp(pi/2, a, c))

def ccx(a: int, b: int, c: int) -> list[Gate]:
    r'''Toffoli, as `H_c\, CCZ\, H_c`.'''
    return h(c) + ccz(a, b, c) + h(c)

## --------------------------------------------------------------------------
## Multi-controlled gates, following Barenco et al.,
## "Elementary gates for quantum computation" (Phys. Rev. A 52, 3457, 1995).
## --------------------------------------------------------------------------
def _mcx_vchain(controls: Sequence[int], target: int, work: Sequence[int]) -> list[Gate]:
    r'''
        Barenco et al. Lemma 7.2: `C^k(X)` with ``k-2`` *dirty* work qubits in `4(k-2)` Toffolis.

        The "V" leg flips ``work[-1]`` by `\text{AND}(controls[:-1])` while leaving the lower work
        qubits dirty; running ``A V A V`` cancels both the dirty contributions and the spurious
        target flips, leaving `X` on the target iff every control is 1 and every work qubit restored.
    '''
    k = len(controls)
    leg = ([(controls[j+1], work[j-1], work[j]) for j in range(k-3, 0, -1)]
           + [(controls[0], controls[1], work[0])])
    V = leg + list(reversed(leg[:-1]))
    A = [(controls[k-1], work[k-3], target)]
    return [gate for (a, b, c) in A + V + A + V for gate in ccx(a, b, c)]

def _mcx_split(controls: Sequence[int], target: int, free: Sequence[int]) -> list[Gate]:
    r'''
        Barenco et al. Lemma 7.3: `C^k(X)` with a *single* borrowed dirty qubit.

        Split the controls into halves ``A`` and ``B``. Each half then has enough spare qubits --
        the other half, plus the target or the ancilla -- for a v-chain. Writing `w` for the
        ancilla's unknown initial value, ``first second first second`` flips the target by
        `B' \wedge ((w \oplus A') \oplus w) = A' \wedge B'` and restores the ancilla.
    '''
    k = len(controls)
    ancilla = free[0]
    p = (k + 1) // 2
    A, B = list(controls[:p]), list(controls[p:])
    first = _mcx(A, ancilla, B + [target])
    second = _mcx(B + [ancilla], target, A)
    return first + second + first + second

def _mcx(controls: Sequence[int], target: int, free: Sequence[int]) -> list[Gate]:
    r'''Multi-controlled `X`; ``free`` lists qubits usable as dirty work space.'''
    controls, free = list(controls), list(free)
    k = len(controls)
    if k == 0: return [x(target)]
    if k == 1: return cx(controls[0], target)
    if k == 2: return ccx(controls[0], controls[1], target)
    if len(free) >= k - 2: return _mcx_vchain(controls, target, free[:k-2])
    if len(free) >= 1: return _mcx_split(controls, target, free)
    raise ValueError(f"C^{k}(X) needs at least one free qubit")

def mcp(phi: float, qubits: Sequence[int]) -> list[Gate]:
    r'''
        Multi-controlled phase: `e^{i\phi}` on `\ket{1\ldots1}`, identity elsewhere. Diagonal, so
        symmetric in ``qubits`` -- there is no distinguished target.

        With `a = \text{AND}(qubits[:-2])` and `b, c` the last two qubits, the counting identity
        `[b \wedge c] - [(b \oplus a) \wedge c] + [a \wedge c]` equals 2 exactly when
        `a = b = c = 1` and 0 otherwise, which gives

        .. MATH::

            MCP(\phi) = CP(\phi/2)_{b,c};\ MCX_{a \to b};\ CP(-\phi/2)_{b,c};\ MCX_{a \to b};\
                        MCP(\phi/2)_{a \cup \{c\}}

        Each level drops one qubit and spends two `MCX` of `O(m)` Toffolis -- `c` is idle during
        them, which is the borrowed work qubit -- so the whole thing is `\Theta(m^2)`.
    '''
    qubits = list(qubits)
    if len(qubits) == 0: return []
    if len(qubits) == 1: return [rz(phi, qubits[0])] # P(phi) up to a global phase
    if len(qubits) == 2: return cp(phi, qubits[0], qubits[1])
    *rest, b, c = qubits
    flip = _mcx(rest, b, [c])
    return cp(phi/2, b, c) + flip + cp(-phi/2, b, c) + flip + mcp(phi/2, rest + [c])

def mcz(qubits: Sequence[int]) -> list[Gate]:
    r'''Multi-controlled `Z`: a phase flip on `\ket{1\ldots1}`.'''
    return mcp(pi, qubits)

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

def hadamard_layer(n: int) -> list[Gate]:
    r'''`H^{\otimes n}`, as gates: the uniform-superposition preparation of Grover and QAOA.'''
    return [gate for q in range(n) for gate in h(q)]

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
