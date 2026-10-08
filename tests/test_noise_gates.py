r'''
Cross-checks for the gate-level circuits in ``papers/noise``.

The stress tests there used to apply noise to whole blocks (Grover's oracle and diffuser, QAOA's
cost and mixer layers) built as single collapsed matrices. They now build the same operators out of
IBM-basis gates so that noise can act on each gate, which is what the NQB draft actually specifies
("noisy quantum circuits arise by composing noisy quantum gates"). These tests pin the
decompositions to the collapsed matrices they replaced.

Global phases are irrelevant throughout: the circuits are only ever used as `rho -> U rho U^dagger`.
'''
import os, sys

import numpy as np
from pytest import approx, mark

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers", "noise"))

from circuits import (  # noqa: E402
    CX, DURATIONS, H, I, SX, X, Z, RZ, ccx, ccz, cp, cx, depolarizing_gate, embed1, embed2, embedN,
    h, hadamard_layer, idle_round, mcp, mcz, noisy_circuit, prepare, rx, rz, rzz, sx, x, zero_state,
    _mcx, _pauli_basis,
)
from stress_ghz import ghz_gates, ghz_state, idle_gates  # noqa: E402
from stress_grover import diffuser, diffuser_gates, initial_state, iteration_gates, oracle, oracle_gates  # noqa: E402
from stress_qaoa_maxcut import (  # noqa: E402
    cost_gates, cost_unitary, layer_gates, mixer_gates, mixer_unitary, random_graph,
)

def dense(matrix) -> np.ndarray:
    return matrix.to_numpy().astype(complex)

def unitary(gates, n: int) -> np.ndarray:
    r'''Dense matrix of a time-ordered gate list: the first gate applied is the rightmost factor.'''
    result = np.eye(2**n, dtype=complex)
    for gate in gates:
        result = dense(embedN(gate.matrix, gate.qubits, n)) @ result
    return result

def assert_same_up_to_phase(got: np.ndarray, want: np.ndarray, tol: float = 1e-8):
    r'''Assert ``got == phase * want`` for one unit scalar ``phase`` shared by every entry.'''
    pivot = np.unravel_index(np.argmax(np.abs(want)), want.shape)
    phase = got[pivot] / want[pivot]
    assert abs(phase) == approx(1, abs=tol)
    assert got == approx(phase * want, abs=tol)

def mcx_matrix(controls, target: int, n: int) -> np.ndarray:
    control_mask = sum(1 << (n-1-q) for q in controls)
    target_bit = 1 << (n-1-target)
    result = np.zeros((2**n, 2**n), dtype=complex)
    for state in range(2**n):
        result[(state ^ target_bit) if (state & control_mask) == control_mask else state, state] = 1
    return result

def random_density(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2**n, 2**n)) + 1j*rng.normal(size=(2**n, 2**n))
    rho = A @ A.conj().T
    return rho / np.trace(rho)

def evolve(operator, rho: np.ndarray, n: int) -> np.ndarray:
    from clue.linalg import SparseRowMatrix
    from clue.numerical_domains import CC
    from clue.quantum_linalg import DensityVector

    state = DensityVector.from_matrix(SparseRowMatrix.from_list(rho.tolist(), CC))
    return dense(state.apply_matrix(operator).as_matrix())

class TestEmbedding:
    @mark.parametrize("qubit, n", [(0, 1), (0, 3), (1, 3), (2, 3)])
    def test_embedN_agrees_with_embed1(self, qubit, n):
        assert dense(embedN(H, (qubit,), n)) == approx(dense(embed1(H, qubit, n)))

    @mark.parametrize("qubit, n", [(0, 2), (0, 4), (1, 4), (2, 4)])
    def test_embedN_agrees_with_embed2(self, qubit, n):
        assert dense(embedN(CX, (qubit, qubit+1), n)) == approx(dense(embed2(CX, qubit, n)))

    def test_embedN_places_non_adjacent_and_reversed_qubits(self):
        r'''What `embed2` cannot do, and what CX on an arbitrary graph edge needs.'''
        assert dense(embedN(CX, (0, 2), 3)) == approx(mcx_matrix([0], 2, 3))
        assert dense(embedN(CX, (2, 0), 3)) == approx(mcx_matrix([2], 0, 3))

class TestIBMBasis:
    def test_sx_squares_to_x(self):
        assert dense(SX) @ dense(SX) == approx(dense(X))

    def test_h_decomposition(self):
        assert_same_up_to_phase(unitary(h(0), 1), dense(H))

    @mark.parametrize("theta", [0.0, 0.7, -1.3, np.pi])
    def test_rx_decomposition(self, theta):
        want = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
        assert_same_up_to_phase(unitary(rx(theta, 0), 1), want)

    @mark.parametrize("theta", [0.6, -2.1])
    def test_rzz_decomposition(self, theta):
        ZZ = np.kron(dense(Z), dense(Z))
        want = np.cos(theta/2)*np.eye(4) - 1j*np.sin(theta/2)*ZZ
        assert_same_up_to_phase(unitary(rzz(theta, 0, 1), 2), want)

    @mark.parametrize("phi", [0.5, np.pi/2, np.pi, -0.9])
    def test_cp_decomposition(self, phi):
        assert_same_up_to_phase(unitary(cp(phi, 0, 1), 2), np.diag([1, 1, 1, np.exp(1j*phi)]))

    def test_ccz_and_ccx(self):
        assert_same_up_to_phase(unitary(ccz(0, 1, 2), 3), np.diag([1, 1, 1, 1, 1, 1, 1, -1]))
        for (a, b, c) in [(0, 1, 2), (2, 0, 1), (1, 2, 0)]:
            assert_same_up_to_phase(unitary(ccx(a, b, c), 3), mcx_matrix([a, b], c, 3))

class TestBarencoMultiControlled:
    r'''
    Barenco et al., "Elementary gates for quantum computation" (Phys. Rev. A 52, 3457, 1995).

    The work qubits these borrow are *dirty*: they hold unknown values and have to be returned
    untouched. Comparing over the whole register rather than a subspace is what makes that
    observable -- a work qubit left flipped cannot hide behind a global phase.
    '''
    @mark.parametrize("n, controls", [(5, 3), (6, 3), (6, 4), (7, 3), (7, 4), (7, 5)])
    def test_mcx_with_borrowed_work_qubits(self, n, controls):
        control_list, target = list(range(controls)), controls
        free = [q for q in range(n) if q not in control_list and q != target]
        assert_same_up_to_phase(unitary(_mcx(control_list, target, free), n),
                                mcx_matrix(control_list, target, n))

    @mark.parametrize("n, controls", [(5, 3), (6, 4), (7, 5)])
    def test_mcx_restores_work_qubits_in_superposition(self, n, controls):
        control_list, target = list(range(controls)), controls
        free = [q for q in range(n) if q not in control_list and q != target]
        got = unitary(_mcx(control_list, target, free), n)
        want = mcx_matrix(control_list, target, n)
        rng = np.random.default_rng(0)
        state = rng.normal(size=2**n) + 1j*rng.normal(size=2**n)
        state /= np.linalg.norm(state)
        assert_same_up_to_phase(np.outer(got @ state, (got @ state).conj()),
                                np.outer(want @ state, (want @ state).conj()))

    @mark.parametrize("n", [2, 3, 4, 5, 6])
    @mark.parametrize("phi", [np.pi, 0.7])
    def test_multi_controlled_phase_uses_no_ancilla(self, n, phi):
        want = np.eye(2**n, dtype=complex)
        want[-1, -1] = np.exp(1j*phi)
        assert_same_up_to_phase(unitary(mcp(phi, range(n)), n), want)

    @mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_mcz(self, n):
        want = np.eye(2**n, dtype=complex)
        want[-1, -1] = -1
        assert_same_up_to_phase(unitary(mcz(range(n)), n), want)

class TestDepolarizingNoise:
    @mark.parametrize("n, gate, lam", [(1, sx(0), 0.1), (2, cx(0, 1), 0.2), (3, x(2), 0.05), (3, cx(2, 0), 0.3)])
    def test_matches_the_closed_form(self, n, gate, lam):
        r'''qiskit's convention: `(1-lam) U rho U^dag + lam tr(rho) I/2^k`, on the gate's own qubits.'''
        k = len(gate.qubits)
        operator = depolarizing_gate(gate, n, lam * DURATIONS["sx"] / gate.duration)
        rho = random_density(n)
        U = dense(embedN(gate.matrix, gate.qubits, n))
        paulis = [dense(embedN(P, gate.qubits, n)) for P in _pauli_basis(k)]
        weight = lam / 4**k
        want = (1 - lam + weight) * (U @ rho @ U.conj().T) + sum(
            weight * (P @ U) @ rho @ (P @ U).conj().T for P in paulis[1:]
        )
        assert evolve(operator, rho, n) == approx(want, abs=1e-10)

    def test_virtual_rz_is_exactly_noiseless(self):
        r'''`RZ` is a phase-frame change: zero duration, so zero noise, with no special case.'''
        operator = depolarizing_gate(rz(0.7, 0), 1, 0.5)
        rho, R = random_density(1), dense(RZ(0.7))
        assert evolve(operator, rho, 1) == approx(R @ rho @ R.conj().T, abs=1e-12)

    @mark.parametrize("n, gates", [(1, h(0)), (2, rzz(0.4, 0, 1)), (3, ccx(0, 1, 2))])
    def test_stays_a_density_matrix(self, n, gates):
        out = evolve(noisy_circuit(gates, n, 0.02), random_density(n), n)
        assert np.trace(out) == approx(1, abs=1e-10)
        assert out == approx(out.conj().T, abs=1e-10)
        assert min(np.linalg.eigvalsh(out)) > -1e-10

    @mark.parametrize("n, gates", [(2, rzz(0.4, 0, 1)), (3, ccx(0, 1, 2)), (4, mcz(range(4)))])
    def test_zero_epsilon_reproduces_the_exact_unitary(self, n, gates):
        rho, U = random_density(n), unitary(gates, n)
        assert evolve(noisy_circuit(gates, n, 0.0), rho, n) == approx(U @ rho @ U.conj().T, abs=1e-9)

class TestStressTestCircuits:
    r'''The gate lists must implement exactly the blocks the noise used to be applied to.'''
    @mark.parametrize("n", [2, 3, 4, 5])
    def test_grover_oracle_for_every_target(self, n):
        for target in range(2**n):
            assert_same_up_to_phase(unitary(oracle_gates(n, target), n), dense(oracle(n, target)))

    @mark.parametrize("n", [2, 3, 4, 5])
    def test_grover_diffuser(self, n):
        assert_same_up_to_phase(unitary(diffuser_gates(n), n), dense(diffuser(n)))

    @mark.parametrize("n", [2, 3, 4, 5])
    @mark.parametrize("gamma", [0.7, -1.4])
    def test_qaoa_cost_layer(self, n, gamma):
        edges = random_graph(n, 1.0, 42)
        assert_same_up_to_phase(unitary(cost_gates(n, edges, gamma), n), dense(cost_unitary(n, edges, gamma)))

    @mark.parametrize("n", [2, 3, 4, 5])
    @mark.parametrize("beta", [0.4, -0.9])
    def test_qaoa_mixer_layer(self, n, beta):
        assert_same_up_to_phase(unitary(mixer_gates(n, beta), n), dense(mixer_unitary(n, beta)))

class TestCommonStructure:
    r'''
    All three stress tests have the same shape: preparation runs ONCE from |0...0> (noisily), and the
    operator handed to the lumping is the *repeated unit* of the circuit. Preparation must not be
    inside that operator, or `A^2` would prepare the state a second time on top of the evolved one.
    '''
    @mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_hadamard_layer_prepares_the_uniform_superposition(self, n):
        got = unitary(hadamard_layer(n), n) @ np.array(zero_state(n).to_list(), dtype=complex)
        want = np.array(initial_state(n).to_list(), dtype=complex)
        assert_same_up_to_phase(np.outer(got, got.conj()), np.outer(want, want.conj()))

    @mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_ghz_gates_prepare_the_ghz_state(self, n):
        got = unitary(ghz_gates(n), n) @ np.array(zero_state(n).to_list(), dtype=complex)
        want = np.array(ghz_state(n).to_list(), dtype=complex)
        assert_same_up_to_phase(np.outer(got, got.conj()), np.outer(want, want.conj()))

    @mark.parametrize("n", [2, 3, 4, 5])
    def test_the_repeated_unit_of_grover_is_one_iteration(self, n):
        r'''One oracle+diffuser step, matching the single Grover operator the predecessor lumps.'''
        target = 2**n - 1
        want = dense(diffuser(n)) @ dense(oracle(n, target))
        assert_same_up_to_phase(unitary(iteration_gates(n, target), n), want)

    @mark.parametrize("n", [2, 3, 4, 5])
    def test_the_repeated_unit_of_qaoa_is_one_layer(self, n):
        edges, gamma, beta = random_graph(n, 1.0, 42), 0.7, 0.4
        want = dense(mixer_unitary(n, beta)) @ dense(cost_unitary(n, edges, gamma))
        assert_same_up_to_phase(unitary(layer_gates(n, edges, gamma, beta), n), want)

    @mark.parametrize("n", [2, 3, 4])
    def test_an_idle_step_is_noiselessly_the_identity(self, n):
        assert unitary(idle_gates(n, 400.0), n) == approx(np.eye(2**n))

    @mark.parametrize("n", [2, 3])
    @mark.parametrize("gates_of", [lambda n: hadamard_layer(n), lambda n: ghz_gates(n)])
    def test_noiseless_preparation_is_the_exact_pure_state(self, n, gates_of):
        gates = gates_of(n)
        got = dense(prepare(gates, n, 0.0).as_matrix())
        pure = unitary(gates, n) @ np.array(zero_state(n).to_list(), dtype=complex)
        assert_same_up_to_phase(got, np.outer(pure, pure.conj()))

    @mark.parametrize("n", [2, 3])
    def test_noisy_preparation_is_a_valid_mixed_state(self, n):
        rho = dense(prepare(hadamard_layer(n), n, 1e-2).as_matrix())
        assert np.trace(rho) == approx(1, abs=1e-10)
        assert rho == approx(rho.conj().T, abs=1e-10)
        assert min(np.linalg.eigvalsh(rho)) > -1e-10
        assert np.trace(rho @ rho).real < 1 - 1e-6 # genuinely mixed, so preparation noise landed
