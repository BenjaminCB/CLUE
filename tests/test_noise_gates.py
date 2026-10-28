r'''
Cross-checks for the gate-level circuits in ``papers/noise``.

The stress tests there used to apply noise to whole blocks (Grover's oracle and diffuser, the
MaxCut cost layer) built as single collapsed matrices. They now write the same operators as qiskit
circuits, transpiled to the IBM basis, so that noise can act on each gate, which is what the NQB
draft actually specifies ("noisy quantum circuits arise by composing noisy quantum gates"). These
tests pin ``from_qiskit`` to qiskit's own matrices and the circuits to the collapsed matrices they
replaced.

Global phases are irrelevant throughout: the circuits are only ever used as `rho -> U rho U^dagger`.
'''
import os, sys

import numpy as np
from pytest import approx, mark, raises
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers", "noise"))

from circuits import (  # noqa: E402
    CX, CZ, ERROR_RATES, H, I, SX, X, RZ, cz, depolarizing_gate, embed1, embed2, embedN,
    from_qiskit, hadamard_layer, noisy_circuit, prepare, rz, sx, x, zero_state, _pauli_basis,
)
from stress_grover import (  # noqa: E402
    diffuser, diffuser_circuit, initial_state, iteration_circuit, oracle, oracle_circuit,
)
from stress_qaoa_maxcut import (  # noqa: E402
    cost_circuit, cost_unitary, cut_difference_count, cut_matrix, cut_values, random_graph, run_size,
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

def circuit(n: int, build) -> QuantumCircuit:
    r'''A fresh ``n``-qubit qiskit circuit, filled in by ``build``.'''
    result = QuantumCircuit(n)
    build(result)
    return result

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

    def test_cz_is_symmetric(self):
        assert dense(embedN(CZ, (0, 1), 2)) == approx(dense(embedN(CZ, (1, 0), 2)))

class TestFromQiskit:
    r'''
    ``from_qiskit`` maps qiskit qubit ``q`` to our ``n-1-q``, so a converted circuit must have
    exactly the matrix ``Operator`` gives the qiskit one, with no qubit reversal. Asymmetric gates
    (CX, CCX with the target in every position) are what would catch a wrong mapping.
    '''
    @mark.parametrize("n, build", [
        (1, lambda c: c.h(0)),
        (1, lambda c: c.rx(0.7, 0)),
        (2, lambda c: c.cx(0, 1)),
        (2, lambda c: c.cx(1, 0)),
        (2, lambda c: c.rzz(-0.6, 0, 1)),
        (2, lambda c: c.cp(0.9, 0, 1)),
        (3, lambda c: c.ccx(0, 1, 2)),
        (3, lambda c: c.ccx(2, 0, 1)),
        (3, lambda c: c.ccx(1, 2, 0)),
        (5, lambda c: c.mcx([0, 1, 2, 3], 4)),
    ])
    def test_matches_qiskits_operator(self, n, build):
        qc = circuit(n, build)
        assert_same_up_to_phase(unitary(from_qiskit(qc), n), Operator(qc).data)

    @mark.parametrize("n, seed", [(1, 0), (2, 1), (3, 2), (4, 3)])
    def test_matches_qiskits_operator_on_random_circuits(self, n, seed):
        qc = random_circuit(n, depth=4, seed=seed)
        assert_same_up_to_phase(unitary(from_qiskit(qc), n), Operator(qc).data)

    def test_qubit_zero_is_the_least_significant_bit(self):
        r'''An ``X`` on qiskit qubit 0 flips bit 0 of the state index, which is our qubit ``n-1``.'''
        gates = from_qiskit(circuit(3, lambda c: c.x(0)))
        assert [(g.name, g.qubits) for g in gates] == [("x", (2,))]

    def test_emits_only_the_requested_basis(self):
        gates = from_qiskit(circuit(3, lambda c: c.ccx(0, 1, 2)), basis=("cz", "rz", "sx"))
        assert {g.name for g in gates} <= {"cz", "rz", "sx"}
        assert_same_up_to_phase(unitary(gates, 3), Operator(circuit(3, lambda c: c.ccx(0, 1, 2))).data)

    def test_error_rates_come_from_the_table(self):
        gates = from_qiskit(circuit(2, lambda c: c.cx(0, 1)))
        assert all(g.error == ERROR_RATES[g.name] for g in gates)

    def test_rejects_a_basis_gate_without_a_matrix(self):
        with raises(ValueError, match="BASIS_GATES"):
            from_qiskit(circuit(1, lambda c: c.h(0)), basis=("cz", "rz", "sx", "x", "h"))

    def test_rejects_non_unitary_instructions(self):
        with raises(ValueError, match="reset"):
            from_qiskit(circuit(1, lambda c: c.reset(0)))

    def test_skips_barriers(self):
        gates = from_qiskit(circuit(2, lambda c: (c.x(0), c.barrier(), c.x(1))))
        assert [g.qubits for g in gates] == [(1,), (0,)]

class TestDepolarizingNoise:
    @mark.parametrize("n, gate, lam", [(1, sx(0), 0.1), (2, cz(0, 1), 0.2), (3, x(2), 0.05), (3, cz(2, 0), 0.3)])
    def test_matches_the_closed_form(self, n, gate, lam):
        r'''qiskit's convention: `(1-lam) U rho U^dag + lam tr(rho) I/2^k`, on the gate's own qubits.'''
        k = len(gate.qubits)
        operator = depolarizing_gate(gate, n, lam / gate.error)
        rho = random_density(n)
        U = dense(embedN(gate.matrix, gate.qubits, n))
        paulis = [dense(embedN(P, gate.qubits, n)) for P in _pauli_basis(k)]
        weight = lam / 4**k
        want = (1 - lam + weight) * (U @ rho @ U.conj().T) + sum(
            weight * (P @ U) @ rho @ (P @ U).conj().T for P in paulis[1:]
        )
        assert evolve(operator, rho, n) == approx(want, abs=1e-10)

    def test_virtual_rz_is_exactly_noiseless(self):
        r'''`RZ` is a phase-frame change: zero reported error, so zero noise, with no special case.'''
        operator = depolarizing_gate(rz(0.7, 0), 1, 0.5)
        rho, R = random_density(1), dense(RZ(0.7))
        assert evolve(operator, rho, 1) == approx(R @ rho @ R.conj().T, abs=1e-12)

    @mark.parametrize("n, build", [(1, lambda c: c.h(0)), (2, lambda c: c.rzz(0.4, 0, 1)), (3, lambda c: c.ccx(0, 1, 2))])
    def test_stays_a_density_matrix(self, n, build):
        gates = from_qiskit(circuit(n, build))
        out = evolve(noisy_circuit(gates, n, 20.0), random_density(n), n)
        assert np.trace(out) == approx(1, abs=1e-10)
        assert out == approx(out.conj().T, abs=1e-10)
        assert min(np.linalg.eigvalsh(out)) > -1e-10

    @mark.parametrize("n, build", [
        (2, lambda c: c.rzz(0.4, 0, 1)), (3, lambda c: c.ccx(0, 1, 2)), (4, lambda c: c.mcx([0, 1, 2], 3)),
    ])
    def test_zero_epsilon_reproduces_the_exact_unitary(self, n, build):
        gates = from_qiskit(circuit(n, build))
        rho, U = random_density(n), unitary(gates, n)
        assert evolve(noisy_circuit(gates, n, 0.0), rho, n) == approx(U @ rho @ U.conj().T, abs=1e-9)

class TestStressTestCircuits:
    r'''The circuits must implement exactly the blocks the noise used to be applied to.'''
    @mark.parametrize("n", [2, 3, 4, 5])
    def test_grover_oracle_for_every_target(self, n):
        for target in range(2**n):
            assert_same_up_to_phase(unitary(from_qiskit(oracle_circuit(n, target)), n), dense(oracle(n, target)))

    @mark.parametrize("n", [2, 3, 4, 5])
    def test_grover_diffuser(self, n):
        assert_same_up_to_phase(unitary(from_qiskit(diffuser_circuit(n)), n), dense(diffuser(n)))

    @mark.parametrize("n", [2, 3, 4, 5])
    @mark.parametrize("gamma", [0.7, -1.4])
    def test_qaoa_cost_layer(self, n, gamma):
        edges = random_graph(n, 1.0, 42)
        assert_same_up_to_phase(unitary(from_qiskit(cost_circuit(n, edges, gamma)), n), dense(cost_unitary(n, edges, gamma)))

class TestCommonStructure:
    r'''
    Both stress tests have the same shape: preparation runs ONCE from |0...0> (noisily), and the
    operator handed to the lumping is the *repeated unit* of the circuit. Preparation must not be
    inside that operator, or `A^2` would prepare the state a second time on top of the evolved one.
    '''
    @mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_hadamard_layer_prepares_the_uniform_superposition(self, n):
        got = unitary(from_qiskit(hadamard_layer(n)), n) @ np.array(zero_state(n).to_list(), dtype=complex)
        want = np.array(initial_state(n).to_list(), dtype=complex)
        assert_same_up_to_phase(np.outer(got, got.conj()), np.outer(want, want.conj()))

    @mark.parametrize("n", [2, 3, 4, 5])
    def test_the_repeated_unit_of_grover_is_one_iteration(self, n):
        r'''One oracle+diffuser step, matching the single Grover operator the predecessor lumps.'''
        target = 2**n - 1
        want = dense(diffuser(n)) @ dense(oracle(n, target))
        assert_same_up_to_phase(unitary(from_qiskit(iteration_circuit(n, target)), n), want)

    @mark.parametrize("n", [2, 3, 4, 5])
    @mark.parametrize("gamma", [1.0, 0.7])
    def test_the_repeated_unit_of_maxcut_is_the_exponential_of_the_cut_matrix(self, n, gamma):
        r'''The noise-free test lumps `cut_matrix` as a generator, so this must be its exponential.'''
        edges = random_graph(n, 1.0, 42)
        want = np.diag(np.exp(-1j * gamma * np.diagonal(dense(cut_matrix(n, edges))).real))
        assert_same_up_to_phase(unitary(from_qiskit(cost_circuit(n, edges, gamma)), n), want)

    @mark.parametrize("n", [2, 3])
    @mark.parametrize("gates_of", [lambda n: from_qiskit(hadamard_layer(n))])
    def test_noiseless_preparation_is_the_exact_pure_state(self, n, gates_of):
        gates = gates_of(n)
        got = dense(prepare(gates, n, 0.0).as_matrix())
        pure = unitary(gates, n) @ np.array(zero_state(n).to_list(), dtype=complex)
        assert_same_up_to_phase(got, np.outer(pure, pure.conj()))

    @mark.parametrize("n", [2, 3])
    def test_noisy_preparation_is_a_valid_mixed_state(self, n):
        rho = dense(prepare(from_qiskit(hadamard_layer(n)), n, 1e-2).as_matrix())
        assert np.trace(rho) == approx(1, abs=1e-10)
        assert rho == approx(rho.conj().T, abs=1e-10)
        assert min(np.linalg.eigvalsh(rho)) > -1e-10
        assert np.trace(rho @ rho).real < 1 - 1e-6 # genuinely mixed, so preparation noise landed

class TestNoiseFreeBaselines:
    r'''
    The reductions the engine must reproduce with the noise switched off. The cost layer is
    diagonal, so in density space the dimension is the number of distinct cut *differences*, not
    the distinct cut *values* the noise-free predecessor expects in state space.
    '''
    @mark.parametrize("n", [2, 3, 4])
    @mark.parametrize("density", [1.0, 0.6])
    def test_the_noise_free_cost_layer_reduces_to_the_cut_differences(self, n, density):
        edges = random_graph(n, density, 42)
        result = run_size(n, density, 42, gamma=1.0, epsilon=0.0, delta=1e-9)

        assert result["reduced_dim"] == cut_difference_count(n, edges)
        assert result["reduced_dim"] < result["ambient_dim"] # it is a genuine reduction

    @mark.parametrize("n", [2, 3, 4])
    def test_the_density_space_reduction_is_not_the_state_space_one(self, n):
        r'''The two agree only for a single cut value or a `\pm` symmetric pair, which these are not.'''
        edges = random_graph(n, 1.0, 42)

        assert cut_difference_count(n, edges) > len(cut_values(n, edges))
