from itertools import product

from hypothesis import given
from numpy import array, trace
from pytest import approx, mark

from clue.quantum_linalg import DensityOperator, DensityVector
from strategies import (density_vector, matrices_with_unitary, mixed_unitary_channels, sparse_matrix,
                        sparse_vector, square_matrices, square_matrix_pairs, square_matrix_triples,
                        state_vector_ensembles, state_vectors)

## Slack allowed when comparing the two sides of an inequality. The entries are exact in
## ``complex128``, so the only error is the rounding of the square roots taken by the norm.
TOLERANCE = 1e-9

def frobenius(lhs: list[list], rhs: list[list]):
    r'''Reference implementation of the Frobenius inner product on dense matrices.'''
    return trace(array(lhs, dtype=complex) @ array(rhs, dtype=complex).conj().T)

def matrix_unit(dim: int, row: int, column: int) -> list[list]:
    r'''Dense representation of the matrix `E_{ij}`: a single 1 at position ``(row, column)``.'''
    entries = [dim * [0] for _ in range(dim)]
    entries[row][column] = 1
    return entries

def dense(vector: DensityVector):
    r'''Dense representation of a density vector, so that we can compare two of them.'''
    return vector.as_matrix().to_numpy()

def evolve(vector: DensityVector, *circuits: list[list], probabilities: tuple = None):
    r'''
        Apply to ``vector`` the super-operator given by the ``circuits`` and the ``probabilities``,
        i.e., the channel `\rho \mapsto \sum_i p_i U_i \rho U_i^\dagger`. Without probabilities we
        take the single circuit with probability 1.
    '''
    probabilities = (1.0,) if probabilities is None else probabilities
    operator = DensityOperator(
        circuits=tuple(sparse_matrix(circuit) for circuit in circuits), probabilities=probabilities
    )
    return vector.apply_matrix(operator)

class TestFrobeniusInnerProduct:
    r'''Unit tests checking the inner product on hand-picked matrices.'''
    def test_orthonormality_of_the_matrix_units(self):
        r'''The matrix units `E_{ij}` are the canonical orthonormal basis of the space.'''
        units = {(i, j): density_vector(matrix_unit(2, i, j)) for i, j in product(range(2), repeat=2)}

        for lhs, rhs in product(units, repeat=2):
            expected = 1 if lhs == rhs else 0
            assert units[lhs].inner_product(units[rhs]) == approx(expected)

    def test_trace_formula_on_a_fixed_example(self):
        r'''On a fixed complex example the result is `\text{tr}(A B^\dagger)`.'''
        lhs = [[1 + 1j, 2], [0, 1j]]
        rhs = [[1, 1 - 1j], [2j, 3]]

        assert density_vector(lhs).inner_product(density_vector(rhs)) == approx(3 + 6j)
        assert frobenius(lhs, rhs) == approx(3 + 6j)

    def test_bilinearity_without_conjugation(self):
        r'''
            Without conjugation we get the bilinear form `\sum_{i,j} A_{ij}B_{ij}`, which is what
            matrix multiplication needs. The keyword is not meant for users, but the rest of the
            library relies on it.
        '''
        lhs = [[1 + 1j, 2], [0, 1j]]
        rhs = [[1, 1 - 1j], [2j, 3]]

        assert density_vector(lhs).inner_product(density_vector(rhs), _conjugate=False) == approx(3 + 2j)

    def test_induced_frobenius_norm(self):
        r'''The induced norm is the square root of the sum of the squared moduli.'''
        entries = [[3, 0], [0, 4j]]

        assert density_vector(entries).norm_squared() == approx(25)
        assert density_vector(entries).norm() == approx(5)

class TestFrobeniusInnerProductProperties:
    r'''
        Property-based tests checking the axioms of a Hermitian inner product. Additivity and
        homogeneity are checked separately: together they are the linearity in the first argument.
    '''
    @given(square_matrix_pairs())
    def test_trace_formula(self, matrices):
        r'''`\langle A, B \rangle = \text{tr}(A B^\dagger)` for every pair of matrices.'''
        lhs, rhs = matrices

        assert density_vector(lhs).inner_product(density_vector(rhs)) == approx(frobenius(lhs, rhs))

    @given(square_matrix_pairs())
    def test_conjugate_symmetry(self, matrices):
        r'''`\langle A, B \rangle = \overline{\langle B, A \rangle}`.'''
        lhs, rhs = (density_vector(entries) for entries in matrices)

        assert lhs.inner_product(rhs) == approx(rhs.inner_product(lhs).conjugate())

    @given(square_matrix_triples())
    def test_additivity_in_the_first_argument(self, matrices):
        r'''`\langle A + B, C \rangle = \langle A, C \rangle + \langle B, C \rangle`.'''
        lhs, rhs, other = (density_vector(entries) for entries in matrices)

        assert (lhs + rhs).inner_product(other) == approx(
            lhs.inner_product(other) + rhs.inner_product(other)
        )

    @given(square_matrix_pairs())
    def test_homogeneity_in_the_first_argument(self, matrices):
        r'''`\langle cA, B \rangle = c\langle A, B \rangle`, with a genuinely complex scalar.'''
        lhs, rhs = (density_vector(entries) for entries in matrices)
        scalar = 2 - 3j

        assert (scalar * lhs).inner_product(rhs) == approx(scalar * lhs.inner_product(rhs))

    @given(square_matrix_pairs())
    def test_conjugate_homogeneity_in_the_second_argument(self, matrices):
        r'''`\langle A, cB \rangle = \bar{c}\langle A, B \rangle`.'''
        lhs, rhs = (density_vector(entries) for entries in matrices)
        scalar = 2 - 3j

        assert lhs.inner_product(scalar * rhs) == approx(scalar.conjugate() * lhs.inner_product(rhs))

    @given(square_matrices())
    def test_positive_definiteness(self, entries):
        r'''`\langle A, A \rangle` is a non-negative real, and vanishes only on the zero matrix.'''
        vector = density_vector(entries)
        squared = vector.inner_product(vector)

        assert squared.imag == approx(0)
        assert squared.real >= 0
        assert (squared.real == approx(0)) == vector.is_zero()

class TestFrobeniusNormProperties:
    r'''
        Property-based tests for the norm induced by the Frobenius inner product. These are the
        laws every norm coming from an inner product must satisfy.
    '''
    @given(square_matrix_pairs())
    def test_cauchy_schwarz_inequality(self, matrices):
        r'''`|\langle A, B \rangle| \leq \|A\|\|B\|`.'''
        lhs, rhs = (density_vector(entries) for entries in matrices)

        assert abs(lhs.inner_product(rhs)) <= lhs.norm() * rhs.norm() + TOLERANCE

    @given(square_matrix_pairs())
    def test_triangle_inequality(self, matrices):
        r'''`\|A + B\| \leq \|A\| + \|B\|`.'''
        lhs, rhs = (density_vector(entries) for entries in matrices)

        assert (lhs + rhs).norm() <= lhs.norm() + rhs.norm() + TOLERANCE

    @given(square_matrix_pairs())
    def test_parallelogram_law(self, matrices):
        r'''`\|A + B\|^2 + \|A - B\|^2 = 2\|A\|^2 + 2\|B\|^2`, the law characterizing the norms
            that come from an inner product.'''
        lhs, rhs = (density_vector(entries) for entries in matrices)

        assert (lhs + rhs).norm_squared() + (lhs - rhs).norm_squared() == approx(
            2 * lhs.norm_squared() + 2 * rhs.norm_squared()
        )

    @given(square_matrices())
    def test_absolute_homogeneity(self, entries):
        r'''`\|cA\| = |c|\|A\|`.'''
        vector = density_vector(entries)
        scalar = 2 - 3j

        assert (scalar * vector).norm() == approx(abs(scalar) * vector.norm())

    @given(square_matrices())
    def test_invariance_under_transposition_and_conjugation(self, entries):
        r'''`\|A^T\| = \|\bar{A}\| = \|A\|`: both rearrange the moduli of the entries.'''
        vector = density_vector(entries)

        assert vector.transpose().norm() == approx(vector.norm())
        assert vector.conjugate().norm() == approx(vector.norm())

class TestDensityVectorRepresentation:
    r'''
        Property-based tests for the isomorphism between the `d \times d` matrices and the vectors
        of dimension `d^2` that :class:`~clue.quantum_linalg.DensityVector` implements.
    '''
    @given(square_matrices())
    def test_matrix_round_trip(self, entries):
        r'''``from_matrix`` and ``as_matrix`` are inverse of each other.'''
        matrix = sparse_matrix(entries)

        assert DensityVector.from_matrix(matrix).as_matrix() == matrix

    @given(square_matrices())
    def test_vector_round_trip(self, entries):
        r'''``from_vector`` and ``as_vector`` are inverse of each other.'''
        vector = density_vector(entries)

        assert dense(DensityVector.from_vector(vector.as_vector())) == approx(dense(vector))

    @given(square_matrices())
    def test_coordinates_are_in_row_major_order(self, entries):
        r'''The entry `A_{ij}` is the coordinate `id + j` of the flattened vector.'''
        vector = density_vector(entries)
        dim = len(entries)
        flattened = vector.as_vector()

        for row, column in product(range(dim), repeat=2):
            index = row * dim + column
            assert vector.coordinate(index) == approx(entries[row][column])
            assert flattened[index] == approx(entries[row][column])

    @given(square_matrices())
    def test_nonzero_coordinates_are_exactly_the_nonzero_entries(self, entries):
        vector = density_vector(entries)
        expected = {index for index, entry in enumerate(vector.as_vector().to_list()) if entry != 0}

        assert vector.nonzero_coordinates() == expected

    @given(square_matrices())
    def test_a_matrix_is_zero_when_all_its_entries_vanish(self, entries):
        vector = density_vector(entries)

        assert vector.is_zero() == all(entry == 0 for row in entries for entry in row)

class TestDensityVectorInvolutions:
    r'''
        Property-based tests for transposition and conjugation, and for the way the Frobenius inner
        product transforms under them.
    '''
    @given(square_matrices())
    def test_transposition_is_an_involution(self, entries):
        r'''`(A^T)^T = A`.'''
        vector = density_vector(entries)

        assert dense(vector.transpose().transpose()) == approx(dense(vector))

    @given(square_matrices())
    def test_conjugation_is_an_involution(self, entries):
        r'''`\bar{\bar{A}} = A`.'''
        vector = density_vector(entries)

        assert dense(vector.conjugate().conjugate()) == approx(dense(vector))

    @given(square_matrices())
    def test_transposition_and_conjugation_commute(self, entries):
        r'''`\overline{A^T} = (\bar{A})^T`, which is what makes the adjoint `A^\dagger` well defined.'''
        vector = density_vector(entries)

        assert dense(vector.transpose().conjugate()) == approx(dense(vector.conjugate().transpose()))

    @given(square_matrix_pairs())
    def test_invariance_of_the_inner_product_under_transposition(self, matrices):
        r'''`\langle A^T, B^T \rangle = \langle A, B \rangle`.'''
        lhs, rhs = (density_vector(entries) for entries in matrices)

        assert lhs.transpose().inner_product(rhs.transpose()) == approx(lhs.inner_product(rhs))

    @given(square_matrix_pairs())
    def test_conjugation_of_the_inner_product(self, matrices):
        r'''`\langle \bar{A}, \bar{B} \rangle = \overline{\langle A, B \rangle}`.'''
        lhs, rhs = (density_vector(entries) for entries in matrices)

        assert lhs.conjugate().inner_product(rhs.conjugate()) == approx(
            lhs.inner_product(rhs).conjugate()
        )

class TestUnitaryEvolutionProperties:
    r'''
        Property-based tests for the evolution `\rho \mapsto U\rho U^\dagger` of a density matrix
        under a unitary gate. These are the defining properties of a closed quantum system.
    '''
    @given(matrices_with_unitary())
    def test_unitary_evolution_preserves_the_frobenius_norm(self, data):
        r'''`\|U \rho U^\dagger\| = \|\rho\|`: a unitary gate does not change the purity.'''
        entries, unitary = data
        vector = density_vector(entries)

        assert evolve(vector, unitary).norm() == approx(vector.norm())

    @given(matrices_with_unitary())
    def test_unitary_evolution_preserves_the_trace(self, data):
        r'''`\text{tr}(U \rho U^\dagger) = \text{tr}(\rho)`, by cyclicity of the trace.'''
        entries, unitary = data
        vector = density_vector(entries)

        assert dense(evolve(vector, unitary)).trace() == approx(dense(vector).trace())

    @given(matrices_with_unitary())
    def test_unitary_evolution_is_reversible(self, data):
        r'''Applying `U` and then `U^\dagger` gives back the original density matrix.'''
        entries, unitary = data
        vector = density_vector(entries)
        adjoint = sparse_matrix(unitary).dagger().to_list()

        assert dense(evolve(evolve(vector, unitary), adjoint)) == approx(dense(vector))

class TestDensityOperatorProperties:
    r'''
        Property-based tests for the super-operators of noisy circuits, i.e., for the channels
        `\rho \mapsto \sum_i p_i U_i \rho U_i^\dagger` given by a mixture of unitary gates.
    '''
    @given(mixed_unitary_channels())
    def test_mixed_unitary_channel_preserves_the_trace(self, data):
        r'''A channel built from unitaries is trace preserving, so it maps states to states.'''
        entries, first, second, probability = data
        vector = density_vector(entries)

        evolved = evolve(vector, first, second, probabilities=(probability, 1 - probability))

        assert dense(evolved).trace() == approx(dense(vector).trace())

    @given(mixed_unitary_channels())
    def test_mixed_unitary_channel_does_not_increase_the_frobenius_norm(self, data):
        r'''
            `\|\Phi(\rho)\| \leq \|\rho\|`: mixing unitaries can only add noise, never purity. It
            follows from the triangle inequality and the unitary invariance of the norm.
        '''
        entries, first, second, probability = data
        vector = density_vector(entries)

        evolved = evolve(vector, first, second, probabilities=(probability, 1 - probability))

        assert evolved.norm() <= vector.norm() + TOLERANCE

    @given(mixed_unitary_channels())
    def test_the_channel_is_the_convex_combination_of_its_branches(self, data):
        r'''`\Phi(\rho) = p\Phi_1(\rho) + (1-p)\Phi_2(\rho)`: the branches act independently.'''
        entries, first, second, probability = data
        vector = density_vector(entries)

        evolved = evolve(vector, first, second, probabilities=(probability, 1 - probability))
        expected = (probability * evolve(vector, first)) + ((1 - probability) * evolve(vector, second))

        assert dense(evolved) == approx(dense(expected))

    @given(mixed_unitary_channels())
    def test_transposition_is_an_involution_on_operators(self, data):
        r'''Transposing a super-operator twice gives back a super-operator acting the same way.'''
        entries, first, second, probability = data
        vector = density_vector(entries)
        operator = DensityOperator(
            circuits=(sparse_matrix(first), sparse_matrix(second)),
            probabilities=(probability, 1 - probability),
        )

        evolved = vector.apply_matrix(operator.transpose().transpose())

        assert dense(evolved) == approx(dense(vector.apply_matrix(operator)))

    @mark.xfail(strict=True, reason="`apply_matrix` returns the zero vector for the empty ensemble")
    @given(square_matrices())
    def test_the_identity_operator_acts_as_the_identity(self, entries):
        vector = density_vector(entries)
        identity = DensityOperator(circuits=(), probabilities=(), dim=vector.dim)

        assert dense(vector.apply_matrix(identity)) == approx(dense(vector))

    @mark.xfail(strict=True, reason="`eye` calls the constructor without `circuits`/`probabilities`")
    @given(square_matrices())
    def test_eye_builds_the_identity_operator(self, entries):
        assert DensityOperator.eye(density_vector(entries).dim).is_identity()

class TestPureStateProperties:
    r'''
        Property-based tests for the density matrix `\rho = |\psi\rangle\langle\psi|` of a pure
        state, built by :func:`~clue.quantum_linalg.DensityVector.from_tensor`.
    '''
    @given(state_vectors())
    def test_the_norm_of_a_pure_state_is_the_squared_norm_of_its_amplitudes(self, amplitudes):
        r'''`\||\psi\rangle\langle\psi|\| = \||\psi\rangle\|^2`.'''
        state = sparse_vector(amplitudes)

        assert DensityVector.from_tensor(state).norm() == approx(state.norm() ** 2)

    @mark.xfail(strict=True, reason="`from_tensor` does not conjugate the second factor")
    @given(state_vectors())
    def test_a_pure_state_density_matrix_is_hermitian(self, amplitudes):
        r'''`|\psi\rangle\langle\psi|` is self-adjoint, as every density matrix must be.'''
        pure = DensityVector.from_tensor(sparse_vector(amplitudes))

        assert dense(pure.transpose().conjugate()) == approx(dense(pure))

    @mark.xfail(strict=True, reason="`from_ensemble` unpacks the vectors and the probabilities swapped")
    @given(state_vector_ensembles())
    def test_an_ensemble_is_the_convex_combination_of_its_pure_states(self, data):
        r'''`\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|`.'''
        first, second, probability = data
        states = (sparse_vector(first), sparse_vector(second))

        ensemble = DensityVector.from_ensemble(states, (probability, 1 - probability))
        expected = (probability * DensityVector.from_tensor(states[0])) + (
            (1 - probability) * DensityVector.from_tensor(states[1])
        )

        assert dense(ensemble) == approx(dense(expected))
