from itertools import product

from hypothesis import given
from numpy import array, trace
from pytest import approx

from strategies import density_vector, square_matrix_pairs, square_matrix_triples, square_matrices

def frobenius(lhs: list[list], rhs: list[list]):
    r'''Reference implementation of the Frobenius inner product on dense matrices.'''
    return trace(array(lhs, dtype=complex) @ array(rhs, dtype=complex).conj().T)

def matrix_unit(dim: int, row: int, column: int) -> list[list]:
    r'''Dense representation of the matrix `E_{ij}`: a single 1 at position ``(row, column)``.'''
    entries = [dim * [0] for _ in range(dim)]
    entries[row][column] = 1
    return entries

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
