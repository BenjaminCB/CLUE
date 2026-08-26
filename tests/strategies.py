r'''
    Hypothesis strategies shared by the property-based tests.

    The strategies here produce *dense* representations (lists of lists) of the objects used
    across :mod:`clue.linalg` and :mod:`clue.quantum_linalg`, together with small helpers that
    turn those representations into the sparse classes of the library. Keeping the dense form
    around makes the tests easy to read and gives Hypothesis something simple to shrink.
'''
from hypothesis import strategies as st

from clue.linalg import SparseRowMatrix, SparseVector
from clue.numerical_domains import CC
from clue.quantum_linalg import DensityVector

## Density matrices grow quadratically with the number of qbits, so even a small bound here
## already covers 1 and 2 qbit systems while keeping the tests fast.
MAX_BASE_DIM = 4

## The entries are small Gaussian integers. They are exactly representable as ``complex128``,
## so the identities below hold without any floating point drift, and the fair amount of zeros
## they produce exercises the sparse code paths.
_coefficients = st.integers(min_value=-4, max_value=4)

complex_entries = st.builds(complex, _coefficients, _coefficients)

## Unit-modulus phases: multiplying the rows of a permutation matrix by them keeps the matrix
## unitary while making it genuinely complex, and all four values are exact in floating point.
_phases = st.sampled_from([1, -1, 1j, -1j])

## Probabilities are dyadic rationals, so that ``p`` and ``1-p`` add up to exactly 1 (which is
## what :class:`~clue.quantum_linalg.DensityOperator` demands of a distribution).
probabilities = st.integers(0, 16).map(lambda numerator: numerator / 16)

def _of_dim(dim: int, entries: st.SearchStrategy):
    r'''Strategy for a dense ``dim`` by ``dim`` matrix with the given entries.'''
    row = st.lists(entries, min_size=dim, max_size=dim)
    return st.lists(row, min_size=dim, max_size=dim)

def square_matrices(entries: st.SearchStrategy = complex_entries, max_dim: int = MAX_BASE_DIM):
    r'''Strategy for a dense square matrix of dimension at most ``max_dim``.'''
    return st.integers(1, max_dim).flatmap(lambda dim: _of_dim(dim, entries))

def square_matrix_pairs(entries: st.SearchStrategy = complex_entries, max_dim: int = MAX_BASE_DIM):
    r'''Strategy for a pair of dense square matrices *sharing the same dimension*.'''
    return st.integers(1, max_dim).flatmap(
        lambda dim: st.tuples(_of_dim(dim, entries), _of_dim(dim, entries))
    )

def square_matrix_triples(entries: st.SearchStrategy = complex_entries, max_dim: int = MAX_BASE_DIM):
    r'''Strategy for a triple of dense square matrices *sharing the same dimension*.'''
    return st.integers(1, max_dim).flatmap(
        lambda dim: st.tuples(*(3 * (_of_dim(dim, entries),)))
    )

def _unitary_of_dim(dim: int):
    r'''
        Strategy for a dense ``dim`` by ``dim`` unitary matrix.

        We use generalized permutation matrices (a permutation matrix whose rows carry a phase of
        modulus one). They are exactly representable, they cover the Pauli `X` and `Z` gates and
        their tensor products, and they are cheap to generate compared to a general unitary.
    '''
    return st.tuples(
        st.permutations(range(dim)), st.lists(_phases, min_size=dim, max_size=dim)
    ).map(lambda choice: _generalized_permutation(dim, *choice))

def _generalized_permutation(dim: int, permutation: list[int], phases: list[complex]) -> list[list]:
    entries = [dim * [0] for _ in range(dim)]
    for row, column in enumerate(permutation):
        entries[row][column] = phases[row]
    return entries

def state_vectors(entries: st.SearchStrategy = complex_entries, max_dim: int = MAX_BASE_DIM):
    r'''Strategy for a dense (not normalized) state vector of dimension at most ``max_dim``.'''
    return st.integers(1, max_dim).flatmap(
        lambda dim: st.lists(entries, min_size=dim, max_size=dim)
    )

def state_vector_ensembles(entries: st.SearchStrategy = complex_entries, max_dim: int = MAX_BASE_DIM):
    r'''Strategy for two state vectors of the same dimension together with a probability.'''
    return st.integers(1, max_dim).flatmap(
        lambda dim: st.tuples(
            st.lists(entries, min_size=dim, max_size=dim),
            st.lists(entries, min_size=dim, max_size=dim),
            probabilities,
        )
    )

def matrices_with_unitary(entries: st.SearchStrategy = complex_entries, max_dim: int = MAX_BASE_DIM):
    r'''Strategy for a dense square matrix together with a unitary matrix of the same dimension.'''
    return st.integers(1, max_dim).flatmap(
        lambda dim: st.tuples(_of_dim(dim, entries), _unitary_of_dim(dim))
    )

def mixed_unitary_channels(entries: st.SearchStrategy = complex_entries, max_dim: int = MAX_BASE_DIM):
    r'''
        Strategy for a dense square matrix together with the ingredients of a two-outcome mixed
        unitary channel acting on it: two unitaries of the same dimension and one probability.
    '''
    return st.integers(1, max_dim).flatmap(
        lambda dim: st.tuples(
            _of_dim(dim, entries), _unitary_of_dim(dim), _unitary_of_dim(dim), probabilities
        )
    )

def sparse_matrix(entries: list[list]) -> SparseRowMatrix:
    r'''Build the :class:`~clue.linalg.SparseRowMatrix` of a dense matrix over the complex field.'''
    return SparseRowMatrix.from_list(entries, CC)

def sparse_vector(entries: list) -> SparseVector:
    r'''Build the :class:`~clue.linalg.SparseVector` of a dense vector over the complex field.'''
    return SparseVector.from_list(entries, CC)

def density_vector(entries: list[list]) -> DensityVector:
    r'''Build the :class:`~clue.quantum_linalg.DensityVector` of a dense square matrix.'''
    return DensityVector.from_matrix(sparse_matrix(entries))
