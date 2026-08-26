r'''
    Hypothesis strategies shared by the property-based tests.

    The strategies here produce *dense* representations (lists of lists) of the objects used
    across :mod:`clue.linalg` and :mod:`clue.quantum_linalg`, together with small helpers that
    turn those representations into the sparse classes of the library. Keeping the dense form
    around makes the tests easy to read and gives Hypothesis something simple to shrink.
'''
from hypothesis import strategies as st

from clue.linalg import SparseRowMatrix
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

def density_vector(entries: list[list]) -> DensityVector:
    r'''Build the :class:`~clue.quantum_linalg.DensityVector` of a dense square matrix.'''
    return DensityVector.from_matrix(SparseRowMatrix.from_list(entries, CC))
