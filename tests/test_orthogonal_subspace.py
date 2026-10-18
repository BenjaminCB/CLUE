r'''
Regression tests for :class:`clue.linalg.OrthogonalSubspace` over a **complex** field.

Its projector was built as `\overline{u[i]}\,u[j]` instead of `u[i]\,\overline{u[j]}`, so over `CC`
a subspace could not recognise its own basis vectors and every complex-valued reduction inflated to
the full ambient dimension. The two agree for real `u`, which is why nothing caught it.
'''
import numpy as np
from pytest import approx, mark

from clue.linalg import NumericalSubspace, find_smallest_common_subspace
from clue.numerical_domains import CC
from clue.linalg import SparseRowMatrix, SparseVector
from clue.quantum_linalg import DensityOperator, DensityVector

def complex_vector(entries) -> SparseVector:
    return SparseVector.from_list([complex(e) for e in entries], CC)

class TestComplexProjector:
    r'''A subspace must contain every scalar multiple of its own basis vectors.'''
    BASIS = [
        [1, 1j, 0.5 - 0.25j, 2],      # genuinely complex
        [1, 2, 0.5, 3],               # real: this one passed before the fix too
        [1j, 1j, 1j, 1j],             # purely imaginary
    ]

    @mark.parametrize("entries", BASIS)
    @mark.parametrize("scale", [1, 3, 1j, 2 + 1j, -0.5j])
    def test_a_multiple_of_the_basis_reduces_to_zero(self, entries, scale):
        subspace = NumericalSubspace(CC, delta=1e-6)
        subspace.absorb_new_vector(complex_vector(entries), force=True)

        residual = subspace.reduce_vector(complex_vector([scale * e for e in entries]))

        assert abs(complex(residual.norm())) == approx(0, abs=1e-12)

    @mark.parametrize("entries", BASIS)
    @mark.parametrize("scale", [1, 3, 1j, 2 + 1j])
    def test_a_multiple_of_the_basis_is_not_absorbed_again(self, entries, scale):
        r'''The dimension must not grow: this is how the inflation showed up in practice.'''
        subspace = NumericalSubspace(CC, delta=1e-6)
        subspace.absorb_new_vector(complex_vector(entries), force=True)

        pivot = subspace.absorb_new_vector(complex_vector([scale * e for e in entries]))

        assert pivot == -1
        assert subspace.dim() == 1

    def test_an_independent_complex_direction_is_still_absorbed(self):
        r'''The fix must not make the subspace *under*-absorb.'''
        subspace = NumericalSubspace(CC, delta=1e-6)
        subspace.absorb_new_vector(complex_vector([1, 1j, 0, 0]), force=True)

        assert subspace.absorb_new_vector(complex_vector([0, 0, 1j, 1])) != -1
        assert subspace.dim() == 2

class TestDiagonalUnitaryChannel:
    r'''
    A closed-form invariant dimension to check the engine against over a complex field: for diagonal
    `U`, the channel `\rho \mapsto U\rho U^\dagger` scales entry `(i,j)` of `\rho` by
    `e^{-i(\theta_i-\theta_j)}`, so the dimension is the number of distinct `\theta_i - \theta_j`.
    '''
    @mark.parametrize("thetas", [
        [0.0, 1.0],                 # 1 difference each way + 0 -> 3
        [0.0, 1.0, 2.0, 3.0],       # differences -3..3 -> 7
        [0.0, 0.0, 1.0, 1.0],       # repeated phases -> still 3
        [0.0, 1.0, 3.0],            # differences 0,+-1,+-2,+-3 -> 7
    ])
    def test_the_invariant_dimension_counts_distinct_phase_differences(self, thetas):
        dimension = len(thetas)
        U = SparseRowMatrix(dimension, CC)
        for i, theta in enumerate(thetas):
            U.increment(i, i, CC(complex(np.cos(theta), -np.sin(theta))))
        channel = DensityOperator(circuits=[U], probabilities=[1.0])

        uniform = complex_vector(dimension * [1 / np.sqrt(dimension)])
        rho = DensityVector.from_tensor(uniform)

        nqb = find_smallest_common_subspace(
            matrices=(channel,), vectors_to_include=(rho,),
            subspace_class=NumericalSubspace, delta=1e-9,
        )

        expected = len({round(a - b, 9) for a in thetas for b in thetas})
        assert nqb.dim() == expected
        assert nqb.check_invariance((channel,))
