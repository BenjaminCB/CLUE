import numpy as np
from hypothesis import given
from pytest import approx

from clue.linalg import NumericalSubspace, find_smallest_common_subspace
from clue.quantum_linalg import DensityOperator, DensityVector
from tests.strategies import Entries, mixed_unitary_channels, sparse_matrix, sparse_vector

_IDENTITY: Entries = [[1, 0], [0, 1]]
_BIT_FLIP: Entries = [[0, 1], [1, 0]]

def bit_flip_channel(p: float) -> DensityOperator:
    return DensityOperator(
        circuits=(sparse_matrix(_IDENTITY), sparse_matrix(_BIT_FLIP)),
        probabilities=(1 - p, p),
    )

def zero_state() -> DensityVector:
    return DensityVector.from_tensor(sparse_vector([1, 0]))

def compute_nqb(channel: DensityOperator, rho: DensityVector, delta: float = 1e-9) -> NumericalSubspace:
    return find_smallest_common_subspace(
        matrices=(channel,), vectors_to_include=(rho,), subspace_class=NumericalSubspace, delta=delta,
    )

class TestBitFlipChannel:
    def test_the_bisimulation_is_a_strict_reduction(self):
        nqb = compute_nqb(bit_flip_channel(0.3), zero_state())

        assert nqb.dim() == 2
        assert nqb.ambient_dimension() == 4

    def test_the_bisimulation_contains_the_whole_orbit(self):
        channel = bit_flip_channel(0.3)
        nqb = compute_nqb(channel, zero_state())

        state = zero_state()
        for _ in range(nqb.ambient_dimension()):
            assert state in nqb
            state = state.apply_matrix(channel)

    def test_the_companion_matrix_reproduces_the_full_dynamics(self):
        channel = bit_flip_channel(0.3)
        rho = zero_state()
        nqb = compute_nqb(channel, rho)
        basis = nqb.basis()

        coordinates_of = lambda v: np.array([v.inner_product(b) for b in basis])

        A_hat = np.array([coordinates_of(b.apply_matrix(channel)) for b in basis]).T
        c = coordinates_of(rho)

        state = rho
        for _ in range(nqb.ambient_dimension()):
            assert c == approx(coordinates_of(state), abs=1e-8)
            state = state.apply_matrix(channel)
            c = A_hat @ c

class TestGeneralNoisyChannels:
    @given(mixed_unitary_channels())
    def test_the_bisimulation_contains_the_whole_orbit(self, data: tuple[Entries, Entries, Entries, float]):
        _, first, second, probability = data
        channel = DensityOperator(
            circuits=(sparse_matrix(first), sparse_matrix(second)),
            probabilities=(probability, 1 - probability),
        )

        rho = DensityVector.from_tensor(sparse_vector(first[0]))
        nqb = compute_nqb(channel, rho)

        state = rho
        for _ in range(len(first) ** 2):
            assert state in nqb
            state = state.apply_matrix(channel)
