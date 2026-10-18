r'''
Theorem 1 of the NQB draft -- `A^k \mu = P^{-1}\hat{C}^k P \mu` with
`(P\mu)_i = \langle \rho_i, \mu\rangle` -- on the operators the stress tests actually lump, with
depolarising noise on and off.

Unlike the older check in ``test_quantum_bisimulation.py``, these run on complex operators, close
the loop in the ambient space instead of in coordinates, cover an arbitrary `\mu` in the subspace,
and assert the orthonormality premise rather than assuming it. `P` is built from the theorem's own
definition; ``find_in`` is checked separately to agree with it.
'''
import os, sys

import numpy as np
from pytest import approx, mark, param, raises

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers", "noise"))

from circuits import hadamard_layer, noisy_circuit, prepare  # noqa: E402
from clue.linalg import NumericalSubspace, find_smallest_common_subspace  # noqa: E402
from clue.quantum_linalg import DensityOperator, DensityVector  # noqa: E402
from stress_grover import iteration_gates  # noqa: E402
from stress_qaoa_maxcut import cost_gates, random_graph  # noqa: E402

def flat(density_vector: DensityVector) -> np.ndarray:
    r'''The `d^2`-dimensional vector of a density matrix, as the subspace sees it.'''
    return np.array(density_vector.as_matrix().to_numpy(), dtype=complex).flatten()

def maxcut(n: int, epsilon: float) -> tuple[DensityOperator, DensityVector]:
    edges = random_graph(n, 1.0, 42)
    return noisy_circuit(cost_gates(n, edges, 1.0), n, epsilon), prepare(hadamard_layer(n), n, epsilon)

def grover(n: int, epsilon: float) -> tuple[DensityOperator, DensityVector]:
    return noisy_circuit(iteration_gates(n, 2**n - 1), n, epsilon), prepare(hadamard_layer(n), n, epsilon)

_DEGENERATE_BASIS = mark.xfail(
    strict=True,
    reason="noisy Grover n=3: the engine returns d=64 of an ambient 64 with cond(L)=4.1e17 while "
           "the dense Krylov rank is 11-20, so Theorem 1's orthonormality premise fails there.",
)

## Specific to NOISY Grover n=3 (at epsilon=0 it is clean), so the mark is per (experiment, epsilon).
EXPERIMENTS = [
    param("maxcut", maxcut, 2, id="maxcut-2"),
    param("maxcut", maxcut, 3, id="maxcut-3"),
    param("grover", grover, 2, id="grover-2"),
    param("grover", grover, 3, id="grover-3", marks=_DEGENERATE_BASIS),
]

DYNAMICS = (
    [param(k, b, n, eps, id=f"{k}-{n}-eps{eps:g}")
     for (k, b, n) in [("maxcut", maxcut, 2), ("maxcut", maxcut, 3), ("grover", grover, 2)]
     for eps in (0.0, 1e-3)]
    + [param("grover", grover, 3, 0.0, id="grover-3-eps0"),
       param("grover", grover, 3, 1e-3, id="grover-3-eps0.001", marks=_DEGENERATE_BASIS)]
)

class ReducedModel:
    r'''`P`, `\hat{C}` and `P^{-1}` for one (channel, rho) pair.'''
    def __init__(self, channel: DensityOperator, rho: DensityVector, delta: float = 1e-9):
        self.channel = channel
        self.nqb = find_smallest_common_subspace(
            matrices=(channel,), vectors_to_include=(rho,),
            subspace_class=NumericalSubspace, delta=delta,
        )
        self.basis = self.nqb.basis()
        self.L = np.array([flat(b) for b in self.basis])  # rows: rho_1..rho_d
        self.C_hat = np.array([self.P(b.apply_matrix(channel)) for b in self.basis]).T

    def P(self, mu: DensityVector) -> np.ndarray:
        r'''`(P\mu)_i = \langle\rho_i, \mu\rangle`, the theorem's own definition.'''
        return self.L.conj() @ flat(mu)

    def embed(self, coordinates: np.ndarray) -> np.ndarray:
        r'''`P^{-1}\hat{v} = \sum_i \hat{v}_i\rho_i`, back in the ambient space.'''
        return coordinates @ self.L

class TestTheorem1:
    @mark.parametrize("name,build,n", EXPERIMENTS)
    def test_the_basis_is_orthonormal(self, name, build, n):
        r'''The premise: `P^{-1}` inverts `P` only for an orthonormal basis.'''
        model = ReducedModel(*build(n, 1e-3))

        assert model.L @ model.L.conj().T == approx(np.eye(model.nqb.dim()), abs=1e-9)
        assert np.linalg.cond(model.L) == approx(1.0, abs=1e-6)

    @mark.parametrize("name,build,n,epsilon", DYNAMICS)
    def test_the_reduced_model_reproduces_the_ambient_dynamics(self, name, build, n, epsilon):
        r'''`A^k\rho = P^{-1}\hat{C}^k P\rho` in the ambient space, for `k` past the dimension.'''
        channel, rho = build(n, epsilon)
        model = ReducedModel(channel, rho)
        assert model.nqb.check_invariance((channel,))

        coordinates = model.P(rho)
        state = rho
        for _ in range(model.nqb.dim() + 2):
            assert model.embed(coordinates) == approx(flat(state), abs=1e-8)
            state = state.apply_matrix(channel)
            coordinates = model.C_hat @ coordinates

    @mark.parametrize("name,build,n", EXPERIMENTS)
    def test_it_holds_for_an_arbitrary_element_of_the_subspace(self, name, build, n):
        r'''The theorem covers every `\mu = P^{-1}\hat{v}`, which need not be a density matrix.'''
        channel, rho = build(n, 1e-3)
        model = ReducedModel(channel, rho)
        d = model.nqb.dim()

        weights = np.array([(-1) ** i * (1 + i) + 1j * (i + 2) for i in range(d)], dtype=complex)
        mu = sum((complex(w) * b for w, b in zip(weights[1:], model.basis[1:])),
                 start=complex(weights[0]) * model.basis[0])

        assert model.P(mu) == approx(weights, abs=1e-8) # P really does invert embed

        coordinates = model.P(mu)
        state = mu
        for _ in range(d + 2):
            assert model.embed(coordinates) == approx(flat(state), abs=1e-8)
            state = state.apply_matrix(channel)
            coordinates = model.C_hat @ coordinates

    @mark.parametrize("name,build,n", EXPERIMENTS)
    def test_the_library_coordinate_map_agrees_with_the_inner_products(self, name, build, n):
        r'''
            ``pinv`` derives the real-field `L^+ = ((LL^T)^{-1}L)^T`, while over `CC` the basis is
            orthonormal only in the Hermitian sense; this pins that ``find_in`` conjugates anyway.
        '''
        channel, rho = build(n, 1e-3)
        model = ReducedModel(channel, rho)

        state = rho
        for _ in range(model.nqb.dim() + 1):
            from_library = np.array(model.nqb.find_in(state).to_list(), dtype=complex)
            assert from_library == approx(model.P(state), abs=1e-10)
            state = state.apply_matrix(channel)

    @mark.parametrize("name,build,n", [p for p in EXPERIMENTS if p.values[0] == "maxcut"])
    def test_the_maxcut_basis_is_genuinely_complex(self, name, build, n):
        r'''
            Without this the test above proves nothing about conjugation: Grover's basis comes out
            numerically real even under noise, so MaxCut's diagonal phases are what exercise it.
        '''
        model = ReducedModel(*build(n, 1e-3))

        assert np.abs(model.L.imag).max() > 1e-3

class TestContainsThreshold:
    r'''
    `NumericalSubspace.contains` uses a hardcoded `norm_squared() < 1e-15` with no reference to the
    subspace's own `delta`, so a residual discarded during construction can later be rejected as
    not in the subspace. This pins the current behaviour.
    '''
    def test_a_residual_between_the_two_thresholds_is_discarded_but_not_contained(self):
        from clue.linalg import SparseVector
        from clue.numerical_domains import CC

        subspace = NumericalSubspace(CC, delta=1e-6) # absorbs only residuals with norm^2 > 1e-12
        subspace.absorb_new_vector(SparseVector.from_list([1 + 0j, 0j, 0j], CC), force=True)

        # norm^2 = 1e-14: below delta^2 = 1e-12, so not absorbed; above the 1e-15 contains bar
        nearly_in = SparseVector.from_list([1 + 0j, 1e-7 + 0j, 0j], CC)

        assert subspace.absorb_new_vector(nearly_in.copy()) == -1 # treated as already present
        assert not subspace.contains(nearly_in.copy())            # yet not contained
        with raises(ValueError, match="not in the current subspace"):
            subspace.find_in(nearly_in.copy())
