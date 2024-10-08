import pytest
import numpy as np

from ptree_gp.spaces import MatchingSpace
from ptree_gp.kernels import PTreeMaternKernel
from ptree_gp.spherical_function import (
    ZonalPolynomialZSF, 
    ApproximationZSF, 
    NaiveZSF
)
from ptree_gp.primitives import (
    Permutation, 
    Matching, 
    matching_distance
)
    
def get_random_permutation(n):
    perm = np.arange(1, n+1)
    return Permutation(*np.random.permutation(perm))


# TODO: add param to pass kwargs to kernels
@pytest.mark.parametrize(
    "kernel_class, zsf_class, n, n_runs",
    [
        # TODO: add approximaion zsf
        # [PTreeHeatKernel, ApproximationZSF, 4, 5],
        # [PTreeMaternKernel, ApproximationZSF, 4, 5],
        # TODO: run tests with different params
        # [PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 1, 1000],
        # [PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 2, 1000],
        # [PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 3, 1000],
        # [PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 4, 1000],
        # [PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 5, 1000],
        # [PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 6, 1000],
        # [PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 7, 10000],
        # [PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 8, 10000],
        [PTreeMaternKernel, ZonalPolynomialZSF, 1, 100],
        [PTreeMaternKernel, ZonalPolynomialZSF, 2, 100],
        [PTreeMaternKernel, ZonalPolynomialZSF, 3, 100],
        [PTreeMaternKernel, ZonalPolynomialZSF, 4, 100],
        [PTreeMaternKernel, ZonalPolynomialZSF, 5, 100],
        [PTreeMaternKernel, ZonalPolynomialZSF, 6, 100],
        [PTreeMaternKernel, ZonalPolynomialZSF, 7, 100],
        [PTreeMaternKernel, ZonalPolynomialZSF, 8, 100],
    ]
)
def test_kernel_pos_semidef(
        kernel_class,
        zsf_class,
        n,
        n_runs, 
        matrix_size_range = [4, 5, 6, 7, 8, 9, 10, 20, 40],
        eps = 1e-9,
        seed = 42,
):
    np.random.seed(seed)

    space = MatchingSpace(n)
    zsf = zsf_class(space)

    kernel = kernel_class(space, zsf)
    params = kernel.init_params()

    for _ in range(n_runs):
        m = np.random.choice(matrix_size_range)
        matrix = np.zeros((m, m))
        permutations = [get_random_permutation(2*n) or _ in range(m)]

        for i, sigma in enumerate(permutations):
            for j, pi in enumerate(permutations):
                permutation = pi.inverse() * sigma
                matrix[i, j] = kernel(params, permutation)

        assert np.all(np.linalg.eigvals(matrix) >= -eps)
        assert np.allclose(matrix, matrix.T)  # Check for symmetry


# # TODO: remove so long line
# @pytest.mark.parametrize(
#     "kernel_class_first, zsf_class_first, kernel_class_second, zsf_class_second, n, n_runs",
#     [
#         [
#             PTreeHeatPrecomputedKernel, ZonalPolynomialZSF, 
#             PTreeHeatKernel, ZonalPolynomialZSF, 
#             4, 1000
#         ],
#         [
#             PTreeMaternPrecomputedKernel, ZonalPolynomialZSF, 
#             PTreeMaternKernel, ZonalPolynomialZSF, 
#             4, 1000
#         ],
#         [
#             PTreeMaternPrecomputedKernel, ZonalPolynomialZSF, 
#             PTreeMaternPrecomputedKernel, NaiveZSF, 
#             4, 1000
#         ],
#     ]
# )
# def test_kernels_equal(
#         kernel_class_first,
#         zsf_class_first,
#         kernel_class_second,
#         zsf_class_second,
#         n,
#         n_runs, 
#         eps = 1e-9,
#         seed = 42,
#         atol=1e-9,
# ):
#     np.random.seed(seed)

#     zsf_first = zsf_class_first().precompute(n)
#     kernel_first = kernel_class_first(
#         n=n, zsf=zsf_first, kappa=1.0).precompute()

#     zsf_second = zsf_class_second().precompute(n)
#     kernel_second = kernel_class_second(
#         n=n, zsf=zsf_second, kappa=1.0).precompute()

#     for _ in range(n_runs):
#         g_array = [get_random_permutation(2*n) or _ in range(n_runs)]

#         for g in g_array:
#             first_value = kernel_first(g)
#             second_value = kernel_second(g)
#             assert abs(first_value - second_value) < atol


# TODO: add tests with fixed values like "test_spherical_function" 