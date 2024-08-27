import pytest
import numpy as np
from typing import List, Set, Tuple

from ptree_gp.spaces import MatchingSpace
from ptree_gp.spherical_function import (
    NaiveZSF, 
    ZonalPolynomialZSF, 
    ApproximationZSF
)
from ptree_gp.utils import (
    iterate_all_partitions
)
from ptree_gp.primitives import (
    Partition, Permutation, Tableau, Matching, matching_distance)
    
    
def get_random_permutation(n):
    perm = np.arange(1, n+1)
    return Permutation(*np.random.permutation(perm))


# TODO: refactor remaining tests to use __call__ instead of .compute_at_dist()

# @pytest.mark.parametrize(
#     "zsf_class, n",
#     [
#         [ZonalPolynomialZSF, 2],
#         [ZonalPolynomialZSF, 3],
#         [ZonalPolynomialZSF, 4],
#         [ZonalPolynomialZSF, 5],
#         [ZonalPolynomialZSF, 6],
#         [ZonalPolynomialZSF, 7],
#     ]
# )
# def test_zsf_first_nontrivial_value(zsf_class, n, atol=1e-9):
#     partitions = list(iterate_all_partitions(n))

#     space = MatchingSpace(n)
#     zsf = zsf_class(space)

#     for zsf_index in partitions:
#         distance = [2]
#         distance += [1 for _ in range(n-2)]
#         distance = Partition(*distance)
#         zsf_value = zsf.compute_at_dist(zsf_index, distance)
#         zsf_index = zsf_index.aslist()
#         true_value = np.sum(
#             zsf_index * (zsf_index - np.arange(1, len(zsf_index)+1))
#         ) / (n*(n-1))
#         assert abs(zsf_value - true_value) < atol
    

# @pytest.mark.parametrize(
#     "zsf_class_first, zsf_class_second, n",
#     [
#         [NaiveZSF, ZonalPolynomialZSF, 1],
#         [NaiveZSF, ZonalPolynomialZSF, 2],
#         [NaiveZSF, ZonalPolynomialZSF, 3],
#         [NaiveZSF, ZonalPolynomialZSF, 4],
#     ]
# )
# def test_zsf_equal(zsf_class_first, zsf_class_second, n, atol=1e-9):
#     partitions = list(iterate_all_partitions(n))
#     space = MatchingSpace(n)

#     zsf_first = zsf_class_first(space)
#     zsf_second = zsf_class_second(space)

#     for zsf_index in partitions:
#         for distance in partitions:
#             first_value = zsf_first.compute_at_dist(zsf_index, distance)
#             second_value = zsf_second.compute_at_dist(zsf_index, distance)
#             assert abs(first_value - second_value) < atol


@pytest.mark.parametrize(
    "zsf_class, n, n_runs",
    [
        [ApproximationZSF, 1, 10],
        [ApproximationZSF, 2, 10],
        [ApproximationZSF, 3, 10],
        [ApproximationZSF, 4, 10],
        # [NaiveZSF, 1, 1000],
        # [NaiveZSF, 2, 1000],
        # [NaiveZSF, 3, 1000],
        # [ZonalPolynomialZSF, 1, 1000],
        # [ZonalPolynomialZSF, 2, 1000],
        # [ZonalPolynomialZSF, 3, 1000],
        # [ZonalPolynomialZSF, 4, 1000],
        # [ZonalPolynomialZSF, 5, 1000],
        # [ZonalPolynomialZSF, 6, 1000],
        # [ZonalPolynomialZSF, 7, 10000],
    ]
)
def test_zsf_pos_def(
        zsf_class,
        n,
        n_runs, 
        matrix_size_range = [4, 5, 6, 7, 8, 9, 10, 20, 40],
        eps=1e-9,
        seed=42,
):
    np.random.seed(seed)

    space = MatchingSpace(n)
    zsf = zsf_class(space)

    partitions = list(iterate_all_partitions(n))

    for _ in range(n_runs):
        zsf_index = np.random.choice(partitions)

        m = np.random.choice(matrix_size_range)
        matrix = np.zeros((m, m))
        permutations = [get_random_permutation(2*n) or _ in range(m)]

        for i, sigma in enumerate(permutations):
            for j, pi in enumerate(permutations):
                permutation = pi.inverse() * sigma
                matrix[i, j] = zsf(zsf_index, permutation)

        assert np.all(np.linalg.eigvals(matrix) >= -eps)
        assert np.allclose(matrix, matrix.T)  # Check for symmetry


# # TODO: move examples to separate file?)
# @pytest.mark.parametrize(
#     "zsf_class",
#     [
#         NaiveZSF,
#         ZonalPolynomialZSF
#     ]
# )
# def test_spherical_function(zsf_class, atol=1e-8):
#     space = MatchingSpace(n = 4)
#     zsf = zsf_class(space)

#     for (zsf_index, distance, ground_truth_value) in [
#         [Partition(4,), Partition(4,), 1.000000000],
#         [Partition(4,), Partition(3, 1), 1.000000000],
#         [Partition(4,), Partition(2, 2), 1.000000000],
#         [Partition(4,), Partition(2, 1, 1), 1.000000000],
#         [Partition(4,), Partition(1, 1, 1, 1), 1.000000000],
#         [Partition(3, 1), Partition(4,), -0.166666667],
#         [Partition(3, 1), Partition(3, 1), 0.125000000],
#         [Partition(3, 1), Partition(2, 2), -0.166666667],
#         [Partition(3, 1), Partition(2, 1, 1), 0.416666667],
#         [Partition(3, 1), Partition(1, 1, 1, 1), 1.000000000],
#         [Partition(2, 2), Partition(4,), -0.041666667],
#         [Partition(2, 2), Partition(3, 1), -0.250000000],
#         [Partition(2, 2), Partition(2, 2), 0.583333333],
#         [Partition(2, 2), Partition(2, 1, 1), 0.166666667],
#         [Partition(2, 2), Partition(1, 1, 1, 1), 1.000000000],
#         [Partition(2, 1, 1), Partition(4,), 0.083333333],
#         [Partition(2, 1, 1), Partition(3, 1), -0.062500000],
#         [Partition(2, 1, 1), Partition(2, 2), -0.166666667],
#         [Partition(2, 1, 1), Partition(2, 1, 1), -0.083333333],
#         [Partition(2, 1, 1), Partition(1, 1, 1, 1), 1.000000000],
#         [Partition(1, 1, 1, 1), Partition(4,), -0.125000000],
#         [Partition(1, 1, 1, 1), Partition(3, 1), 0.250000000],
#         [Partition(1, 1, 1, 1), Partition(2, 2), 0.250000000],
#         [Partition(1, 1, 1, 1), Partition(2, 1, 1), -0.500000000],
#         [Partition(1, 1, 1, 1), Partition(1, 1, 1, 1), 1.000000000],
#     ]:
#         zsf_value = zsf.compute_at_dist(zsf_index, distance)
#         assert abs(zsf_value - ground_truth_value) < atol
