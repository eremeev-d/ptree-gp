import math

import pytest

from gp_phylogenetic_trees.utils import (
    check_covers, half_partition, compute_sphere_size,
    iterate_all_permutations, iterate_column_stabilizer,
    iterate_all_partitions, iterate_all_matchings)
from gp_phylogenetic_trees.primitives import (
    Partition, Permutation, Tableau, Matching,
    matching_distance)


def test_partition():
    assert Partition(1, 2, 3) == Partition(3, 2, 1)
    assert Partition(3, 2, 1) != Partition(2, 2, 2)

    assert Partition(1, 2, 2).size == 5
    assert Partition(3, 2, 1).size == 6
    assert Partition(2, 3, 3).size == 8

    assert Partition(1, 2, 2).length == 3
    assert Partition(4).length == 1
    assert Partition(1, 2, 3, 2, 2).length == 5

    assert Partition(4, 2, 1).conjugate() == Partition(3, 2, 1, 1)
    assert Partition(4, 2).conjugate() == Partition(2, 2, 1, 1)

    assert Partition(1, 2, 3)[0] == 3
    assert Partition(1, 2, 3)[1] == 2
    assert Partition(1, 2, 3)[2] == 1
    assert Partition(1, 2, 2, 2)[0] == 2
    assert Partition(1, 2, 2, 2)[1] == 2
    assert Partition(1, 2, 2, 2)[2] == 2
    assert Partition(1, 2, 2, 2)[3] == 1
    

def test_permutation():
    assert Permutation(1, 2, 3) != Permutation(3, 2, 1)
    assert Permutation(1, 2, 3).sign() == 1
    assert Permutation(2, 1, 3).sign() == -1
    assert len(Permutation(3, 2, 1)) == 3
    assert (Permutation(2, 5, 3, 4, 1) * Permutation(2, 4, 3, 1, 5)) == Permutation(5, 4, 3, 2, 1)
    assert (Permutation(1, 2, 3, 4, 5) * Permutation(2, 4, 3, 1, 5)) == Permutation(2, 4, 3, 1, 5)
    assert Permutation(2, 5, 3, 4, 1) * Permutation(2, 5, 3, 4, 1).inverse() == Permutation(1, 2, 3, 4, 5)
    assert Permutation(1, 2, 3, 4, 5).inverse() == Permutation(1, 2, 3, 4, 5)


def test_tableau():
    assert Tableau(
        diagram=Partition(4, 2, 1),
        values=Permutation(5, 6, 3, 4, 1, 2, 7)).get_columns() == ({1, 5, 7}, {2, 6}, {3}, {4})


def test_matching():
    assert Permutation(2, 3, 4, 1) * Matching((1, 2), (3, 4)) == Matching((2, 3), (4, 1))
    assert Permutation(2, 1, 3, 4) * Matching((1, 2), (3, 4)) == Matching((1, 2), (3, 4))
    assert Permutation(4, 3, 1, 2) * Matching((1, 2), (3, 4)) == Matching((1, 2), (3, 4))


def test_check_covers():
    assert check_covers(
        x=Matching((1, 2), (3, 4), (5, 6)),
        t=Tableau(Partition(4, 2), Permutation(1, 2, 3, 4, 5, 6)))
    assert check_covers(
        x=Matching((1, 2), (3, 4), (5, 6)),
        t=Tableau(Partition(4, 2), Permutation(1, 2, 3, 4, 5, 6)))
    assert not check_covers(
        x=Matching((1, 2), (3, 4), (5, 6)),
        t=Tableau(Partition(4, 2), Permutation(6, 2, 3, 4, 5, 1)))


def test_half_partition():
    assert half_partition(Partition(4, 4)) == Partition(2, 2)
    assert half_partition(Partition(8, 2)) == Partition(4, 1)
    assert half_partition(Partition(2, 2, 2)) == Partition(1, 1, 1)
    assert half_partition(Partition(8, 4, 4, 2)) == Partition(4, 2, 2, 1)


def test_permutation_partition():
    assert Permutation(1).partition() == Partition(1)
    assert Permutation(1, 2, 3, 4).partition() == Partition(1, 1, 1, 1)
    assert Permutation(2, 1, 3, 4).partition() == Partition(2, 1, 1)
    assert Permutation(2, 1, 4, 3).partition() == Partition(2, 2)
    assert Permutation(2, 3, 4, 1).partition() == Partition(4)
    assert Permutation(2, 1, 4, 3, 5).partition() == Partition(2, 2, 1)


@pytest.mark.parametrize("t", [
    Tableau(
        diagram=Partition(2, 2),
        values=Permutation(4, 2, 1, 3)
    ),
    Tableau(
        diagram=Partition(3, 3, 2),
        values=Permutation(4, 5, 2, 8, 1, 7, 3, 6)
    ),
    Tableau(
        diagram=Partition(4, 2, 2), 
        values=Permutation(1, 2, 3, 4, 5, 6, 7, 8)
    )
])
def test_iterate_column_stabilizer(t: Tableau):
    sigmas_gt = set()
    t_cols = t.get_columns()
    n = t.get_diagram().size
    for sigma in iterate_all_permutations(n):
        sigma_t = sigma * t
        if sigma_t.get_columns() == t_cols:
            sigmas_gt.add(sigma)

    sigmas = set()
    for sigma in iterate_column_stabilizer(t):
        sigmas.add(sigma)

    assert sigmas == sigmas_gt


def test_iterate_all_partitions():
    assert list(iterate_all_partitions(0)) == [Partition()]

    assert list(iterate_all_partitions(1)) == [Partition(1)]

    assert list(iterate_all_partitions(2)) == [
        Partition(2), Partition(1, 1)]

    assert list(iterate_all_partitions(3)) == [
        Partition(3), Partition(2, 1), Partition(1, 1, 1)]

    assert list(iterate_all_partitions(4)) == [
        Partition(4), Partition(3, 1), Partition(2, 2), 
        Partition(2, 1, 1), Partition(1, 1, 1, 1)]

    assert list(iterate_all_partitions(5)) == [
        Partition(5), Partition(4, 1), Partition(3, 2),
        Partition(3, 1, 1), Partition(2, 2, 1),
        Partition(2, 1, 1, 1), Partition(1, 1, 1, 1, 1)]

    for n in range(10):
        partitions = [mu.aslist() for mu in iterate_all_partitions(n)]
        assert sorted(partitions, reverse=True) == partitions

    assert len(list(iterate_all_partitions(20))) == 627
    assert len(list(iterate_all_partitions(25))) == 1958
    assert len(list(iterate_all_partitions(30))) == 5604

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_iterate_all_matchings(n: int):  # TODO: add tests for values?
    matchings = set(iterate_all_matchings(n))
    assert len(matchings) == (math.factorial(2*n) / ((2**n) * math.factorial(n)))
    

@pytest.mark.parametrize("x, y, d", [
    [
        Matching((1, 2), (3, 4)),
        Matching((1, 2), (3, 4)),
        Partition(1, 1)
    ],
    [
        Matching((1, 2), (3, 4)),
        Matching((2, 3), (1, 4)),
        Partition(2)
    ],
    [
        Matching((1, 2), (3, 4), (5, 6)),
        Matching((2, 3), (1, 4), (5, 6)),
        Partition(2, 1)
    ],
    [
        Matching((1, 2), (3, 4), (5, 6), (7, 8)),
        Matching((2, 3), (1, 4), (5, 7), (6, 8)),
        Partition(2, 2)
    ],
])
def test_matching_distance(x: Matching, y: Matching, d: Partition):
    for sigma in iterate_all_permutations(2 * x.get_number_of_pairs()):
        assert matching_distance(sigma * x, sigma * y) == d


@pytest.mark.parametrize("d, true_size", [
    [Partition(1,), 1],

    [Partition(2,), 2],
    [Partition(1, 1), 1],

    [Partition(3,), 8],
    [Partition(2, 1), 6],
    [Partition(1, 1, 1), 1],

    [Partition(4,), 48],
    [Partition(3, 1), 32],
    [Partition(2, 2), 12],
    [Partition(2, 1, 1), 12],
    [Partition(1, 1, 1, 1), 1],

    [Partition(5,), 384],
    [Partition(4, 1), 240],
    [Partition(3, 2), 160],
    [Partition(3, 1, 1), 80],
    [Partition(2, 2, 1), 60],
    [Partition(2, 1, 1, 1), 20],
    [Partition(1, 1, 1, 1, 1), 1],
    
    [Partition(2, 1, 1, 1, 1), 6*5],
    [Partition(2, 1, 1, 1, 1, 1), 7*6],
    [Partition(2, 1, 1, 1, 1, 1, 1), 8*7],
    [Partition(2, 1, 1, 1, 1, 1, 1, 1), 9*8],
])
def test_compute_sphere_size(d: Partition, true_size: int) -> None:
    assert compute_sphere_size(d) == true_size