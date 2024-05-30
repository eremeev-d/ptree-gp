from functools import partial

from gp_phylogenetic_trees.utils import (
    half_partition, check_covers, 
    iterate_x0_stabilizer, iterate_all_permutations
)
from gp_phylogenetic_trees.spherical_function import SphericalFunction
from gp_phylogenetic_trees.primitives import (
    Partition, Permutation, Tableau, Matching
)


def compute_projection(character, n, g):
    x0 = Matching(*tuple((2*h-1, 2*h) for h in range(1, n+1)))
    K = list(iterate_x0_stabilizer(x0))
    value = 0.
    for k1 in K:
        for k2 in K:
            value += character(k1 * g * k2)
    return value / (len(K)**2)


def check_proj_is_identical_zero(character_row, n, eps=1e-8):
    character = lambda g: character_row[g.partition()]
    for g in iterate_all_permutations(2*n):
        value = compute_projection(character, n, g)
        if abs(value) > eps:
            return False
    return True


def check_proj_equals_spherical_func(two_lamb, character_row, n, eps=1e-8):
    x0 = Matching(*tuple((2*h-1, 2*h) for h in range(1, n+1)))
    character = lambda g: character_row[g.partition()]
    sp = SphericalFunction(two_lamb)

    for g in iterate_all_permutations(2*n):
        proj_val = compute_projection(character, n, g)
        sp_val = sp(g * x0)
        if abs(proj_val - sp_val) > eps:
            return False

    return True


def main():
    n = 2

    A = Partition(4)
    B = Partition(3, 1)
    C = Partition(2, 2)
    D = Partition(2, 1, 1)
    E = Partition(1, 1, 1, 1)

    characters_table = {
        A: {E: 1, D: 1, C: 1, B: 1, A: 1},
        B: {E: 3, D: 1, C: -1, B: 0, A: -1},
        C: {E: 2, D: 0, C: 2, B: -1, A: 0},
        D: {E: 3, D: -1, C: -1, B: 0, A: 1},
        E: {E: 1, D: -1, C: 1, B: 1, A: -1}
    }

    assert not check_proj_is_identical_zero(characters_table[A], n)
    assert check_proj_is_identical_zero(characters_table[B], n)
    assert not check_proj_is_identical_zero(characters_table[C], n)
    assert check_proj_is_identical_zero(characters_table[D], n)
    assert check_proj_is_identical_zero(characters_table[E], n)

    assert check_proj_equals_spherical_func(A, characters_table[A], n)
    assert check_proj_equals_spherical_func(C, characters_table[C], n)

    print("All is OK!")


if __name__ == "__main__":
    main()