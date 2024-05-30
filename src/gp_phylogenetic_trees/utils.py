from __future__ import annotations
from typing import List, Set, Tuple, Optional
from math import factorial

import itertools

from gp_phylogenetic_trees.primitives import (
    Partition, Permutation, Tableau, Matching
)


def check_covers(x: Matching, t: Tableau):
    index_to_row = dict()
    row = 0
    cnt = 0
    for index in t._values._value:
        index_to_row[index] = row
        cnt += 1
        if cnt == t._diagram._value[row]:
            row += 1
            cnt = 0
    
    for pair in x._value:
        if index_to_row[pair[0]] != index_to_row[pair[1]]:
            return False
    return True


def half_partition(double_partition: Partition):
    double_value = double_partition._value
    value = []
    for x in double_value:
        assert x % 2 == 0
        value.append(x // 2)
    return Partition(*value)


def double_partition(partition: Partition):  # TODO: add tests
    return Partition(*[2*h for h in partition])


def iterate_all_permutations(n: int):
    for sigma in itertools.permutations(range(1, n+1)):
        yield Permutation(*sigma)


# TODO: optimize with not converting to partition and back
def iterate_all_partitions(n: int, m: Optional[int] = None):
    """Iterate over all partitions of n, elements of partition must be <= m"""
    if m is None:
        m = n
    m = min(m, n)

    if n == 0:
        yield Partition()
        return
    if n == 1:
        yield Partition(1)
        return

    for first_element in range(1, m+1)[::-1]:
        for sub_partition in iterate_all_partitions(
            n - first_element, first_element
        ):
            partition = [first_element] + sub_partition.aslist()
            yield Partition(*partition)


# TODO: optimize even more?
def iterate_all_matchings(
        n: int, 
        is_available: Optional[List[bool]] = None
):
    if is_available is None:
        is_available = list([True for _ in range(2*n)])
        is_initial_call = True
    else:
        is_initial_call = False

    if sum(is_available) == 0:
        yield []

    for i in range(2*n):
        if not is_available[i]:
            continue
        is_available[i] = False

        for j in range(i+1, 2*n):
            if not is_available[j]:
                continue
            is_available[j] = False

            for sub_matching in iterate_all_matchings(n, is_available):
                sub_matching.append((i+1, j+1))
                if is_initial_call:
                    yield Matching(*sub_matching)
                else:
                    yield sub_matching

            is_available[j] = True

        is_available[i] = True
        

def iterate_column_stabilizer(t: Tableau, fix_even: bool = False):
    t_cols = t.get_columns()
    n = t.get_diagram().size

    col_iterators = [
        iter(itertools.permutations(col)) for col in t_cols
    ]

    if fix_even:
        for i in range(len(col_iterators)):
            if i % 2 == 0:
                col_iterators[i] = [t_cols[i]]

    iterator = itertools.product(*col_iterators)

    for col_sigmas in iterator:
        sigma = [0 for _ in range(n)]
        for col, col_sigma in zip(t_cols, col_sigmas):
            for i, j in zip(col, col_sigma):
                sigma[i-1] = j
        yield Permutation(*sigma)


def iterate_x0_stabilizer(x0: Matching):
    n = x0.get_number_of_pairs()
    for k in iterate_all_permutations(2*n):
        if k * x0 == x0:
            yield k


def compute_sphere_size(d: Partition) -> int:
    n = d.size

    cnt = dict()
    for d_i in d:
        if d_i not in cnt:
            cnt[d_i] = 0
        cnt[d_i] += 1

    num = factorial(d.size) * 2**n
    
    den = 1
    for k, cnt_k in cnt.items():
        den *= factorial(cnt_k)
        den *= (2 * k) ** cnt_k

    assert num % den == 0
    return num // den