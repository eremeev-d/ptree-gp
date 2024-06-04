import pytest

from ptree_gp.sn_characters import SnCharactersTable
from ptree_gp.primitives import Partition


def test_full_n_equal_4():
    partitions = [
        Partition(4),
        Partition(3, 1),
        Partition(2, 2),
        Partition(2, 1, 1),
        Partition(1, 1, 1, 1)
    ]

    ground_truth_table = [
        [1, 1, 1, 1, 1],
        [3, 1, -1, 0, -1],
        [2, 0, 2, -1, 0],
        [3, -1, -1, 0, 1],
        [1, -1, 1, 1, -1]
    ]

    table = SnCharactersTable()

    for lamb_index, lamb in enumerate(partitions):
        for rho_index, rho in enumerate(partitions[::-1]):
            value = table.get_value(character=lamb, rho=rho)
            ground_truth_value = ground_truth_table[lamb_index][rho_index]
            assert value == ground_truth_value


def test_bigger():
    table = SnCharactersTable()

    assert table.get_value(
        character=Partition(4, 1, 1),
        rho=Partition(3, 3)
    ) == 1

    assert table.get_value(
        character=Partition(3, 3),
        rho=Partition(4, 1, 1)
    ) == -1

    assert table.get_value(
        character=Partition(3, 2, 1, 1, 1, 1, 1),
        rho=Partition(5, 1, 1, 1, 1, 1)
    ) == 5

    assert table.get_value(
        character=Partition(5, 3, 1, 1),
        rho=Partition(4, 2, 2, 1, 1)
    ) == -1

    assert table.get_value(
        character=Partition(4, 3, 2, 2, 2, 1, 1),
        rho=Partition(3, 3, 3, 2, 2, 2)
    ) == 9

    assert table.get_value(
        character=Partition(4, 3, 2, 2, 2, 1, 1),
        rho=Partition(3, 3, 3, 3, 1, 1, 1)
    ) == 0