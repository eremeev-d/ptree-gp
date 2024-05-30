import math
from typing import Iterable, List

from gp_phylogenetic_trees.primitives import (
    Partition, Permutation, Tableau, Matching
)
from gp_phylogenetic_trees.utils import (
    iterate_all_partitions
)


class BorderStrip:
    def __init__(self, ksi: List[int]) -> None:
        self._ksi = ksi

    def height(self):
        height = 0
        for n in self._ksi:
            if n > 0:
                height += 1
        return height - 1

    def __iter__(self):
        return iter(self._ksi)

    def __rsub__(self, lamb: Partition) -> Partition:
        result = []
        for ksi_i, lamb_i in zip(self._ksi, lamb):
            new_lamb_i = lamb_i - ksi_i
            if new_lamb_i > 0:
                result.append(new_lamb_i)
            else:
                break  # We assume that lamb \ ksi is connected
        return Partition(*result)


# https://en.wikipedia.org/wiki/Murnaghan–Nakayama_rule
class SnCharactersTable:
    def __init__(self):
        self.clear_cache()

    def _compute_value(self, lamb: Partition, rho: Partition) -> int:
        # If rho == (1, 1, ..., 1), simply use hook-length formula instead
        if rho.size == rho.length:
            return self._hook_length_formula(lamb)

        value = 0

        for ksi in self._iterate_bs(lamb, rho[0]):
            new_lamb = lamb - ksi
            new_rho = Partition(*rho.aslist()[1:])
            if ksi.height() % 2:
                value -= self.get_value(new_lamb, new_rho)
            else:
                value += self.get_value(new_lamb, new_rho)

        return value

    def _is_correct_bs(self, lamb: Partition, ksi: List[int]) -> bool:
        new_lamb = [lamb_i - ksi_i for lamb_i, ksi_i in zip(lamb, ksi)]
        for i in range(len(new_lamb) - 1):
            if new_lamb[i] < new_lamb[i+1]:
                return False
        for new_lamb_i in new_lamb:
            if new_lamb_i < 0:
                return False
        return True


    def _iterate_bs(self, lamb: Partition, size: int) -> Iterable[BorderStrip]:
        for start_row in range(lamb.length):
            ksi = [0 for _ in range(lamb.length)]

            for row in range(start_row, lamb.length):
                ksi[row] = size - sum(ksi)  # TODO: optimize?
                ksi[row] = max(ksi[row], 0)
                if row + 1 != lamb.length:
                    ksi[row] = min(ksi[row], lamb[row] - lamb[row+1] + 1)

            if sum(ksi) != size:
                continue
            if self._is_correct_bs(lamb, ksi):
                yield BorderStrip(ksi)

    # https://en.wikipedia.org/wiki/Hook_length_formula
    def _hook_length_formula(self, lamb: Partition) -> int:
        lamb_conj = lamb.conjugate()
        d = math.factorial(lamb.size)  # d_\lambda

        for i in range(1, lamb.length+1):
            for j in range(1, lamb[i - 1]+1):
                hook_length = 1  # Cell (i, j) itself
                hook_length += lamb[i - 1] - j  # Cells left to (i, j)
                hook_length += lamb_conj[j - 1] - i
                assert d % hook_length == 0
                d //= hook_length
        return d

    # TODO: rename rho -> partition
    def get_value(self, character: Partition, rho: Partition) -> int:
        assert character.size == rho.size
        if (character, rho) not in self._cache:
            value = self._compute_value(character, rho)
            self._cache[(character, rho)] = value
        return self._cache[(character, rho)]

    def clear_cache(self):
        self._cache = dict()
        self._cache[(Partition(1), Partition(1))] = 1
        self._cache[(Partition(), Partition())] = 1


if __name__ == "__main__":
    table = SnCharactersTable()
    n = 30
    v = 0.0
    c = 0

    for lamb in iterate_all_partitions(n):
        c += 1
        rho = [2]
        for _ in range(n-rho[0]):
            rho.append(1)
        rho = Partition(*rho)
        v += table.get_value(character=lamb, rho=rho)**2

    print(v)
    print(c)
    print(len(table._cache))