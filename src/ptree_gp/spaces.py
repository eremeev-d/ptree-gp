from typing import Optional
from math import factorial

from ptree_gp.primitives import Matching, Tableau, Partition, Permutation
from ptree_gp.utils import double_partition


class MatchingSpace:
    def __init__(self, n: int = None, x0: Optional[Matching] = None):
        if x0 is not None:
            if n is not None:
                assert x0.get_number_of_pairs() == n
            n = x0.get_number_of_pairs()
        else:
            assert n is not None
            x0 = Matching(*tuple((2*h-1, 2*h) for h in range(1, n+1)))
        self._x0 = x0
        self._n = n

    @property
    def x0(self) -> Matching:
        return self._x0
    
    @property
    def n(self) -> int:
        return self._n

    @property
    def group_size(self) -> int:
        return factorial(2 * self.n)

    def get_good_tableau(self, lamb: Partition) -> Tableau:
        permutation = []
        for pair in self.x0.iterate_pairs():
            permutation.extend(pair)
        return Tableau(double_partition(lamb), Permutation(*permutation))