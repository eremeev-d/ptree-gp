from __future__ import annotations
from typing import List, Set, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np
import math

from gp_phylogenetic_trees.utils import (
    half_partition, double_partition, check_covers, 
    iterate_column_stabilizer, iterate_x0_stabilizer, 
    iterate_all_matchings, iterate_all_partitions, 
    iterate_all_permutations, compute_sphere_size
)
from gp_phylogenetic_trees.zonal_polynomials import (
    ZpCoeffComputer, MsfToPsConverter, PsWeightedSum
)
from gp_phylogenetic_trees.primitives import (
    Partition, Permutation, Tableau, Matching,
    matching_distance
)
from gp_phylogenetic_trees.sn_characters import SnCharactersTable

# NOTE (!!!): Currently it is assumed that x0 = {{1, 2}, {3, 4}, ..., {2n-1, 2n}}
# TODO: add description of methods
# TODO: move n to init? and probably replace with HomogenousSpace?
class ZonalSphericalFunctionBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _precompute(
            self, 
            n: int, 
            zsf_indexes: List[Partition]
    ) -> None:
        ...

    # TODO: add some checks for zsf_indxes.size == n?
    def precompute(
            self, 
            n: int, 
            zsf_indexes: Optional[List[Partition]] = None,
            **kwargs
    ) -> ZonalSphericalFunction:
        if zsf_indexes is None:
            zsf_indexes = list(iterate_all_partitions(n))
        self._precompute(n=n, zsf_indexes=zsf_indexes, **kwargs)
        return self

    @abstractmethod
    def _compute(
            self, 
            zsf_index: Partition, 
            permutation: Permutation
    ) -> float:
        ...

    # NOTE: if changed, dont forget to compute_at_dist too
    def __call__(
            self, 
            zsf_index: Partition, 
            permutation: Permutation
    ) -> float:
        return self._compute(zsf_index=zsf_index, permutation=permutation)


class DistanceBasedZSFBase(ZonalSphericalFunctionBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _compute_at_dist(
            self, 
            zsf_index: Partition, 
            distance: Partition
    ) -> float:
        ...

    def _compute(
            self, 
            zsf_index: Partition, 
            permutation: Permutation
    ) -> float:
        n = zsf_index.size
        x0 = Matching(*tuple((2*h-1, 2*h) for h in range(1, n+1)))
        distance = matching_distance(permutation * x0, x0)
        return self._compute_at_dist(zsf_index, distance)

    def compute_at_dist(
            self, 
            zsf_index: Partition, 
            distance: Partition
    ) -> float:
        return self._compute_at_dist(zsf_index=zsf_index, distance=distance)


# TODO: iterate all matchings more efficiently (!!!)
class NaiveZSF(DistanceBasedZSFBase):
    def _precompute(
            self, 
            n: int, 
            zsf_indexes: List[Partition]
    ) -> None:
        self._numerator = dict()
        self._denominator = dict()

        for lamb in zsf_indexes:
            self._precompute_lamb(n, lamb)

    def _precompute_lamb(self, n: int, lamb: Partition) -> None:
        two_lamb = double_partition(lamb)
        x0 = Matching(*tuple((2*h-1, 2*h) for h in range(1, n+1)))
        t = Tableau(two_lamb, Permutation(*range(1, 2*n+1)))

        for x in iterate_all_matchings(n):
            d = matching_distance(x, x0)
            key = (lamb, d)

            if key not in self._denominator:
                self._denominator[key] = 0
            self._denominator[key] = self._denominator[key] + 1

            for sigma in iterate_column_stabilizer(t, fix_even=True):
                if check_covers(x, sigma * t):
                    if key not in self._numerator:
                        self._numerator[key] = 0
                    self._numerator[key] = self._numerator[key] + sigma.sign()

    def _compute_at_dist(
            self, 
            zsf_index: Partition, 
            distance: Partition
    ) -> float:
        key = (zsf_index, distance)
        return self._numerator[key] / self._denominator[key]


class ZonalPolynomialZSF(DistanceBasedZSFBase):
    def _precompute(
            self, 
            n: int, 
            zsf_indexes: List[Partition]
    ) -> None:
        self._zp_coefs = dict()
        self._zp = ZpCoeffComputer(n)
        self._mtop = MsfToPsConverter()

        for lamb in zsf_indexes:
            self._precompute_lamb(n, lamb)

    def _precompute_lamb(self, n: int, lamb: Partition) -> None:
        x0 = Matching(*tuple((2*h-1, 2*h) for h in range(1, n+1)))

        # TODO: rewrite it
        zp_coefs = PsWeightedSum({})
        for msf in iterate_all_partitions(n):
            t = {}
            for msf_i in msf:
                if msf_i not in t:
                    t[msf_i] = 0
                t[msf_i] += 1

            den = 1.0
            for i, t_i in t.items():
                den *= math.factorial(t_i)

            monom = (self._zp(kappa=lamb, lamb=msf) / den) \
                * self._mtop.monom(msf)
            zp_coefs += monom

        zp_coefs = zp_coefs.asdict()
        norm = zp_coefs[Partition(*[1 for _ in range(n)])]
        for key in list(zp_coefs.keys()):
            zp_coefs[key] /= norm

        self._zp_coefs[lamb] = zp_coefs

    def _compute_at_dist(
            self, 
            zsf_index: Partition, 
            distance: Partition
    ) -> float:
        zp_coefs = self._zp_coefs[zsf_index]
        if distance in zp_coefs:
            return zp_coefs[distance] / compute_sphere_size(distance)
        else:
            return 0.0


class ApproximationZSF(ZonalSphericalFunctionBase):
    def _precompute(
            self, 
            n: int, 
            zsf_indexes: List[Partition],
            n_stab_samples = 100
    ) -> None:
        self._zsf_value = dict()
        self._characters = SnCharactersTable()
        self._stab_samples = [
            self._get_random_stabilizer_permutation(n) 
            for _ in range(n_stab_samples)]

    def _get_random_stabilizer_permutation(self, n: int):
        pairs_order = np.random.permutation(n)
        permutation = []
        for i in pairs_order:
            if np.random.randint(2):
                permutation.append(2 * i + 1)
                permutation.append(2 * i + 2)
            else:
                permutation.append(2 * i + 2)
                permutation.append(2 * i + 1)
        return Permutation(*permutation)

    def _compute(self, zsf_index: Partition, permutation: Permutation) -> float:
        zsf_value = 0.0
        rho = double_partition(zsf_index)
        for stab_1 in self._stab_samples:
            for stab_2 in self._stab_samples:
                g = stab_2.inverse() * permutation * stab_1
                zsf_value += self._characters.get_value(
                    character=rho, rho = g.partition())
        zsf_value /= len(self._stab_samples)**2
        return zsf_value