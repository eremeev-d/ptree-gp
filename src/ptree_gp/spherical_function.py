from __future__ import annotations
from typing import List, Set, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np
import math

from ptree_gp.spaces import MatchingSpace
from ptree_gp.utils import (
    half_partition, double_partition, check_covers, 
    iterate_column_stabilizer, 
    iterate_all_matchings, iterate_all_partitions, 
    iterate_all_permutations, compute_sphere_size
)
from ptree_gp.zonal_polynomials import (
    ZpCoeffComputer, MsfToPsConverter, PsWeightedSum
)
from ptree_gp.primitives import (
    Partition, Permutation, Tableau, Matching,
    matching_distance
)
from ptree_gp.sn_characters import SnCharactersTable


class ZonalSphericalFunctionBase(ABC):
    def __init__(self, space: MatchingSpace) -> None:
        self._space = space

    @abstractmethod
    def __call__(
            self, 
            zsf_index: Partition, 
            permutation: Permutation
    ) -> float:
        raise NotImplementedError


class DistanceBasedZSFBase(ZonalSphericalFunctionBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _compute_at_dist(
            self, 
            zsf_index: Partition, 
            distance: Partition
    ) -> float:
        raise NotImplementedError

    def __call__(
            self, 
            zsf_index: Partition, 
            permutation: Permutation
    ) -> float:
        x0 = self._space.x0
        distance = matching_distance(permutation * x0, x0)
        return self._compute_at_dist(zsf_index, distance)

    def compute_at_dist(
            self, 
            zsf_index: Partition, 
            distance: Partition
    ) -> float:
        return self._compute_at_dist(zsf_index=zsf_index, distance=distance)


class ZonalPolynomialZSF(DistanceBasedZSFBase):
    def __init__(self, space: MatchingSpace) -> None:
        super().__init__(space)
        self._zp = ZpCoeffComputer(space.n)
        self._mtop = MsfToPsConverter()

    def _compute_at_dist(
            self, 
            zsf_index: Partition, 
            distance: Partition
    ) -> float:

        msf_coefs = self._zp(zsf_index)
        identity_partition = Partition(*[1 for _ in range(self._space.n)])
        
        num = 0.0
        den = 0.0

        # TODO: optimize to save only non-zero coefs and iterate over them?
        for kappa in iterate_all_partitions(self._space.n):
            num = num + msf_coefs[kappa] * self._mtop(kappa, distance)
            den = den + msf_coefs[kappa] * self._mtop(kappa, identity_partition)

        den = den * compute_sphere_size(distance)
        return num / den


# TODO: iterate all matchings more efficiently (!!!)
class NaiveZSF(DistanceBasedZSFBase):
    def __init__(self, space: MatchingSpace) -> None:
        super().__init__(space)

        self._numerator = dict()
        self._denominator = dict()

        for lamb in iterate_all_partitions(space.n):
            self._precompute_lamb(lamb)

    def _precompute_lamb(self, lamb: Partition) -> None:
        n = self._space.n
        x0 = self._space.x0
        t = self._space.get_good_tableau(lamb)

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


class ApproximationZSF(ZonalSphericalFunctionBase):
    def __init__(
            self, 
            space: MatchingSpace, 
            n_stab_samples: int = 100
    ) -> None:
        super().__init__(space)
        self._zsf_value = dict()
        self._characters = SnCharactersTable()
        self._stab_samples = [
            self._get_random_stabilizer_permutation(space.n) 
            for _ in range(n_stab_samples)]

    def _get_random_stabilizer_permutation(self, n: int):
        ### First, sample from stabilizer of x1 = {{1, 2}, {3, 4}, ..., {2n-1, 2n}} 
        pairs_order = np.random.permutation(n)
        permutation = []
        for i in pairs_order:
            if np.random.randint(2):
                permutation.append(2 * i + 1)
                permutation.append(2 * i + 2)
            else:
                permutation.append(2 * i + 2)
                permutation.append(2 * i + 1)
        permutation = Permutation(*permutation)

        ### If x0 = g x1, then Stab(x0) = g Stab(x1) g^{-1}
        g = []
        for pair in self._space.x0.iterate_pairs():
            g.extend(pair)
        g = Permutation(*g)
        return g * permutation * g.inverse()


    def __call__(
            self, 
            zsf_index: Partition, 
            permutation: Permutation
    ) -> float:
        zsf_value = 0.0
        rho = double_partition(zsf_index)
        for stab_1 in self._stab_samples:
            for stab_2 in self._stab_samples:
                g = stab_2.inverse() * permutation * stab_1
                zsf_value += self._characters.get_value(
                    character=rho, rho = g.partition())
        zsf_value /= len(self._stab_samples)**2
        return zsf_value