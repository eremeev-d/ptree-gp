from __future__ import annotations
from typing import List, Set, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import math

from ptree_gp.utils import (
    half_partition, check_covers, 
    iterate_column_stabilizer, iterate_x0_stabilizer, 
    iterate_all_matchings, iterate_all_partitions
)
from ptree_gp.primitives import (
    Partition, Permutation, Tableau, Matching,
    matching_distance
)


class ZpCoeffComputer:
    def __init__(self, n, eps=1e-12):
        self._n = n
        self._eps = eps

    # NOTE: computes coefs up to some unknown constant that depends on zp_index!
    def __call__(self, zp_index: Partition) -> Dict[Partition, float]:
        coefs = dict()
        coefs[zp_index] = 1.0
        n = self._n

        partitions = [  # TODO: just add comparator to Partition class?
            lamb.astuple() 
            for lamb 
            in iterate_all_partitions(n)
        ]
        partitions.sort(reverse=True)
        partitions = [Partition(*lamb) for lamb in partitions]

        for zp_coef in partitions:
            if zp_coef == zp_index:
                continue
            elif self._less(zp_index, zp_coef):
                coefs[zp_coef] = 0.0
                continue

            result = 0.0

            for s, zp_coef_s in enumerate(zp_coef):
                for r, zp_coef_r in enumerate(zp_coef):
                    if r >= s:
                        break
                    for t in range(1, zp_coef_s + 1):
                        mu = list(zp_coef.aslist()).copy()
                        mu[s] -= t
                        mu[r] += t
                        mu = list(filter(lambda mu_i: mu_i >= 1, mu))
                        mu = Partition(*mu)

                        if self._less(zp_coef, mu) and self._lesseq(mu, zp_index):
                            result += (zp_coef_r + t - zp_coef_s + t) * coefs[mu]

            if abs(result) > self._eps:
                den = self._rho(zp_index) - self._rho(zp_coef)
                result = result / den
            else:
                result = 0.0

            coefs[zp_coef] = result

        return coefs

    def _less(self, a: Partition, b: Partition) -> bool:
        return a.aslist() < b.aslist()
        a_sum = 0.0
        b_sum = 0.0
        for a_i, b_i in zip(a, b):
            a_sum += a_i
            b_sum += b_i
            if a_sum > b_sum + self._eps :
                return False
        return True

    def _lesseq(self, a: Partition, b: Partition) -> bool:
        return self._less(a, b) or (a == b)

    def _rho(self, lamb: Partition) -> float:
        result = 0.0
        for i, lamb_i in enumerate(lamb):
            result += lamb_i * (lamb_i - (i + 1))
        return result


def check_ps_dict_degree(ps_dict: dict[Partition, int]):
    degree = None
    for key in ps_dict.keys():
        if degree is None:
            degree = key.size
        else:
            if degree != key.size:
                return False
    return True


class PsWeightedSum:
    def __init__(self, value: dict[Partition, int]) -> None:
        self._value = value

    def __repr__(self) -> str:
        return repr(self._value)

    def asdict(self) -> dict:
        return self._value

    def __mul__(self, other: PsWeightedSum) -> PsWeightedSum:
        result = {}

        for key_first, value_first in self._value.items():
            for key_second, value_second in other._value.items():
                key = key_first._value + key_second._value  # TODO: dont use _value
                key = Partition(*key)
                if key not in result:
                    result[key] = 0
                result[key] += value_first * value_second

        assert check_ps_dict_degree(result), "mul"
        return PsWeightedSum(result)

    def __isub__(self, other: PsWeightedSum) -> PsWeightedSum:
        for key in other._value.keys():
            if key not in self._value:
                self._value[key] = 0
            self._value[key] -= other._value[key]
            if self._value[key] == 0:
                self._value.pop(key)
        return self

    def __iadd__(self, other: PsWeightedSum) -> PsWeightedSum:
        for key in other._value.keys():
            if key not in self._value:
                self._value[key] = 0.0
            self._value[key] += other._value[key]
            if self._value[key] == 0.0:
                self._value.pop(key)
        return self

    def __rmul__(self, const: float) -> float:
        result = deepcopy(self._value)
        for key in result:
            result[key] *= const
        return PsWeightedSum(result)


class MsfToPsConverter:
    def __init__(self) -> None:
        self._cache = dict()

    def monom(self, a: Partition) -> PsWeightedSum:
        if a in self._cache:
            return self._cache[a]
        else:
            value = self._monom(a)
            self._cache[a] = value
            return value

    # TODO: dont use _value of Partition
    def _monom(self, a: Partition) -> PsWeightedSum:
        n = a.length

        if n == 1:
            return PsWeightedSum({ Partition(a[0]): 1 })
        if n == 2:
            return PsWeightedSum({
                Partition(a[0], a[1]): 1,
                Partition(a[0] + a[1]): -1
            })
        
        c = Partition(*a._value[:-1])
        s = PsWeightedSum({Partition(a[n-1]): 1}) * self.monom(c)

        for i in range(n-1):
            b = list(deepcopy(c)._value)
            b[i] += a[n-1]
            s -= self.monom(Partition(*b))

        return s

    def __call__(self, kappa: Partition, mu: Partition):
        monom = self.monom(kappa).asdict()
        if mu in monom:
            coef = monom[mu]
            t = {}
            for kappa_i in kappa:
                if kappa_i not in t:
                    t[kappa_i] = 0
                t[kappa_i] += 1

            den = 1.0
            for i, t_i in t.items():
                den *= math.factorial(t_i)
            return coef / den
        else:
            return 0.0