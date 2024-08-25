from __future__ import annotations
from typing import List, Set, Tuple, Optional
from math import factorial, exp
from abc import ABC, abstractmethod

import numpy as np

from ptree_gp.spaces import MatchingSpace
from ptree_gp.primitives import (
    Partition, Permutation, Matching, matching_distance)
from ptree_gp.spherical_function import (
    ZonalSphericalFunctionBase, ZonalPolynomialZSF)
from ptree_gp.sn_characters import (
    SnCharactersTable)
from ptree_gp.utils import (
    double_partition, iterate_all_partitions)


class PTreeMaternKernel:
    def __init__(
            self, 
            space: MatchingSpace, 
            zsf: ZonalSphericalFunctionBase, 
            zsf_indexes: Optional[List[Partition]] = None
    ):
        self._space = space
        self._zsf = zsf
        if zsf_indexes is None:
            zsf_indexes = list(iterate_all_partitions(space.n))
        self._zsf_indexes = zsf_indexes
        self._characters = SnCharactersTable()

    def init_params(self) -> dict:
        return {
            "nu": np.inf,
            "lengthscale": 1.0
        }

    # TODO: optimize to cache k(identity)?
    def __call__(
            self, 
            params: dict, 
            permutation: Permutation, 
            normalize: bool = True
    ) -> float:
        if normalize:
            identity = Permutation(*list(range(1, 2*self._space.n+1)))
            return self._compute(params, permutation) / self._compute(params, identity)
        else:
            return self._compute(params, permutation)

    def _compute(self, params: dict, permutation: Permutation) -> float:
        n = self._space.n
        group_size = self._space.group_size

        kernel_value = 0.0

        for zsf_index in self._zsf_indexes:
            eigenvalue = self._compute_eigenvalue(zsf_index)
            phi_eigenvalue = self._phi(params, eigenvalue)
            dim_rho = self._compute_dim(zsf_index)
            kernel_value += phi_eigenvalue * (dim_rho / group_size) \
                * self._zsf(zsf_index=zsf_index, permutation=permutation)

        return kernel_value

    # TODO: add lru_cache?
    def _compute_dim(self, zsf_index: Partition) -> int:
        identity_partition = Partition(*[1 for _ in range(2 * self._space.n)])
        return self._characters.get_value(
            character=double_partition(zsf_index), rho=identity_partition)

    # NOTE: this is eigenvalue of *normalized* Laplacian
    def _compute_eigenvalue(self, zsf_index: Partition) -> float:
        n = self._space.n

        kappa = [2] + [1 for _ in range(2 * self._space.n - 2)]
        kappa = Partition(*kappa)

        dim_rho = self._compute_dim(zsf_index)
        chi_value = self._characters.get_value(
            character=double_partition(zsf_index), rho=kappa)
        eigenvalue = (dim_rho - chi_value) / dim_rho

        return eigenvalue

    def _phi(self, params: dict, eigenvalue: float) -> float:
        if params["nu"] == np.inf:
            return np.exp(
                - 0.5 * (params["lengthscale"]**2) * eigenvalue
            )
        else:
            return (
                (2 * params["nu"] / (params["lengthscale"] ** 2)) \
                + eigenvalue
            ) ** (-params["nu"])