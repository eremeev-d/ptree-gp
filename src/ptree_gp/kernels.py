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
            num_zsf_indexes: Optional[int] = None
    ):
        self._space = space
        self._zsf = zsf
        self._num_zsf_indexes = num_zsf_indexes
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

    # TODO: add lru_cache?
    # TODO: add some tests?
    # TODO: check formula for impact
    # NOTE: returns |G| * |H| * impact(rho)
    def _get_normalized_impact(self, params: dict, zsf_index: Partition):
        n = self._space.n
        two_rho = double_partition(zsf_index)

        identity_partition = Partition(*[1 for _ in range(2 * n)])
        kappa = [2] + [1 for _ in range(2 * n - 2)]
        kappa = Partition(*kappa)

        dim_rho = self._characters.get_value(
            character=two_rho, rho=identity_partition)

        chi_value = self._characters.get_value(
            character=two_rho, rho=kappa)

        eigenvalue = (dim_rho - chi_value) / dim_rho
        phi_eigenvalue = self._phi(params, eigenvalue)
        return (phi_eigenvalue ** 2) * dim_rho

    def _get_zsf_indexes(self, params: dict):
        zsf_indexes = list(iterate_all_partitions(self._space.n))
        if self._num_zsf_indexes is None:
            return zsf_indexes
        else:
            zsf_indexes.sort(
                lambda zsf_index: self.get_normalized_impact(params, zsf_index))
            return zsf_indexes[:self._num_zsf_indexes]
        

    def _compute(self, params: dict, permutation: Permutation) -> float:
        n = self._space.n
        group_size = self._space.group_size

        kernel_value = 0.0

        for zsf_index in self._get_zsf_indexes(params):
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