from __future__ import annotations
from typing import List, Set, Tuple, Optional
from math import factorial, exp
from abc import ABC, abstractmethod

from ptree_gp.primitives import (
    Partition, Permutation, Matching, matching_distance)
from ptree_gp.spherical_function import (
    ZonalSphericalFunctionBase, ZonalPolynomialZSF)
from ptree_gp.sn_characters import (
    SnCharactersTable)
from ptree_gp.utils import (
    double_partition, iterate_all_partitions)


# NOTE (!!!): Currently it is assumed that x0 = {{1, 2}, {3, 4}, ..., {2n-1, 2n}}
# TODO: refactor by moving all sum_rho machinery to separate method that takes zsf_values method
# TODO: refactor by PTreeKernelBase -> PTreeDistanceKernelBase -> PTreePrecomputedKernelBase
# TODO: add description of methods
# TODO: add support for kernels without precompute?
#       by moving this class to smth like PTreeKernelPrecomputed
class PTreeKernelBase(ABC):
    def __init__(self, n: int, zsf: ZonalSphericalFunctionBase) -> None:
        self._n = n
        self._zsf = zsf
        self._characters = SnCharactersTable()

    # NOTE: this is eigenvalue of *normalized* laplace operator
    def _compute_eigenvalue(self, two_rho: Partition) -> float:
        n = self._n

        identity_partition = Partition(*[1 for _ in range(2 * n)])
        kappa = [2] + [1 for _ in range(2 * self._n - 2)]
        kappa = Partition(*kappa)

        dim_rho = self._characters.get_value(
            character=two_rho, rho=identity_partition)

        chi_value = self._characters.get_value(
            character=two_rho, rho=kappa)

        eigenvalue = (dim_rho - chi_value) / dim_rho
        return eigenvalue
        
    def precompute(self) -> PTreeKernelBase:
        for rho in iterate_all_partitions(self._n):
            two_rho = double_partition(rho)
            self._compute_eigenvalue(two_rho)
        return self

    @abstractmethod
    def _phi(self, eigenvalue: float) -> float:
        ...

    def _compute_unnormalized(self, permutation: Permutation) -> float:
        n = self._n

        identity_partition = Partition(*[1 for _ in range(2 * n)])
        group_size = factorial(2 * n)

        kernel_value = 0.0

        for rho in iterate_all_partitions(n):
            two_rho = double_partition(rho)
            eigenvalue = self._compute_eigenvalue(two_rho)
            phi_eigenvalue = self._phi(eigenvalue)
            
            dim_rho = self._characters.get_value(
                character=two_rho, rho=identity_partition)

            kernel_value += phi_eigenvalue * (dim_rho / group_size) \
                * self._zsf(zsf_index=rho, permutation=permutation)

        return kernel_value

    def __call__(self, permutation: Permutation) -> float:
        n = self._n
        identity_permutation = Permutation(*range(1, 2*n+1))
        return self._compute_unnormalized(permutation) \
            / self._compute_unnormalized(identity_permutation)


# TODO: refactor so it doesnt copypastes
class PTreePrecomputedKernelBase(PTreeKernelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = dict()

    def precompute(self) -> PTreeKernelBase:
        for distance in iterate_all_partitions(self._n):
            self._compute_unnormalized(distance)
        return self

    def _compute_unnormalized(self, distance: Partition) -> float:
        if distance in self._cache:
            return self._cache[distance]
        n = self._n

        identity_partition = Partition(*[1 for _ in range(2 * n)])
        group_size = factorial(2 * n)

        kernel_value = 0.0

        for rho in iterate_all_partitions(n):
            two_rho = double_partition(rho)
            eigenvalue = self._compute_eigenvalue(two_rho)
            phi_eigenvalue = self._phi(eigenvalue)
            
            dim_rho = self._characters.get_value(
                character=two_rho, rho=identity_partition)

            kernel_value += phi_eigenvalue * (dim_rho / group_size) \
                * self._zsf.compute_at_dist(
                    zsf_index=rho, distance=distance)

        self._cache[distance] = kernel_value
        return kernel_value

    def __call__(self, permutation: Permutation) -> float:
        n = self._n
        x0 = Matching(*tuple((2*h-1, 2*h) for h in range(1, n+1)))
        distance = matching_distance(permutation * x0, x0)
        identity_distance = Partition(*[1 for _ in range(n)])
        return self._compute_unnormalized(distance) \
            / self._compute_unnormalized(identity_distance)
        


# TODO: refactor so that it is not duplicated
class PTreeHeatPrecomputedKernel(PTreePrecomputedKernelBase):
    def __init__(
            self, 
            n: int, 
            zsf: ZonalSphericalFunctionBase, 
            kappa: float = 1.0
    ) -> None:
        super().__init__(n=n, zsf=zsf)
        self._kappa = kappa

    def _phi(self, eigenvalue: float) -> float:
        return exp(-0.5 * (self._kappa**2) * eigenvalue)


class PTreeMaternPrecomputedKernel(PTreePrecomputedKernelBase):
    def __init__(
            self, 
            n: int, 
            zsf: ZonalSphericalFunctionBase, 
            kappa: float = 1.0,
            nu: float = 1.5,
    ) -> None:
        super().__init__(n=n, zsf=zsf)
        self._kappa = kappa
        self._nu = nu

    def _phi(self, eigenvalue: float) -> float:
        return ((2 * self._nu / (self._kappa**2)) + eigenvalue) ** (-self._nu)


class PTreeHeatKernel(PTreeKernelBase):
    def __init__(
            self, 
            n: int, 
            zsf: ZonalSphericalFunctionBase, 
            kappa: float = 1.0
    ) -> None:
        super().__init__(n=n, zsf=zsf)
        self._kappa = kappa

    def _phi(self, eigenvalue: float) -> float:
        return exp(-0.5 * (self._kappa**2) * eigenvalue)


class PTreeMaternKernel(PTreeKernelBase):
    def __init__(
            self, 
            n: int, 
            zsf: ZonalSphericalFunctionBase, 
            kappa: float = 1.0,
            nu: float = 1.5,
    ) -> None:
        super().__init__(n=n, zsf=zsf)
        self._kappa = kappa
        self._nu = nu

    def _phi(self, eigenvalue: float) -> float:
        return ((2 * self._nu / (self._kappa**2)) + eigenvalue) ** (-self._nu)