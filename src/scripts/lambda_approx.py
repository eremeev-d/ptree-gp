import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from math import exp

from ptree_gp.sn_characters import SnCharactersTable
from ptree_gp.kernels import (
    PTreeHeatPrecomputedKernel, PTreeMaternPrecomputedKernel,
    PTreeHeatKernel, PTreeMaternKernel)
from ptree_gp.spherical_function import (
    NaiveZSF, ZonalPolynomialZSF, ApproximationZSF)
from ptree_gp.primitives import (
    Permutation, Partition, Matching, matching_distance)
from ptree_gp.utils import (
    double_partition, iterate_all_partitions)


# TODO: dont copy-paste from kernels.py
def phi_heat(**kwargs):
    return exp(-0.5 * (kwargs["phi_kappa"]**2) * kwargs["eigenvalue"])


# TODO: dont copy-paste from kernels.py
# NOTE: returns |G| * |H| * impact(rho)
def compute_normalized_impact(characters, rho: Partition, phi_kappa: float):
    n = rho.size
    two_rho = double_partition(rho)

    identity_partition = Partition(*[1 for _ in range(2 * n)])
    kappa = [2] + [1 for _ in range(2 * n - 2)]
    kappa = Partition(*kappa)

    dim_rho = characters.get_value(
        character=two_rho, rho=identity_partition)

    chi_value = characters.get_value(
        character=two_rho, rho=kappa)

    eigenvalue = (dim_rho - chi_value) / dim_rho
    phi_eigenvalue = phi_heat(eigenvalue=eigenvalue, phi_kappa=phi_kappa)
    return (phi_eigenvalue ** 2) * dim_rho


def plot_impacts(impacts, n, filepath):
    impacts.append(0)
    impacts = np.sort(impacts)
    impacts_cumsum = np.cumsum(impacts)
    fig = plt.figure(figsize=(16, 8))
    plt.grid()
    plt.plot(impacts_cumsum[::-1] / impacts_cumsum[-1])
    plt.title(f"n = {n}")
    plt.savefig(filepath)
    plt.show()


def main():
    ### Params
    parser = argparse.ArgumentParser()
    parser.add_argument("n_list", metavar="n", type=int, nargs="+")
    parser.add_argument("--phi_kappa", type=float, default=1.0)
    parser.add_argument("--base_path", type=str, required=True)
    args = parser.parse_args()

    n_list = args.n_list
    phi_kappa = args.phi_kappa
    base_path = args.base_path
    
    characters = SnCharactersTable()
    for n in n_list:
        impacts = []

        for rho in iterate_all_partitions(n):
            impacts.append(compute_normalized_impact(
                characters=characters, rho=rho, phi_kappa=phi_kappa))
    
        plot_impacts(impacts, n, os.path.join(base_path, f"lambda_{n}.png"))


if __name__ == "__main__":
    main()