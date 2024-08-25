from ptree_gp.primitives import Permutation, matching_distance
from ptree_gp.kernels import PTreeMaternKernel
from ptree_gp.spaces import MatchingSpace
from ptree_gp.spherical_function import ZonalPolynomialZSF
from ptree_gp.zonal_polynomials import ZpCoeffComputer
from ptree_gp.utils import iterate_all_permutations, iterate_all_partitions


def test_kernel(n = 3):
    space = MatchingSpace(n)
    x0 = space.x0
    zsf = ZonalPolynomialZSF(space)

    kernel = PTreeMaternKernel(space, zsf)
    params = kernel.init_params()
    params["nu"] = 2.0

    results = dict()

    for sigma in iterate_all_permutations(2 * n):
        value = kernel(params, sigma)
        key = matching_distance(sigma * x0, x0)
        if key not in results:
            results[key] = value
        else:
            assert results[key] == value

    print(results)


def test_zp(n = 4):
    zpcc = ZpCoeffComputer(n)
    partitions = list(iterate_all_partitions(n))

    print(partitions)

    for zp_index in partitions:
        coefs = zpcc(zp_index)
        for zp_coef in partitions:
            print(f"{coefs[zp_coef]:.4f}", end=" ")
        print()


# TODO: remove this file
if __name__ == "__main__":
    test_zp()
    # test_kernel()