import argparse
import time
import json
import os

import numpy as np
import pandas as pd

from ptree_gp.kernels import (
    PTreeHeatPrecomputedKernel, PTreeMaternPrecomputedKernel,
    PTreeHeatKernel, PTreeMaternKernel)
from ptree_gp.spherical_function import (
    NaiveZSF, ZonalPolynomialZSF, ApproximationZSF)
from ptree_gp.primitives import (
    Permutation, Matching, matching_distance)


def get_random_permutation(n):  # TODO: move this to utils
    perm = np.arange(1, n+1)
    return Permutation(*np.random.permutation(perm))


def get_x0(n):  # TODO: move this to utils
    return Matching(*tuple((2*h-1, 2*h) for h in range(1, n+1)))


def compute_stats(results):
    return {
        "size": len(results),
        "mean": np.mean(results),
        "std": np.std(results),
        "median": np.median(results),
        "25%": np.quantile(results, 0.25),
        "75%": np.quantile(results, 0.75),
    }


def bench_precompute_time(
    kernel_class,
    kernel_kwargs,
    zsf_class,
    zsf_kwargs,
    n,
    num_runs,
):
    results = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        zsf = zsf_class().precompute(n=n, **zsf_kwargs)
        kernel = kernel_class(n=n, zsf=zsf, **kernel_kwargs).precompute()
        end_time = time.perf_counter()
        results.append(end_time - start_time)
    return compute_stats(results)


# TODO: add warmup
def bench_percall_time(
    kernel_class,
    kernel_kwargs,
    zsf_class,
    zsf_kwargs,
    n,
    num_calls_per_run,
    num_runs,
):
    results = []

    for _ in range(num_runs):
        zsf = zsf_class().precompute(n=n, **zsf_kwargs)
        kernel = kernel_class(n=n, zsf=zsf, **kernel_kwargs).precompute()

        x0 = get_x0(n)

        x_array = [get_random_permutation(2*n) for _ in range(num_calls_per_run)]
        y_array = [get_random_permutation(2*n) for _ in range(num_calls_per_run)]

        start_time = time.perf_counter()
        for x, y in zip(x_array, y_array):
            value = kernel(y.inverse() * x)
        end_time = time.perf_counter()
        results.append((end_time - start_time) / num_calls_per_run)

    return compute_stats(results)


def run_time_benchmarks(config):
    results = []

    for benchmark in config["benchmarks"]:

        method = benchmark["method"]
        n = benchmark["n"]
        num_runs = benchmark["num_runs"]
        num_calls_per_run = benchmark["num_calls_per_run"]
        if "zsf_kwargs" not in benchmark:
            benchmark["zsf_kwargs"] = {}
        zsf_kwargs = benchmark["zsf_kwargs"]

        print(f"Processing {method}, n={n}")

        zsf_class = {
            "Naive": NaiveZSF,
            "ZonalPolynomial": ZonalPolynomialZSF,
            "MonteCarloApproximation": ApproximationZSF
        }[method]

        if zsf_class == ApproximationZSF:
            kernel_class = PTreeHeatKernel
        else:
            kernel_class = PTreeHeatPrecomputedKernel

        precompute_results = bench_precompute_time(
            kernel_class=kernel_class,
            kernel_kwargs={},
            zsf_class=zsf_class,
            zsf_kwargs=zsf_kwargs,
            n=n,
            num_runs=num_runs
        )

        for key, value in precompute_results.items():
            benchmark["Precompute " + key] = value

        query_results = bench_percall_time(
            kernel_class=kernel_class,
            kernel_kwargs={},
            zsf_class=zsf_class,
            zsf_kwargs=zsf_kwargs,
            n=n,
            num_runs=num_runs,
            num_calls_per_run=num_calls_per_run
        )

        for key, value in query_results.items():
            benchmark["Query " + key] = value

        results.append(benchmark)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    benchmark_results = run_time_benchmarks(config)

    with open(args.save_filepath, "w") as f:
        json.dump(benchmark_results, f)


if __name__ == "__main__":
    main()