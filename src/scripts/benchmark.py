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
    n_runs,
):
    results = []
    for _ in range(n_runs):
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
    n_calls_per_run,
    n_runs,
):
    results = []

    for _ in range(n_runs):
        zsf = zsf_class().precompute(n=n, **zsf_kwargs)
        kernel = kernel_class(n=n, zsf=zsf, **kernel_kwargs).precompute()

        x0 = get_x0(n)

        x_array = [get_random_permutation(2*n) for _ in range(n_calls_per_run)]
        y_array = [get_random_permutation(2*n) for _ in range(n_calls_per_run)]

        start_time = time.perf_counter()
        for x, y in zip(x_array, y_array):
            value = kernel(y.inverse() * x)
        end_time = time.perf_counter()
        results.append((end_time - start_time) / n_calls_per_run)

    return compute_stats(results)


def bench_accuracy(
    kernel_class,
    kernel_kwargs,
    zsf_approx_class,
    zsf_approx_kwargs,
    zsf_exact_class,
    zsf_exact_kwargs,
    n,
    n_matchings,
    n_runs,
):
    results = []
    for _ in range(n_runs):
        zsf_approx = zsf_approx_class().precompute(n=n, **zsf_approx_kwargs)
        kernel_approx = kernel_class(n=n, zsf=zsf_approx, **kernel_kwargs).precompute()

        zsf_exact = zsf_exact_class().precompute(n=n, **zsf_exact_kwargs)
        kernel_exact = kernel_class(n=n, zsf=zsf_exact, **kernel_kwargs).precompute()

        x0 = get_x0(n)
        g_array = [get_random_permutation(2*n) for _ in range(n_matchings)]

        k_approx = np.zeros((n_matchings, n_matchings))
        k_exact = np.zeros((n_matchings, n_matchings))

        for i, g in enumerate(g_array):
            for j, h in enumerate(g_array):
                permutation = h.inverse() * g
                k_approx[i, j] = kernel_approx(permutation)
                k_exact[i, j] = kernel_exact(permutation)

        results.append(np.linalg.norm(k_approx - k_exact, ord="fro"))
    return compute_stats(results)


def format_num(num):
    num = f"{num:.2E}"
    a, b = num.split("E")
    return a + " \cdot 10^{" + str(int(b)) + "}"


def format_stats(stats):
    mean = stats["mean"]
    std = stats["std"] / np.sqrt(stats["size"])
    return f"${format_num(mean)}$"
    # return f"${format_num(mean)} \\pm {format_num(3 * std)}$"


# TODO: rewrite
def run_time_benchmark(
    save_path,
    methods_list = [
        "Наивный метод", 
        "Метод на основе ЗП", 
        "Аппроксимация, L=10",
        # "Аппроксимация, L=100"
    ],
    n_list = [5, 15],  # TODO: at least [5, 15]
    n_runs_list = [20, 20],  # TODO: at least 20
    n_calls_per_run_list = [100, 100],  # 1000
):
    results = dict()
    for method in methods_list:
        results[method] = dict()
        for n, n_runs, n_calls_per_run in zip(
                n_list, n_runs_list, n_calls_per_run_list
        ):
            print(f"Processing {method}, n={n}")

            zsf_class = {
                "Наивный метод": NaiveZSF,
                "Метод на основе ЗП": ZonalPolynomialZSF,
                "Аппроксимация, L=10": ApproximationZSF,
                "Аппроксимация, L=100": ApproximationZSF
            }[method]

            if (zsf_class == NaiveZSF) and (n > 5):
                continue

            if zsf_class == ApproximationZSF:
                kernel_class = PTreeHeatKernel
            else:
                kernel_class = PTreeHeatPrecomputedKernel

            zsf_kwargs = {
                "Наивный метод": {},
                "Метод на основе ЗП": {},
                "Аппроксимация, L=10": {"n_stab_samples": 10},
                "Аппроксимация, L=100": {"n_stab_samples": 100},
            }[method]

            results[method][f"Преодобработка, n={n}"] = format_stats(
                bench_precompute_time(
                    kernel_class=kernel_class,
                    kernel_kwargs={},
                    zsf_class=zsf_class,
                    zsf_kwargs=zsf_kwargs,
                    n=n,
                    n_runs=n_runs
                )
            )

            results[method][f"Запрос, n={n}"] = format_stats(
                bench_percall_time(
                    kernel_class=kernel_class,
                    kernel_kwargs={},
                    zsf_class=zsf_class,
                    zsf_kwargs=zsf_kwargs,
                    n=n,
                    n_runs=n_runs,
                    n_calls_per_run=n_calls_per_run
                )
            )

    with open(os.path.join(save_path, "bench-compute-time.json"), "w") as f:
        json.dump(results, f)

    table = pd.DataFrame.from_dict(results, orient="index")
    columns = [
        "Преодобработка, n=5",
        "Преодобработка, n=15",
        "Запрос, n=5",
        "Запрос, n=15",
    ]
    try:
        table = table[columns]
    except:
        for _ in range(10):
            print("ATTENTION !!! ERROR")
        raise

    table = table.to_latex()
    table = table.replace("\\toprule", "\\hline")
    table = table.replace("\\midrule", "\\hline")
    table = table.replace("\\bottomrule", "\\hline")
    table = table.replace("NaN", "---")

    with open(os.path.join(save_path, "bench-compute-time.txt"), "w") as f:
        f.write(table)


# TODO: add argparse?
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    run_time_benchmark(save_path=args.save_path)


if __name__ == "__main__":
    main()