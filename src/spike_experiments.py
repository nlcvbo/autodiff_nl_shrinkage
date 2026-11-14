"""Spike experiments: outliers created by population spikes.

Location of the largest sample eigenvalues and shrinkage of the outliers.

Run ``uv run spike_experiments.py [scale] [n_jobs]``; the summaries are written
to ``spike_results.json``. ``scale`` multiplies the numbers of Monte-Carlo runs
(``1`` uses the numbers of the article, e.g. ``0.05`` for a quick check) and
``n_jobs`` is the number of worker processes.
"""

import json
import os
import sys
import time
import warnings

import numpy as np

from heavy_utils import T_ATOMS, WT_ATOMS, monte_carlo, population, right_edge, scaled_runs, summarize, weight_law
from spike_utils import outlier_location


def experiment_spike_location(
    taus=(10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0),
    alphas=(0.0, 2.0),
    sizes=((400, 800, 100), (1600, 3200, 24)),
    scale=1.0,
    n_jobs=1,
):
    """Location of the outlier created by one population spike (Figure "spikes" (a)).

    One eigenvalue 10 of ``Sigma`` is replaced by ``tau_s``. The largest sample
    eigenvalue is compared with the prediction of
    :func:`outlier_location` above the threshold ``tau_c`` (where the
    prediction reaches the right edge of the support), and with the edge
    below it, with exponential (``alpha = 2``) and equal (``alpha = 0``) weights.
    """
    print("\n=== Spike location ===")
    results = []
    for alpha in alphas:
        _, weights, w_args, d, wd = weight_law(1000, alpha)
        edge = right_edge(T_ATOMS, WT_ATOMS, d, wd, 0.5, weights, w_args)
        grid = np.geomspace(10.5, 60, 300)
        curve = np.array([outlier_location(ts, alpha, 0.5) for ts in grid])
        tau_c = float(grid[np.argmin(curve)])
        print(f"alpha={alpha:g}: right edge {edge:.3f}, threshold tau_c {tau_c:.2f}", flush=True)
        for p, n, n_runs in sizes:
            for tau_s in taus:
                tau_pop = population(p)
                tau_pop[-1] = tau_s
                predicted = outlier_location(tau_s, alpha, p / n) if tau_s > tau_c else edge
                cfg = {"p": p, "n": n, "alpha": alpha, "kind": "gauss", "tau_pop": tau_pop, "location_only": True}
                runs_ = monte_carlo(cfg, scaled_runs(n_runs, scale), 40_000 + p + round(10 * tau_s), n_jobs)
                results.append(
                    summarize(
                        f"alpha={alpha:g}, p={p}, tau_s={tau_s:g}, predicted {predicted:.3f}",
                        runs_,
                        {"alpha": alpha, "p": p, "n": n, "tau_s": tau_s, "predicted": predicted, "edge": edge, "tau_c": tau_c},
                    )
                )
    return results


def experiment_spike_estimation(
    rs=(3, 10),
    taus=(12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0),
    p=400,
    n=800,
    runs=30,
    n_epochs=300,
    scale=1.0,
    n_jobs=1,
):
    """Shrinkage of the outliers created by ``r`` spikes (Figure "spikes" (b, c)).

    ``r`` eigenvalues 10 of ``Sigma`` are replaced by ``tau_s``; ``H`` is
    unchanged. Estimators: formulas with the limit ``H`` on every eigenvalue
    (``"h"``), outliers left unshrunk (``"keep"``) or replaced by ``-1/X``
    (``"inv"``), formulas with the finite-p spectrum including the spikes
    (``"Hn"``), and the bona fide estimator (``"ret"``). Without spike
    (``r = 0``) as a reference.
    """
    print("\n=== Spike estimation ===")
    _, weights, w_args, d, wd = weight_law(n, 2.0)
    edge = right_edge(T_ATOMS, WT_ATOMS, d, wd, p / n, weights, w_args)
    results = []
    for r, tau_s in [(0, 0.0)] + [(r, ts) for r in rs for ts in taus]:
        tau_pop = population(p)
        if r:
            tau_pop[-r:] = tau_s
        cfg = {
            "p": p,
            "n": n,
            "alpha": 2.0,
            "kind": "gauss",
            "tau_pop": tau_pop,
            "edge": edge,
            "estimators": ["h", "keep", "inv", "Hn", "ret"],
            "n_epochs": n_epochs,
            "n_top": max(r, 1),
        }
        runs_ = monte_carlo(cfg, scaled_runs(runs, scale), 50_000 + 1000 * r + round(tau_s), n_jobs)
        results.append(summarize(f"r={r}, tau_s={tau_s:g}", runs_, {"r": r, "tau_s": tau_s, "p": p, "n": n}))
    return results


def spike_experiments(scale=1.0, n_jobs=1, which=None, save=None):
    """Run the spike experiments and return their summaries.

    Parameters
    ----------
    scale : float, default=1.0
        Multiplies the numbers of Monte-Carlo runs.
    n_jobs : int, default=1
        Number of worker processes for the Monte-Carlo runs.
    which : iterable of str, optional
        Subset of ``"spike_location"``, ``"spike_estimation"`` (all by default).
    save : str, optional
        Path of a JSON file where the summaries are written.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # one BLAS thread per worker process
    common = {"scale": scale, "n_jobs": n_jobs}
    experiments = {
        "spike_location": lambda: experiment_spike_location(**common),
        "spike_estimation": lambda: experiment_spike_estimation(**common),
    }
    results = {}
    for name in which or experiments:
        start_time = time.time()
        results[name] = experiments[name]()
        print(f"[{name}] done in {time.time() - start_time:.0f} s", flush=True)
        if save:
            with open(save, "w") as file:
                json.dump(results, file, indent=1, default=float)
    return results


if __name__ == "__main__":
    # uv run spike_experiments.py [scale] [n_jobs]
    warnings.filterwarnings("ignore")
    spike_experiments(
        scale=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0,
        n_jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 1,
        save="spike_results.json",
    )
