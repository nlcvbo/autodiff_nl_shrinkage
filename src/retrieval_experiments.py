"""Stability of the H retrieval to the random seed and to the initialization.

One weighted sample covariance is drawn once: Gaussian noise, Sigma with the
spectrum H of the article (atoms 1, 3, 10 in the proportions 0.2, 0.4, 0.4),
exponentially decaying weights (alpha = 2), p = 200 and n = 400 (c = 1/2).
On this fixed sample, the population spectrum is retrieved by
:meth:`Autodiff.fit_from_S` (Adam, learning rate 5e-2, one simulated spectrum
per step) for

* 10 seeds: the seed drives the simulated noise of the Wasserstein objective
  and the random parts of the initializations, so the spread over the seeds
  measures the algorithm alone, not the sampling noise;
* three initializations: the spectrum of the weighted linear shrinkage
  (default of the other experiments), the sample spectrum, and p draws from
  the uniform law on [0, 1];
* 100 and 1000 epochs.

The covariance is then shrunk with the formulas evaluated at the retrieved
spectrum. For each initialization and number of epochs, the script reports
the mean and standard deviation over the seeds of the excess loss over the
ideal oracle (``u_i^T Sigma u_i``) and of the PRIAL against ``S``.

Run ``uv run retrieval_experiments.py [n_jobs]``; the results are written to
``retrieval_results.json``.
"""

import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from autodiff import Autodiff
from heavy_utils import T_ATOMS, WT_ATOMS, losses, noise, population, shrinkage, weight_law
from utils import initial_tau_guess
from weighted_LWO_estimator import wLWO_estimator

INITS = ("lwo", "sample", "uniform")
INIT_LABELS = {"lwo": "linear shrinkage (default)", "sample": "sample spectrum", "uniform": "uniform on [0, 1]"}


def fixed_sample(p, n, alpha, data_seed):
    """The fixed sample: ``S = X^T diag(W) X / n`` with ``X = Z diag(tau_pop)^(1/2)``, Gaussian ``Z``."""
    tau_pop = population(p)
    W, weights, w_args, d, wd = weight_law(n, alpha)
    Z, _ = noise("gauss", n, p, np.random.default_rng(data_seed))
    X = Z * np.sqrt(tau_pop)[None, :]
    S = X.T @ (W[:, None] * X) / n
    return tau_pop, X, W, S, (weights, w_args, d, wd)


def initial_spectrum(init, X, W, lambda_, p, n):
    """Initial population spectrum of the retrieval (float64 tensor of length ``p``)."""
    if init == "lwo":
        # Spectrum of the weighted linear shrinkage, as in heavy_utils.monte_carlo_run
        # (initial_tau_guess resamples it with numpy's global generator)
        S_lwo = wLWO_estimator(X, W / n, assume_centered=True, S_r=None, est=False)
        return initial_tau_guess(np.linalg.eigvalsh(S_lwo), p)
    if init == "sample":
        return torch.tensor(np.sort(lambda_), dtype=torch.float64)
    if init == "uniform":
        return torch.sort(torch.rand(p, dtype=torch.float64)).values
    raise ValueError(f"Unknown initialization: {init!r}")


def stability_run(task):
    """One retrieval on the fixed sample, then the losses of the shrinkage with the retrieved spectrum."""
    warnings.filterwarnings("ignore")
    cfg, init, n_epochs, seed = task
    torch.set_num_threads(1)
    torch.manual_seed(seed)  # simulated noise of the objective, uniform initialization
    np.random.seed(seed)  # resampling in initial_tau_guess
    p, n = cfg["p"], cfg["n"]
    tau_pop, X, W, S, (weights, w_args, d, wd) = fixed_sample(p, n, cfg["alpha"], cfg["data_seed"])
    lambda_, U = np.linalg.eigh(S)
    tau_init = initial_spectrum(init, X, W, lambda_, p, n)
    t0 = time.time()
    estimator = Autodiff(bias=True, assume_centered=True).fit_from_S(
        S, W, tau_init=tau_init, method="Adam", n_epochs=n_epochs, b=1, lr=5e-2, verbose=False
    )
    tau_hat = estimator.get_tau().detach()
    seconds = time.time() - t0
    h = shrinkage(lambda_, tau_hat, torch.ones(p) / p, d, wd, p / n, weights, w_args)
    res, _ = losses(tau_pop, lambda_, U, {"ret": h})
    return {"init": init, "n_epochs": n_epochs, "seed": seed, **res, "tau_hat_max": float(tau_hat.max()), "seconds": seconds}


def retrieval_experiments(p=200, n=400, alpha=2.0, seeds=tuple(range(10)), inits=INITS, epochs=(100, 1000), data_seed=0,
                          n_jobs=1, save="retrieval_results.json"):
    """Seed and initialization stability of the retrieval (Table "retrieval stability" of the article).

    Returns the per-run results and, for each (initialization, epochs), the
    mean and standard deviation over the seeds of the excess loss and of the
    PRIAL, with the references of the fixed sample (ideal oracle and formulas
    with the true ``H``).
    """
    warnings.filterwarnings("ignore")
    cfg = {"p": p, "n": n, "alpha": alpha, "data_seed": data_seed}
    print(f"\n=== Stability of the H retrieval, p={p}, n={n}, alpha={alpha:g}, {len(seeds)} seeds ===")

    # References on the fixed sample: ideal oracle and formulas with the true H
    tau_pop, _, _, S, (weights, w_args, d, wd) = fixed_sample(p, n, alpha, data_seed)
    lambda_, U = np.linalg.eigh(S)
    ref, _ = losses(tau_pop, lambda_, U, {"h": shrinkage(lambda_, T_ATOMS, WT_ATOMS, d, wd, p / n, weights, w_args)})
    reference = {"prial_ideal": 1 - ref["L_ideal"] / ref["L_S"], "excess_h": ref["L_h"] / ref["L_ideal"] - 1,
                 "prial_h": 1 - ref["L_h"] / ref["L_S"]}
    print(f"reference: PRIAL ideal {100 * reference['prial_ideal']:.2f}%, formulas with the true H: "
          f"excess {100 * reference['excess_h']:.2f}%, PRIAL {100 * reference['prial_h']:.2f}%", flush=True)

    tasks = [(cfg, init, m, s) for init in inits for m in epochs for s in seeds]
    if n_jobs <= 1:
        runs = [stability_run(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            runs = list(pool.map(stability_run, tasks))

    summaries = []
    for init in inits:
        for m in epochs:
            rs = [r for r in runs if r["init"] == init and r["n_epochs"] == m]
            excess = np.array([r["L_ret"] / r["L_ideal"] - 1 for r in rs])
            prial = np.array([1 - r["L_ret"] / r["L_S"] for r in rs])
            s = {"init": init, "n_epochs": m, "seeds": len(rs),
                 "excess_mean": float(excess.mean()), "excess_std": float(excess.std(ddof=1)),
                 "prial_mean": float(prial.mean()), "prial_std": float(prial.std(ddof=1)),
                 "tau_hat_max_mean": float(np.mean([r["tau_hat_max"] for r in rs])),
                 "seconds_mean": float(np.mean([r["seconds"] for r in rs]))}
            summaries.append(s)
            print(f"{INIT_LABELS[init]:28s} {m:5d} epochs | excess {100 * s['excess_mean']:.2f} +- {100 * s['excess_std']:.2f}% | "
                  f"PRIAL {100 * s['prial_mean']:.2f} +- {100 * s['prial_std']:.2f}% | max tau_hat {s['tau_hat_max_mean']:.1f} | "
                  f"{s['seconds_mean']:.1f} s", flush=True)

    result = {"config": cfg, "seeds": list(seeds), "reference": reference, "summaries": summaries, "runs": runs}
    if save:
        with open(save, "w") as f:
            json.dump(result, f, indent=1)
    return result


if __name__ == "__main__":
    retrieval_experiments(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
