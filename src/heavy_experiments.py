"""Heavy-tail experiments: robustness of the shrinkage formulas.

Rank deficiency and ``c`` close to 1, singular population covariance, heavy
tails (Student, Pareto and elliptical noise) with the true and the retrieved
spectrum.

Run ``uv run heavy_experiments.py [scale] [n_jobs]``; the summaries are written
to ``heavy_results.json``. ``scale`` multiplies the numbers of Monte-Carlo runs
(``1`` uses the numbers of the article, e.g. ``0.05`` for a quick check) and
``n_jobs`` is the number of worker processes. With the default sizes the full
set takes hours.
"""

import json
import os
import sys
import time
import warnings

import torch

from heavy_utils import NOISES, T_ATOMS, WT_ATOMS, limit_cdfs, monte_carlo, population, right_edge, scaled_runs, summarize, weight_law


def experiment_concentration(
    cs=(0.25, 0.5, 0.8, 0.95, 1.05, 1.25, 2.0, 4.0),
    n=400,
    sweeps=((0.95, (100, 200, 400, 800, 1600)), (2.0, (50, 100, 200, 400, 800))),
    runs=100,
    scale=1.0,
    n_jobs=1,
):
    """Rank deficiency and ``c`` close to 1 (Gaussian noise, true ``H``).

    The theorems hold for ``c != 1``, including ``c > 1``, where ``S`` has a
    null space (atom of F at 0). For each ``c`` (``n`` samples, ``n / 2`` for
    ``c = 4``), the PRIAL of the formulas is compared with that of the ideal
    oracle and, for ``c > 1``, the predicted ``h(0)`` with the observed mean of
    ``u_i^T Sigma u_i`` on the null space (``h(0)`` does not depend on the
    weights). Then the gap to the ideal oracle is followed as ``n`` grows, at
    ``c = 0.95`` and ``c = 2``. Configurations with ``p > 800`` use a quarter of
    the runs.
    """
    print("\n=== Concentration ratio c, rank-deficient S (c > 1) ===")
    configs = [(c, n if c < 4 else n // 2) for c in cs] + [(c, m) for c, ms in sweeps for m in ms]
    results = []
    for c, m in configs:
        p = round(c * m)
        cfg = {"p": p, "n": m, "alpha": 2.0, "kind": "gauss", "tau_pop": population(p), "estimators": ["h"]}
        n_runs = scaled_runs(runs if p <= 800 else runs // 4, scale)
        results.append(summarize(f"c={c:g}, p={p}, n={m}", monte_carlo(cfg, n_runs, 1000 + m, n_jobs), {"c": c, "p": p, "n": m}))
    return results


def experiment_singular(p=400, n=800, zero_fraction=0.2, runs=100, scale=1.0, n_jobs=1):
    """Singular population covariance: a fraction ``zero_fraction`` of zero eigenvalues.

    ``H`` gets an atom at 0. The non-zero spectrum of ``S`` is that of the
    restricted model, and the formulas return 0 on the null space of ``S``.
    """
    print("\n=== Singular population covariance ===")
    t = torch.tensor([0.0, 1.0, 3.0, 10.0])
    wt = torch.tensor([zero_fraction, *((1 - zero_fraction) * WT_ATOMS.numpy()).tolist()])
    cfg = {"p": p, "n": n, "alpha": 2.0, "kind": "gauss", "tau_pop": population(p, t, wt), "t": t, "wt": wt, "estimators": ["h"]}
    runs_ = monte_carlo(cfg, scaled_runs(runs, scale), 4000, n_jobs)
    return [summarize(f"zero fraction {zero_fraction}, p={p}, n={n}", runs_, {"p": p, "n": n, "zero_fraction": zero_fraction})]


def experiment_heavy_tails(
    noises=NOISES,
    n_list=(100, 200, 400, 800, 1600, 3200),
    runs=(1600, 800, 400, 160, 80, 40),
    alpha=2.0,
    c=0.5,
    kolmogorov=True,
    scale=1.0,
    n_jobs=1,
):
    """Heavy tails with the true ``H`` (Figure "heavy" and Table "heavy_set" of the article).

    For each noise (:func:`noise`) and number of samples ``n``
    (``p = c n``):

    * excess loss of the formulas over the ideal oracle (``"h"``), with the
      eigenvalues beyond the right edge replaced by the spike estimate
      ``-1/X`` (``"inv"``), and, for elliptical noise, with the effective
      weights ``W_t s_t^2`` (``"effw"``);
    * Kolmogorov distances of the spectral distribution and of ``Delta_n`` to
      their limits (``dK_F``, ``dK_Delta``);
    * number of eigenvalues beyond the right edge (``n_out``), ratio
      ``lambda_1 / edge`` and participation ratio of the top eigenvector
      (``ipr``: about 6 if delocalized here, ``p`` if localized on one
      coordinate).
    """
    print(f"\n=== Heavy tails, true H, alpha={alpha:g}, c={c:g} ===")
    _, weights, w_args, d, wd = weight_law(1000, alpha)  # the limiting weight law does not depend on n
    edge = right_edge(T_ATOMS, WT_ATOMS, d, wd, c, weights, w_args)
    limits = limit_cdfs(T_ATOMS, WT_ATOMS, d, wd, c, weights, w_args) if kolmogorov else None
    results = []
    for m, n_runs in zip(n_list, runs, strict=True):
        p = round(c * m)
        for kind in noises:
            estimators = ["h", "inv"] + (["effw"] if kind.startswith("ell") else [])
            cfg = {
                "p": p,
                "n": m,
                "alpha": alpha,
                "kind": kind,
                "tau_pop": population(p),
                "edge": edge,
                "limits": limits,
                "estimators": estimators,
            }
            runs_ = monte_carlo(cfg, scaled_runs(n_runs, scale), 10_000 + m, n_jobs)
            results.append(summarize(f"{kind}, p={p}, n={m}", runs_, {"kind": kind, "p": p, "n": m, "alpha": alpha, "c": c, "edge": edge}))
    return results


def experiment_bona_fide(
    noises=("gauss", "t5", "t4", "t3", "t2.5", "pareto3", "ell5"),
    p=400,
    n=800,
    runs=60,
    n_epochs=300,
    first_round_cases=True,
    scale=1.0,
    n_jobs=1,
):
    """Bona fide estimator: ``H`` retrieved by :class:`Autodiff`, then the formulas (Table "heavy").

    For each noise: excess loss with the true ``H`` (``"h"``), with the
    outliers replaced by ``-1/X`` (``"inv"``), with the retrieved spectrum
    (``"ret"``), and the largest retrieved atom (``tau_hat_max``, against
    ``max tau = 10``). Under heavy tails, the retrieval reads the localized
    outliers as population spikes. For elliptical noise, also with the
    effective weights ``W_t s_t^2``: formulas with the true ``H`` (``"effw"``)
    and bona fide estimator (``"effw_ret"``, largest atom
    ``tau_hat_eff_max``). ``first_round_cases`` adds Gaussian noise
    with ``c = 2`` (``n = p / 2``) and three spikes ``tau_s = 50``.
    """
    print("\n=== Bona fide estimator (retrieved H) ===")
    _, weights, w_args, d, wd = weight_law(n, 2.0)
    results = []
    cases = [(kind, p, n, population(p)) for kind in noises]
    if first_round_cases:
        spiked = population(p)
        spiked[-3:] = 50.0
        cases += [("gauss", p, p // 2, population(p)), ("gauss", p, n, spiked)]
    for kind, p_, n_, tau_pop in cases:
        edge = right_edge(T_ATOMS, WT_ATOMS, d, wd, p_ / n_, weights, w_args)
        estimators = ["h", "inv", "ret"] + (["effw", "effw_ret"] if kind.startswith("ell") else [])
        cfg = {
            "p": p_,
            "n": n_,
            "alpha": 2.0,
            "kind": kind,
            "tau_pop": tau_pop,
            "edge": edge,
            "estimators": estimators,
            "n_epochs": n_epochs,
        }
        label = f"{kind}, p={p_}, n={n_}" + (", 3 spikes 50" if tau_pop.max() > 10 else "")
        results.append(summarize(label, monte_carlo(cfg, scaled_runs(runs, scale), 30_000, n_jobs), {"kind": kind, "p": p_, "n": n_}))
    return results


def heavy_experiments(scale=1.0, n_jobs=1, which=None, save=None):
    """Run the heavy experiments and return their summaries.

    Parameters
    ----------
    scale : float, default=1.0
        Multiplies the numbers of Monte-Carlo runs.
    n_jobs : int, default=1
        Number of worker processes for the Monte-Carlo runs.
    which : iterable of str, optional
        Subset of ``"concentration"``, ``"singular"``, ``"heavy_tails"``, ``"heavy_settings"``, ``"bona_fide"`` (all by default).
    save : str, optional
        Path of a JSON file where the summaries are written.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # one BLAS thread per worker process
    common = {"scale": scale, "n_jobs": n_jobs}
    settings_noises = ("gauss", "t5", "t4", "t3", "t2.5")
    experiments = {
        "concentration": lambda: experiment_concentration(**common),
        "singular": lambda: experiment_singular(**common),
        "heavy_tails": lambda: experiment_heavy_tails(**common),
        # other weights (alpha = 0, 5) and ratios (c = 0.25, 2)
        "heavy_settings": lambda: (
            experiment_heavy_tails(settings_noises, (1600,), (80,), alpha=0.0, **common)
            + experiment_heavy_tails(settings_noises, (1600,), (80,), alpha=5.0, **common)
            + experiment_heavy_tails(settings_noises, (1600,), (80,), c=0.25, **common)
            + experiment_heavy_tails(settings_noises, (800,), (80,), c=2.0, **common)
        ),
        "bona_fide": lambda: experiment_bona_fide(**common),
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
    # uv run heavy_experiments.py [scale] [n_jobs]
    warnings.filterwarnings("ignore")
    heavy_experiments(
        scale=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0,
        n_jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 1,
        save="heavy_results.json",
    )
