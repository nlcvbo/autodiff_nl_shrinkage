"""Shared machinery of the heavy-tail and spike experiments.

Population, weights, noise models, shrinkage with safeguards, losses, Monte-Carlo
runs and summaries. Notation: p is the dimension, n the number of samples and
c = p / n (the article writes n for the dimension and N for the number of
samples). Unless stated otherwise:

* Sigma = diag(tau_pop), where tau_pop has the atoms {1, 3, 10} in the
  proportions (0.2, 0.4, 0.4) (the spectrum H of the article);
* the sample weights decay exponentially with rate alpha = 2;
* an estimator is sum_i h_i u_i u_i^T, where (lambda_i, u_i) are the eigenpairs
  of S = X^T diag(W) X / n. Its Frobenius loss L is compared with the loss L_S
  of S (PRIAL = 1 - E[L] / E[L_S]) and with the loss L_ideal of the "ideal"
  rotation-invariant oracle h_i = u_i^T Sigma u_i, which cannot be computed in
  practice (excess loss = E[L] / E[L_ideal] - 1);
* intervals are 95% paired bootstrap intervals over the Monte-Carlo runs.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from autodiff import Autodiff
from nl_formulas import nl_cov_shrinkage
from nl_utils import asymptotic_density, m_X, support_u
from utils import initial_tau_guess, make_weight_distribution, make_weights
from weighted_LWO_estimator import wLWO_estimator

T_ATOMS = torch.tensor([1.0, 3.0, 10.0])  # atoms of H (float32, as in the other experiments)
WT_ATOMS = torch.tensor([0.2, 0.4, 0.4])  # probabilities of the atoms of H
KT = 30  # number of gaps of H searched by the support identification (w_args[1])
NOISES = ("gauss", "t12", "t5", "t4", "t3", "t2.5", "pareto3", "ell5")


def population(p, t=T_ATOMS, wt=WT_ATOMS):
    """Population eigenvalues: the atoms ``t`` repeated in the proportions ``wt``."""
    counts = np.floor(wt.numpy().astype(float) * p + 1e-6).astype(int)
    counts[-1] += p - counts.sum()
    return np.repeat(t.numpy().astype(float), counts)


def weight_law(n, alpha):
    """Sample weights and the matching weight law for the formulas.

    ``alpha > 0``: exponentially decaying weights, whose limiting law has the
    closed form used by ``weights="ewma"``. ``alpha = 0``: equal weights,
    described as a one-atom discrete law (``weights="Ndiracs"``).

    Returns
    -------
    W, weights, w_args, d, wd
        Weight vector of length ``n`` and the arguments of the formulas.
    """
    if alpha > 0:
        d, wd = make_weight_distribution()
        return make_weights("ewma", n, alpha, d, wd), "ewma", [alpha, KT], d, wd
    one = torch.ones(1, dtype=torch.float64)
    return np.ones(n), "Ndiracs", [-1, KT], one, one


def noise(kind, n, p, rng):
    """Centered noise matrix of shape ``(n, p)`` with unit-variance entries.

    Parameters
    ----------
    kind : str
        ``"gauss"``; ``"t<nu>"``: Student t with ``nu`` degrees of freedom
        (moments finite only below order ``nu``); ``"pareto<a>"``: centered
        Pareto law with tail index ``a`` (skewed); ``"ell<nu>"``: elliptical
        t, i.e. Gaussian samples multiplied by a random scale ``s_t`` common to
        the ``p`` coordinates of each sample, so that the entries are not
        independent.
    n, p : int
        Number of samples and dimension.
    rng : numpy.random.Generator

    Returns
    -------
    Z : ndarray of shape (n, p)
    s2 : ndarray of shape (n,) or None
        Squared scales ``s_t^2`` of the elliptical noise (``E[s_t^2] = 1``).
    """
    if kind == "gauss":
        return rng.standard_normal((n, p)), None
    if kind.startswith("t"):
        nu = float(kind[1:])
        return rng.standard_t(nu, size=(n, p)) * np.sqrt((nu - 2) / nu), None
    if kind.startswith("pareto"):
        a = float(kind[6:])
        P = rng.pareto(a, size=(n, p)) + 1.0
        mean, sd = a / (a - 1), np.sqrt(a / ((a - 1) ** 2 * (a - 2)))
        return (P - mean) / sd, None
    if kind.startswith("ell"):
        nu = float(kind[3:])
        s2 = (nu - 2) / rng.chisquare(nu, size=n)
        return rng.standard_normal((n, p)) * np.sqrt(s2)[:, None], s2
    raise ValueError(f"Unknown noise: {kind!r}")


def shrinkage(lambda_, t, wt, d, wd, c, weights, w_args):
    """Covariance shrinkage ``h(lambda_i)`` of :func:`nl_cov_shrinkage`, with two safeguards.

    * Numerical zeros of ``lambda_`` (null space of ``S`` when ``c > 1`` or
      when ``Sigma`` is singular) are set to 0.
    * :func:`nl_cov_shrinkage` solves the equation of ``X`` eigenvalue after
      eigenvalue with warm starts. When a solve fails, :func:`nl_utils.m_X`
      falls back to ``m_tilde = -i exp(-10)``, i.e. ``X = i exp(-10) / lambda``,
      almost 0, and ``h`` is then close to ``sum(wt t^2) / sum(wt t)``. These
      eigenvalues, recognised by comparison with the exact fallback value, are
      solved again one by one, from a cold start. Without this safeguard, a
      single wrong eigenvalue per run noticeably increases the loss.

    ``lambda_`` must be sorted in increasing order; ``h`` has the same order.
    """
    lambda_ = np.where(lambda_ > 1e-9 * lambda_.max(), lambda_, 0.0)
    h = nl_cov_shrinkage(lambda_, t, wt, d, wd, c=c, weights=weights, w_args=w_args).real.numpy().astype(float)
    # value of h at the fallback X = i exp(-10) / lambda, for each eigenvalue
    t_np, wt_np = t.numpy().astype(float), wt.numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = 1 + t_np[None, :] ** 2 * (np.exp(-10) / lambda_[:, None]) ** 2  # |t X + 1|^2
    h_fallback = (wt_np * t_np**2 / a).sum(axis=1) / (wt_np * t_np / a).sum(axis=1)
    for i in np.flatnonzero((lambda_ > 0) & np.isclose(h, h_fallback, rtol=1e-5, atol=0.0)):
        h[i] = float(nl_cov_shrinkage(lambda_[i : i + 1], t, wt, d, wd, c=c, weights=weights, w_args=w_args).real[-1])
    return h


def compute_X(lambda_, t, wt, d, wd, c, weights, w_args):
    """``X(lambda_i)`` computed by :func:`nl_utils.m_X` (real outside the support of F)."""
    return m_X(torch.as_tensor(np.asarray(lambda_, dtype=float)), weights, w_args, d, wd, c, t, wt)[1].numpy()


def right_edge(t, wt, d, wd, c, weights, w_args):
    """Right edge of the support of the limiting sample spectrum F (:func:`support_u`)."""
    support = support_u(t.numpy(), wt.numpy(), d.numpy(), wd.numpy(), c, weights, w_args, verbose=False)[0]
    return float(support[-1])


def limit_cdfs(t, wt, d, wd, c, weights, w_args, omega=400):
    """Limiting cdfs of the sample spectrum F and of ``Delta_n`` on a grid.

    ``F`` comes from :func:`asymptotic_density`, which gives the same mass to
    each atom it receives, so ``H`` is passed as a list of 100 atoms. ``Delta``
    (Theorem "FEg" of the article with ``g = Id``) has the density
    ``h(lambda) f(lambda)``; it is normalised to mass 1, with an atom
    ``(1 - 1/c) h(0) / sum(wt t)`` at 0 when ``c > 1``.

    Returns
    -------
    dict with keys ``grid``, ``F``, ``Delta`` (ndarrays).
    """
    tau_atoms = torch.tensor(population(100, t, wt), dtype=torch.float32)
    xi, f, F = asymptotic_density(tau_atoms, d, wd, c, 0.1, weights, w_args, omega, "root")
    xi, f, F = xi.detach().numpy().astype(float), f.detach().numpy().astype(float), F.detach().numpy().astype(float)
    # When c > 1, nl_cov_shrinkage only evaluates the ceil(len / c) largest
    # positive points: pad the grid with zeros (which return h(0)) so that
    # every grid point is evaluated.
    n_pad = int(np.ceil(max(c - 1, 0) * xi.shape[0])) + 1
    h = shrinkage(np.concatenate([np.zeros(n_pad), xi]), t, wt, d, wd, c, weights, w_args)
    h0, h = h[0], h[n_pad:]
    density = f * h
    Delta = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(xi))])
    atom0 = (1 - 1 / c) * h0 / float((wt * t).sum()) if c > 1 else 0.0
    Delta = atom0 + (1 - atom0) * Delta / Delta[-1]
    return {"grid": xi, "F": F, "Delta": Delta}


def kolmogorov(points, jumps, grid, cdf):
    """Kolmogorov distance between a step function and a limiting cdf.

    The step function jumps by ``jumps[i]`` at ``points[i]``; the limit is
    linear on ``grid``, equal to ``cdf[0]`` below it (possible atom at 0) and to
    ``cdf[-1]`` above it. Points below ``grid[0]`` are numerical zeros (null
    space of ``S``) and are merged into a single jump at 0.
    """
    order = np.argsort(points)
    x = np.asarray(points)[order]
    right = np.cumsum(np.asarray(jumps)[order])
    left = np.concatenate([[0.0], right[:-1]])
    G = np.interp(x, grid, cdf, left=cdf[0], right=cdf[-1])
    low = x < grid[0]
    keep_right = ~low
    if low.any():
        keep_right[np.flatnonzero(low)[-1]] = True
    dist_left = np.abs(left - G)[~low].max() if (~low).any() else 0.0
    return float(max(np.abs(right - G)[keep_right].max(), dist_left))


def losses(tau_pop, lambda_, U, estimates):
    """Frobenius losses of ``S``, of the ideal oracle and of each estimator.

    Parameters
    ----------
    tau_pop : ndarray of shape (p,)
        Population eigenvalues (``Sigma = diag(tau_pop)``).
    lambda_, U : ndarray
        Eigenpairs of ``S``.
    estimates : dict
        Mapping from name to the shrunk eigenvalues ``h`` (same order as ``lambda_``).

    Returns
    -------
    losses : dict
        ``L_S``, ``L_ideal`` and ``L_<name>`` for each estimator.
    d : ndarray of shape (p,)
        Ideal shrinkage ``u_i^T Sigma u_i``.
    """
    d = (tau_pop[:, None] * U**2).sum(axis=0)
    base = (tau_pop**2).sum()
    losses = {"L_S": float(base - 2 * (lambda_ * d).sum() + (lambda_**2).sum()), "L_ideal": float(base - (d**2).sum())}
    for name, h in estimates.items():
        losses["L_" + name] = float(base - 2 * (h * d).sum() + (h**2).sum())
    return losses, d


def monte_carlo_run(task):
    """One Monte-Carlo run of an experiment.

    Module-level function, so that it can be sent to worker processes.

    Parameters
    ----------
    task : tuple (cfg, seed)
        ``cfg`` is a dict with the keys

        * ``p``, ``n``, ``alpha``, ``kind`` (see :func:`noise`) and
          ``tau_pop`` (population eigenvalues);
        * ``t``, ``wt`` (optional): the limit ``H`` given to the formulas
          (default ``T_ATOMS``, ``WT_ATOMS``);
        * ``location_only`` (optional): only return the two largest sample
          eigenvalues;
        * ``estimators``: subset of

          - ``"h"``: formulas with the limit ``H``,
          - ``"keep"``: same, eigenvalues beyond the right edge left unshrunk,
          - ``"inv"``: same, eigenvalues beyond the right edge replaced by the
            spike estimate ``-1 / Re X(lambda_i)``,
          - ``"Hn"``: formulas with the finite-p spectrum of ``tau_pop``
            (spikes included),
          - ``"ret"``: formulas with the spectrum retrieved by
            :class:`Autodiff` (bona fide estimator),
          - ``"effw"``: elliptical noise only, formulas with the effective
            weights ``W_t s_t^2``;
          - ``"effw_ret"``: elliptical noise only, bona fide estimator with
            the effective weights ``W_t s_t^2``, in the retrieval and in the
            formulas;

        * ``edge``: right edge of the support of F; ``limits`` (optional):
          output of :func:`limit_cdfs`; ``n_top`` (optional): number of
          largest eigenvalues to return; ``n_epochs``: epochs of the retrieval.

    Returns
    -------
    dict
        Losses (``L_*``) and diagnostics of the run.
    """
    warnings.filterwarnings("ignore")
    cfg, seed = task
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    p, n, alpha = cfg["p"], cfg["n"], cfg["alpha"]
    c = p / n
    tau_pop = np.asarray(cfg["tau_pop"], dtype=float)
    t, wt = cfg.get("t", T_ATOMS), cfg.get("wt", WT_ATOMS)
    W, weights, w_args, d, wd = weight_law(n, alpha)

    # Weighted sample covariance S = X^T diag(W) X / n, with X = Z diag(tau_pop)^(1/2)
    Z, s2 = noise(cfg.get("kind", "gauss"), n, p, rng)
    X = Z * np.sqrt(tau_pop)[None, :]
    S = X.T @ (W[:, None] * X) / n
    if cfg.get("location_only"):
        lambda_ = np.linalg.eigvalsh(S)
        return {"lambda_1": float(lambda_[-1]), "lambda_2": float(lambda_[-2])}
    lambda_, U = np.linalg.eigh(S)

    edge = cfg.get("edge", np.inf)
    out_right = lambda_ > edge  # eigenvalues beyond the right edge of the support of F
    estimators = cfg["estimators"]
    h = shrinkage(lambda_, t, wt, d, wd, c, weights, w_args)
    est = {"h": h}
    if "keep" in estimators:
        est["keep"] = np.where(out_right, lambda_, h)
    if "inv" in estimators:
        h_inv = h.copy()
        if out_right.any():
            X_out = compute_X(lambda_[out_right], t, wt, d, wd, c, weights, w_args).real
            # -1 / Xcheck estimates the population spike; keep lambda_i if the solve failed (X = 0)
            h_inv[out_right] = np.where(X_out < 0, -1 / np.minimum(X_out, -1e-12), lambda_[out_right])
        est["inv"] = h_inv
    if "Hn" in estimators:
        atoms, counts = np.unique(tau_pop, return_counts=True)
        t_n = torch.tensor(atoms, dtype=torch.float32)
        wt_n = torch.tensor(counts / p, dtype=torch.float32)
        est["Hn"] = shrinkage(lambda_, t_n, wt_n, d, wd, c, weights, w_args)
    if "ret" in estimators:
        # Initial spectrum from the weighted linear shrinkage, as in Autodiff.fit
        # (positive even when S is singular)
        S_lwo = wLWO_estimator(X, W / n, assume_centered=True, S_r=None, est=False)
        tau_init = initial_tau_guess(np.linalg.eigvalsh(S_lwo), p)
        estimator = Autodiff(bias=True, assume_centered=True).fit_from_S(
            S, W, tau_init=tau_init, method="Adam", n_epochs=cfg.get("n_epochs", 300), b=1, lr=5e-2, verbose=False
        )
        tau_hat = estimator.get_tau().detach()
        est["ret"] = shrinkage(lambda_, tau_hat, torch.ones(p) / p, d, wd, c, weights, w_args)
    if ("effw" in estimators or "effw_ret" in estimators) and s2 is not None:
        # Elliptical noise with known scales s_t is Gaussian noise with the weights W_t s_t^2
        d_eff = torch.tensor(np.quantile(W * s2, (np.arange(200) + 0.5) / 200))
        wd_eff = torch.ones(200, dtype=torch.float64) / 200
        if "effw" in estimators:
            est["effw"] = shrinkage(lambda_, t, wt, d_eff, wd_eff, c, "Ndiracs", [-1, KT])
        if "effw_ret" in estimators:
            S_lwo = wLWO_estimator(X, W / n, assume_centered=True, S_r=None, est=False)
            tau_init = initial_tau_guess(np.linalg.eigvalsh(S_lwo), p)
            estimator = Autodiff(bias=True, assume_centered=True).fit_from_S(
                S, W * s2, tau_init=tau_init, method="Adam", n_epochs=cfg.get("n_epochs", 300), b=1, lr=5e-2, verbose=False
            )
            tau_hat_eff = estimator.get_tau().detach()
            est["effw_ret"] = shrinkage(lambda_, tau_hat_eff, torch.ones(p) / p, d_eff, wd_eff, c, "Ndiracs", [-1, KT])
    if "h" not in estimators:
        del est["h"]

    res, d_sigma = losses(tau_pop, lambda_, U, est)
    res["ipr"] = float(p * (U[:, -1] ** 4).sum())  # participation ratio of the top eigenvector
    if np.isfinite(edge):
        excess = (h - d_sigma) ** 2  # excess of the formulas over the ideal oracle, per eigenvalue
        res.update(
            n_out=int(out_right.sum()),
            lambda_1_edge=float(lambda_[-1] / edge),
            share_out=float(excess[out_right].sum() / excess.sum()),  # share of the excess beyond the edge
        )
    null = lambda_ <= 1e-9 * lambda_.max()
    if null.any():
        res["h_null"] = float(h[null][0])  # h(0), predicted value of u_i^T Sigma u_i on the null space
        res["d_null"] = float(d_sigma[null].mean())  # observed mean of u_i^T Sigma u_i on the null space
    if cfg.get("limits") is not None:
        limits = cfg["limits"]
        res["dK_F"] = kolmogorov(lambda_, np.full(p, 1 / p), limits["grid"], limits["F"])
        res["dK_Delta"] = kolmogorov(lambda_, d_sigma / d_sigma.sum(), limits["grid"], limits["Delta"])
    if "ret" in estimators:
        res["tau_hat_max"] = float(tau_hat.max())
    if "effw_ret" in est:
        res["tau_hat_eff_max"] = float(tau_hat_eff.max())
    if cfg.get("n_top"):
        res["lambda_top"] = lambda_[-cfg["n_top"] :].tolist()
        res["d_top"] = d_sigma[-cfg["n_top"] :].tolist()
    return res


def monte_carlo(cfg, n_runs, seed0, n_jobs=1):
    """Run ``n_runs`` independent runs of :func:`monte_carlo_run`, on ``n_jobs`` processes."""
    tasks = [(cfg, seed0 + i) for i in range(n_runs)]
    if n_jobs <= 1:
        return [monte_carlo_run(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(monte_carlo_run, tasks, chunksize=max(1, n_runs // (4 * n_jobs))))


def ratio(L, L_ref, kind, n_boot=1000, seed=0):
    """``E[L] / E[L_ref] - 1`` (``kind="excess"``) or ``1 - E[L] / E[L_ref]`` (``kind="prial"``).

    Returns the point estimate and a 95% paired bootstrap interval.
    """
    L, L_ref = np.asarray(L, dtype=float), np.asarray(L_ref, dtype=float)
    sign = 1.0 if kind == "excess" else -1.0

    def stat(idx):
        return sign * (L[idx].mean() / L_ref[idx].mean() - 1)

    boots = [stat(idx) for idx in np.random.default_rng(seed).integers(0, len(L), size=(n_boot, len(L)))]
    return [stat(np.arange(len(L))), *np.percentile(boots, [2.5, 97.5]).tolist()]


def summarize(label, runs, extra=None):
    """Summarise and print the runs of one configuration.

    For each estimator: excess loss over the ideal oracle and PRIAL (with 95%
    intervals); for each scalar diagnostic: mean and median over the runs.
    """
    summary = {"label": label, "runs": len(runs), **(extra or {})}
    if "L_S" in runs[0]:
        L_S = [r["L_S"] for r in runs]
        L_ideal = [r["L_ideal"] for r in runs]
        summary["prial_ideal"] = ratio(L_ideal, L_S, "prial")
    for key in runs[0]:
        if key.startswith("L_") and key not in ("L_S", "L_ideal"):
            losses = [r[key] for r in runs]
            summary["excess_" + key[2:]] = ratio(losses, L_ideal, "excess")
            summary["prial_" + key[2:]] = ratio(losses, L_S, "prial")
        elif not key.startswith("L_") and np.isscalar(runs[0][key]):
            values = np.array([r[key] for r in runs], dtype=float)
            boots = values[np.random.default_rng(0).integers(0, len(values), size=(1000, len(values)))].mean(axis=1)
            summary[key] = float(values.mean())
            summary[key + "_ci"] = np.percentile(boots, [2.5, 97.5]).tolist()  # 95% bootstrap interval of the mean
            summary[key + "_median"] = float(np.median(values))
    text = [f"{label} ({len(runs)} runs)"]
    if "prial_ideal" in summary:
        text.append(f"PRIAL ideal {100 * summary['prial_ideal'][0]:.1f}%")
    for key, value in summary.items():
        if key.startswith("excess_"):
            name = key[7:]
            prial = 100 * summary["prial_" + name][0]
            text.append(f"{name}: excess {100 * value[0]:.2f}% [{100 * value[1]:.2f}, {100 * value[2]:.2f}], PRIAL {prial:.1f}%")
    for key in ("n_out", "lambda_1_edge", "ipr", "dK_F", "dK_Delta", "h_null", "d_null", "tau_hat_max", "tau_hat_eff_max", "lambda_1"):
        if key in summary:
            text.append(f"{key} {summary[key]:.4g}")
    print(" | ".join(text), flush=True)
    return summary


def scaled_runs(runs, scale):
    """Number of runs after scaling (at least 2)."""
    return max(2, round(runs * scale))
