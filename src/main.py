"""Numerical experiments for the autodiff non-linear shrinkage estimator.

Run ``uv run main.py`` from the ``src`` directory to execute all experiments:

* :func:`experiment_1` compares covariance and precision estimators on
  a three-atom population spectrum.
* :func:`experiment_2` compares estimators of the regression vector
  ``Sigma^{-1} v``.
* :func:`experiment_3` identifies continuous population spectra
  (four shapes ``H_1`` to ``H_4``) and shows Q-Q plots.

Run ``uv run main.py review [scale] [n_jobs]`` to execute instead the
robustness experiments of the article (rank deficiency, spikes and outliers,
heavy tails), see :func:`review_experiments`.
"""

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq

from autodiff import Autodiff
from nl_formulas import nl_cov_shrinkage, nl_prec_shrinkage
from nl_utils import f_xi, m_X, support_u
from support.ewma import mld_exp
from utils import initial_tau_guess
from weighted_LWO_estimator import wLWO_estimator

# ---------------------------------------------------------------------------
# Helpers shared by the experiments
# ---------------------------------------------------------------------------


def make_weight_distribution(n_atoms=200):
    """Return the support ``d`` and probabilities ``wd`` of the weight law.

    The support is uniform on ``[1, 10]``, rescaled to have unit mean.
    """
    d = torch.tensor(np.linspace(1, 10, n_atoms))
    wd = torch.ones(n_atoms, dtype=torch.float64)
    wd = wd / wd.sum(axis=0)
    d = d / (d * wd).sum(axis=0)
    return d, wd


def make_weights(weights, n, alpha, d, wd):
    """Build the vector of ``n`` sample weights.

    Parameters
    ----------
    weights : {"ewma", "unif", "Ndiracs"}
        ``"ewma"``: exponentially decaying weights with rate ``alpha``;
        ``"unif"``: linear weights, ``alpha`` in ``[0, 2]``;
        ``"Ndiracs"``: weights drawn from the discrete law ``(d, wd)``.
    n : int
        Number of samples.
    alpha : float
        Parameter of the ``"ewma"`` and ``"unif"`` families.
    d, wd : torch.Tensor
        Support and probabilities of the ``"Ndiracs"`` law.

    Returns
    -------
    W : ndarray of shape (n,)
        Sample weights, with unit mean.
    """
    index = np.arange(n)
    if weights == "ewma":
        beta = alpha / (1 - np.exp(-alpha))
        return np.exp(-alpha * index / n) * beta
    if weights == "unif":
        return alpha * (index + 1 - (n + 1) / 2) / n + 1
    if weights == "Ndiracs":
        d, wd = d.numpy(), wd.numpy()
        W = []
        for j in range(1, d.shape[0]):
            W += [d[j]] * int(n * wd[j])
        W += [d[0]] * (n - len(W))
        return np.array(W)
    raise ValueError(f"Unknown weights family: {weights!r}")


def simulate_weighted_sample(tau_pop, W):
    """Draw a weighted Gaussian sample and compute its covariance matrices.

    Parameters
    ----------
    tau_pop : ndarray of shape (p,)
        Eigenvalues of the (diagonal) population covariance.
    W : ndarray of shape (n,)
        Sample weights.

    Returns
    -------
    S : ndarray of shape (p, p)
        Weighted sample covariance matrix ``X^T diag(W) X / n``.
    S_lwo : ndarray of shape (p, p)
        Weighted Ledoit-Wolf-type linear shrinkage of ``S`` towards a multiple
        of the identity.
    lambda_, U : ndarray
        Eigenvalues and eigenvectors of ``S``.
    e_lwo : ndarray of shape (p,)
        Eigenvalues of ``S_lwo``.
    """
    n, p = W.shape[0], tau_pop.shape[0]
    sqrt_tau = np.sqrt(np.diag(tau_pop))

    if n <= p <= 10000:
        X_sample = np.random.normal(size=(n, p)) @ sqrt_tau
        S = X_sample.T @ (W[:, None] * X_sample) / n
        S_lwo = wLWO_estimator(X_sample, W / n, assume_centered=True, S_r=None, est=False)
    else:
        # Accumulate S by blocks of at most p samples to bound memory usage and
        # compute the shrinkage intensity from the weighted fourth moments.
        n_left = n
        n_used = 0
        X2sum = np.zeros(n)
        S = np.zeros((p, p))
        while n_left > 0:
            n_sample = min(n_left, p)
            X_sample = np.random.normal(size=(n_sample, p)) @ sqrt_tau
            S += X_sample.T @ (W[n_used : n_used + n_sample, None] * X_sample) / n
            X2sum[n_used : n_used + n_sample] = (X_sample**2).sum(axis=1)
            n_used += n_sample
            n_left -= n_sample

        mu_S = np.trace(S) / p
        delta2 = np.linalg.norm(S - mu_S * np.eye(p), ord="fro") ** 2 / p
        S2 = np.linalg.norm(S, ord="fro") ** 2 / p
        beta2 = (X2sum**2 * W**2).sum() / p / n**2 - S2 * (W**2).sum() / n**2
        beta2 = beta2 / (1 - (W**2).sum() / n**2)
        beta2 = min(beta2, delta2)

        shrinkage = beta2 / delta2
        S_lwo = shrinkage * mu_S * np.eye(p) + (1 - shrinkage) * S

    start_time = time.time()
    lambda_, U = np.linalg.eigh(S)
    print("Diagonalization time:", time.time() - start_time, "s.")
    e_lwo = np.linalg.eigvalsh(S_lwo)
    return S, S_lwo, lambda_, U, e_lwo


def frobenius_loss(A, B):
    """Normalised squared Frobenius distance ``||A - B||_F^2 / p``."""
    return np.linalg.norm(A - B, ord="fro") ** 2 / A.shape[0]


def print_losses(label, norm, loss_ref, losses, digits=3):
    """Print the loss and the PRIAL (relative improvement over ``loss_ref``).

    Parameters
    ----------
    label : str
        Name of the estimated quantity.
    norm : float
        Normalisation of the loss (squared norm of the true quantity).
    loss_ref : float
        Loss of the reference estimator (the sample estimator).
    losses : dict
        Mapping from estimator name to loss.
    digits : int
        Number of printed digits.
    """
    print(f"{label} matrix estimation:")
    print("Norm:           \t", np.around(norm, digits))
    print("Loss/PRIAL ref: \t", np.around(loss_ref, digits), "\t|\t", 0)
    for name, loss in losses.items():
        print(f"Loss/PRIAL {name}: \t", np.around(loss, digits), "\t|\t", np.around(1 - loss / loss_ref, digits))
    print()


def fit_autodiff(S, W, tau_init, n_epochs, lr, b=1):
    """Fit an :class:`Autodiff` estimator and report the compute time."""
    start_time = time.time()
    estimator = Autodiff(bias=True, assume_centered=True).fit_from_S(
        S, W, tau_init=tau_init, method="Adam", n_epochs=n_epochs, b=b, lr=lr, momentum=0.0, verbose=True
    )
    print("Compute time:", time.time() - start_time, "s.")
    return estimator


def asymptotic_density(tau, d, wd, c, mu, weights, w_args, omega, method):
    """Compute the asymptotic sample density ``f`` and cdf ``F`` for ``tau``.

    Returns
    -------
    xi, f, F : torch.Tensor
        Grid, density and cdf of the asymptotic sample spectrum.
    """
    output = f_xi.apply(tau, None, d, wd, c, mu, weights, w_args, omega, method, False)
    nu = output[-1].to(int)
    output = output[: -nu - 1]
    third = output.shape[0] // 3
    return output[2 * third :], output[:third], output[third : 2 * third]


def setup_matplotlib():
    """Use an interactive backend when one is available."""
    try:
        matplotlib.use("tkagg")
        plt.ion()
    except (ImportError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def experiment_1(tau_init=None, p=2000, n=20000, plots=False):
    """Covariance and precision estimation with a three-atom spectrum.

    The population spectrum is ``{1, 3, 10}`` with probabilities
    ``(0.2, 0.4, 0.4)`` and the samples have exponentially decaying weights.
    The autodiff estimator is compared to the sample covariance, to linear
    shrinkage and to the oracle non-linear shrinkage.

    Parameters
    ----------
    tau_init : array-like, optional
        Initial population spectrum (computed from the data if None).
    p, n : int
        Dimension and number of samples.
    plots : bool, default=False
        If True, plot the fitted densities and cdfs.

    Returns
    -------
    estimator : Autodiff
        Fitted estimator.
    tau_init : torch.Tensor
        Initial population spectrum.
    S : ndarray of shape (p, p)
        Sample covariance matrix.
    """
    c = p / n

    # Population spectrum
    t = torch.tensor([1.0, 3.0, 10.0])
    wt = torch.tensor([0.2, 0.4, 0.4])
    tau_pop = np.array([1] * (p // 5) + [3] * (2 * p // 5) + [10] * (p - p // 5 - 2 * p // 5))

    # Weights
    weights = "ewma"
    alpha = 5.0
    Kt = 30
    method = "root"
    d, wd = make_weight_distribution()
    w_args = [alpha, Kt]
    W = make_weights(weights, n, alpha, d, wd)

    # Optimisation parameters
    n_epochs = 10
    lr = 5e-2
    omega = min(400, min(p, n))
    mu = 0.1

    # Sampling
    S, S_lwo, lambda_, U, e_lwo = simulate_weighted_sample(tau_pop, W)

    if tau_init is None:
        tau_init = initial_tau_guess(e_lwo)
    tau_init = torch.as_tensor(tau_init)

    # Covariance estimation
    start_time = time.time()
    h_oracle = nl_cov_shrinkage(lambda_, t, wt, d, wd, c=c, weights=weights, w_args=w_args).numpy()
    print("NL shrinkage time:", time.time() - start_time, "s.")
    S_oracle = U @ np.diag(h_oracle) @ U.T
    cov = np.diag(tau_pop)

    # Precision estimation
    prec = np.diag(1 / tau_pop)
    t_oracle = nl_prec_shrinkage(lambda_, t, wt, d, wd, c=c, weights=weights, w_args=w_args).numpy()
    P_oracle = U @ np.diag(t_oracle) @ U.T

    # Optimise tau
    estimator = fit_autodiff(S, W, tau_init, n_epochs, lr)
    tau_fit = estimator.get_tau()
    S_nl = estimator.get_covariance(d=d, wd=wd, weights=weights, w_args=w_args)
    P_nl = estimator.get_precision(d=d, wd=wd, weights=weights, w_args=w_args)

    print_losses(
        "Covariance",
        frobenius_loss(cov, 0 * cov),
        frobenius_loss(S, cov),
        {
            "linear shrinkage": frobenius_loss(S_lwo, cov),
            "autodiff non-linear": frobenius_loss(S_nl, cov),
            "oracle non-linear": frobenius_loss(S_oracle, cov),
        },
    )
    print_losses(
        "Precision",
        frobenius_loss(prec, 0 * prec),
        frobenius_loss(np.linalg.pinv(S), prec),
        {
            "linear shrinkage": frobenius_loss(np.linalg.pinv(S_lwo), prec),
            "autodiff non-linear": frobenius_loss(P_nl, prec),
            "oracle non-linear": frobenius_loss(P_oracle, prec),
        },
        digits=5,
    )

    if plots:
        title = f"c={c}, {weights}, alpha={alpha}, omega={omega}"
        scale = wt.max().item()

        xi, f, F = asymptotic_density(tau_fit, d, wd, c, mu, weights, w_args, omega, method)
        xi_i, _, F_i = asymptotic_density(tau_init, d, wd, c, mu, weights, w_args, omega, method)
        f_max = f.detach().numpy().max()
        tau_fit_np = tau_fit.detach().numpy()

        plt.figure()
        plt.plot(xi.detach().numpy(), f.detach().numpy(), label="(autodiff) Asymptotic sample density")
        plt.hist(lambda_, bins=100, density=True, label="Sample density")
        plt.hist(
            tau_fit_np,
            weights=3 * np.ones(tau_fit_np.shape[0]) / tau_fit_np.shape[0] * f_max / scale,
            bins=100,
            label="(autodiff) Asymptotic population density",
        )
        plt.hist(t.numpy(), weights=4 * wt.numpy() * f_max / scale, bins=150, label="Population density")
        plt.title(title)
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(xi.detach().numpy(), F.detach().numpy(), label="Fit (autodiff) sample cdf")
        plt.plot(xi_i.detach().numpy(), F_i.detach().numpy(), label="Init sample cdf")
        plt.hist(lambda_, bins=200, cumulative=True, histtype="step", density=True, label="Sample cdf")
        plt.title(title)
        plt.legend()
        plt.show()

        plt.figure()
        plt.hist(tau_fit_np, bins=200, cumulative=True, histtype="step", density=True, label="Fit (autodiff) population cdf")
        plt.hist(tau_init.numpy(), bins=200, cumulative=True, histtype="step", density=True, label="Init population cdf")
        plt.hist(t.numpy(), weights=wt.numpy(), cumulative=True, histtype="step", bins=200, label="Population density")
        plt.title(title)
        plt.legend()
        plt.show()

    return estimator, tau_init, S


def experiment_2(tau_init=None, p=2000, n=20000):
    """Estimation of the regression vector ``Sigma^{-1} v``.

    The autodiff precision estimator is applied to a random unit vector ``v``
    and compared to the oracle non-linear shrinkage, linear shrinkage and the
    sample precision matrix. For each estimator, the relative squared error
    and the PRIAL with respect to the sample estimator are printed.

    Parameters
    ----------
    tau_init : array-like, optional
        Initial population spectrum (computed from the data if None).
    p, n : int
        Dimension and number of samples.

    Returns
    -------
    estimator : Autodiff
        Fitted estimator.
    tau_init : torch.Tensor
        Initial population spectrum.
    S : ndarray of shape (p, p)
        Sample covariance matrix.
    """
    c = p / n

    # Population spectrum
    t = torch.tensor([1.0, 3.0, 10.0])
    wt = torch.tensor([0.2, 0.4, 0.4])
    tau_pop = np.array([1] * (p // 5) + [3] * (2 * p // 5) + [10] * (p - p // 5 - 2 * p // 5))

    # Weights
    weights = "ewma"
    alpha = 5.0
    Kt = 30
    d, wd = make_weight_distribution()
    w_args = [alpha, Kt]
    W = make_weights(weights, n, alpha, d, wd)

    # Optimisation parameters
    n_epochs = 10
    lr = 5e-2

    # Random unit vector for the regression
    v = np.random.normal(size=p)
    v = v / np.linalg.norm(v)

    # Sampling
    S, S_lwo, lambda_, U, e_lwo = simulate_weighted_sample(tau_pop, W)

    if tau_init is None:
        tau_init = initial_tau_guess(e_lwo, p)
    tau_init = torch.as_tensor(tau_init)

    # Autodiff estimator
    estimator = fit_autodiff(S, W, tau_init, n_epochs, lr)
    start_time = time.time()
    w_autodiff = estimator.get_precision(d=d, wd=wd, weights=weights, w_args=w_args) @ v
    print("w_autodiff compute time:", time.time() - start_time, "s.")

    # Oracle non-linear shrinkage: P_oracle v with the true population spectrum
    start_time = time.time()
    t_oracle = nl_prec_shrinkage(lambda_, t, wt, d, wd, c=c, weights=weights, w_args=w_args).numpy()
    w_oracle = U @ (t_oracle * (U.T @ v))
    print("w_oracle compute time:", time.time() - start_time, "s.")

    # Sample and linear shrinkage estimators
    start_time = time.time()
    w_sample = np.linalg.solve(S, v)
    print("w_sample compute time:", time.time() - start_time, "s.")

    start_time = time.time()
    w_linear = np.linalg.solve(S_lwo, v)
    print("w_linear compute time:", time.time() - start_time, "s.")

    w_pop = v / tau_pop
    error_ref = np.linalg.norm(w_sample - w_pop) ** 2

    print("norm:", np.linalg.norm(w_pop) ** 2 / p)
    for name, w_hat in [
        ("autodiff", w_autodiff),
        ("oracle", w_oracle),
        ("linear", w_linear),
        ("sample", w_sample),
    ]:
        error = np.linalg.norm(w_hat - w_pop) ** 2
        print(f"{name}: \t", np.around(error / np.linalg.norm(w_pop) ** 2, 8), " \t | \t ", np.around(1 - error / error_ref, 6))

    return estimator, tau_init, S


def _quantile_spectrum(H_distrib, p):
    """Atoms ``t`` of the population spectrum with cdf shape ``H_{H_distrib}``."""
    x = np.linspace(0, 1, p)
    H1 = 1 - (1 - x**3) ** (1 / 3)
    H2 = (1 - (1 - x) ** 3) ** (1 / 3)
    H3 = (1 - np.abs(1 - 2 * x) ** 3) ** (1 / 3) / 2
    H3[x > 1 / 2] = 1 - H3[x > 1 / 2]
    H4 = 1 / 2 - (1 - (2 * x) ** 3) ** (1 / 3) / 2
    H4[x > 1 / 2] = 1 / 2 + (1 - (2 - 2 * x[x > 1 / 2]) ** 3) ** (1 / 3) / 2
    H = {1: H1, 2: H2, 3: H3}.get(H_distrib, H4)

    t = torch.zeros(p)
    for kappa in range(p):
        below = (((kappa + 1 / 2) - p * H) < 0).astype(int)
        j_k = max(np.argmax(below) - 1, 0)
        j_kp = max(np.argmax(below), 0)
        if H[j_kp] - H[j_k] > 0:
            t[kappa] = (((kappa + 1 / 2) / p - H[j_k]) * x[j_k] + (H[j_kp] - (kappa + 1 / 2) / p) * x[j_kp]) / (H[j_kp] - H[j_k])
        else:
            t[kappa] = (x[j_k] + x[j_kp]) / 2
    return 1 + 9 * t


def experiment_3(H_distrib=1, tau_init=None, p=2000, n=20000, plots=True):
    """Identification of a continuous population spectrum.

    Parameters
    ----------
    H_distrib : {1, 2, 3, 4}, default=1
        Shape of the population spectral distribution.
    tau_init : array-like, optional
        Initial population spectrum (computed from the data if None).
    p, n : int
        Dimension and number of samples.
    plots : bool, default=True
        If True, plot the Q-Q plot and the densities of the fitted spectrum.

    Returns
    -------
    estimator : Autodiff
        Fitted estimator.
    """
    c = p / n

    # Population spectrum (any value other than 1, 2, 3 selects H_4)
    H_label = H_distrib if H_distrib in (1, 2, 3) else 4
    tau_pop = _quantile_spectrum(H_distrib, p)
    wt = torch.ones(p) / p

    # Weights
    weights = "unif"
    alpha = 1.0
    d, wd = make_weight_distribution()
    W = make_weights(weights, n, alpha, d, wd)

    # Optimisation parameters
    n_epochs = 200
    lr = 2e-3

    if plots:
        plt.figure()
        plt.hist(tau_pop.numpy(), bins=100, density=True, label=f"H{H_label} histogram")
        plt.show()

    # Sampling
    S, _, lambda_, _, e_lwo = simulate_weighted_sample(tau_pop.numpy(), W)

    if tau_init is None:
        tau_init = initial_tau_guess(e_lwo, p)
    tau_init = torch.as_tensor(tau_init)

    # Optimise tau
    print("autodiff:")
    estimator = fit_autodiff(S, W, tau_init, n_epochs, lr)
    tau_fit = estimator.get_tau().detach().numpy()

    if plots:
        plt.figure()
        plt.plot(tau_pop.numpy(), np.sort(tau_fit), label="(autodiff) algorithm")
        plt.plot(tau_pop.numpy(), np.sort(lambda_), label="Sample spectrum")
        plt.plot(tau_pop.numpy(), tau_pop.numpy(), label="Population spectrum")
        plt.xlabel(r"$H$")
        plt.ylabel(r"$\hat H$")
        plt.title(rf"Q-Q plot for $H_{H_label}$")
        plt.legend()
        plt.show()

        plt.figure()
        plt.hist(tau_fit, bins=200, cumulative=False, histtype="step", density=True, label="Fit (autodiff) population pdf")
        plt.hist(lambda_, bins=200, cumulative=False, histtype="step", density=True, label="Sample pdf")
        plt.hist(tau_pop.numpy(), weights=wt.numpy(), cumulative=False, density=True, histtype="step", bins=200, label="Population density")
        plt.title(f"c={c}, {weights}, alpha={alpha}")
        plt.legend()
        plt.show()

    return estimator


# ---------------------------------------------------------------------------
# Review experiments: rank deficiency, spikes and outliers, heavy tails
# ---------------------------------------------------------------------------
#
# These experiments were added for the review of the article. They support the
# subsections "Convergence with heavy tails: empirical observations" and
# "Spikes and outliers", and the checks on c (rank-deficient sample covariance
# when c > 1) and on a singular population covariance. They only call the
# functions of this repository (nl_formulas, nl_utils, autodiff, utils), which
# are not modified.
#
# Notation of this section: p is the dimension, n the number of samples and
# c = p / n (the article writes n for the dimension and N for the number of
# samples). Unless stated otherwise:
#   * Sigma = diag(tau_pop), where tau_pop has the atoms {1, 3, 10} in the
#     proportions (0.2, 0.4, 0.4) (the spectrum H of the article);
#   * the sample weights decay exponentially with rate alpha = 2;
#   * an estimator is sum_i h_i u_i u_i^T, where (lambda_i, u_i) are the
#     eigenpairs of S = X^T diag(W) X / n. Its Frobenius loss L is compared with
#     the loss L_S of S (PRIAL = 1 - E[L] / E[L_S]) and with the loss L_ideal of
#     the "ideal" rotation-invariant oracle h_i = u_i^T Sigma u_i, which cannot
#     be computed in practice (excess loss = E[L] / E[L_ideal] - 1);
#   * intervals are 95% paired bootstrap intervals over the Monte-Carlo runs.

REVIEW_T = torch.tensor([1.0, 3.0, 10.0])  # atoms of H (float32, as in the other experiments)
REVIEW_WT = torch.tensor([0.2, 0.4, 0.4])  # probabilities of the atoms of H
REVIEW_KT = 30  # number of gaps of H searched by the support identification (w_args[1])
REVIEW_NOISES = ("gauss", "t12", "t5", "t4", "t3", "t2.5", "pareto3", "ell5")


def review_population(p, t=REVIEW_T, wt=REVIEW_WT):
    """Population eigenvalues: the atoms ``t`` repeated in the proportions ``wt``."""
    counts = np.floor(wt.numpy().astype(float) * p + 1e-6).astype(int)
    counts[-1] += p - counts.sum()
    return np.repeat(t.numpy().astype(float), counts)


def review_weight_law(n, alpha):
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
        return make_weights("ewma", n, alpha, d, wd), "ewma", [alpha, REVIEW_KT], d, wd
    one = torch.ones(1, dtype=torch.float64)
    return np.ones(n), "Ndiracs", [-1, REVIEW_KT], one, one


def review_noise(kind, n, p, rng):
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


def review_shrinkage(lambda_, t, wt, d, wd, c, weights, w_args):
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


def review_X(lambda_, t, wt, d, wd, c, weights, w_args):
    """``X(lambda_i)`` computed by :func:`nl_utils.m_X` (real outside the support of F)."""
    return m_X(torch.as_tensor(np.asarray(lambda_, dtype=float)), weights, w_args, d, wd, c, t, wt)[1].numpy()


def review_right_edge(t, wt, d, wd, c, weights, w_args):
    """Right edge of the support of the limiting sample spectrum F (:func:`support_u`)."""
    support = support_u(t.numpy(), wt.numpy(), d.numpy(), wd.numpy(), c, weights, w_args, verbose=False)[0]
    return float(support[-1])


def review_outlier_location(tau_s, alpha, c, t=REVIEW_T, wt=REVIEW_WT):
    """Heuristic location of the outlier created by a population spike ``tau_s``.

    The outlier ``lambda_s`` solves ``f_lambda(-1/tau_s) = 0``, i.e.
    ``Xcheck(lambda_s) = -1/tau_s`` in the theorem on the computation of
    ``Xcheck`` of the article, extended outside the support. With
    ``theta = c sum(wt t / (1 - t / tau_s))``, this reads
    ``E[W / (W - lambda_s / theta)] = -theta / tau_s``, whose left-hand side is
    :func:`mld_exp` for exponential weights and ``1 / (1 - lambda_s / theta)``
    for equal weights (``alpha = 0``). It is a prediction, not a theorem; it
    only makes sense above the threshold where it reaches the right edge.
    """
    theta = c * float((wt * t / (1 - t / tau_s)).sum())
    if alpha == 0:
        return theta + tau_s
    beta = alpha / (1 - np.exp(-alpha))  # largest weight of the limiting weight law

    def gap(u):
        # mld_exp increases from -inf to 0 on ]beta, inf[
        return mld_exp(u, alpha) + theta / tau_s

    hi = 2 * beta
    while gap(hi) < 0:
        hi *= 2
    return brentq(gap, beta * (1 + 1e-12), hi, xtol=1e-12) * theta


def review_limit_cdfs(t, wt, d, wd, c, weights, w_args, omega=400):
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
    tau_atoms = torch.tensor(review_population(100, t, wt), dtype=torch.float32)
    xi, f, F = asymptotic_density(tau_atoms, d, wd, c, 0.1, weights, w_args, omega, "root")
    xi, f, F = xi.detach().numpy().astype(float), f.detach().numpy().astype(float), F.detach().numpy().astype(float)
    # When c > 1, nl_cov_shrinkage only evaluates the ceil(len / c) largest
    # positive points: pad the grid with zeros (which return h(0)) so that
    # every grid point is evaluated.
    n_pad = int(np.ceil(max(c - 1, 0) * xi.shape[0])) + 1
    h = review_shrinkage(np.concatenate([np.zeros(n_pad), xi]), t, wt, d, wd, c, weights, w_args)
    h0, h = h[0], h[n_pad:]
    density = f * h
    Delta = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(xi))])
    atom0 = (1 - 1 / c) * h0 / float((wt * t).sum()) if c > 1 else 0.0
    Delta = atom0 + (1 - atom0) * Delta / Delta[-1]
    return {"grid": xi, "F": F, "Delta": Delta}


def review_kolmogorov(points, jumps, grid, cdf):
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


def review_losses(tau_pop, lambda_, U, estimates):
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


def review_run(task):
    """One Monte-Carlo run of a review experiment.

    Module-level function, so that it can be sent to worker processes.

    Parameters
    ----------
    task : tuple (cfg, seed)
        ``cfg`` is a dict with the keys

        * ``p``, ``n``, ``alpha``, ``kind`` (see :func:`review_noise`) and
          ``tau_pop`` (population eigenvalues);
        * ``t``, ``wt`` (optional): the limit ``H`` given to the formulas
          (default ``REVIEW_T``, ``REVIEW_WT``);
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

        * ``edge``: right edge of the support of F; ``limits`` (optional):
          output of :func:`review_limit_cdfs`; ``n_top`` (optional): number of
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
    t, wt = cfg.get("t", REVIEW_T), cfg.get("wt", REVIEW_WT)
    W, weights, w_args, d, wd = review_weight_law(n, alpha)

    # Weighted sample covariance S = X^T diag(W) X / n, with X = Z diag(tau_pop)^(1/2)
    Z, s2 = review_noise(cfg.get("kind", "gauss"), n, p, rng)
    X = Z * np.sqrt(tau_pop)[None, :]
    S = X.T @ (W[:, None] * X) / n
    if cfg.get("location_only"):
        lambda_ = np.linalg.eigvalsh(S)
        return {"lambda_1": float(lambda_[-1]), "lambda_2": float(lambda_[-2])}
    lambda_, U = np.linalg.eigh(S)

    edge = cfg.get("edge", np.inf)
    out_right = lambda_ > edge  # eigenvalues beyond the right edge of the support of F
    estimators = cfg["estimators"]
    h = review_shrinkage(lambda_, t, wt, d, wd, c, weights, w_args)
    est = {"h": h}
    if "keep" in estimators:
        est["keep"] = np.where(out_right, lambda_, h)
    if "inv" in estimators:
        h_inv = h.copy()
        if out_right.any():
            X_out = review_X(lambda_[out_right], t, wt, d, wd, c, weights, w_args).real
            # -1 / Xcheck estimates the population spike; keep lambda_i if the solve failed (X = 0)
            h_inv[out_right] = np.where(X_out < 0, -1 / np.minimum(X_out, -1e-12), lambda_[out_right])
        est["inv"] = h_inv
    if "Hn" in estimators:
        atoms, counts = np.unique(tau_pop, return_counts=True)
        t_n = torch.tensor(atoms, dtype=torch.float32)
        wt_n = torch.tensor(counts / p, dtype=torch.float32)
        est["Hn"] = review_shrinkage(lambda_, t_n, wt_n, d, wd, c, weights, w_args)
    if "ret" in estimators:
        # Initial spectrum from the weighted linear shrinkage, as in Autodiff.fit
        # (positive even when S is singular)
        S_lwo = wLWO_estimator(X, W / n, assume_centered=True, S_r=None, est=False)
        tau_init = initial_tau_guess(np.linalg.eigvalsh(S_lwo), p)
        estimator = Autodiff(bias=True, assume_centered=True).fit_from_S(
            S, W, tau_init=tau_init, method="Adam", n_epochs=cfg.get("n_epochs", 300), b=1, lr=5e-2, verbose=False
        )
        tau_hat = estimator.get_tau().detach()
        est["ret"] = review_shrinkage(lambda_, tau_hat, torch.ones(p) / p, d, wd, c, weights, w_args)
    if "effw" in estimators and s2 is not None:
        d_eff = torch.tensor(np.quantile(W * s2, (np.arange(200) + 0.5) / 200))
        wd_eff = torch.ones(200, dtype=torch.float64) / 200
        est["effw"] = review_shrinkage(lambda_, t, wt, d_eff, wd_eff, c, "Ndiracs", [-1, REVIEW_KT])
    if "h" not in estimators:
        del est["h"]

    res, d_sigma = review_losses(tau_pop, lambda_, U, est)
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
        res["dK_F"] = review_kolmogorov(lambda_, np.full(p, 1 / p), limits["grid"], limits["F"])
        res["dK_Delta"] = review_kolmogorov(lambda_, d_sigma / d_sigma.sum(), limits["grid"], limits["Delta"])
    if "ret" in estimators:
        res["tau_hat_max"] = float(tau_hat.max())
    if cfg.get("n_top"):
        res["lambda_top"] = lambda_[-cfg["n_top"] :].tolist()
        res["d_top"] = d_sigma[-cfg["n_top"] :].tolist()
    return res


def review_monte_carlo(cfg, n_runs, seed0, n_jobs=1):
    """Run ``n_runs`` independent runs of :func:`review_run`, on ``n_jobs`` processes."""
    tasks = [(cfg, seed0 + i) for i in range(n_runs)]
    if n_jobs <= 1:
        return [review_run(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(review_run, tasks, chunksize=max(1, n_runs // (4 * n_jobs))))


def review_ratio(L, L_ref, kind, n_boot=1000, seed=0):
    """``E[L] / E[L_ref] - 1`` (``kind="excess"``) or ``1 - E[L] / E[L_ref]`` (``kind="prial"``).

    Returns the point estimate and a 95% paired bootstrap interval.
    """
    L, L_ref = np.asarray(L, dtype=float), np.asarray(L_ref, dtype=float)
    sign = 1.0 if kind == "excess" else -1.0

    def stat(idx):
        return sign * (L[idx].mean() / L_ref[idx].mean() - 1)

    boots = [stat(idx) for idx in np.random.default_rng(seed).integers(0, len(L), size=(n_boot, len(L)))]
    return [stat(np.arange(len(L))), *np.percentile(boots, [2.5, 97.5]).tolist()]


def review_summary(label, runs, extra=None):
    """Summarise and print the runs of one configuration.

    For each estimator: excess loss over the ideal oracle and PRIAL (with 95%
    intervals); for each scalar diagnostic: mean and median over the runs.
    """
    summary = {"label": label, "runs": len(runs), **(extra or {})}
    if "L_S" in runs[0]:
        L_S = [r["L_S"] for r in runs]
        L_ideal = [r["L_ideal"] for r in runs]
        summary["prial_ideal"] = review_ratio(L_ideal, L_S, "prial")
    for key in runs[0]:
        if key.startswith("L_") and key not in ("L_S", "L_ideal"):
            losses = [r[key] for r in runs]
            summary["excess_" + key[2:]] = review_ratio(losses, L_ideal, "excess")
            summary["prial_" + key[2:]] = review_ratio(losses, L_S, "prial")
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
    for key in ("n_out", "lambda_1_edge", "ipr", "dK_F", "dK_Delta", "h_null", "d_null", "tau_hat_max", "lambda_1"):
        if key in summary:
            text.append(f"{key} {summary[key]:.4g}")
    print(" | ".join(text), flush=True)
    return summary


def review_runs(runs, scale):
    """Number of runs after scaling (at least 2)."""
    return max(2, round(runs * scale))


def review_experiment_concentration(
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
        cfg = {"p": p, "n": m, "alpha": 2.0, "kind": "gauss", "tau_pop": review_population(p), "estimators": ["h"]}
        n_runs = review_runs(runs if p <= 800 else runs // 4, scale)
        results.append(
            review_summary(f"c={c:g}, p={p}, n={m}", review_monte_carlo(cfg, n_runs, 1000 + m, n_jobs), {"c": c, "p": p, "n": m})
        )
    return results


def review_experiment_singular(p=400, n=800, zero_fraction=0.2, runs=100, scale=1.0, n_jobs=1):
    """Singular population covariance: a fraction ``zero_fraction`` of zero eigenvalues.

    ``H`` gets an atom at 0. The non-zero spectrum of ``S`` is that of the
    restricted model, and the formulas return 0 on the null space of ``S``.
    """
    print("\n=== Singular population covariance ===")
    t = torch.tensor([0.0, 1.0, 3.0, 10.0])
    wt = torch.tensor([zero_fraction, *((1 - zero_fraction) * REVIEW_WT.numpy()).tolist()])
    cfg = {"p": p, "n": n, "alpha": 2.0, "kind": "gauss", "tau_pop": review_population(p, t, wt), "t": t, "wt": wt, "estimators": ["h"]}
    runs_ = review_monte_carlo(cfg, review_runs(runs, scale), 4000, n_jobs)
    return [review_summary(f"zero fraction {zero_fraction}, p={p}, n={n}", runs_, {"p": p, "n": n, "zero_fraction": zero_fraction})]


def review_experiment_heavy_tails(
    noises=REVIEW_NOISES,
    n_list=(100, 200, 400, 800, 1600, 3200),
    runs=(1600, 800, 400, 160, 80, 40),
    alpha=2.0,
    c=0.5,
    kolmogorov=True,
    scale=1.0,
    n_jobs=1,
):
    """Heavy tails with the true ``H`` (Figure "heavy" and Table "heavy_set" of the article).

    For each noise (:func:`review_noise`) and number of samples ``n``
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
    _, weights, w_args, d, wd = review_weight_law(1000, alpha)  # the limiting weight law does not depend on n
    edge = review_right_edge(REVIEW_T, REVIEW_WT, d, wd, c, weights, w_args)
    limits = review_limit_cdfs(REVIEW_T, REVIEW_WT, d, wd, c, weights, w_args) if kolmogorov else None
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
                "tau_pop": review_population(p),
                "edge": edge,
                "limits": limits,
                "estimators": estimators,
            }
            runs_ = review_monte_carlo(cfg, review_runs(n_runs, scale), 10_000 + m, n_jobs)
            results.append(
                review_summary(f"{kind}, p={p}, n={m}", runs_, {"kind": kind, "p": p, "n": m, "alpha": alpha, "c": c, "edge": edge})
            )
    return results


def review_experiment_bona_fide(
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
    outliers as population spikes. ``first_round_cases`` adds Gaussian noise
    with ``c = 2`` (``n = p / 2``) and three spikes ``tau_s = 50``.
    """
    print("\n=== Bona fide estimator (retrieved H) ===")
    _, weights, w_args, d, wd = review_weight_law(n, 2.0)
    results = []
    cases = [(kind, p, n, review_population(p)) for kind in noises]
    if first_round_cases:
        spiked = review_population(p)
        spiked[-3:] = 50.0
        cases += [("gauss", p, p // 2, review_population(p)), ("gauss", p, n, spiked)]
    for kind, p_, n_, tau_pop in cases:
        edge = review_right_edge(REVIEW_T, REVIEW_WT, d, wd, p_ / n_, weights, w_args)
        estimators = ["h", "inv", "ret"] + (["effw"] if kind.startswith("ell") else [])
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
        results.append(
            review_summary(label, review_monte_carlo(cfg, review_runs(runs, scale), 30_000, n_jobs), {"kind": kind, "p": p_, "n": n_})
        )
    return results


def review_experiment_spike_location(
    taus=(10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0),
    alphas=(0.0, 2.0),
    sizes=((400, 800, 100), (1600, 3200, 24)),
    scale=1.0,
    n_jobs=1,
):
    """Location of the outlier created by one population spike (Figure "spikes" (a)).

    One eigenvalue 10 of ``Sigma`` is replaced by ``tau_s``. The largest sample
    eigenvalue is compared with the prediction of
    :func:`review_outlier_location` above the threshold ``tau_c`` (where the
    prediction reaches the right edge of the support), and with the edge
    below it, with exponential (``alpha = 2``) and equal (``alpha = 0``) weights.
    """
    print("\n=== Spike location ===")
    results = []
    for alpha in alphas:
        _, weights, w_args, d, wd = review_weight_law(1000, alpha)
        edge = review_right_edge(REVIEW_T, REVIEW_WT, d, wd, 0.5, weights, w_args)
        grid = np.geomspace(10.5, 60, 300)
        curve = np.array([review_outlier_location(ts, alpha, 0.5) for ts in grid])
        tau_c = float(grid[np.argmin(curve)])
        print(f"alpha={alpha:g}: right edge {edge:.3f}, threshold tau_c {tau_c:.2f}", flush=True)
        for p, n, n_runs in sizes:
            for tau_s in taus:
                tau_pop = review_population(p)
                tau_pop[-1] = tau_s
                predicted = review_outlier_location(tau_s, alpha, p / n) if tau_s > tau_c else edge
                cfg = {"p": p, "n": n, "alpha": alpha, "kind": "gauss", "tau_pop": tau_pop, "location_only": True}
                runs_ = review_monte_carlo(cfg, review_runs(n_runs, scale), 40_000 + p + round(10 * tau_s), n_jobs)
                results.append(
                    review_summary(
                        f"alpha={alpha:g}, p={p}, tau_s={tau_s:g}, predicted {predicted:.3f}",
                        runs_,
                        {"alpha": alpha, "p": p, "n": n, "tau_s": tau_s, "predicted": predicted, "edge": edge, "tau_c": tau_c},
                    )
                )
    return results


def review_experiment_spike_estimation(
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
    _, weights, w_args, d, wd = review_weight_law(n, 2.0)
    edge = review_right_edge(REVIEW_T, REVIEW_WT, d, wd, p / n, weights, w_args)
    results = []
    for r, tau_s in [(0, 0.0)] + [(r, ts) for r in rs for ts in taus]:
        tau_pop = review_population(p)
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
        runs_ = review_monte_carlo(cfg, review_runs(runs, scale), 50_000 + 1000 * r + round(tau_s), n_jobs)
        results.append(review_summary(f"r={r}, tau_s={tau_s:g}", runs_, {"r": r, "tau_s": tau_s, "p": p, "n": n}))
    return results


def review_experiments(scale=1.0, n_jobs=1, which=None, save=None):
    """Run the robustness experiments of the article and return their summaries.

    Parameters
    ----------
    scale : float, default=1.0
        Multiplies the numbers of Monte-Carlo runs; ``scale=1`` uses the
        numbers of the article, e.g. ``scale=0.05`` gives a quick check.
    n_jobs : int, default=1
        Number of worker processes for the Monte-Carlo runs.
    which : iterable of str, optional
        Subset of ``"concentration"``, ``"singular"``, ``"heavy_tails"``,
        ``"heavy_settings"``, ``"bona_fide"``, ``"spike_location"``,
        ``"spike_estimation"`` (all by default).
    save : str, optional
        Path of a JSON file where the summaries are written.

    Notes
    -----
    With the default sizes the full set takes hours on one process: most of
    the time is spent in :func:`nl_cov_shrinkage` (one root search per sample
    eigenvalue) and in the retrieval (:meth:`Autodiff.fit_from_S`).
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # one BLAS thread per worker process
    common = {"scale": scale, "n_jobs": n_jobs}
    settings_noises = ("gauss", "t5", "t4", "t3", "t2.5")
    experiments = {
        "concentration": lambda: review_experiment_concentration(**common),
        "singular": lambda: review_experiment_singular(**common),
        "heavy_tails": lambda: review_experiment_heavy_tails(**common),
        # other weights (alpha = 0, 5) and ratios (c = 0.25, 2), Table "heavy_set"
        "heavy_settings": lambda: (
            review_experiment_heavy_tails(settings_noises, (1600,), (80,), alpha=0.0, **common)
            + review_experiment_heavy_tails(settings_noises, (1600,), (80,), alpha=5.0, **common)
            + review_experiment_heavy_tails(settings_noises, (1600,), (80,), c=0.25, **common)
            + review_experiment_heavy_tails(settings_noises, (800,), (80,), c=2.0, **common)
        ),
        "bona_fide": lambda: review_experiment_bona_fide(**common),
        "spike_location": lambda: review_experiment_spike_location(**common),
        "spike_estimation": lambda: review_experiment_spike_estimation(**common),
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


def main():
    """Run all the experiments."""
    setup_matplotlib()
    warnings.filterwarnings("ignore")

    experiment_1(tau_init=None, p=100, n=1000, plots=True)
    experiment_2(tau_init=None, p=100, n=1000)
    for H_distrib in (1, 2, 3, 4):
        experiment_3(H_distrib=H_distrib, tau_init=None, p=200, n=2000)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "review":
        # uv run main.py review [scale] [n_jobs]
        warnings.filterwarnings("ignore")
        review_experiments(
            scale=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0,
            n_jobs=int(sys.argv[3]) if len(sys.argv) > 3 else 1,
            save="review_results.json",
        )
    else:
        main()
