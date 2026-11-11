"""Numerical experiments for the autodiff non-linear shrinkage estimator.

Run ``uv run main.py`` from the ``src`` directory to execute all experiments:

* :func:`WeSpeR_experiment_1` compares covariance and precision estimators on
  a three-atom population spectrum.
* :func:`WeSpeR_experiment_2` compares estimators of the regression vector
  ``Sigma^{-1} v``.
* :func:`WeSpeR_experiment_3` identifies continuous population spectra
  (four shapes ``H_1`` to ``H_4``) and shows Q-Q plots.
"""

import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from autodiff import Autodiff
from nl_formulas import nl_cov_shrinkage, nl_prec_shrinkage
from nl_utils import f_xi
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


def WeSpeR_experiment_1(tau_init=None, p=2000, n=20000, plots=False):
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


def WeSpeR_experiment_2(tau_init=None, p=2000, n=20000):
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


def WeSpeR_experiment_3(H_distrib=1, tau_init=None, p=2000, n=20000, plots=True):
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


def main():
    """Run all the experiments."""
    setup_matplotlib()
    warnings.filterwarnings("ignore")

    WeSpeR_experiment_1(tau_init=None, p=100, n=1000, plots=True)
    WeSpeR_experiment_2(tau_init=None, p=100, n=1000)
    for H_distrib in (1, 2, 3, 4):
        WeSpeR_experiment_3(H_distrib=H_distrib, tau_init=None, p=200, n=2000)


if __name__ == "__main__":
    main()
