"""Non-linear shrinkage formulas of the sample eigenvalues.

Given the sample eigenvalues and a population spectrum ``(t, wt)``, these
functions return the oracle non-linear shrinkage of the eigenvalues of the
covariance matrix (:func:`nl_cov_shrinkage`) and of its inverse
(:func:`nl_prec_shrinkage`) for weighted samples.
"""

import numpy as np
import torch

from nl_utils import m_X


def _transforms(lambda_, t, wt, d, wd, c, weights, w_args):
    """Evaluate the transform ``X`` at the positive sample eigenvalues.

    Returns
    -------
    lambda_ : ndarray
        Sample eigenvalues, sorted.
    lambda_p : ndarray
        Positive sample eigenvalues that are used (the ``n`` largest if
        ``c > 1``).
    X0 : torch.Tensor or None
        Value of ``X`` at zero (only needed when ``c > 1``).
    X_lambda : torch.Tensor
        Value of ``X`` at ``lambda_p``.
    """
    lambda_ = np.sort(lambda_)
    n = int(np.ceil(lambda_.shape[0] / c))

    if c > 1:
        _, X0 = m_X(torch.zeros(1), weights, w_args, d, wd, c, t, wt)
        lambda_p = lambda_[lambda_ > 0][-n:]
    else:
        X0 = None
        lambda_p = lambda_[lambda_ > 0]

    _, X_lambda = m_X(lambda_p, weights, w_args, d, wd, c, t, wt)
    return lambda_, lambda_p, X0, X_lambda


def nl_cov_shrinkage(lambda_, t, wt, d, wd, c=0.1, weights="ewma", w_args=None):
    """Oracle non-linear shrinkage of the eigenvalues of the covariance matrix.

    Parameters
    ----------
    lambda_ : array-like of shape (p,)
        Sample eigenvalues.
    t, wt : torch.Tensor
        Atoms and probabilities of the population spectrum.
    d, wd : torch.Tensor
        Support and probabilities of the weight distribution.
    c : float, default=0.1
        Concentration ratio ``p / n``.
    weights : {"ewma", "unif", "Ndiracs"}, default="ewma"
        Family of the sample weights.
    w_args : list, optional
        Parameters of the weight family (``[alpha, Kt]`` for ``"ewma"`` and
        ``"unif"``, ``[Kd, Kt]`` for ``"Ndiracs"``).

    Returns
    -------
    h_lambda : torch.Tensor of shape (p,)
        Shrunk eigenvalues, in increasing order of the sample eigenvalues.
    """
    lambda_, lambda_p, X0, X_lambda = _transforms(lambda_, t, wt, d, wd, c, weights, w_args)

    if c > 1:
        h_lambda = 1 / (c - 1) / X0 * torch.ones(lambda_.shape)
    else:
        h_lambda = torch.zeros(lambda_.shape)

    numerator = (wt[None, :] * t[None, :] ** 2 / torch.abs(t[None, :] * X_lambda[:, None] + 1) ** 2).sum(axis=1)
    denominator = (wt[None, :] * t[None, :] / torch.abs(t[None, :] * X_lambda[:, None] + 1) ** 2).sum(axis=1)
    h_lambda[-lambda_p.shape[0] :] = numerator / denominator

    return h_lambda


def nl_prec_shrinkage(lambda_, t, wt, d, wd, c=0.1, weights="ewma", w_args=None):
    """Oracle non-linear shrinkage of the eigenvalues of the precision matrix.

    See :func:`nl_cov_shrinkage` for the parameters.

    Returns
    -------
    t_lambda : torch.Tensor of shape (p,)
        Shrunk inverse eigenvalues, in increasing order of the sample
        eigenvalues.
    """
    lambda_, lambda_p, X0, X_lambda = _transforms(lambda_, t, wt, d, wd, c, weights, w_args)

    if c > 1:
        t_lambda = (-X0 + c / (c - 1) * (wt / t).sum()) * torch.ones(lambda_.shape)
    else:
        t_lambda = torch.zeros(lambda_.shape)

    numerator = (wt[None, :] / torch.abs(t[None, :] * X_lambda[:, None] + 1) ** 2).sum(axis=1)
    denominator = (wt[None, :] * t[None, :] / torch.abs(t[None, :] * X_lambda[:, None] + 1) ** 2).sum(axis=1)
    t_lambda[-lambda_p.shape[0] :] = numerator / denominator

    return t_lambda
