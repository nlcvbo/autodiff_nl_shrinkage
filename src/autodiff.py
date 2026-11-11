"""Scikit-learn compatible estimator based on automatic differentiation.

The population spectrum is identified by matching simulated and observed
sample spectra with gradient descent (see :func:`utils.autodiff_minimization`),
and is then plugged into the non-linear shrinkage formulas of
:mod:`nl_formulas`.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, validate_data

from nl_formulas import nl_cov_shrinkage, nl_prec_shrinkage
from utils import autodiff_minimization, initial_tau_guess
from weighted_LWO_estimator import wLWO_estimator


class Autodiff(BaseEstimator, TransformerMixin):
    """Non-linear shrinkage estimator of weighted covariance matrices.

    Parameters
    ----------
    bias : bool, default=True
        If True, normalize by ``n``. If False, correct the normalization by
        the effective number of samples.
    assume_centered : bool, default=False
        If True, data will not be centered before computing the covariance.

    Attributes
    ----------
    model_ : AutodiffModel
        Trained sampling model.
    tau_fit_ : torch.Tensor of shape (p,)
        Estimated population eigenvalues.
    loss : ndarray
        Training loss at each epoch.
    lambda_ : torch.Tensor of shape (p,)
        Sample eigenvalues.
    U : torch.Tensor of shape (p, p)
        Sample eigenvectors.
    c : float
        Concentration ratio ``p / n`` (corrected by the ``bias`` option).
    covariance_, precision_ : ndarray of shape (p, p)
        Estimated covariance / precision matrices, set by
        :meth:`get_covariance` / :meth:`get_precision`.
    """

    def __init__(self, bias=True, assume_centered=False):
        self.bias = bias
        self.assume_centered = assume_centered

    def _store_fit(self, minimization_output):
        """Store the outputs of :func:`utils.autodiff_minimization`."""
        model, tau_fit, loss, _, lambda_, U = minimization_output
        self.model_ = model
        self.tau_fit_ = tau_fit
        self.loss = loss
        self.lambda_ = lambda_
        self.U = U
        return self

    def fit_from_F(
        self,
        lambda_,
        U,
        W,
        c=None,
        p_tau=1000,
        y=None,
        tau_init=None,
        method="Adam",
        n_epochs=100,
        b=1,
        lr=5e-2,
        momentum=0.0,
        verbose=True,
    ):
        """Estimate the population spectrum from the eigendecomposition of S.

        The optimisation is run on a subsampled problem of dimension ``p_tau``.

        Parameters
        ----------
        lambda_ : array-like of shape (n_features,)
            Eigenvalues of the sample covariance.
        U : array-like of shape (n_features, n_features)
            Eigenvectors of the sample covariance.
        W : ndarray of shape (n_samples,)
            Training weight vector.
        c : float, optional
            Concentration ratio. Defaults to ``p / n`` (see ``bias``).
        p_tau : int or None, default=1000
            Dimension of the subsampled problem (``None`` keeps ``p``).
        y : Ignored
            Not used, present for API consistency by convention.
        tau_init : torch.Tensor, optional
            Initial population spectrum (computed from ``lambda_`` if None).
        method : {"Adam", "SGD"}, default="Adam"
            Optimiser.
        n_epochs : int, default=100
            Number of optimisation steps.
        b : int, default=1
            Batch size of the simulated sample spectra.
        lr : float, default=5e-2
            Learning rate.
        momentum : float, default=0.0
            Momentum (SGD only).
        verbose : bool, default=True
            If True, print the training loss.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        p = U.shape[0]
        n = W.shape[0]
        self.n, self.p = n, p

        if p_tau is None:
            p_tau = p
        self.p_tau = p_tau

        ddof = 0 if self.bias else (W**2).sum() / n**2
        self.c = p / (n - ddof) if c is None else c
        n_tau = int(p_tau / self.c)
        self.n_tau = n_tau

        if tau_init is None:
            tau_init = initial_tau_guess(lambda_, p_tau)

        lambda_tau = np.sort(np.random.choice(lambda_, p_tau))
        W_tau = np.sort(np.random.choice(W, n_tau))

        output = autodiff_minimization(
            torch.tensor(np.diag(lambda_tau)),
            n_tau,
            p_tau,
            tau_init,
            torch.sqrt(torch.tensor(W_tau)),
            method=method,
            n_epochs=n_epochs,
            b=b,
            lr=lr,
            momentum=momentum,
            save_all_tau=False,
            verbose=verbose,
        )
        # The optimisation ran on a subsampled problem: keep the eigendecomposition
        # of the full problem for the shrinkage step.
        self._store_fit(output)
        self.lambda_ = lambda_
        self.U = U
        return self

    def fit_from_S(
        self,
        S,
        W,
        y=None,
        tau_init=None,
        method="Adam",
        n_epochs=100,
        b=1,
        lr=5e-2,
        momentum=0.0,
        verbose=True,
    ):
        """Estimate the population spectrum from a weighted sample covariance.

        Parameters
        ----------
        S : ndarray of shape (n_features, n_features)
            Weighted sample covariance.
        W : ndarray of shape (n_samples,)
            Training weight vector.
        y : Ignored
            Not used, present for API consistency by convention.
        tau_init : torch.Tensor, optional
            Initial population spectrum (computed from ``S`` if None).
        method : {"Adam", "SGD"}, default="Adam"
            Optimiser.
        n_epochs : int, default=100
            Number of optimisation steps.
        b : int, default=1
            Batch size of the simulated sample spectra.
        lr : float, default=5e-2
            Learning rate.
        momentum : float, default=0.0
            Momentum (SGD only).
        verbose : bool, default=True
            If True, print the training loss.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        p = S.shape[0]
        n = W.shape[0]
        self.n, self.p = n, p

        ddof = 0 if self.bias else (W**2).sum() / n**2
        self.c = p / (n - ddof)

        if tau_init is None:
            tau_init = initial_tau_guess(np.linalg.eigvalsh(S), p)

        output = autodiff_minimization(
            torch.tensor(S),
            n,
            p,
            tau_init,
            torch.sqrt(torch.tensor(W)),
            method=method,
            n_epochs=n_epochs,
            b=b,
            lr=lr,
            momentum=momentum,
            save_all_tau=False,
            verbose=verbose,
        )
        return self._store_fit(output)

    def fit(
        self,
        X,
        W,
        y=None,
        tau_init=None,
        method="Adam",
        n_epochs=100,
        b=1,
        lr=5e-2,
        momentum=0.0,
        verbose=True,
    ):
        """Estimate the population spectrum from data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        W : ndarray of shape (n_samples,)
            Training weight vector.
        y : Ignored
            Not used, present for API consistency by convention.
        tau_init : torch.Tensor, optional
            Initial population spectrum (computed from a weighted
            Ledoit-Wolf-type shrinkage estimator if None).
        method : {"Adam", "SGD"}, default="Adam"
            Optimiser.
        n_epochs : int, default=100
            Number of optimisation steps.
        b : int, default=1
            Batch size of the simulated sample spectra.
        lr : float, default=5e-2
            Learning rate.
        momentum : float, default=0.0
            Momentum (SGD only).
        verbose : bool, default=True
            If True, print the training loss.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(self, X, dtype=np.float64)
        n, p = X.shape
        self.n, self.p = n, p

        if not self.assume_centered:
            self.location_ = X.mean(axis=0)
            X = X - self.location_[None, :]
        else:
            self.location_ = np.zeros(X.shape[1], dtype=X.dtype)

        ddof = 0 if self.bias else (W**2).sum() / n**2
        S = X.T @ (W[:, None] * X) / (n - ddof)
        self.c = p / (n - ddof)

        if tau_init is None:
            S_lwo = wLWO_estimator(X, W / n, assume_centered=self.assume_centered, S_r=None, est=False)
            tau_init = initial_tau_guess(np.linalg.eigvalsh(S_lwo), p)

        output = autodiff_minimization(
            torch.tensor(S),
            n,
            p,
            tau_init,
            torch.sqrt(torch.tensor(W)),
            method=method,
            n_epochs=n_epochs,
            b=b,
            lr=lr,
            momentum=momentum,
            save_all_tau=False,
            verbose=verbose,
        )
        return self._store_fit(output)

    def transform(self, X):
        """Center the data using the fitted ``location_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_centered : ndarray of shape (n_samples, n_features)
        """
        check_is_fitted(self, "location_")
        X = validate_data(self, X, reset=False)

        if not self.assume_centered:
            return X - self.location_
        return X

    def _uniform_weights(self):
        """Weights of the estimated population spectrum (uniform atoms)."""
        n_atoms = self.tau_fit_.shape[0]
        return torch.ones(n_atoms) / n_atoms

    def get_covariance(self, d=None, wd=None, weights="Ndiracs", w_args=None):
        """Return the estimated covariance matrix.

        Parameters
        ----------
        d, wd : torch.Tensor, optional
            Support and probabilities of the weight distribution
            (``weights="Ndiracs"`` and generic case).
        weights : {"ewma", "unif", "Ndiracs"}, default="Ndiracs"
            Family of the sample weights.
        w_args : list, optional
            Parameters of the weight family (``[alpha, Kt]`` for ``"ewma"`` and
            ``"unif"``, ``[Kd, Kt]`` for ``"Ndiracs"``).

        Returns
        -------
        covariance_ : ndarray of shape (n_features, n_features)
        """
        check_is_fitted(self, "model_")
        try:
            check_is_fitted(self, "covariance_")
        except NotFittedError:
            h_lambda = nl_cov_shrinkage(
                self.lambda_,
                self.tau_fit_.detach(),
                self._uniform_weights(),
                d,
                wd,
                c=self.c,
                weights=weights,
                w_args=w_args,
            ).real.numpy()
            self.h_lambda_ = h_lambda
            self.covariance_ = self.U @ np.diag(h_lambda) @ self.U.T
        return self.covariance_

    def get_precision(self, d=None, wd=None, weights="Ndiracs", w_args=None):
        """Return the estimated precision matrix.

        See :meth:`get_covariance` for the parameters.

        Returns
        -------
        precision_ : ndarray of shape (n_features, n_features)
        """
        check_is_fitted(self, "model_")
        try:
            check_is_fitted(self, "precision_")
        except NotFittedError:
            t_lambda = nl_prec_shrinkage(
                self.lambda_,
                self.tau_fit_.detach(),
                self._uniform_weights(),
                d,
                wd,
                c=self.c,
                weights=weights,
                w_args=w_args,
            ).real.numpy()
            self.precision_ = self.U @ np.diag(t_lambda) @ self.U.T
        return self.precision_

    def get_tau(self):
        """Return the estimated population eigenvalues."""
        check_is_fitted(self, "tau_fit_")
        return self.tau_fit_

    def error_norm(self, comp_cov, norm="frobenius"):
        """Compute the error norm between the estimated covariance and a given one.

        Parameters
        ----------
        comp_cov : ndarray of shape (n_features, n_features)
            The covariance to compare against.
        norm : {'frobenius', 'spectral'}, default='frobenius'
            The norm used to compute the error.

        Returns
        -------
        norm_value : float
        """
        diff = self.get_covariance() - comp_cov

        if norm == "frobenius":
            return np.sqrt(np.sum(diff**2))
        if norm == "spectral":
            return np.linalg.norm(diff, ord=2)
        raise ValueError("Unsupported norm type.")
