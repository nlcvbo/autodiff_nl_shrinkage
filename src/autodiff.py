import numpy as np
import scipy as sp
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from autodiff_nl_shrinkage.src.utils import WeSpeR_LD_model, WeSpeR_LD_minimization
from nl_formulas import nl_cov_shrinkage, nl_prec_shrinkage
from weighted_LWO_estimator import wLWO_estimator

class WeSpeR_LD(BaseEstimator, TransformerMixin):
    """
    A Scikit-Learn-compatible covariance estimator template.
    
    Parameters
    ----------
    bias : bool, default=False
        If True, normalize by N. If False, normalize by N-1.
    assume_centered : bool, default=False
        If True, data will not be centered before computing covariance.
    """

    def __init__(self, bias=True, assume_centered=False):
        self.bias = bias
        self.assume_centered = assume_centered

    def fit_from_F(self, lambda_, U, W, c = None, p_tau = 1000, y=None, tau_init = None, method = "Adam", n_epochs = 100, b = 1, assume_centered = False, lr = 5e-2, momentum=0., verbose = True):
        """
        Estimate the covariance matrix from sample covariance.

        Parameters
        ----------
        S : array-like of shape (n_features, n_features)
            Training sample covariance.
        
        W : array-like of shape (n_samples)
            Training weight vector.

        y : Ignored
            Not used, present for API consistency by convention.

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
        n_tau = int(p_tau/c) 
        self.n_tau = n_tau

        ddof = 0 if self.bias else (W**2).sum()/n**2
        if c is None:
            self.c = p/(n-ddof)
        else:
            self.c = c

        if tau_init is None:
            e_S = lambda_
            if self.c >= 1:
                tau_init = np.unique(np.around(e_S,6))
                tau_add = np.linspace(np.min(e_S), np.max(e_S), e_S.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_S.mean()
            else:
                tau_init = np.unique(np.around(e_S,6))
                tau_add = np.linspace(np.min(e_S), np.max(e_S), e_S.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_S.mean()
            tau_init = np.sort(np.random.choice(tau_init, p_tau))
            tau_init = torch.tensor(tau_init)

        lambda_tau = np.sort(np.random.choice(lambda_, p_tau))
        W_tau = np.sort(np.random.choice(W, n_tau))

        model, tau_fit, loss, _, _, _ = WeSpeR_LD_minimization(torch.tensor(np.diag(lambda_tau)), n_tau, p_tau, tau_init, torch.sqrt(torch.tensor(W_tau)), method = method, n_epochs = n_epochs, b = b, lr = lr, momentum = momentum, save_all_tau = False, verbose = verbose)

        self.model_ = model
        self.tau_fit_ = tau_fit
        self.loss = loss
        self.lambda_ = lambda_
        self.U = U
        return self

    def fit_from_S(self, S, W, y=None, tau_init = None, method = "Adam", n_epochs = 100, b = 1, assume_centered = False, lr = 5e-2, momentum=0., verbose = True):
        """
        Estimate the covariance matrix from sample covariance.

        Parameters
        ----------
        S : array-like of shape (n_features, n_features)
            Training sample covariance.
        
        W : array-like of shape (n_samples)
            Training weight vector.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        p = S.shape[0]
        n = W.shape[0]
        self.n, self.p = n, p

        ddof = 0 if self.bias else (W**2).sum()/n**2
        self.c = p/(n-ddof)

        if tau_init is None:
            e_S = np.linalg.eigvalsh(S)
        
            if self.c >= 1:
                tau_init = np.unique(np.around(e_S,6))
                tau_add = np.linspace(np.min(e_S), np.max(e_S), e_S.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_S.mean()
            else:
                tau_init = np.unique(np.around(e_S,6))
                tau_add = np.linspace(np.min(e_S), np.max(e_S), e_S.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_S.mean()
            tau_init = np.sort(np.random.choice(tau_init, p))
            tau_init = torch.tensor(tau_init)

        model, tau_fit, loss, _, lambda_, U = WeSpeR_LD_minimization(torch.tensor(S), n, p, tau_init, torch.sqrt(torch.tensor(W)), method = method, n_epochs = n_epochs, b = b, lr = lr, momentum = momentum, save_all_tau = False, verbose = verbose)

        self.model_ = model
        self.tau_fit_ = tau_fit
        self.loss = loss
        self.lambda_ = lambda_
        self.U = U
        return self

    def fit(self, X, W, y=None, tau_init = None, method = "Adam", n_epochs = 100, b = 1, assume_centered = False, lr = 5e-2, momentum=0., verbose = True):
        """
        Estimate the covariance matrix from data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        W : array-like of shape (n_samples)
            Training weight vector.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(X, dtype=np.float64)
        n, p = X.shape
        self.n, self.p = n, p

        if not self.assume_centered:
            self.location_ = X.mean(axis=0)
            X = X - self.location_[None,:]
        else:
            self.location_ = np.zeros(X.shape[1], dtype=X.dtype)

        ddof = 0 if self.bias else (W**2).sum()/n**2
        S = X.T @  (W[:, None] * X)/(n-ddof)
        self.c = p/(n-ddof)

        if tau_init is None:
            SLWO = wLWO_estimator(X, W/n, assume_centered = assume_centered, S_r = None, est = False)
            e_SLWO = np.linalg.eigvalsh(SLWO)
        
            if self.c >= 1:
                tau_init = np.unique(np.around(e_SLWO,6))
                tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
            else:
                tau_init = np.unique(np.around(e_SLWO,6))
                tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
            tau_init = np.sort(np.random.choice(tau_init, p))
            tau_init = torch.tensor(tau_init)

        model, tau_fit, loss, _, lambda_, U = WeSpeR_LD_minimization(torch.tensor(S), n, p, tau_init, torch.sqrt(torch.tensor(W)), method = method, n_epochs = n_epochs, b = b, lr = lr, momentum = momentum, save_all_tau = False, verbose = verbose)

        self.model_ = model
        self.tau_fit_ = tau_fit
        self.loss = loss
        self.lambda_ = lambda_
        self.U = U
        return self

    def transform(self, X):
        """
        Optionally center the data using the location_.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_centered : ndarray
        """
        check_is_fitted(self, "location_")
        X = self._validate_data(X, reset=False)

        if not self.assume_centered:
            return X - self.location_
        return X

    def get_covariance(self, d = None, wd = None, weights = 'Ndiracs', w_args = [], method = 'root', verbose = False):
        """Return the estimated covariance matrix."""
        check_is_fitted(self, "model_")
        try:
            check_is_fitted(self, "covariance_")
        except NotFittedError:
            h_lambda = nl_cov_shrinkage(self.lambda_, self.tau_fit_.detach(), torch.ones(self.tau_fit_.shape[0])/self.tau_fit_.shape[0], d, wd, c = self.c, weights = weights, w_args = w_args, method = method, verbose = False).real.numpy()
            self.h_lambda_ = h_lambda
            self.covariance_ = self.U @ np.diag(h_lambda) @ self.U.T
        return self.covariance_
    
    def get_precision(self, d = None, wd = None, weights = 'Ndiracs', w_args = [], method = 'root', verbose = False):
        """Return the estimated precision matrix."""
        check_is_fitted(self, "model_")
        try:
            check_is_fitted(self, "precision_")
        except NotFittedError:            
            t_lambda = nl_prec_shrinkage(self.lambda_, self.tau_fit_.detach(), torch.ones(self.tau_fit_.shape[0])/self.tau_fit_.shape[0], d, wd, c = self.c, weights = weights, w_args = w_args, method = method, verbose = False).real.numpy()
            self.precision_ = self.U @ np.diag(t_lambda) @ self.U.T
        return self.precision_
    
    def get_tau(self):
        """Return the estimated population eigenvalues."""
        check_is_fitted(self, "tau_fit_")
        return self.tau_fit_

    def error_norm(self, comp_cov, norm='frobenius'):
        """
        Compute the error norm between estimated covariance and a given one.

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
        est_cov = self.get_covariance()
        diff = est_cov - comp_cov

        if norm == 'frobenius':
            return np.sqrt(np.sum(diff ** 2))
        elif norm == 'spectral':
            return np.linalg.norm(diff, ord=2)
        else:
            raise ValueError("Unsupported norm type.")
