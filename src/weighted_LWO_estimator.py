"""Weighted Ledoit-Wolf-type linear shrinkage estimators.

The estimators shrink the weighted sample covariance matrix towards a
reference matrix (by default a multiple of the identity). Data are passed as
``(n_samples, n_features)`` arrays together with the sample weights ``w``
(summing to one).
"""

import numpy as np


def _reference_matrix(S_r, p):
    """Return the shrinkage target, normalised to ``||S_r||_F^2 = p``.

    ``None`` or an all-zero matrix select the identity.
    """
    if S_r is None or not np.any(S_r):
        S_r = np.eye(p)
    return S_r / (np.linalg.norm(S_r, ord="fro") / np.sqrt(p))


def wLWO_estimator_oracle(X, w, cov, assume_centered=False, S_r=None, est=False):
    """Oracle linear shrinkage, using the true covariance ``cov``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data.
    w : ndarray of shape (n_samples,)
        Sample weights.
    cov : ndarray of shape (n_features, n_features)
        True covariance matrix.
    assume_centered : bool, default=False
        If True, the data are not centered.
    S_r : ndarray, optional
        Shrinkage target (identity by default).
    est : bool, default=False
        If True, also return the shrinkage coefficients.

    Returns
    -------
    S_opt : ndarray of shape (n_features, n_features)
        Oracle estimator ``a * S + b * S_r``.
    coefficients : ndarray of shape (2,)
        ``[a, b]`` (only if ``est``).
    """
    X = X.T
    p = X.shape[0]

    W = np.diag(w)

    if not assume_centered:
        X_mean = (X @ W).sum(axis=1)[:, None] / W.sum()
        X = X - X_mean
        S = X @ W @ X.T / (1 - (W**2).sum())
    else:
        S = X @ W @ X.T

    S_r = _reference_matrix(S_r, p)

    mu = np.trace(S @ S_r.T) / p
    S2 = np.linalg.norm(S, ord="fro") ** 2
    delta2_opt = S2 / p - mu**2
    alpha2_opt = np.trace(S @ cov.T) / p - mu * np.trace(cov @ S_r.T) / p
    beta2_opt = -np.trace(S @ cov.T) / p * np.trace(S @ S_r.T) / p + S2 / p * np.trace(S_r @ cov.T) / p
    S_opt = alpha2_opt / delta2_opt * S + beta2_opt / delta2_opt * S_r

    if est:
        return S_opt, np.array([alpha2_opt / delta2_opt, beta2_opt / delta2_opt])
    return S_opt


def wLWO_estimator(X, w, assume_centered=False, S_r=None, est=False):
    """Weighted Ledoit-Wolf-type linear shrinkage estimator of the covariance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data.
    w : ndarray of shape (n_samples,)
        Sample weights.
    assume_centered : bool, default=False
        If True, the mean is known to be zero; otherwise it is estimated.
    S_r : ndarray, optional
        Shrinkage target (identity by default).
    est : bool, default=False
        If True, also return the shrinkage coefficients.

    Returns
    -------
    S_star : ndarray of shape (n_features, n_features)
        Shrunk covariance ``shrinkage * mu * S_r + (1 - shrinkage) * S``.
    coefficients : ndarray of shape (2,)
        ``[1 - shrinkage, shrinkage * mu]`` (only if ``est``).
    """
    X = X.T
    if assume_centered:
        return wLWO_estimator_known_mean(X, w, S_r, est)
    return wLWO_estimator_unknown_mean(X, w, S_r, est)


def wLWO_estimator_known_mean(X, w, S_r=None, est=False):
    """Linear shrinkage when the mean is known to be zero.

    Note that ``X`` has shape ``(n_features, n_samples)``. See
    :func:`wLWO_estimator` for the other parameters and the return values.
    """
    p = X.shape[0]
    W = np.diag(w)
    S = X @ W @ X.T

    S_r = _reference_matrix(S_r, p)

    mu = np.trace(S @ S_r.T) / p
    delta2 = np.linalg.norm(S - mu * S_r, ord="fro") ** 2 / p
    S2 = np.linalg.norm(S, ord="fro") ** 2 / p
    beta2 = ((X**2).sum(axis=0) ** 2 * w**2).sum() / p - S2 * (w**2).sum()
    beta2 = beta2 / (1 - (w**2).sum())
    beta2 = min(beta2, delta2)

    shrinkage = beta2 / delta2
    if est:
        c = np.array([1 - shrinkage, shrinkage * mu])
        return shrinkage * mu * S_r + (1 - shrinkage) * S, c
    return shrinkage * mu * S_r + (1 - shrinkage) * S


def wLWO_estimator_unknown_mean(X, w, S_r=None, est=False):
    """Linear shrinkage with a weighted estimate of the mean.

    Note that ``X`` has shape ``(n_features, n_samples)``. See
    :func:`wLWO_estimator` for the other parameters and the return values.
    """
    p = X.shape[0]
    W = np.diag(w)
    X_mean = (X @ W).sum(axis=1)[:, None] / W.sum()
    X = X - X_mean
    S = X @ W @ X.T / (1 - (W**2).sum())

    S_r = _reference_matrix(S_r, p)

    C = 1 - (W**2).sum()

    C11 = (w**2 * (1 - w) ** 4 - w**6).sum() + (w**2).sum() * (w**4).sum()
    C22 = (
        (w**2).sum() ** 3
        - 3 * (w**2).sum() * (w**4).sum()
        + 2 * (w**6).sum()
        + (w**2).sum() * (2 * w**2 * (1 - w) ** 2).sum()
        - 2 * (w**4 * (1 - w) ** 2).sum()
    )
    C33 = (
        (w**2).sum() * (4 * w**2 * (1 - w) ** 2).sum()
        - (4 * w**4 * (1 - w) ** 2).sum()
        + 2 * (w**2).sum() ** 3
        - 6 * (w**2).sum() * (w**4).sum()
        + 4 * (w**6).sum()
    )

    q1 = 2 * (w**2 * (1 - w)).sum() ** 2 - 2 * (w**4 * (1 - w) ** 2).sum()
    q2 = (
        (w**2).sum() ** 2
        - 4 * (w**3).sum() * (w**2).sum()
        - (w**4).sum()
        + 4 * (w**5).sum()
        + 2 * (w**3).sum() ** 2
        - ((w**2).sum() ** 2 * w**2 - 4 * w**4 * (w**2).sum() - w**2 * (w**4).sum() + 6 * w**6).sum()
    )
    q3 = (
        -4 * (w**2).sum() * (w**2 * (1 - w)).sum()
        + 4 * (w**4 * (1 - w)).sum()
        + 4 * (w**3).sum() * (w**2 * (1 - w)).sum()
        + (w**2).sum() * (4 * w**3 * (1 - w)).sum()
        - 8 * (w**5 * (1 - w)).sum()
    )
    q4 = (
        2 * (w * (1 - 2 * w) ** 2 * ((w**2).sum() - w**2)).sum()
        - 2 * (w**3 * (1 - 2 * w) ** 2).sum()
        - 2 * (w**2 * (1 - 2 * w) ** 2 * ((w**2).sum() - 2 * w**2)).sum()
    )

    D11 = (
        2 * (w**3 * (1 - w) ** 2).sum()
        - 2 * (w**4 * (1 - w) ** 2).sum()
        + (w**4).sum()
        - 2 * (w**5).sum()
        - (w**2).sum() * (w**4).sum()
        + 2 * (w**6).sum()
    )
    D22 = q1 + q2 + q3
    D33 = (
        2 * q2
        + q4
        + (w**3).sum() ** 2
        + 2 * (w**2 * (1 - w)).sum() ** 2
        + (w * (1 - w) ** 2).sum() ** 2
        - (w**2 * (w**2 + (1 - w) ** 2) ** 2).sum()
    )

    E11 = (
        2 * (w**3 * (1 - w) ** 2).sum()
        + (w**4).sum()
        - 2 * (w**5).sum()
        - 2 * (w**4 * (1 - w) ** 2).sum()
        - (w**2).sum() * (w**4).sum()
        + 2 * (w**6).sum()
    )
    E22 = (
        q2
        + (w * (1 - w) ** 2).sum() ** 2
        + (w**3).sum() ** 2
        + 2 * (w * ((1 - w) ** 2 + w**2) * ((w**2).sum() - w**2 - (w**3).sum())).sum()
        - (w**2 * (1 - w) ** 4 + w**6 + 2 * w**2 * (w**2 + (1 - w) ** 2) * ((w**2).sum() - 2 * w**2)).sum()
    )
    E33 = (
        2 * q2
        + 4 * (w**2 * (1 - w)).sum() ** 2
        - 4 * (w**4 * (1 - w) ** 2).sum()
        - 8 * (w**2 * (1 - w) * ((w**2).sum() - w**2 - (w**3).sum())).sum()
        + 8 * (w**3 * (1 - w) * ((w**2).sum() - 2 * w**2)).sum()
    )

    P = np.array([[C11, C11 + D11, C11 + E11], [C22, C22 + D22, C22 + E22], [C33, C33 + D33, C33 + E33]]) / C**2
    g = np.array([0, 0, -1])[:, None]
    h = np.linalg.pinv(P) @ g

    a = h[0]
    b = h[1] + 1
    c = h[2]

    X2 = ((X**2).sum(axis=0) ** 2 * w**2 / C**2).sum()
    S2 = np.linalg.norm(S, ord="fro") ** 2
    tS = np.trace(S)

    mu = np.trace(S @ S_r.T) / p
    delta2 = np.linalg.norm(S - mu * S_r, ord="fro") ** 2
    beta2 = a * X2 + b * S2 + c * tS**2
    beta2 = max(beta2, 0)
    beta2 = min(beta2, delta2)

    shrinkage = beta2 / delta2
    if est:
        c = np.array([1 - shrinkage, shrinkage * mu])
        return shrinkage * mu * S_r + (1 - shrinkage) * S, c
    return shrinkage * mu * S_r + (1 - shrinkage) * S
