"""Support identification of the sample spectrum for linearly weighted samples.

The weights are uniformly distributed on ``[1 - alpha / 2, 1 + alpha / 2]``.
"""

import numpy as np
from scipy.optimize import root_scalar

from WeSpeR_root_finding import m_idx_tab, root_brentq_l, root_brentq_m, root_brentq_p, root_brentq_r


def mld_unif(x, alpha):
    """Transform ``E[W / (W - x)]`` of the limiting weight distribution."""
    return 1 + x / alpha * np.log(1 + alpha / (1 - alpha / 2 - x))


def mld_unifp(x, alpha):
    """First derivative of :func:`mld_unif`."""
    return np.log(1 + alpha / (1 - alpha / 2 - x)) / alpha + x / ((1 - alpha / 2 - x) ** 2 + alpha * (1 - alpha / 2 - x))


def mld_unifpp(x, alpha):
    """Second derivative of :func:`mld_unif`."""
    return (
        2 / ((1 - alpha / 2 - x) ** 2 + alpha * (1 - alpha / 2 - x))
        - 2 * x * (x - 1) / ((1 - alpha / 2 - x) ** 2 + alpha * (1 - alpha / 2 - x)) ** 2
    )


def _solve_mldm(g1, alpha):
    """Solve ``mld_unif(m, alpha) = g1`` for ``m`` with Newton's method.

    The starting point is the solution for two Diracs at ``1 -/+ alpha / 2``.
    """
    alpha_d = alpha / 2
    m_init = 1 - 1 / 2 / g1 - np.sqrt(1 + 4 * g1 * (g1 - 1) * alpha_d**2) / 2 / g1
    return root_scalar(
        lambda x: mld_unif(x, alpha) - g1,
        x0=m_init,
        fprime=lambda x: mld_unifp(x, alpha),
        method="newton",
    ).root


def x_unif(w, t, alpha, c):
    """Return the function ``u -> x_k(u)`` whose critical values are the edges of the support.

    With ``return_m=True`` the returned function also returns ``m(u)``.
    """

    def xk(u, return_m=False):
        g1 = c * (w * t / (t - u)).sum()
        m = _solve_mldm(g1, alpha)

        rst = -u * g1 * m
        if return_m:
            return rst, m
        return rst

    return xk


def x_unif_p(w, t, alpha, c):
    """Return the first derivative of ``u -> x_k(u)``."""

    def xkp(u):
        g1 = c * (w * t / (t - u)).sum()
        m = _solve_mldm(g1, alpha)

        tw = c * (w * t / (t - u)).sum()
        twp = c * (w * t / (t - u) ** 2).sum()
        v = m
        vp = twp / mld_unifp(m, alpha)

        rst = -tw * v - u * twp * v - u * tw * vp
        return rst

    return xkp


def x_unif_pp(w, t, alpha, c):
    """Return the second derivative of ``u -> x_k(u)``."""

    def xkpp(u):
        g1 = c * (w * t / (t - u)).sum()
        m = _solve_mldm(g1, alpha)

        tw = c * (w * t / (t - u)).sum()
        twp = c * (w * t / (t - u) ** 2).sum()
        twpp = 2 * c * (w * t / (t - u) ** 3).sum()
        v = m
        vp = twp / mld_unifp(m, alpha)
        vpp = twpp / mld_unifp(m, alpha) - twp**2 * mld_unifpp(m, alpha) / mld_unifp(m, alpha) ** 3

        rst = -twp * v - tw * vp - twp * v - u * twpp * v - u * twp * vp - tw * vp - u * twp * vp - u * tw * vpp
        return rst

    return xkpp


def tw(u, w, t, d, wd, c):
    """Return ``t(u) = c * sum_j w_j t_j / (t_j - u)`` (``w``: probabilities of the atoms ``t``)."""
    return c * (w * t / (t - u)).sum()


def twp(u, w, t, d, wd, c):
    """Return the first derivative of :func:`tw` with respect to ``u``."""
    return c * (w * t / (t - u) ** 2).sum()


def twpp(u, w, t, d, wd, c):
    """Return the second derivative of :func:`tw` with respect to ``u``."""
    return 2 * c * (w * t / (t - u) ** 3).sum()


def twppp(u, w, t, d, wd, c):
    """Return the third derivative of :func:`tw` with respect to ``u``."""
    return 6 * c * (w * t / (t - u) ** 4).sum()


def v(u, k, w, t, d, wd, c, m):
    """Return ``m``, the value of the weight transform at the solution of ``mld(m) = t(u)``."""
    return m


def vp(u, k, w, t, d, wd, c, m):
    """Return the first derivative of ``m`` with respect to ``u``."""
    return twp(u, w, t, d, wd, c) / (wd * d / (d - m) ** 2).sum()


def vpp(u, k, w, t, d, wd, c, m):
    """Return the second derivative of ``m`` with respect to ``u``."""
    return (
        twpp(u, w, t, d, wd, c) / (wd * d / (d - m) ** 2).sum()
        - twp(u, w, t, d, wd, c) ** 2 * 2 * (wd * d / (d - m) ** 3).sum() / (wd * d / (d - m) ** 2).sum() ** 3
    )


def vppp(u, k, w, t, d, wd, c, m):  # see p.135
    """Return the third derivative of ``m`` with respect to ``u`` (see p.135 of the thesis)."""
    return (
        twppp(u, w, t, d, wd, c) / (wd * d / (d - m) ** 2).sum()
        - 3 * twp(u, w, t, d, wd, c) * twpp(u, w, t, d, wd, c) * 2 * (wd * d / (d - m) ** 3).sum() / (wd * d / (d - m) ** 2).sum() ** 3
        + 3 * twp(u, w, t, d, wd, c) ** 3 * 2 * (wd * d / (d - m) ** 3).sum() ** 2 / (wd * d / (d - m) ** 2).sum() ** 5
        - twp(u, w, t, d, wd, c) ** 3 * 6 * (wd * d / (d - m) ** 4).sum() / (wd * d / (d - m) ** 2).sum() ** 4
    )


def find_support_unif(c, alpha, t, w, Kt=-1, eps=1e-1, method=None, verbose=False):
    """Return the edges of the support of the sample spectrum.

    Parameters
    ----------
    c : float
        Concentration ratio ``p / n``.
    alpha : float
        Spread of the weights, in ``[0, 2]``.
    t, w : ndarray
        Atoms of the population spectrum (sorted) and their probabilities.
    Kt : int, default=-1
        Number of largest gaps of ``t`` considered (all if non-positive).
    eps : float
        Geometric step of the bracketing line searches.
    method : str, optional
        Unused, kept for API consistency.
    verbose : bool, default=False
        If True, print the intermediate critical points.

    Returns
    -------
    support : ndarray
        Sorted edges ``[l_1, r_1, l_2, r_2, ...]`` of the support intervals.
    """
    # eps >= 1e-1 necessary
    support = []

    t_idx = m_idx_tab(Kt, t)
    t_idx = [i - 1 for i in t_idx]

    for i in t_idx[:-1]:
        xkpp = x_unif_pp(w, t, alpha, c)
        xkp = x_unif_p(w, t, alpha, c)
        xk = x_unif(w, t, alpha, c)

        try:
            u0 = root_brentq_p(xkpp, t[i], t[i + 1], eps)
            if xkp(u0) == 0:
                support += [xk(u0), xk(u0)]
                if verbose:
                    print(u0, xk(u0), xkp(u0))
            elif xkp(u0) > 0:
                try:
                    u1 = root_brentq_m(xkp, t[i], u0, eps)
                    u2 = root_brentq_p(xkp, u0, t[i + 1], eps)
                    support += [xk(u1), xk(u2)]
                    if verbose:
                        print(u1, xk(u1), xkp(u1))
                        print(u2, xk(u2), xkp(u2))
                except ValueError:
                    if verbose:
                        print("Error", t[i], t[i + 1])
        except ValueError:
            if verbose:
                print("Error:", t[i], t[i + 1])
        # otherwise, no spectral gap

    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M if c<1
    if c < 1:
        xkp = x_unif_p(w, t, alpha, c)
        xk = x_unif(w, t, alpha, c)
        try:
            u0 = root_brentq_p(xkp, 0, t[0], eps)
            xk0 = xk(u0)
            support += [0, xk0]
            if verbose:
                print(u0, xk(u0), xkp(u0))
        except ValueError:
            if verbose:
                print("Error", 0, t[0])
    else:
        xkpp = x_unif_pp(w, t, alpha, c)
        xkp = x_unif_p(w, t, alpha, c)
        xk = x_unif(w, t, alpha, c)

        try:
            u0 = root_brentq_l(xkp, 0, t[0], eps)
            xk0 = xk(u0)
            support += [0, xk0]
            if verbose:
                print(u0, xk(u0), xkp(u0))
        except ValueError:
            if verbose:
                print("Error:", -np.inf, t[0])

    # for the right exterior intervals, ]tau_p,infty[, for k == M
    xkpp = x_unif_pp(w, t, alpha, c)
    xkp = x_unif_p(w, t, alpha, c)
    xk = x_unif(w, t, alpha, c)

    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1] + 1, eps)
        support += [xk(u0)]
        if verbose:
            print(u0, xk(u0), xkp(u0))
    except ValueError:
        if verbose:
            print("Error:", t[-1], np.inf)

    support.sort()
    support = support[:-1]
    if support.shape[0] % 2 != 0:
        support = np.concatenate([support, [t[-1] * (1 + alpha / 2) * (1 + np.sqrt(c)) ** 2]])

    return support


def find_support_unif_u(c, alpha, t, w, Kt=-1, eps=1e-1, method=None, verbose=False):
    """Return the support of the sample spectrum with the matching points ``u``.

    Parameters
    ----------
    c : float
        Concentration ratio ``p / n``.
    alpha : float
        Spread of the weights, in ``[0, 2]``.
    t, w : ndarray
        Atoms of the population spectrum (sorted) and their probabilities.
    Kt : int, default=-1
        Number of largest gaps of ``t`` considered (all if non-positive).
    eps : float
        Geometric step of the bracketing line searches.
    method : str, optional
        Unused, kept for API consistency.
    verbose : bool, default=False
        If True, print the intermediate critical points.

    Returns
    -------
    support : ndarray
        Sorted edges of the support intervals.
    u : ndarray
        Points ``u`` at which each edge is attained.
    mldm : ndarray
        Value of ``m`` at each ``u``.
    tildeomegai : ndarray
        Mass of each support interval.
    """
    # eps >= 1e-1 necessary
    support = []
    u = []
    mldm = []
    tildeomegai = []

    t_idx = m_idx_tab(Kt, t, 1e-5)
    t_idx = [i - 1 for i in t_idx]

    for i in t_idx[:-1]:
        xkpp = x_unif_pp(w, t, alpha, c)
        xkp = x_unif_p(w, t, alpha, c)
        xk = x_unif(w, t, alpha, c)

        try:
            u0 = root_brentq_p(xkpp, t[i], t[i + 1], eps)
            if xkp(u0) == 0:
                xk0, m0 = xk(u0, return_m=True)
                support += [xk0, xk0]
                u += [u0, u0]
                mldm += [m0, m0]
                if verbose:
                    print(u0, xk(u0), xkp(u0))
            elif xkp(u0) > 0:
                try:
                    u1 = root_brentq_m(xkp, t[i], u0, eps)
                    u2 = root_brentq_p(xkp, u0, t[i + 1], eps)
                    xk1, m1 = xk(u1, return_m=True)
                    xk2, m2 = xk(u2, return_m=True)
                    support += [xk1, xk2]
                    u += [u1, u2]
                    mldm += [m1, m2]
                    if verbose:
                        print(u1, xk(u1), xkp(u1))
                        print(u2, xk(u2), xkp(u2))
                except ValueError:
                    if verbose:
                        print("Error", t[i], t[i + 1])
        except ValueError:
            if verbose:
                print("Error:", t[i], t[i + 1])
        # otherwise, no spectral gap

    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M if c<1
    if c < 1:
        xkp = x_unif_p(w, t, alpha, c)
        xk = x_unif(w, t, alpha, c)
        try:
            u0 = root_brentq_p(xkp, 0, t[0], eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            if verbose:
                print(u0, xk(u0), xkp(u0))
        except ValueError:
            if verbose:
                print("Error", 0, t[0])
    else:
        xkpp = x_unif_pp(w, t, alpha, c)
        xkp = x_unif_p(w, t, alpha, c)
        xk = x_unif(w, t, alpha, c)

        try:
            u0 = root_brentq_l(xkp, 0, t[0], eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            if verbose:
                print(u0, xk(u0), xkp(u0))
        except ValueError:
            if verbose:
                print("Error:", -np.inf, t[0])

    # for the right exterior intervals, ]tau_p,infty[, for k == M
    xkpp = x_unif_pp(w, t, alpha, c)
    xkp = x_unif_p(w, t, alpha, c)
    xk = x_unif(w, t, alpha, c)

    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1] + 1, eps)
        xk0, m0 = xk(u0, return_m=True)
        support += [xk0]
        u += [u0]
        mldm += [m0]
        if verbose:
            print(u0, xk(u0), xkp(u0))
    except ValueError:
        if verbose:
            print("Error:", t[-1], np.inf)

    support, u, mldm = map(list, zip(*sorted(zip(support, u, mldm, strict=True)), strict=True))

    support = np.array(support[1:])
    u = np.array(u[1:])
    mldm = np.array(mldm[1:])

    if support.shape[0] % 2 != 0:
        support = np.concatenate([support, [t[-1] * (1 + alpha / 2) * (1 + np.sqrt(c)) ** 2]])
        u = np.concatenate([u, [t[-1] + 1]])
        mldm = np.concatenate([mldm, [(1 + alpha / 2) + 1]])

        nu = support.shape[0] // 2
        tildeomegai = np.ones(nu) / nu
        return support, u, mldm, tildeomegai

    nu = support.shape[0] // 2
    tildeomegai = np.ones(nu)
    for i in range(nu):
        tildeomegai[i] = w[(t > u[2 * i]) & (t < u[2 * i + 1])].sum()
    return support, u, mldm, tildeomegai
