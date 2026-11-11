"""Support identification of the sample spectrum for discrete weight distributions.

The sample weights follow a discrete law with atoms ``d`` and probabilities
``wd``; the population spectrum has atoms ``t`` and probabilities ``w``.
"""

import numpy as np
from numpy.polynomial import polynomial as poly

from WeSpeR_root_finding import m_idx_tab, root_brentq_l, root_brentq_lm, root_brentq_m, root_brentq_p, root_brentq_r


def mld_Ndiracs(x, wd, d):
    """Transform ``sum_j wd_j d_j / (d_j - x)`` of the weight distribution.

    ``x`` is a 1D or 2D array; the sum is taken over the atoms ``d``.
    """
    if len(x.shape) == 1:
        return (wd[None, :] * d[None, :] / (d[None, :] - x[:, None])).sum(axis=1)
    else:
        return (wd[None, None, :] * d[None, None, :] / (d[None, None, :] - x[:, :, None])).sum(axis=2)


def mld_Ndiracsp(x, wd, d):
    """First derivative of :func:`mld_Ndiracs`."""
    if len(x.shape) == 1:
        return (wd[None, :] * d[None, :] / (d[None, :] - x[:, None]) ** 2).sum(axis=1)
    else:
        return (wd[None, None, :] * d[None, None, :] / (d[None, None, :] - x[:, :, None]) ** 2).sum(axis=2)


def mld_Ndiracspp(x, wd, d):
    """Second derivative of :func:`mld_Ndiracs`."""
    if len(x.shape) == 1:
        return 2 * (wd[None, :] * d[None, :] / (d[None, :] - x[:, None]) ** 3).sum(axis=1)
    else:
        return 2 * (wd[None, None, :] * d[None, None, :] / (d[None, None, :] - x[:, :, None]) ** 3).sum(axis=2)


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


def mldm_root(u, k, w, t, d, wd, c, P, Q, eps=1e-2, verbose=False):
    """Solve ``mld_Ndiracs(m) = t(u)`` for ``m`` in the ``k``-th interval between atoms of ``d``, by root finding.

    Returns ``inf`` if no root is found.
    """
    g1 = tw(u, w, t, d, wd, c)
    if not np.isfinite(g1):
        if k < d.shape[0]:
            m = d[k - 1] if g1 > 0 else d[k]
        elif g1 >= 0:
            m = d[0]
        else:
            m = d[-1]
        return m

    def f(x):
        return mld_Ndiracs(x * np.ones(1), wd, d)[0] - g1

    try:
        if k < d.shape[0]:
            ul = d[k - 1]
            ur = d[k]
            m = root_brentq_m(f, ul, ur, eps)
        elif g1 >= 0:
            ul = d[0] - 1
            ur = d[0]
            m = root_brentq_lm(f, ul, ur, eps)
        else:
            ul = d[-1]
            ur = d[-1] + 1
            m = root_brentq_r(f, ul, ur, eps)

    except Exception:
        if verbose:
            print("No roots found, u=", u, ", t(u)=", g1)
        m = np.inf

    return m


def mldm_poly(u, k, w, t, d, wd, c, P, Q, verbose=False):
    """Solve ``mld_Ndiracs(m) = t(u)`` for ``m`` through the roots of a polynomial.

    Returns ``inf`` if no root is found.
    """
    g1 = tw(u, w, t, d, wd, c)
    if np.isfinite(g1):
        R = poly.polyadd(P, poly.polymul(poly.Polynomial(-g1), Q))[0]
    else:
        idx = np.argmin(np.abs(u - t))
        R = poly.polymul(poly.Polynomial(-1), Q)[0] if t[idx] - u > 0 else Q  # g1 > 0 in the first case
    try:
        roots = np.sort(R.roots().real)
        if roots[0] < 0:
            roots = np.concatenate([roots[1:], roots[:1]])
        m = roots[k - 1]

    except Exception:
        if verbose:
            print("No roots found, t(u)=", g1)
        m = np.inf

    return m


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


def x_Ndiracs(k, w, t, d, wd, c, P, Q, mldm):
    """Return the function ``u -> x_k(u)`` whose critical values are the edges of the support.

    ``k`` indexes the group of weight atoms, ``(P, Q)`` are the polynomials used by
    the ``"poly"`` method and ``mldm`` is the solver of the weight-transform
    equation. With ``return_m=True`` the returned function also returns ``m(u)``.
    """

    def xk(u, return_m=False):
        m = mldm(u, k, w, t, d, wd, c, P, Q)
        rst = -u * tw(u, w, t, d, wd, c) * v(u, k, w, t, d, wd, c, m)
        if return_m:
            return rst, m
        return rst

    return xk


def x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldm):
    """Return the first derivative of ``u -> x_k(u)``. See :func:`x_Ndiracs`."""

    def xkp(u):
        m = mldm(u, k, w, t, d, wd, c, P, Q)
        return (
            -tw(u, w, t, d, wd, c) * v(u, k, w, t, d, wd, c, m)
            - u * twp(u, w, t, d, wd, c) * v(u, k, w, t, d, wd, c, m)
            - u * tw(u, w, t, d, wd, c) * vp(u, k, w, t, d, wd, c, m)
        )

    return xkp


def x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldm):
    """Return the second derivative of ``u -> x_k(u)``. See :func:`x_Ndiracs`."""

    def xkpp(u):
        m = mldm(u, k, w, t, d, wd, c, P, Q)
        return (
            -twp(u, w, t, d, wd, c) * v(u, k, w, t, d, wd, c, m)
            - tw(u, w, t, d, wd, c) * vp(u, k, w, t, d, wd, c, m)
            - twp(u, w, t, d, wd, c) * v(u, k, w, t, d, wd, c, m)
            - u * twpp(u, w, t, d, wd, c) * v(u, k, w, t, d, wd, c, m)
            - u * twp(u, w, t, d, wd, c) * vp(u, k, w, t, d, wd, c, m)
            - tw(u, w, t, d, wd, c) * vp(u, k, w, t, d, wd, c, m)
            - u * twp(u, w, t, d, wd, c) * vp(u, k, w, t, d, wd, c, m)
            - u * tw(u, w, t, d, wd, c) * vpp(u, k, w, t, d, wd, c, m)
        )

    return xkpp


def x_Ndiracs_ppp(k, w, t, d, wd, c, P, Q, mldm):
    """Return the third derivative of ``u -> x_k(u)``. See :func:`x_Ndiracs`."""

    def xkpp(u):
        m = mldm(u, k, w, t, d, wd, c, P, Q)
        return (
            -3 * twpp(u, w, t, d, wd, c) * v(u, k, w, t, d, wd, c, m)
            - 6 * twp(u, w, t, d, wd, c) * vp(u, k, w, t, d, wd, c, m)
            - 3 * tw(u, w, t, d, wd, c) * vpp(u, k, w, t, d, wd, c, m)
            - 3 * u * twpp(u, w, t, d, wd, c) * vp(u, k, w, t, d, wd, c, m)
            - 3 * u * twp(u, w, t, d, wd, c) * vpp(u, k, w, t, d, wd, c, m)
            - u * twppp(u, w, t, d, wd, c) * v(u, k, w, t, d, wd, c, m)
            - u * tw(u, w, t, d, wd, c) * vppp(u, k, w, t, d, wd, c, m)
        )

    return xkpp


def xX_Ndiracs(k, w, t, d, wd, c, P, Q, mldm):
    """Return ``X -> x_k(-1 / X)``. See :func:`x_Ndiracs`."""

    # X \mapsto x_Ndiracs(k, w, t, d, wd, c, P, Q)(-1/X)
    def xk(X, return_m=False):
        return x_Ndiracs(k, w, t, d, wd, c, P, Q, mldm)(-1 / X, return_m=return_m)

    return xk


def xX_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldm):
    """Return the first derivative of ``X -> x_k(-1 / X)``. See :func:`x_Ndiracs`."""

    def xkp(X):
        return x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldm)(-1 / X) / X**2

    return xkp


def xX_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldm):
    """Return the second derivative of ``X -> x_k(-1 / X)``. See :func:`x_Ndiracs`."""

    def xkpp(X):
        return x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldm)(-1 / X) / X**4 - 2 / X**3 * x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldm)(-1 / X)

    return xkpp


def find_support_Ndiracs(c, wd, d, t, w, eps=1e-2, Kd=-1, Kt=-1, method="root", verbose=False):
    """Return the edges of the support of the sample spectrum.

    On each interval ``]tau_i, tau_{i+1}[`` the zero of ``x_k''`` is found (for each
    ``k``); if ``x_k'`` is non-negative there, the zeros of ``x_k'`` are searched on
    the left and right sub-intervals. On ``]-inf, tau_1[`` and ``]tau_p, inf[`` only
    the zero of ``x_M'`` is searched. When ``p`` or ``M`` are large, only the
    largest gaps of ``t`` (``Kt``) and ``d`` (``Kd``) are considered.

    Parameters
    ----------
    c : float
        Concentration ratio ``p / n``.
    wd, d : ndarray
        Probabilities and atoms of the weight distribution.
    t, w : ndarray
        Atoms of the population spectrum (sorted) and their probabilities.
    eps : float
        Geometric step of the bracketing line searches.
    Kd, Kt : int, default=-1
        Number of largest gaps of ``d`` / ``t`` considered (all if non-positive).
    method : {"root", "poly"}, default="root"
        Way of solving the weight-transform equation.
    verbose : bool, default=False
        If True, print the intermediate critical points.

    Returns
    -------
    support : ndarray
        Sorted edges ``[l_1, r_1, l_2, r_2, ...]`` of the support intervals.
    """
    M = d.shape[0]
    support = []

    m_idx = m_idx_tab(Kd, d)
    t_idx = m_idx_tab(Kt, t)
    t_idx = [i - 1 for i in t_idx]

    if method == "poly":
        P = poly.Polynomial(0)
        for i in range(M):
            R = poly.Polynomial(wd[i] * d[i])
            for j in range(M):
                if j != i:
                    R = poly.polymul(R, poly.Polynomial((d[j], -1)))
            P = poly.polyadd(P, R)[0]

        Q = poly.Polynomial(1)
        for i in range(M):
            Q = poly.polymul(Q, poly.Polynomial((d[i], -1)))[0]
        mldmf = mldm_poly

    elif method == "root":
        P, Q = None, None
        mldmf = mldm_root

    else:
        P, Q = None, None
        mldmf = mldm_root

    for i in t_idx[:-1]:
        for k in m_idx:
            xkpp = x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
            xkp = x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
            xk = x_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)

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
                    print("Error:", k, t[i], t[i + 1])
            # otherwise, no spectral gap

    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for all k != M
    for k in m_idx[:-1]:
        xkpp = x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
        xkp = x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)

        try:
            u0 = root_brentq_p(xkp, 0, t[0], eps)
            support += [0, xk(u0)]
            if verbose:
                print(u0, xk(u0), xkp(u0))
        except ValueError:
            if verbose:
                print("Error", k, 0, t[0])

    # for the right exterior intervals, ]tau_p, infty[, for all k != M, consider X = -1/u
    for k in m_idx[:-1]:
        xkppX = xX_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
        xkpX = xX_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
        xkX = xX_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)

        try:
            X0 = root_brentq_p(xkppX, -1 / t[-1], 0, eps)
            if xkpX(X0) == 0:
                support += [xkX(X0), xkX(X0)]
                if verbose:
                    print(X0, xkX(X0), xkX(X0))
            elif xkpX(X0) > 0:
                try:
                    X1 = root_brentq_m(xkpX, -1 / t[-1], X0, eps)
                    X2 = root_brentq_p(xkpX, X0, 0, eps)
                    support += [xkX(X1), xkX(X2)]
                    if verbose:
                        print(-1 / X1, -1 / X2, xkX(X1), xkX(X2))
                except ValueError:
                    if verbose:
                        print("Error", t[i], t[i + 1])
        except ValueError:
            if verbose:
                print("Error:", t[i], t[i + 1])

    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M if c<1
    if c < 1:
        xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)
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
        xkpp = x_Ndiracs_pp(M, w, t, d, wd, c, P, Q, mldmf)
        xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)

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
    xkpp = x_Ndiracs_pp(M, w, t, d, wd, c, P, Q, mldmf)
    xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
    xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)

    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1] + 1, eps)
        support += [xk(u0)]
        if verbose:
            print(u0, xk(u0), xkp(u0))
    except ValueError:
        if verbose:
            print("Error:", M, t[-1], np.inf, eps, xkp(t[-1] * (1 + eps)), xkp(2 * (t[-1] * (1 + eps))))

    support.sort()
    return support[1:]


def find_support_Ndiracs_u(c, wd, d, t, w, eps=1e-2, Kd=-1, Kt=-1, method="root", verbose=False):
    """Return the support of the sample spectrum with the matching points ``u``.

    On each interval ``]tau_i, tau_{i+1}[`` the zero of ``x_k''`` is found (for each
    ``k``); if ``x_k'`` is non-negative there, the zeros of ``x_k'`` are searched on
    the left and right sub-intervals. On ``]-inf, tau_1[`` and ``]tau_p, inf[`` only
    the zero of ``x_M'`` is searched. When ``p`` or ``M`` are large, only the
    largest gaps of ``t`` (``Kt``) and ``d`` (``Kd``) are considered.

    Parameters
    ----------
    c : float
        Concentration ratio ``p / n``.
    wd, d : ndarray
        Probabilities and atoms of the weight distribution.
    t, w : ndarray
        Atoms of the population spectrum (sorted) and their probabilities.
    eps : float
        Geometric step of the bracketing line searches.
    Kd, Kt : int, default=-1
        Number of largest gaps of ``d`` / ``t`` considered (all if non-positive).
    method : {"root", "poly"}, default="root"
        Way of solving the weight-transform equation.
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
    M = d.shape[0]
    support = []
    u = []
    mldm = []
    m = []
    tildeomegai = []

    m_idx = m_idx_tab(Kd, d, 1e-5)
    t_idx = m_idx_tab(Kt, t, 1e-5)
    t_idx = [i - 1 for i in t_idx]

    if method == "poly":
        P = poly.Polynomial(0)
        for i in range(M):
            R = poly.Polynomial(wd[i] * d[i])
            for j in range(M):
                if j != i:
                    R = poly.polymul(R, poly.Polynomial((d[j], -1)))
            P = poly.polyadd(P, R)[0]

        Q = poly.Polynomial(1)
        for i in range(M):
            Q = poly.polymul(Q, poly.Polynomial((d[i], -1)))[0]
        mldmf = mldm_poly

    elif method == "root":
        P, Q = None, None
        mldmf = mldm_root

    else:
        P, Q = None, None
        mldmf = mldm_root

    for i in t_idx[:-1]:
        for k in m_idx:
            xkpp = x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
            xkp = x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
            xk = x_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)

            try:
                u0 = root_brentq_p(xkpp, t[i], t[i + 1], eps)
                if xkp(u0) == 0:
                    xk0, m0 = xk(u0, return_m=True)
                    support += [xk0, xk0]
                    u += [u0, u0]
                    mldm += [m0, m0]
                    m += [k, k]
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
                        m += [k, k]
                        if verbose:
                            print(u1, xk(u1), xkp(u1))
                            print(u2, xk(u2), xkp(u2))
                    except ValueError:
                        if verbose:
                            print("Error", t[i], t[i + 1])
            except ValueError:
                if verbose:
                    print("Error:", k, t[i], t[i + 1])
            # otherwise, no spectral gap

    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for all k != M
    for k in m_idx[:-1]:
        xkpp = x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
        xkp = x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)

        try:
            u0 = root_brentq_p(xkp, 0, t[0], eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            m += [k, k]
            if verbose:
                print(u0, xk(u0), xkp(u0))
        except ValueError:
            if verbose:
                print("Error", k, 0, t[0])

    # for the right exterior intervals, ]tau_p, infty[, for all k != M, consider X = -1/u
    for k in m_idx[:-1]:
        xkppX = xX_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
        xkpX = xX_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
        xkX = xX_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)

        try:
            X0 = root_brentq_p(xkppX, -1 / t[-1], 0, eps)
            if xkpX(X0) == 0:
                xk0, m0 = xkX(X0, return_m=True)
                u0 = -1 / X0
                support += [xk0, xk0]
                u += [u0, u0]
                mldm += [m0, m0]
                m += [k, k]
                if verbose:
                    print(X0, xkX(X0), xkX(X0))
            elif xkpX(X0) > 0:
                try:
                    X1 = root_brentq_m(xkpX, -1 / t[-1], X0, eps)
                    X2 = root_brentq_p(xkpX, X0, 0, eps)
                    xk1, m1 = xkX(X1, return_m=True)
                    xk2, m2 = xkX(X2, return_m=True)
                    support += [xk1, xk2]
                    u += [-1 / X1, -1 / X2]
                    mldm += [m1, m2]
                    if verbose:
                        print(-1 / X1, -1 / X2, xkX(X1), xkX(X2))
                except ValueError:
                    if verbose:
                        print("Error", t[i], t[i + 1])
        except ValueError:
            if verbose:
                print("Error:", t[i], t[i + 1])

    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M if c<1
    if c < 1:
        xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)
        try:
            u0 = root_brentq_p(xkp, 0, t[0], eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            m += [M, M]
            if verbose:
                print(u0, xk(u0), xkp(u0))
        except ValueError:
            if verbose:
                print("Error", 0, t[0])
    else:
        xkpp = x_Ndiracs_pp(M, w, t, d, wd, c, P, Q, mldm)
        xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldm)
        xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldm)

        try:
            u0 = root_brentq_l(xkp, 0, t[0], eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            m += [M, M]
            if verbose:
                print(u0, xk(u0), xkp(u0))
        except ValueError:
            if verbose:
                print("Error:", -np.inf, t[0])

    # for the right exterior intervals, ]tau_p,infty[, for k == M
    xkpp = x_Ndiracs_pp(M, w, t, d, wd, c, P, Q, mldmf)
    xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
    xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)

    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1] + 1, eps)
        xk0, m0 = xk(u0, return_m=True)
        support += [xk0]
        u += [u0]
        mldm += [m0]
        m += [M]
        if verbose:
            print(u0, xk(u0), xkp(u0), tw(u0, w, t, d, wd, c), m0)

    except ValueError:
        if verbose:
            print("Error:", M, t[-1], np.inf, eps, xkp(t[-1] * (1 + eps)), xkp(2 * (t[-1] * (1 + eps))))

    support, u, mldm, m = map(list, zip(*sorted(zip(support, u, mldm, m, strict=True)), strict=True))

    support = np.array(support[1:])
    u = np.array(u[1:])
    mldm = np.array(mldm[1:])
    m = np.array(m[1:]).astype(int)

    if support.shape[0] % 2 != 0:
        print("Support failed")
        support = np.concatenate([support, [t[-1] * d[-1] * (1 + np.sqrt(c)) ** 2]])
        u = np.concatenate([u, [t[-1] + 1]])
        mldm = np.concatenate([mldm, [d[-1] + 1]])
        m = np.concatenate([m, [M]])

        nu = support.shape[0] // 2
        tildeomegai = np.ones(nu) / nu
        return support, u, mldm, tildeomegai

    nu = support.shape[0] // 2
    tildeomegai = np.ones(nu) / nu
    for i in range(nu):
        m1, m2 = m[2 * i] - 1, m[2 * i + 1] - 1
        if m1 == M and mldm[2 * i] <= d[0]:
            m1 = 0
        if m2 == M and mldm[2 * i + 1] <= d[0]:
            m2 = 0
        tildeomegai[i] = w[(t > u[2 * i]) & (t < u[2 * i + 1])].sum() * wd[m1 : m2 + 1].sum()
    if tildeomegai.sum() > 0:
        tildeomegai = tildeomegai / tildeomegai.sum()
    return support, u, mldm, tildeomegai
