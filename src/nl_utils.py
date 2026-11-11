"""Numerical tools for the non-linear shrinkage formulas.

This module solves the Marchenko-Pastur-type equations satisfied by the
Stieltjes transform of the sample spectrum for several weight families
(``"ewma"``, ``"unif"``, ``"Ndiracs"``), and computes the asymptotic sample
density together with its derivative with respect to the population spectrum
(:class:`f_xi`).
"""

from functools import partial

import numpy as np
import torch
from scipy.optimize import minimize

from WeSpeR_support_identification_ewma import find_support_exp_u, mld_exp, mld_expp
from WeSpeR_support_identification_Ndiracs import (
    find_support_Ndiracs_u,
    mld_Ndiracs,
    mld_Ndiracsp,
)
from WeSpeR_support_identification_unif import find_support_unif_u, mld_unif, mld_unifp

# A solution of the fixed-point equation is accepted if its squared residual is
# below this tolerance.
RESIDUAL_TOL = 1e-4


def _split_ab(ab):
    """Split a flat numpy vector ``[a_1..a_k, b_1..b_k]`` into two tensors."""
    ab = torch.from_numpy(ab)
    half = ab.shape[0] // 2
    return ab[:half], ab[half:]


def m_tilde_loss_w(ab, x, w, ww, c, t, wt):
    """Squared residual of the equation satisfied by ``m_tilde`` (general weights).

    ``m_tilde = a - i exp(b)`` is parametrised by ``ab = [a, b]``; ``x`` is
    the evaluation point, ``(w, ww)`` the support and probabilities of the
    weight distribution, ``c`` the concentration ratio and ``(t, wt)`` the
    support and probabilities of the population spectrum.
    """
    a, b = _split_ab(ab)
    sq = (t[None, :] * a[:, None] - x[:, None]) ** 2 + t[None, :] ** 2 * torch.exp(b)[:, None] ** 2
    u = 1 + c * w[None, :] * (wt[None, :] * (t[None, :] * a[:, None] - x[:, None]) * t[None, :] / sq).sum(axis=1)[:, None]
    v = c * w[None, :] * (wt[None, :] * t[None, :] ** 2 * torch.exp(b)[:, None] / sq).sum(axis=1)[:, None]
    loss = (a - (w[None, :] * ww[None, :] * u / (u**2 + v**2)).sum(axis=1)) ** 2 + (
        -torch.exp(b) + (w[None, :] * ww[None, :] * v / (u**2 + v**2)).sum(axis=1)
    ) ** 2
    return loss.mean().item()


def X_loss_w(ab, x, w, ww, c, t, wt):
    """Squared residual of the equation satisfied by ``X`` (general weights)."""
    a, b = _split_ab(ab)
    X = a + torch.exp(b) * 1j
    denominator = x[:, None] - c * w[None, :] * (wt[None, :] * t[None, :] / (1 + t[None, :] * X[:, None])).sum(axis=1)[:, None]
    loss = torch.abs(X + (ww[None, :] * w[None, :] / denominator).sum(axis=1)) ** 2
    return loss.mean().item()


def _m_tilde_loss_mld(ab, x, mld, alpha, c, t, wt):
    """Squared residual of the ``m_tilde`` equation for a closed-form weight law."""
    a, b = _split_ab(ab)
    m_tilde = a - torch.exp(b) * 1j
    theta = c * (wt[None, :] * t[None, :] / (t[None, :] * m_tilde[:, None] - x[:, None])).sum(axis=1)
    integral = mld(-1 / theta, alpha) / theta
    return (torch.abs(m_tilde - integral) ** 2).mean().item()


def _X_loss_mld(ab, x, mld, alpha, c, t, wt):
    """Squared residual of the ``X`` equation for a closed-form weight law."""
    a, b = _split_ab(ab)
    X = a + torch.exp(b) * 1j
    theta = c * (wt[None, :] * t[None, :] / (t[None, :] * X[:, None] + 1)).sum(axis=1)
    integral = mld(x / theta, alpha) / theta
    return (torch.abs(X - integral) ** 2).mean().item()


def _solve_m_X(x, t, wt, m_tilde_loss, X_loss, X_start, z_imag):
    """Solve for ``m_tilde`` and ``X`` at each point of ``x`` by minimisation.

    For ``x != 0`` the residual of ``m_tilde`` is minimised (warm-started with
    the previous solution) and ``X = -m_tilde / x``. At ``x == 0`` the
    residual of ``X`` is minimised instead, starting from its solution at
    ``z = i * z_imag``.

    Parameters
    ----------
    x : torch.Tensor of shape (k,)
        Evaluation points.
    t, wt : torch.Tensor
        Support and probabilities of the population spectrum.
    m_tilde_loss, X_loss : callable ``(ab, x) -> float``
        Residuals to minimise, as functions of the flat parameter vector
        ``[a, b]`` and of the evaluation point.
    X_start : ndarray of shape (2,)
        Starting point for the ``X`` equation at ``z = i * z_imag``.
    z_imag : float
        Imaginary part of the point where the ``X`` equation is first solved.

    Returns
    -------
    m : torch.Tensor of shape (k,), complex
        Stieltjes transform of the sample spectrum.
    X : torch.Tensor of shape (k,), complex
        Auxiliary Stieltjes-type transform.
    """
    X = torch.zeros(x.shape, dtype=torch.cfloat)
    m_tilde = torch.zeros(x.shape, dtype=torch.cfloat)
    x_last = np.zeros(2)
    for i in range(x.shape[0]):
        if x[i] != 0:
            res = minimize(m_tilde_loss, x0=x_last, args=(x[i : i + 1],))
            if res.fun < RESIDUAL_TOL:
                ab = torch.from_numpy(res.x)
                x_last = res.x
            else:
                ab = torch.tensor([0, -1e1], dtype=torch.float)
                x_last = np.zeros(2)
            m_tilde[i] = ab[0] - torch.exp(ab[1]) * 1j
            X[i] = -m_tilde[i] / x[i]
        else:
            z = z_imag * 1j * torch.ones(1, dtype=torch.cfloat)
            res = minimize(X_loss, x0=X_start, args=(z,))
            res = minimize(X_loss, x0=res.x[:1], args=(x[i : i + 1],))
            x_last = np.zeros(2)
            m_tilde[i] = 0
            X[i] = torch.from_numpy(res.x)[0]
    m = torch.zeros(x.shape, dtype=torch.cfloat)
    m[m_tilde != 0] = (wt[None, :] / (t[None, :] * m_tilde[:, None] - x[:, None]).to(torch.cfloat)[m_tilde != 0]).sum(axis=1)
    return m, X


def m_X_w(x, w, ww, c, t, wt):
    """Stieltjes transforms ``(m, X)`` of the sample spectrum, general weights.

    Parameters
    ----------
    x : torch.Tensor of shape (k,)
        Evaluation points on the real axis.
    w, ww : torch.Tensor
        Support and probabilities of the weight distribution.
    c : float
        Concentration ratio ``p / n``.
    t, wt : torch.Tensor
        Support and probabilities of the population spectrum.
    """
    return _solve_m_X(
        x,
        t,
        wt,
        partial(m_tilde_loss_w, w=w, ww=ww, c=c, t=t, wt=wt),
        partial(X_loss_w, w=w, ww=ww, c=c, t=t, wt=wt),
        X_start=np.zeros(2),
        z_imag=1e-5,
    )


def m_X_w_ewma(x, alpha, c, t, wt):
    """Stieltjes transforms ``(m, X)`` for exponentially weighted samples.

    Same as :func:`m_X_w` with weights ``W_i ~ exp(-alpha * i / n)``.
    """
    return _solve_m_X(
        x,
        t,
        wt,
        partial(_m_tilde_loss_mld, mld=mld_exp, alpha=alpha, c=c, t=t, wt=wt),
        partial(_X_loss_mld, mld=mld_exp, alpha=alpha, c=c, t=t, wt=wt),
        X_start=np.array([float((wt * t).sum()), 0.0]),
        z_imag=1e-3,
    )


def m_X_w_unif(x, alpha, c, t, wt):
    """Stieltjes transforms ``(m, X)`` for linearly weighted samples.

    Same as :func:`m_X_w` with weights uniformly distributed on
    ``[1 - alpha/2, 1 + alpha/2]``.
    """
    return _solve_m_X(
        x,
        t,
        wt,
        partial(_m_tilde_loss_mld, mld=mld_unif, alpha=alpha, c=c, t=t, wt=wt),
        partial(_X_loss_mld, mld=mld_unif, alpha=alpha, c=c, t=t, wt=wt),
        X_start=np.zeros(2),
        z_imag=1e-5,
    )


def m_X(x, weights, w_args, d, wd, c, t, wt):
    """Dispatch to :func:`m_X_w_ewma`, :func:`m_X_w_unif` or :func:`m_X_w`.

    Parameters
    ----------
    weights : {"ewma", "unif", "Ndiracs"}
        Family of the sample weights; any other value is treated as a general
        discrete weight distribution ``(d, wd)``.
    w_args : list
        Parameters of the weight family; ``w_args[0]`` is ``alpha`` for
        ``"ewma"`` and ``"unif"``.
    d, wd : torch.Tensor
        Support and probabilities of the weight distribution.

    See :func:`m_X_w` for the other parameters.
    """
    if weights == "ewma":
        return m_X_w_ewma(x, w_args[0], c, t, wt)
    if weights == "unif":
        return m_X_w_unif(x, w_args[0], c, t, wt)
    return m_X_w(x, d, wd, c, t, wt)


def WeSpeR_support_u(tau, wtau, d, wd, c, weights, w_args, method=None, verbose=True):
    """Identify the support of the asymptotic sample spectrum.

    Parameters
    ----------
    tau, wtau : ndarray
        Atoms and probabilities of the population spectrum (repeated atoms
        are merged).
    d, wd : ndarray
        Support and probabilities of the weight distribution.
    c : float
        Concentration ratio ``p / n``.
    weights : {"ewma", "unif", "Ndiracs", "test"}
        Family of the sample weights.
    w_args : list
        Parameters of the weight family: ``[alpha, Kt]`` for ``"ewma"`` and
        ``"unif"``, ``[Kd, Kt]`` for ``"Ndiracs"``.
    method : str, optional
        Root-finding method.
    verbose : bool, default=True
        Verbosity of the support identification.

    Returns
    -------
    support, u, mldm, tildeomegai : torch.Tensor
        Edges of the support intervals, the corresponding points ``u``, the
        values of the weight transform at ``u`` and the mass of each interval.
    """
    t = np.unique(tau)
    wt = np.zeros(t.shape)
    for i in range(len(t)):
        wt[i] = wtau[tau == t[i]].sum()

    if weights in ("ewma", "unif"):
        alpha, Kt = w_args[0], w_args[1]
        find_support = find_support_exp_u if weights == "ewma" else find_support_unif_u
        support, u, mldm, tildeomegai = find_support(c, alpha, t, wt, Kt=Kt, eps=1e-1, method=method, verbose=verbose)
    elif weights == "Ndiracs":
        Kd, Kt = w_args[0], w_args[1]
        if wd.shape != d.shape:
            wd = np.ones(d.shape[0]) / d.shape[0]
        support, u, mldm, tildeomegai = find_support_Ndiracs_u(c, wd, d, t, wt, Kt=Kt, Kd=Kd, eps=1e-2, method=method, verbose=verbose)
    elif weights == "test":
        support = np.array([0.1, 10])
        u = np.array([-1 / tau[0] - 1, -1 / tau[-1] + 1])
        mldm = np.array([1 - 1 / (-1 / tau[0] - 1), 1 - 1 / (-1 / tau[-1] + 1)])
        tildeomegai = np.ones(1)
    else:
        raise ValueError(f"Unknown weights family: {weights!r}")

    return torch.from_numpy(support), torch.from_numpy(u), torch.from_numpy(mldm), torch.from_numpy(tildeomegai)


def WeSpeR_density(omega, support, d, wd, c, tau, tildeomegai, mu=0.1, wt=None, weights="test", w_args=None):
    """Compute the asymptotic sample density and cdf on an adaptive grid.

    Parameters
    ----------
    omega : int
        Total number of grid points.
    support : torch.Tensor
        Edges of the support intervals (as returned by :func:`WeSpeR_support_u`).
    d, wd : torch.Tensor
        Support and probabilities of the weight distribution.
    c : float
        Concentration ratio ``p / n``.
    tau : torch.Tensor
        Atoms of the population spectrum.
    tildeomegai : torch.Tensor
        Mass of each support interval.
    mu : float, default=0.1
        Fraction of the grid points spread uniformly over the intervals
        (the rest is allocated proportionally to ``tildeomegai``).
    wt : torch.Tensor, optional
        Probabilities of the atoms of ``tau`` (uniform if None).
    weights : str, default="test"
        Family of the sample weights.
    w_args : list, optional
        Parameters of the weight family.

    Returns
    -------
    xi, f, F, G, X, omegai, i_idx, j_idx, sin2, nu
        Grid, density, cdf, un-normalised cdf, ``X`` on the grid, number of
        points per interval, interval / position indices, ``sin(.)**2``
        grid coordinates and number of intervals.
    """
    if support.shape[0] % 2 != 0:
        support = np.concatenate([support, [tau[-1] * d[-1] * (1 + np.sqrt(c)) ** 2]])

    nu = support.shape[0] // 2

    mu = max(0, min(mu, 1))

    # Number of points per interval
    omegai = (mu / nu + (1 - mu) * tildeomegai) * omega
    if omegai.sum() > 0:
        omegai = torch.round(omegai * omega / omegai.sum() + 0.5).to(int)
    else:
        omegai = torch.round(omega * torch.ones(nu) / nu + 0.5).to(int)

    if wt is None or wt.shape != tau.shape:
        wt = torch.ones(tau.shape[0]) / tau.shape[0]

    # Grid
    grid = torch.zeros(0)
    density = torch.zeros(0)
    X = torch.zeros(0)
    i_idx = torch.zeros(0, dtype=int)
    j_idx = torch.zeros(0, dtype=int)
    sin2 = torch.zeros(0)
    for i in range(nu):
        ul = support[2 * i]
        ur = support[2 * i + 1]
        j = torch.ones(omegai[i]).cumsum(axis=0)
        xij = ul + (ur - ul) * torch.sin(np.pi * j / 2 / (omegai[i] + 1)) ** 2

        m, Xij = m_X(xij, weights, w_args, d, wd, c, tau, wt)
        f = torch.abs(m.imag) / np.pi

        grid = torch.concatenate([grid, ul * torch.ones(1), xij, ur * torch.ones(1)])
        density = torch.concatenate([density, torch.zeros(1), f, torch.zeros(1)])
        X = torch.concatenate([X, torch.zeros(1), Xij, torch.zeros(1)])
        j_idx = torch.concatenate([j_idx, torch.ones(omegai[i] + 2, dtype=int).cumsum(axis=0) - 1])
        i_idx = torch.concatenate([i_idx, 2 * i * torch.ones(omegai[i] + 2, dtype=int)])
        sin2 = torch.concatenate([sin2, torch.zeros(1), torch.sin(np.pi * j / 2 / (omegai[i] + 1)) ** 2, torch.ones(1)])

    xi = grid.to(tau.dtype)
    f = density.to(tau.dtype)

    F0 = max(0, 1 - 1 / c)
    G = torch.zeros(xi.shape, dtype=tau.dtype)
    F = torch.zeros(xi.shape, dtype=tau.dtype)
    G[0] = F0
    F[0] = F0
    for i in range(nu):
        j = torch.ones(omegai[i] + 1, dtype=int).cumsum(0)
        offset = omegai[:i].sum(axis=0) + 2 * i
        if offset > 0:
            G[offset] = F[offset - 1]
            F[offset] = F[offset - 1]
        G[j + offset] = G[offset] + 1 / 2 * ((xi[j + offset] - xi[j - 1 + offset]) * (f[j + offset] + f[j - 1 + offset])).cumsum(axis=0)
        if tildeomegai.sum(axis=0) > 0:
            if (G[offset + j][-1] - F[offset]).to(float) > 0:
                F[j + offset] = F[offset] + tildeomegai[i] * (1 - F[0]) * (G[offset + j] - F[offset]) / (G[offset + j][-1] - F[offset])
            else:
                F[j + offset] = F[offset] + tildeomegai[i] * (1 - F[0]) * j / j[-1]
        else:
            F[j + offset] = G[j + offset]
    if tildeomegai.sum(axis=0) == 0:
        F = G / G[-1]
    return xi, f, F, G, X, omegai, i_idx, j_idx, sin2, nu


class f_xi(torch.autograd.Function):  # noqa: N801
    """Differentiable asymptotic sample density and cdf.

    ``f_xi.apply(tau, wt, d, wd, c, mu, weights, w_args, omega, method, verbose)``
    returns the flat tensor ``[f, F, xi, omegai, nu]`` where ``xi`` is the grid
    of points, ``f`` the density and ``F`` the cdf on that grid, ``omegai`` the
    number of grid points per support interval and ``nu`` the number of
    intervals. The backward pass differentiates the output with respect to
    the atoms ``tau`` of the population spectrum.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        wt=None,
        d=None,
        wd=None,
        c=0.1,
        mu=0.1,
        weights="test",
        w_args=None,
        omega=100,
        method="root",
        verbose=True,
    ):
        """Compute the density and cdf of the sample spectrum.

        Parameters
        ----------
        input : torch.Tensor of shape (p,)
            Atoms ``tau`` of the population spectrum, sorted.
        wt : torch.Tensor, optional
            Probabilities of the atoms (uniform if None).
        d, wd : torch.Tensor, optional
            Support and probabilities of the weight distribution.
        c : float
            Concentration ratio ``p / n``.
        mu : float
            Fraction of the grid points spread uniformly (see
            :func:`WeSpeR_density`).
        weights : str
            Family of the sample weights.
        w_args : list
            Parameters of the weight family.
        omega : int
            Number of grid points.
        method : str
            Root-finding method for the support identification.
        verbose : bool
            Verbosity.
        """
        tau = input

        if d is None:
            d = torch.ones(1)
        if wd is None or wd.shape != d.shape:
            wd = torch.ones(d.shape[0]) / d.shape[0]
        if wt is None:
            wt = torch.ones(tau.shape[0]) / tau.shape[0]

        support, u, mldm, tildeomegai = WeSpeR_support_u(
            tau.detach().numpy(), wt.numpy(), d.numpy(), wd.numpy(), c, weights, w_args, method=method, verbose=verbose
        )

        xi, f, F, G, X, omegai, i_idx, j_idx, sin2, nu = WeSpeR_density(
            omega, support, d, wd, c, tau, tildeomegai, mu=mu, wt=wt, weights=weights, w_args=w_args
        )

        output = torch.concatenate([f, F, xi, omegai, nu * torch.ones(1)])
        ctx.save_for_backward(input, output)
        ctx.wt = wt
        ctx.d = d
        ctx.wd = wd
        ctx.c = c
        ctx.weights = weights
        ctx.w_args = w_args
        ctx.omega = omega
        ctx.u = u
        ctx.mldm = mldm
        ctx.tildeomegai = tildeomegai
        ctx.omegai = omegai
        ctx.i_idx = i_idx
        ctx.j_idx = j_idx
        ctx.sin2 = sin2
        ctx.X = X
        ctx.xi = xi
        ctx.f = f
        ctx.G = G
        ctx.F = F
        ctx.nu = nu
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Differentiate the output of :meth:`forward` with respect to ``tau``."""
        tau, _ = ctx.saved_tensors
        wt = ctx.wt
        d = ctx.d
        wd = ctx.wd
        c = ctx.c
        weights = ctx.weights
        w_args = ctx.w_args
        u = ctx.u
        mldm = ctx.mldm
        omegai = ctx.omegai
        tildeomegai = ctx.tildeomegai
        i_idx = ctx.i_idx
        sin2 = ctx.sin2
        X = ctx.X
        f = ctx.f
        xi = ctx.xi
        nu = ctx.nu
        G = ctx.G
        F = ctx.F

        tu = c * (wt[None, :] * tau[None, :] / (tau[None, :] - u[:, None])).sum(axis=1)
        theta = c * (tau[None, :] * wt[None, :] / (tau[None, :] * X[:, None] + 1)).sum(axis=1)

        if weights == "ewma":
            alpha = w_args[0]
            mldmp = mld_expp(mldm, alpha)
            integral1 = (mld_exp(xi / theta, alpha) + xi / theta * mld_expp(xi / theta, alpha)) / theta**2
            integral3 = mld_expp(xi / theta, alpha) / theta**2

        elif weights == "unif":
            alpha = w_args[0]
            mldmp = mld_unifp(mldm, alpha)
            integral1 = (mld_unif(xi / theta, alpha) + xi / theta * mld_unifp(xi / theta, alpha)) / theta**2
            integral3 = mld_unifp(xi / theta, alpha) / theta**2

        elif weights == "Ndiracs":
            mldmp = mld_Ndiracsp(mldm, wd, d)
            integral1 = (mld_Ndiracs(xi / theta, wd, d) + xi / theta * mld_Ndiracsp(xi / theta, wd, d)) / theta**2
            integral3 = mld_Ndiracsp(xi / theta, wd, d) / theta**2

        else:
            mldmp = (wd[None, :] * d[None, :] / (d[None, :] - mldm[:, None]) ** 2).sum(axis=1)
            integral1 = (wd[None, :] * d[None, :] ** 2 / (xi[:, None] - d[None, :] * theta[None, :]) ** 2).sum(axis=1)
            integral3 = (wd[None, :] * d[None, :] / (xi[:, None] - d[None, :] * theta[None, :]) ** 2).sum(axis=1)

        dsidtk = c * u[:, None] ** 2 * wt[None, :] / (tau[None, :] - u[:, None]) ** 2 * (mldm[:, None] + tu[:, None] / mldmp[:, None])

        dxiidtk = (1 - sin2[:, None]) * dsidtk[i_idx, :] + sin2[:, None] * dsidtk[i_idx + 1, :]

        integral2 = c * (wt[None, :] * tau[None, :] ** 2 / (tau[None, :] * X[:, None] + 1) ** 2).sum(axis=1)
        dhatGammaidtk = c * wt[None, :] / (tau[None, :] * X[:, None] + 1) ** 2 * integral1[:, None]
        dhatGammaidXi = 1 - integral2 * integral1
        dhatGammaidxii = -integral3
        dhatXidtk = -dhatGammaidtk / dhatGammaidXi[:, None]
        dXidxii = -dhatGammaidxii / dhatGammaidXi
        dXidtk = dhatXidtk + dXidxii[:, None] * dxiidtk

        dXidtk[torch.logical_not(torch.isfinite(dXidtk))] = 0

        integral4 = (tau[None, :] * wt[None, :] / (tau[None, :] * X[:, None] + 1) ** 2).sum(axis=1)
        integral5 = (wt[None, :] / (tau[None, :] * X[:, None] + 1)).sum(axis=1)
        dtildefidXi = integral4 / xi
        dtildefidxii = integral5 / xi**2
        dtildefidtk = wt[None, :] * X[:, None] / (tau[None, :] * X[:, None] + 1) ** 2 / xi[:, None]
        dfidtk = (dtildefidXi[:, None] * dXidtk + dtildefidxii[:, None] * dxiidtk + dtildefidtk).imag / np.pi

        dfidtk[torch.logical_not(torch.isfinite(dfidtk))] = 0

        dGidtk = torch.zeros(dxiidtk.shape, dtype=dxiidtk.dtype)
        dFidtk = torch.zeros(dxiidtk.shape, dtype=dxiidtk.dtype)
        for i in range(nu):
            j = torch.ones(omegai[i] + 1, dtype=int).cumsum(0)
            offset = omegai[:i].sum(axis=0) + 2 * i
            dGidtk[j + offset] = 1 / 2 * (
                (dxiidtk[j + offset, :] - dxiidtk[j - 1 + offset, :]) * (f[j + offset, None] + f[j - 1 + offset, None])
            ).cumsum(axis=0) + 1 / 2 * (
                (xi[j + offset, None] - xi[j - 1 + offset, None]) * (dfidtk[j + offset, :] + dfidtk[j - 1 + offset, :])
            ).cumsum(axis=0)
            if tildeomegai.sum(axis=0) > 0:
                if (G[offset + j][-1] - F[offset]).to(float) > 0:
                    dFidtk[j + offset] = (
                        tildeomegai[i] * (1 - F[0]) / (G[offset + j[-1]] - F[offset]) * dGidtk[j + offset, :]
                        - tildeomegai[i]
                        * (1 - F[0])
                        / (G[offset + j[-1]] - F[offset]) ** 2
                        * (G[offset + j, None] - F[offset])
                        * dGidtk[offset + j[-1:], :]
                    )
                else:
                    dFidtk[j + offset] = 0
                dFidtk[j[-1] + offset] = 0
            else:
                dFidtk[j + offset] = dGidtk[j + offset]
        if tildeomegai.sum(axis=0) == 0:
            dFidtk = dGidtk / G[-1] - G[:, None] / G[-1] ** 2 * dGidtk[-1:, :]

        domegaidtk = torch.zeros((omegai.shape[0] + 1, wt.shape[0]))
        doutidtk = torch.concatenate([dfidtk, dFidtk, dxiidtk, domegaidtk], axis=0)

        return (grad_output[:, None] * doutidtk,) + (None,) * 10
