"""Helpers specific to the spike experiments."""

import numpy as np
from scipy.optimize import brentq

from heavy_utils import T_ATOMS, WT_ATOMS
from support.ewma import mld_exp


def outlier_location(tau_s, alpha, c, t=T_ATOMS, wt=WT_ATOMS):
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
