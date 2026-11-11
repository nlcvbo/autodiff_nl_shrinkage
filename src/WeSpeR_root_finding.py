"""Bracketing and root-finding helpers for the support identification.

The ``line_search_*`` functions look for an interval ``[ul, ur]`` inside
``(a, b)`` on which a function ``f`` changes sign; the ``root_brentq_*``
functions then locate the root with Brent's method.
"""

import numpy as np
from scipy.optimize import root_scalar

MAX_ITER = 1e3


def m_idx_tab(K, d, eps=1e-5):
    """Indices (1-based) of the ends of the groups of nearby atoms of ``d``.

    Atoms of the sorted vector ``d`` separated by at most ``eps`` belong to the
    same group. If ``K >= 1``, only the ``K`` largest gaps are kept.

    Parameters
    ----------
    K : int or None
        Maximum number of groups (all groups if None or non-positive).
    d : ndarray of shape (M,)
        Sorted atoms.
    eps : float, default=1e-5
        Minimum gap between two groups.

    Returns
    -------
    list of int
        Position of the last atom of each group (always ending with ``M``).
    """
    M = d.shape[0]
    d_diff = np.diff(d, axis=0)
    if K is None or K <= 0:
        idx = np.ones(d.shape, dtype=int).cumsum(axis=0) - 1
        return [*(idx[:-1][d_diff > eps] + 1).tolist(), M]

    if K >= 2:
        d_idx = np.argsort(d_diff, axis=0)
        return [*(d_idx[-K + 1 :][d_diff[d_idx[-K + 1 :]] > eps] + 1).tolist(), M]
    return [M]


def _first_finite_point(f, a, b, eps, direction):
    """Move from the midpoint of ``(a, b)`` until ``f`` is finite.

    The point moves geometrically (ratio ``1 + eps``), according to
    ``direction``: ``"from_a"`` towards ``a``, ``"from_b"`` towards ``b``,
    ``"away_from_a"`` away from ``a`` or ``"away_from_b"`` away from ``b``.
    """
    um = (a + b) / 2
    n_iter = 0
    while not np.isfinite(f(um)) and n_iter < MAX_ITER:
        if direction == "from_b":
            um = b - (b - um) / (1 + eps)
        elif direction == "from_a":
            um = a + (um - a) / (1 + eps)
        elif direction == "away_from_a":
            um = a + (um - a) * (1 + eps)
        else:  # "away_from_b"
            um = b - (b - um) * (1 + eps)
        n_iter += 1
    return um


def line_search_p(f, a, b, eps=1e-1):
    """Bracket a root of ``f`` on ``(a, b)``.

    Starting from the midpoint, ``ul`` is moved towards ``a`` while ``f(ul) < 0``
    and ``ur`` towards ``b`` while ``f(ur) > 0``, with geometric steps ``eps``.

    Returns
    -------
    ur, ul : float
        Right and left ends of the bracketing interval.
    success : bool
        False if the maximum number of iterations was reached.
    """
    n_iter = 0
    ur = (a + b) / 2
    ul = (a + b) / 2
    while f(ul) < 0 and n_iter < MAX_ITER:
        ul = a + (ul - a) / (1 + eps)
        n_iter += 1

    while f(ur) > 0 and n_iter < MAX_ITER:
        ur = b - (b - ur) / (1 + eps)
        n_iter += 1

    return ur, ul, n_iter < MAX_ITER


def line_search_m(f, a, b, eps=1e-1):
    """Bracket a root of ``f`` on ``(a, b)``, for ``f`` positive then negative.

    The starting point is first moved towards ``a`` until ``f`` is finite, then
    ``ul`` is moved towards ``a`` while ``f(ul) > 0`` and ``ur`` towards ``b``
    while ``f(ur) < 0``. Returns ``(ur, ul, success)`` as :func:`line_search_p`.
    """
    um = _first_finite_point(f, a, b, eps, "from_a")

    n_iter1 = 0
    n_iter2 = 0
    ur = um
    ul = um
    while f(ul) > 0 and n_iter1 < MAX_ITER:
        ul = a + (ul - a) / (1 + eps)
        n_iter1 += 1

    while f(ur) < 0 and n_iter2 < MAX_ITER:
        ur = b - (b - ur) / (1 + eps)
        n_iter2 += 1
    return ur, ul, (n_iter1 < MAX_ITER) and (n_iter2 < MAX_ITER)


def line_search_r(f, a, b, eps=1e-1):
    """Bracket a root of ``f`` located to the right of ``a``.

    ``ul`` is moved towards ``a`` while ``f(ul) > 0`` and ``ur`` away from
    ``a`` while ``f(ur) < 0``. Returns ``(ur, ul, success)`` as
    :func:`line_search_p`.
    """
    um = _first_finite_point(f, a, b, eps, "away_from_a")

    ur = um
    ul = um

    n_iter1 = 0
    while f(ul) > 0 and n_iter1 < MAX_ITER:
        ul = a + (ul - a) / (1 + eps)
        n_iter1 += 1

    n_iter2 = 0
    while f(ur) < 0 and n_iter2 < MAX_ITER:
        ur = a + (ur - a) * (1 + eps)
        n_iter2 += 1

    return ur, ul, (n_iter1 < MAX_ITER) and (n_iter2 < MAX_ITER)


def line_search_l(f, a, b, eps=1e-1):
    """Bracket a root of ``f`` located to the left of ``b``.

    ``ul`` is moved away from ``b`` while ``f(ul) < 0`` and ``ur`` towards
    ``b`` while ``f(ur) > 0``. Returns ``(ur, ul, success)`` as
    :func:`line_search_p`.
    """
    um = _first_finite_point(f, a, b, eps, "away_from_b")

    ur = um
    ul = um

    n_iter1 = 0
    while f(ul) < 0 and n_iter1 < MAX_ITER:
        ul = b - (b - ul) * (1 + eps)
        n_iter1 += 1

    n_iter2 = 0
    while f(ur) > 0 and n_iter2 < MAX_ITER:
        ur = b - (b - ur) / (1 + eps)
        n_iter2 += 1

    return ur, ul, (n_iter1 < MAX_ITER) and (n_iter2 < MAX_ITER)


def _brentq(f, line_search, a, b, eps):
    """Bracket a root with ``line_search`` and refine it with Brent's method.

    Raises
    ------
    ValueError
        If the bracketing interval does not contain a sign change.
    """
    ur, ul, _ = line_search(f, a, b, eps)
    return root_scalar(f, bracket=[ul, ur], method="brentq").root


def root_brentq_p(f, a, b, eps):
    """Root of ``f`` on ``(a, b)`` bracketed by :func:`line_search_p`."""
    return _brentq(f, line_search_p, a, b, eps)


def root_brentq_m(f, a, b, eps):
    """Root of ``f`` on ``(a, b)`` bracketed by :func:`line_search_m`."""
    return _brentq(f, line_search_m, a, b, eps)


def root_brentq_r(f, a, b, eps):
    """Root of ``f`` on ``(a, b)`` bracketed by :func:`line_search_r`."""
    return _brentq(f, line_search_r, a, b, eps)


def root_brentq_l(f, a, b, eps):
    """Root of ``f`` on ``(a, b)`` bracketed by :func:`line_search_l`."""
    return _brentq(f, line_search_l, a, b, eps)


def root_brentq_lm(f, a, b, eps):
    """Root of ``-f`` on ``(a, b)`` bracketed by :func:`line_search_l`."""
    return _brentq(lambda x: -f(x), line_search_l, a, b, eps)


def root_brentq_rm(f, a, b, eps):
    """Root of ``-f`` on ``(a, b)`` bracketed by :func:`line_search_r`."""
    return _brentq(lambda x: -f(x), line_search_r, a, b, eps)
