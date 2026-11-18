"""
Edge-safe, high-order numerical differentiation of uniformly sampled series
(e.g. GRACE(-FO) residual range -> range-rate -> range-acceleration).

Method
------
Finite-difference weights from Fornberg's algorithm (Fornberg 1988, Math.
Comp. 51, 699-706) on an n_points stencil:
  * interior : centred stencil, accuracy order n_points-1 (1st deriv.) /
               n_points-1 (2nd deriv., n_points odd)
  * edges    : the SAME number of points, shifted to stay inside the data
               (one-sided / off-centre). No samples are lost, and the edge
               points keep the same polynomial exactness as the interior,
               unlike np.gradient's 1st/2nd-order edge formulas.
  * gaps     : if t is given, the series is split at gaps (or at NaNs) and
               each segment is differentiated separately, so no stencil spans
               a gap.
The second derivative uses a dedicated 2nd-derivative stencil rather than
applying the first derivative twice (which widens the stencil and lowers
accuracy).

The interior operator is a linear, zero-phase (antisymmetric / symmetric) FIR
filter, so its transfer function is known exactly: see fd_response().
"""

import numpy as np


def fornberg_weights(z, x, m):
    """
    Weights c[j, k] such that the k-th derivative at z is approx
    sum_j c[j, k] f(x[j]), for k = 0..m (Fornberg 1988).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    c = np.zeros((n, m + 1))
    c1, c4 = 1.0, x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def _diff_segment(y, dt, deriv, n_points, edge):
    N = y.size
    p = min(n_points, N if N % 2 else N - 1)  # shrink for short segments
    if p < deriv + 1:
        return np.full(N, np.nan)
    h = p // 2
    out = np.empty(N)

    # interior: centred stencil as a correlation
    w = fornberg_weights(0.0, np.arange(-h, h + 1), deriv)[:, deriv]
    if N > 2 * h:
        out[h:N - h] = np.correlate(y, w, mode="valid")

    if edge == "nan":
        out[:h] = np.nan
        out[N - h:] = np.nan
        return out / dt ** deriv
    # edges: same-width stencil shifted inside the data
    for i in list(range(min(h, N))) + list(range(max(N - h, h), N)):
        j0 = min(max(i - h, 0), N - p)
        nodes = np.arange(j0, j0 + p) - i
        wi = fornberg_weights(0.0, nodes, deriv)[:, deriv]
        out[i] = wi @ y[j0:j0 + p]
    return out / dt ** deriv


def derivative(y, dt, deriv=1, n_points=9, t=None, gap_factor=1.5,
               edge="shift"):
    """
    Edge-safe high-order derivative of a uniformly sampled series.

    Parameters
    ----------
    y          : 1-D array (column vectors are accepted)
    dt         : sampling interval [s]
    deriv      : 1 or 2 (any order < n_points works)
    n_points   : stencil width (odd). 9 -> 8th-order accurate interior.
                 Larger = closer to the ideal i*omega up to higher frequency,
                 but more high-frequency noise passes through.
    t          : optional time tags; the series is split where diff(t) >
                 gap_factor*dt. NaNs in y also split the series.
    edge       : "shift" (default) - same-width off-centre stencils at the
                 n_points//2 samples next to every segment end; exact for
                 smooth data but amplifies white noise there (~10x for
                 n_points=9).
                 "nan" - return NaN at those samples instead.
    Returns
    -------
    1-D array of the same length as y (NaN where y is NaN).
    """
    y = np.asarray(y, dtype=float).ravel()
    if edge not in ("shift", "nan"):
        raise ValueError("edge must be 'shift' or 'nan'")
    if n_points % 2 != 1:
        raise ValueError("n_points must be odd")
    brk = ~np.isfinite(y)
    if t is not None:
        t = np.asarray(t, dtype=float).ravel()
        gap = np.diff(t) > gap_factor * dt
    else:
        gap = np.zeros(y.size - 1, bool)

    out = np.full(y.size, np.nan)
    start = 0
    for i in range(y.size + 1):
        end_here = (i == y.size) or brk[i] or (i > 0 and gap[i - 1])
        if end_here:
            seg_end = i
            if seg_end > start and not brk[start]:
                out[start:seg_end] = _diff_segment(y[start:seg_end], dt,
                                                   deriv, n_points, edge)
            start = i + 1 if (i < y.size and brk[i]) else i
    return out


def fd_response(f, dt, deriv=1, n_points=9):
    """
    Complex frequency response of the interior (centred) stencil,
    H(f) = sum_j w_j exp(i 2 pi f j dt) / dt^deriv, to compare with the
    ideal (i 2 pi f)^deriv. Use this in an analytical transfer function.
    """
    h = n_points // 2
    j = np.arange(-h, h + 1)
    w = fornberg_weights(0.0, j, deriv)[:, deriv]
    f = np.atleast_1d(np.asarray(f, dtype=float))
    return np.exp(2j * np.pi * np.outer(f, j) * dt) @ w / dt ** deriv


def remove_1cpr(t, y, period=5640.0, n_rev=2, drift=False, linear_cpr=False,
                gap_factor=1.5):
    """
    Remove bias, drift and 1-CPR terms by least squares, arc by arc.

    Per arc of `n_rev` orbital revolutions (arcs also break at data gaps and
    NaNs) the model
        a0 + a1*tau + c*cos(w tau) + s*sin(w tau)   [+ tau*(c2 cos + s2 sin)]
    is fitted and subtracted, with w = 2 pi / period and tau = time from the
    arc midpoint. Over whole revolutions these terms are (nearly) orthogonal
    to 2 CPR and higher, so those are left essentially untouched.

    Parameters
    ----------
    t, y       : time [s] and signal (same length; columns are accepted)
    period     : orbital period [s] (~5640 s for GRACE; better: from the orbit)
    n_rev      : arc length in revolutions. 2 (default) keeps >99% of 2 CPR;
                 1 rev with drift=True removes ~20% of 2 CPR.
    drift      : include a linear trend per arc (costs some 2 CPR on short arcs)
    linear_cpr : allow the 1-CPR amplitude to vary linearly within an arc
    Returns
    -------
    y_clean, y_model : arrays of the same length as y (NaN stays NaN)
    """
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    if t.shape != y.shape:
        raise ValueError(f"t {t.shape} and y {y.shape} must have equal length")
    dt = np.median(np.diff(t))
    w = 2 * np.pi / period
    arc_len = n_rev * period

    # arc boundaries: gaps, NaNs, and every n_rev revolutions
    ok = np.isfinite(y)
    new = np.r_[True, np.diff(t) > gap_factor * dt]
    new |= np.r_[True, np.diff(np.floor((t - t[0]) / arc_len)) != 0]
    arc_id = np.cumsum(new)

    model = np.full_like(y, np.nan)
    n_par = 2 + drift + 2 * linear_cpr
    for a in np.unique(arc_id):
        sel = (arc_id == a) & ok
        if sel.sum() < 2 * n_par:  # too short to fit reliably
            continue
        tau = t[sel] - t[sel].mean()
        cols = [np.ones_like(tau)]
        if drift:
            cols.append(tau)
        cols += [np.cos(w * tau), np.sin(w * tau)]
        if linear_cpr:
            cols += [tau * np.cos(w * tau), tau * np.sin(w * tau)]
        A = np.column_stack(cols)
        coef = np.linalg.lstsq(A, y[sel], rcond=None)[0]
        model[sel] = A @ coef

    return y - model, model