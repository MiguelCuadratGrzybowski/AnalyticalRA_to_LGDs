"""
Utility_functions_TimeFilt_new.py

This module adds a robust, arc-aware preprocessing layer for filtering + differentiation:
  - Split the time series into coarse "arcs" (default: 1 day)
  - Within each arc, split into "pieces" by gaps (tol_gap seconds)
  - Discard pieces that are too short for the requested filtering/differentiation sequence
    (minimum size derived from edge/padding requirements)
  - Apply a simple robust outlier detector (N * sigma_robust with MAD; default N=7)
    and fill short NaN runs by linear interpolation on the regular grid
  - Run the requested filter/diff sequence piecewise and concatenate results

All original functions are kept untouched by importing them from Utility_functions_TimeFilt.
New/updated functions are provided with a `_new` suffix (or clearly named helpers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd

from scipy.signal import butter, bessel, cheby2, sosfiltfilt
from scipy import ndimage

# Import *all* original functions unchanged
from .Utility_functions_TimeFilt import (  # noqa: F401
    get_time_pieces,
    interpolate_array_to_size,
    _robust_dt_from_x,
    cdtaylor,
    fdtaylor,
    bdtaylor,
    _padlen_for_sos,
)

# ---------------------------------------------------------------------
# New helpers: arc splitting, piece filtering, robust outliers
# ---------------------------------------------------------------------

def split_time_into_arcs_new(t: np.ndarray, arc_seconds: float = 86400.0) -> List[Tuple[int, int]]:
    """
    Split a monotonic increasing time array into coarse arcs of fixed duration.

    Parameters
    ----------
    t : ndarray
        Time stamps in seconds (preferably sorted ascending).
    arc_seconds : float
        Arc duration in seconds. Default: 86400 (1 day).

    Returns
    -------
    list of (start_idx, end_idx) inclusive
    """
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return []
    if t.size == 1:
        return [(0, 0)]
    if arc_seconds <= 0:
        return [(0, t.size - 1)]

    arcs: List[Tuple[int, int]] = []
    start = 0
    t0 = t[0]
    # Assign each sample to an arc index based on (t - t0) // arc_seconds
    arc_id = np.floor((t - t0) / arc_seconds).astype(int)
    # arc boundaries occur where arc_id changes
    change = np.where(np.diff(arc_id) != 0)[0]
    if change.size == 0:
        return [(0, t.size - 1)]
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change, [t.size - 1]))
    for s, e in zip(starts, ends):
        arcs.append((int(s), int(e)))
    return arcs


def _stable_interp_grid_new(x: np.ndarray, dx: float) -> np.ndarray:
    """
    Build a regular grid from x[0] to x[-1] with step dx using integer count
    to avoid float overshoot.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return x.copy()
    if dx <= 0:
        raise ValueError("dx must be > 0")

    x0 = float(x[0])
    x1 = float(x[-1])
    span = x1 - x0
    if not np.isfinite(span) or span < 0:
        return x.copy()

    n = int(np.floor(span / dx)) + 1
    if n < 2:
        return x.copy()
    return x0 + dx * np.arange(n, dtype=np.float64)


def interpolate_array_new(y_array, x_array, dx: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updated interpolation (kept separate from the original).

    Fixes:
      - correct type checks
      - stable grid construction (no overshoot)
      - reverses if strictly decreasing

    Returns
    -------
    (y_int, x_int)
    """
    # Convert inputs
    if not isinstance(y_array, (np.ndarray, np.generic)):
        try:
            y_array = y_array.to_numpy()
        except Exception:
            y_array = np.asarray(y_array)
    if not isinstance(x_array, (np.ndarray, np.generic)):
        try:
            x_array = x_array.to_numpy()
        except Exception:
            x_array = np.asarray(x_array)

    y = np.asarray(y_array, dtype=np.float64)
    x = np.asarray(x_array, dtype=np.float64)

    if x.size <= 1 or y.size <= 1:
        return y.copy(), x.copy()

    if x[-1] < x[0]:
        x = x[::-1].copy()
        y = y[::-1].copy()

    x_int = _stable_interp_grid_new(x, float(dx))
    if x_int.size <= 1:
        return y.copy(), x.copy()
    y_int = np.interp(x_int, x, y)
    return y_int, x_int


def robust_sigma_mad_new(x: np.ndarray) -> float:
    """
    Robust sigma estimate using MAD:
        sigma ~= 1.4826 * median(|x - median(x)|)

    Returns 0 if undefined.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig):
        return 0.0
    return float(sig)


def _fill_short_nan_runs_new(y: np.ndarray, max_run: int) -> np.ndarray:
    """
    Linearly interpolate NaN runs of length <= max_run.
    Larger NaN runs are left as NaN (to be treated as gaps later if desired).
    """
    y = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    if n == 0 or not np.any(~np.isfinite(y)):
        return y

    isn = ~np.isfinite(y)
    idx = np.arange(n, dtype=np.int64)

    run_starts = np.where(isn & ~np.r_[False, isn[:-1]])[0]
    run_ends   = np.where(isn & ~np.r_[isn[1:], False])[0]

    for s, e in zip(run_starts, run_ends):
        L = int(e - s + 1)
        if L > int(max_run):
            continue
        left = s - 1
        right = e + 1
        if left < 0 or right >= n:
            continue
        if not (np.isfinite(y[left]) and np.isfinite(y[right])):
            continue
        y[s:e+1] = np.interp(idx[s:e+1], [left, right], [y[left], y[right]])
    return y


def detect_and_fix_outliers_new(
    y: np.ndarray,
    *,
    N: float = 7.0,
    max_run: int = 12,
    center: str = "median",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limited outlier detection: mark |y - center| > N * sigma_robust as NaN,
    where sigma_robust is MAD-based. Then interpolate short NaN runs.

    Parameters
    ----------
    y : ndarray
        Input series on a regular grid (recommended).
    N : float
        Threshold multiplier (default 7).
    max_run : int
        Max consecutive NaNs to interpolate back (default 12 -> 60s at 5s).
    center : {"median","mean"}
        Center for residual.

    Returns
    -------
    (y_clean, outlier_mask)
    """
    y = np.asarray(y, dtype=np.float64)
    y_clean = y.copy()

    finite = np.isfinite(y_clean)
    if finite.sum() < 10:
        return y_clean, np.zeros_like(y_clean, dtype=bool)

    c = np.median(y_clean[finite]) if center == "median" else float(np.mean(y_clean[finite]))
    sig = robust_sigma_mad_new(y_clean[finite])
    if sig <= 0:
        return y_clean, np.zeros_like(y_clean, dtype=bool)

    resid = y_clean - c
    out = np.isfinite(resid) & (np.abs(resid) > float(N) * sig)
    if np.any(out):
        y_clean[out] = np.nan
        y_clean = _fill_short_nan_runs_new(y_clean, max_run=max_run)

    return y_clean, out


# ---------------------------------------------------------------------
# Minimum-size logic derived from filter/diff edge effects
# ---------------------------------------------------------------------

@dataclass
class MinSizeRequirements:
    min_samples: int
    reason: str


def _min_samples_for_diff_new(n: int) -> MinSizeRequirements:
    """
    Central difference scheme uses a 2n+1 stencil for the interior.
    We require at least 2n+1 samples; in practice a bit more helps, but keep minimal.
    """
    n = int(round(n))
    min_samp = max(2 * n + 1, 3)
    return MinSizeRequirements(min_samples=min_samp, reason=f"diff requires >= {min_samp} samples (2n+1 stencil)")


def _min_samples_for_sosfiltfilt_new(sos: np.ndarray) -> int:
    """
    sosfiltfilt requires padlen < n. We use the same conservative padlen cap as _padlen_for_sos,
    then require n >= padlen + 2 to be safe.
    """
    # This is the padlen used in the original module (capped).
    # For min-size planning, take default heuristic: 3 * n_sections.
    n_sections = int(sos.shape[0])
    default_pad = 3 * n_sections
    # Need n-1 >= padlen -> n >= padlen+1 ; add +1 safety
    return int(default_pad + 2)


def compute_min_piece_samples_new(
    *,
    methods: Sequence[str],
    Ns: Sequence[int] | None,
    btypes: Sequence[str] | None,
    cutoffs: Sequence[float] | None,
    fs: float,
    n_diff: int,
    filter_method: str = "butw",
) -> MinSizeRequirements:
    """
    Compute a conservative minimum piece length in samples needed for the requested processing.
    """
    min_req = _min_samples_for_diff_new(n_diff).min_samples if "diff" in methods else 1
    reasons = []
    if "diff" in methods:
        reasons.append(_min_samples_for_diff_new(n_diff).reason)

    sos_methods_dic = {"butw": butter, "bessel": bessel, "cheby2": cheby2}

    if Ns is None: Ns = []
    if btypes is None: btypes = []
    if cutoffs is None: cutoffs = []

    # butterworth step index
    j = 0
    for m in methods:
        if m == "butw":
            N = int(Ns[j])
            bt = btypes[j]
            fc = float(cutoffs[j])
            sos = sos_methods_dic[filter_method](N, fc, bt, fs=float(fs), output="sos")
            mmin = _min_samples_for_sosfiltfilt_new(sos)
            if mmin > min_req:
                min_req = mmin
            reasons.append(f"butw(N={N}) needs >= {mmin} samples (filtfilt padding)")
            j += 1

    if not reasons:
        reasons = ["no min-size requirement (methods empty)"]

    return MinSizeRequirements(min_samples=int(min_req), reason="; ".join(reasons))


# ---------------------------------------------------------------------
# New piecewise filter + diff wrappers
# ---------------------------------------------------------------------

def get_time_pieces_new(time_array: np.ndarray, tol_gap: float, divider: bool = False):
    """
    Wrapper around original get_time_pieces with a clearer parameter name.
    """
    return get_time_pieces(time_array, tol=tol_gap, divider=divider)


def filter_array_new(
    array: np.ndarray,
    t_array: Optional[np.ndarray] = None,
    *,
    method: str = "butw",
    N: int = 1,
    btype: str = "low",
    cutoff: float = 1e-2,
    fs: float = 0.2,
    tol_gap: float = 30.0,
    int_flag: bool = True,
    dx_interp: float = 5.0,
    outlier_N: float = 7.0,
    outlier_maxrun: int = 12,
    outliers: bool = True,
    min_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter array with:
      - arc/piece splitting by gaps (tol_gap)
      - optional interpolation onto regular grid (dx_interp)
      - optional robust outlier handling (MAD-based N*sigma)
      - discard pieces shorter than min_samples (if provided)

    Returns (filtered, t_filtered). If t_array is None, returns (filtered, None).
    """
    sos_methods_dic = {"butw": butter, "bessel": bessel, "cheby2": cheby2}
    y = np.asarray(array, dtype=np.float64)

    if t_array is None:
        sos = sos_methods_dic[method](int(N), float(cutoff), str(btype), fs=float(fs), output="sos")
        padlen = _padlen_for_sos(sos, y.size)
        if y.size <= 1:
            return y.copy(), None
        if min_samples is not None and y.size < int(min_samples):
            return np.array([], dtype=np.float64), None
        y_f = sosfiltfilt(sos, y, padlen=padlen, padtype="even")
        return y_f, None

    t = np.asarray(t_array, dtype=np.float64)
    (starts, ends), _ = get_time_pieces_new(t, tol_gap=tol_gap, divider=False)

    out_y: List[np.ndarray] = []
    out_t: List[np.ndarray] = []

    for s, e in zip(starts, ends):
        y_piece = y[s:e+1]
        t_piece = t[s:e+1]
        if y_piece.size == 0:
            continue
        if min_samples is not None and y_piece.size < int(min_samples):
            continue

        # interpolate onto regular grid (recommended)
        if int_flag and t_piece.size > 1:
            y_int, t_int = interpolate_array_new(y_piece, t_piece, dx=float(dx_interp))
        else:
            y_int, t_int = y_piece.copy(), t_piece.copy()

        if min_samples is not None and y_int.size < int(min_samples):
            continue

        # outliers on the working grid (pre-filter)
        if outliers:
            y_int, _ = detect_and_fix_outliers_new(y_int, N=outlier_N, max_run=outlier_maxrun)

        # infer fs from grid if needed
        dt_eff = _robust_dt_from_x(t_int)
        fs_eff = (1.0 / dt_eff) if (np.isfinite(dt_eff) and dt_eff > 0) else float(fs)

        sos = sos_methods_dic[method](int(N), float(cutoff), str(btype), fs=float(fs_eff), output="sos")
        padlen = _padlen_for_sos(sos, y_int.size)
        if y_int.size <= 1:
            continue
        y_f = sosfiltfilt(sos, y_int, padlen=padlen, padtype="even")

        out_y.append(y_f)
        out_t.append(t_int)

    if len(out_y) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    return np.concatenate(out_y), np.concatenate(out_t)


def differentiate_central_scheme_new(
    y_array: np.ndarray,
    x_array: np.ndarray,
    *,
    n: int,
    fs: Optional[float] = None,
    tol_gap: float = 30.0,
    int_flag: bool = True,
    dx_interp: float = 5.0,
    outliers: bool = True,
    outlier_N: float = 7.0,
    outlier_maxrun: int = 12,
    min_samples: Optional[int] = None,
    edge_mode: str = "mirror",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    New differentiation:
      - split by gaps (tol_gap)
      - interpolate onto regular grid (dx_interp)
      - robust outlier handling before differentiation
      - discard short pieces (min_samples)
      - edge_mode defaults to 'mirror' (usually safer than 'nearest' for derivatives)
    """
    y = np.asarray(y_array, dtype=np.float64)
    x = np.asarray(x_array, dtype=np.float64)

    if y.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    (starts, ends), _ = get_time_pieces_new(x, tol_gap=tol_gap, divider=False)

    coeff_cd_1 = cdtaylor(int(n), 1)
    coeff_fd_1 = fdtaylor(int(n), 1)
    coeff_bd_1 = bdtaylor(int(n), 1)

    out_dy: List[np.ndarray] = []
    out_x: List[np.ndarray] = []

    for s, e in zip(starts, ends):
        y_piece = y[s:e+1]
        x_piece = x[s:e+1]
        if y_piece.size == 0:
            continue
        if min_samples is not None and y_piece.size < int(min_samples):
            continue

        # interpolate
        if int_flag and x_piece.size > 1:
            y_int, x_int = interpolate_array_new(y_piece, x_piece, dx=float(dx_interp))
        else:
            y_int, x_int = y_piece.copy(), x_piece.copy()

        if min_samples is not None and y_int.size < int(min_samples):
            continue

        # outliers on working grid
        if outliers:
            y_int, _ = detect_and_fix_outliers_new(y_int, N=outlier_N, max_run=outlier_maxrun)

        size = y_int.size
        if size <= 1:
            continue

        # dt/fs
        dt = _robust_dt_from_x(x_int)
        if not np.isfinite(dt) or dt <= 0:
            dt = (1.0 / float(fs)) if (fs is not None and np.isfinite(fs) and fs > 0) else float(dx_interp)
        fs_eff = 1.0 / dt

        # too short for full stencil: fall back to simple diff
        if size <= 2 * int(n):
            if size < 2:
                dy = np.zeros(size, dtype=np.float64)
            else:
                dy_raw = fs_eff * np.diff(y_int)
                dy = interpolate_array_to_size(dy_raw, x_int[:-1], x_int)
            out_dy.append(dy)
            out_x.append(x_int)
            continue

        # forward/backward edges
        nn = int(n)
        dy_front = np.zeros(nn, dtype=np.float64)
        for i in range(nn):
            slice_y = y_int[i:i + len(coeff_fd_1)]
            dy_front[i] = fs_eff * np.dot(coeff_fd_1, slice_y)

        dy_back = np.zeros(nn, dtype=np.float64)
        for i, idx in enumerate(range(size - nn, size)):
            slice_y = y_int[idx - len(coeff_bd_1) + 1: idx + 1]
            dy_back[i] = fs_eff * np.dot(coeff_bd_1, slice_y)

        # central
        dy_central = fs_eff * ndimage.convolve1d(
            y_int, -coeff_cd_1, axis=-1, mode=edge_mode
        )

        dy = np.concatenate((dy_front, dy_central[nn:size - nn], dy_back))
        out_dy.append(dy)
        out_x.append(x_int)

    if len(out_dy) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    return np.concatenate(out_dy), np.concatenate(out_x)


def perform_filtering_sequence_new(
    array_to_be_filtered: np.ndarray,
    t_array: np.ndarray,
    *,
    methods: Sequence[str],
    Ns: Sequence[int],
    btypes: Sequence[str],
    cutoffs: Sequence[float],
    fs: float = 0.2,
    n: int = 4,
    arc_seconds: float = 86400.0,   # default: 1 day
    tol_gap: float = 30.0,          # nicer than 20s; can tune
    dx_interp: float = 5.0,
    outliers: bool = True,
    outlier_N: float = 7.0,
    outlier_maxrun: int = 12,
    int_flag: bool = True,
    min_piece_seconds: Optional[float] = None,
    discard_short: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arc-aware, robust filter+diff sequence.

    Workflow
    --------
    1) Split into coarse arcs (default: daily).
    2) Within each arc, split into pieces by gaps >= tol_gap.
    3) Discard pieces that are too short for the requested scheme:
         - computed from filter/diff edge + padding needs
         - optionally also enforce min_piece_seconds (duration-based)
    4) Apply robust outlier handling on the interpolated working grid (MAD; N=7).
    5) Run filtering/differentiation sequentially on each piece.
    6) Concatenate all processed pieces and return.

    Notes
    -----
    - This function is designed to replace *only* the diff/filter stage.
    - It intentionally does NOT compute LGD.
    """
    y0 = np.asarray(array_to_be_filtered, dtype=np.float64)
    t0 = np.asarray(t_array, dtype=np.float64)

    if y0.size != t0.size:
        raise ValueError("array_to_be_filtered and t_array must have the same length")

    if y0.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Ensure sorted by time (critical)
    order = np.argsort(t0)
    if not np.all(order == np.arange(order.size)):
        t0 = t0[order]
        y0 = y0[order]

    # Determine min samples based on the requested processing sequence
    min_req = compute_min_piece_samples_new(
        methods=methods, Ns=Ns, btypes=btypes, cutoffs=cutoffs,
        fs=float(fs), n_diff=int(n)
    )
    min_samples = min_req.min_samples

    # Optionally enforce duration-based minimum as well
    min_piece_s = float(min_piece_seconds) if (min_piece_seconds is not None) else None

    out_all_y: List[np.ndarray] = []
    out_all_t: List[np.ndarray] = []

    arcs = split_time_into_arcs_new(t0, arc_seconds=float(arc_seconds))
    # Precompute butterworth step indices mapping
    # (btypes/Ns/cutoffs are indexed by BUTW steps, not method index)
    for a_s, a_e in arcs:
        t_arc = t0[a_s:a_e+1]
        y_arc = y0[a_s:a_e+1]
        if t_arc.size == 0:
            continue

        # Split arc into pieces by gaps
        (p_starts, p_ends), _ = get_time_pieces_new(t_arc, tol_gap=tol_gap, divider=False)

        for ps, pe in zip(p_starts, p_ends):
            # piece indices in global coordinates
            s = a_s + int(ps)
            e = a_s + int(pe)

            t_piece_raw = t0[s:e+1]
            y_piece_raw = y0[s:e+1]
            if y_piece_raw.size == 0:
                continue

            # duration-based criterion
            if min_piece_s is not None and t_piece_raw.size >= 2:
                if (t_piece_raw[-1] - t_piece_raw[0]) < min_piece_s:
                    if discard_short:
                        continue

            # sample-based criterion (before interpolation)
            if y_piece_raw.size < min_samples and discard_short:
                continue

            # Work on an interpolated grid right away (so min_samples is meaningful)
            if t_piece_raw.size > 1:
                y_work, t_work = interpolate_array_new(y_piece_raw, t_piece_raw, dx=float(dx_interp))
            else:
                continue

            if y_work.size < min_samples and discard_short:
                continue

            # Optional outlier clean on the working grid (pre-sequence)
            if outliers:
                y_work, _ = detect_and_fix_outliers_new(y_work, N=outlier_N, max_run=outlier_maxrun)

            # Now run the sequence locally on this piece, updating y_work/t_work
            y_curr = y_work
            t_curr = t_work

            i_butw = 0
            for m in methods:
                if m == "butw":
                    y_curr, t_curr = filter_array_new(
                        y_curr, t_curr,
                        method="butw",
                        N=int(Ns[i_butw]),
                        btype=str(btypes[i_butw]),
                        cutoff=float(cutoffs[i_butw]),
                        fs=float(fs),
                        tol_gap=tol_gap,
                        int_flag=False,         # already on regular grid
                        dx_interp=float(dx_interp),
                        outliers=outliers,      # optionally re-run mild outliers after each stage
                        outlier_N=outlier_N,
                        outlier_maxrun=outlier_maxrun,
                        min_samples=min_samples,
                    )
                    i_butw += 1

                elif m == "diff":
                    y_curr, t_curr = differentiate_central_scheme_new(
                        y_curr, t_curr,
                        n=int(n),
                        fs=float(fs),
                        tol_gap=tol_gap,
                        int_flag=False,         # already on regular grid
                        dx_interp=float(dx_interp),
                        outliers=outliers,
                        outlier_N=outlier_N,
                        outlier_maxrun=outlier_maxrun,
                        min_samples=min_samples,
                        edge_mode="mirror",
                    )

                else:
                    raise ValueError(f"Unknown method '{m}' in perform_filtering_sequence_new")

                if y_curr.size == 0:
                    break

            if y_curr.size == 0:
                continue

            out_all_y.append(y_curr)
            out_all_t.append(t_curr)

    if len(out_all_y) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    return np.concatenate(out_all_y), np.concatenate(out_all_t)




def transfer_function(frequency_array: np.ndarray) -> np.ndarray:
    """
    Analytical transfer function Z(f) from range-acceleration residuals to LGD,
    following the piecewise form used in Ghobadi-Far et al. (2018), with a smooth
    log-linear bridge between 0.3 and 0.7 mHz.

    Ghobadi-Far, K., Han, S.-C., Weller, S., Loomis, B. D., Luthcke, S. B., Mayer-Gürr, T., & Behzadpour, S. (2018).
    A Transfer Function Between Line-of-Sight Gravity Difference and GRACE Intersatellite Ranging Data and
    an Application to Hydrological Surface Mass Variation.
    Journal of Geophysical Research: Solid Earth, 123(10), 9186–9201. https://doi.org/10.1029/2018JB016088

    Piecewise:
      - f < f1              : Z(f) = 0
      - f1 <= f < f2        : Z(f) = a*log10(f) + b, with continuity at f2
      - f >= f2             : Z(f) = 1 + c * f^(-p)

    Parameters
    ----------
    frequency_array : ndarray
        Frequencies in Hz (can include 0).

    Returns
    -------
    ndarray
        Z(f) evaluated at frequency_array (same shape as input).
    """
    f = np.asarray(frequency_array, dtype=np.float64)
    Z = np.zeros_like(f, dtype=np.float64)

    # thresholds (Hz)
    f1 = 0.3e-3
    f2 = 0.7e-3

    # high-frequency branch parameters (as in your code)
    c = 3.45e-4
    p = 1.04

    # Value at f2 from HF branch (continuity target)
    Z_f2 = 1.0 + c * (f2 ** (-p))

    # log-linear bridge coefficients a,b such that:
    #   Z(f1)=0  and  Z(f2)=Z_f2
    # a*(log10(f1)-log10(f2)) = -Z_f2  ->  a = Z_f2/(log10(f2)-log10(f1))
    denom = (np.log10(f2) - np.log10(f1))
    a = Z_f2 / denom
    b = -a * np.log10(f1)

    # Masks
    m_hf = f >= f2
    m_mid = (f >= f1) & (f < f2)
    # m_lf = f < f1 already zeros

    # Mid band: avoid log10(0) safely by masking (0 not included here)
    Z[m_mid] = a * np.log10(f[m_mid]) + b

    # High band
    Z[m_hf] = 1.0 + c * (f[m_hf] ** (-p))

    # Anything non-finite -> 0 (defensive)
    Z[~np.isfinite(Z)] = 0.0
    return Z


def compute_lgd(
    drho_dotdot_array: np.ndarray,
    time_array=None,
    *,
    tol: float = 30.0,
    dt: float | None = None,
    detrend_mean: bool = True,
) -> np.ndarray:
    """
    Frequency-domain estimate of residual LGD (dg_AB^LOS) from range-acceleration residuals.

        dg_AB^LOS(t) ≈ F^{-1}{ Z(f) * F{drho_ddot(t)} }

    Key improvements over your original:
      - Vectorized / no repeated code paths.
      - dt is NOT hard-coded to 5 s:
          * if dt is provided -> use it
          * else if time_array provided -> infer dt per piece (median)
          * else -> fall back to 5 s (for backward compatibility)
      - Handles DC explicitly (Z(0)=0 already), optional mean removal to reduce edge ringing.
      - Uses gap splitting when time_array is provided.

    Parameters
    ----------
    drho_dotdot_array : ndarray
        Range-acceleration residual time series.
    time_array : ndarray or None
        Timestamps in seconds. If provided, splits at gaps >= tol and can infer dt.
    tol : float
        Gap threshold in seconds for piece splitting.
    dt : float or None
        Sampling interval in seconds. If given, overrides inferred dt.
    detrend_mean : bool
        If True, subtract mean per piece before FFT and add it back after IFFT (reduces leakage).

    Returns
    -------
    ndarray
        LGD estimate on the (possibly piecewise) original time support.
        If time_array is provided, output length matches input length.
    """
    y = np.asarray(drho_dotdot_array, dtype=np.float64)

    # Split into pieces if we have time_array
    if time_array is not None and isinstance(time_array, (np.ndarray, np.generic, pd.Series, pd.DataFrame)):
        t = np.asarray(time_array).squeeze()
        if isinstance(time_array, pd.DataFrame):
            # If someone passed a single-column DataFrame
            if t.ndim != 1:
                t = np.asarray(time_array.iloc[:, 0]).squeeze()

        pieces_indices, _ = get_time_pieces_new(t, tol_gap=tol)
        start_indices, end_indices = pieces_indices

        y_pieces = [y[s:e + 1] for s, e in zip(start_indices, end_indices)]
        t_pieces = [t[s:e + 1] for s, e in zip(start_indices, end_indices)]

        out_pieces = []
        for y_piece, t_piece in zip(y_pieces, t_pieces):
            if y_piece.size == 0:
                continue
            if y_piece.size == 1:
                out_pieces.append(np.zeros_like(y_piece, dtype=np.float64))
                continue

            # Determine dt for this piece
            if dt is not None:
                dt_use = float(dt)
            else:
                dt_use = _robust_dt_from_x(t_piece)
                if not np.isfinite(dt_use) or dt_use <= 0:
                    dt_use = 5.0  # backward-compatible fallback

            # Mean removal (optional, helps reduce ringing)
            mean_piece = float(np.mean(y_piece)) if detrend_mean else 0.0
            y0 = y_piece - mean_piece

            # FFT pipeline
            F = np.fft.rfft(y0, n=y0.size)
            f = np.fft.rfftfreq(y0.size, d=dt_use)
            Z = transfer_function(f)
            x = np.fft.irfft(Z * F, n=y0.size)

            if detrend_mean:
                x = x + mean_piece  # keep DC behavior consistent with original signal support

            out_pieces.append(x)

        if len(out_pieces) == 0:
            return np.array([], dtype=np.float64)

        return np.concatenate(out_pieces)

    # No time_array: single piece
    if y.size == 0:
        return np.array([], dtype=np.float64)
    if y.size == 1:
        return np.zeros_like(y, dtype=np.float64)

    dt_use = float(dt) if dt is not None else 5.0  # keep your legacy default
    mean_all = float(np.mean(y)) if detrend_mean else 0.0
    y0 = y - mean_all

    F = np.fft.rfft(y0, n=y0.size)
    f = np.fft.rfftfreq(y0.size, d=dt_use)
    Z = transfer_function(f)
    x = np.fft.irfft(Z * F, n=y0.size)

    if detrend_mean:
        x = x + mean_all

    return x


def compute_lgd_new(
    drho_dotdot_array: np.ndarray,
    time_array=None,
    *,
    tol: float = 30.0,
    dt: float | None = None,
    remove_mean: bool = True,
    add_mean_back: bool = False,   # default False for consistency when Z(0)=0 / LF suppressed
) -> np.ndarray:
    """
    Frequency-domain estimate of residual LGD (dg_AB^LOS) from range-acceleration residuals:

        dg_AB^LOS(t) ≈ F^{-1}{ Z(f) * F{drho_ddot(t)} }

    Consistent mean handling:
      - If remove_mean=True, subtract per-piece mean before FFT to reduce leakage/ringing.
      - By default we DO NOT add the mean back, because many Z(f) definitions
        intentionally suppress DC / very-low-frequency content (e.g., Z(0)=0).
      - If you *really* want to preserve the original mean, set add_mean_back=True.

    Parameters
    ----------
    drho_dotdot_array : ndarray
        Range-acceleration residual time series.
    time_array : ndarray or None
        Timestamps in seconds. If provided, splits at gaps >= tol and can infer dt per piece.
    tol : float
        Gap threshold in seconds for piece splitting.
    dt : float or None
        Sampling interval in seconds. If given, overrides inferred dt.
    remove_mean : bool
        Remove per-piece mean before FFT (recommended).
    add_mean_back : bool
        Add mean back after IFFT. Default False to stay consistent with LF-suppressing Z(f).

    Returns
    -------
    ndarray
        LGD estimate; if time_array is provided, output length matches input length.
    """
    y = np.asarray(drho_dotdot_array, dtype=np.float64)

    def _process_piece(y_piece: np.ndarray, t_piece: np.ndarray | None) -> np.ndarray:
        if y_piece.size == 0:
            return y_piece
        if y_piece.size == 1:
            return np.zeros_like(y_piece, dtype=np.float64)

        # Determine dt
        if dt is not None:
            dt_use = float(dt)
        else:
            if t_piece is not None:
                dt_use = _robust_dt_from_x(t_piece)
            else:
                dt_use = np.nan
            if not np.isfinite(dt_use) or dt_use <= 0:
                dt_use = 5.0  # legacy fallback

        mean_piece = float(np.mean(y_piece)) if remove_mean else 0.0
        y0 = y_piece - mean_piece if remove_mean else y_piece

        F = np.fft.rfft(y0, n=y0.size)
        f = np.fft.rfftfreq(y0.size, d=dt_use)
        Z = transfer_function(f)

        x = np.fft.irfft(Z * F, n=y0.size)

        # IMPORTANT: only add mean back if explicitly requested
        if remove_mean and add_mean_back:
            x = x + mean_piece

        return x

    # Piecewise if time_array provided
    if time_array is not None and isinstance(time_array, (np.ndarray, np.generic, pd.Series, pd.DataFrame)):
        t = np.asarray(time_array).squeeze()
        if isinstance(time_array, pd.DataFrame):
            if t.ndim != 1:
                t = np.asarray(time_array.iloc[:, 0]).squeeze()

        (start_indices, end_indices), _ = get_time_pieces_new(t, tol_gap=tol)

        out_pieces = []
        for s, e in zip(start_indices, end_indices):
            out_pieces.append(_process_piece(y[s:e + 1], t[s:e + 1]))

        return np.concatenate(out_pieces) if out_pieces else np.array([], dtype=np.float64)

    # Single piece
    return _process_piece(y, None)
