"""
lgd_synthesis.py -- synthesise GRACE-FO line-of-sight gravity differences (LGD)
from spherical-harmonic (SH) coefficients and the positions of both satellites.

Python port of Compute_LGD.f90 (PROGRAM LOSACC, (c) Shin-Chan Han, 16SEP21),
including its subroutines GET_ACC_VEC_SH, LGDR2 and FILLMAT.

What it computes
----------------
For each epoch the gravitational acceleration a = grad(V) of the SH model

    V(r, theta, lambda) = GM/R * sum_{l=lmin}^{lmax} (R/r)^(l+1)
                          * sum_m Pbar_lm(cos theta) [C_lm cos(m lambda) + S_lm sin(m lambda)]

is evaluated at both satellites (in the Earth-fixed frame) and projected on the
line of sight:

    LGD = (a_D - a_C) . e_CD,       e_CD = (r_D - r_C) / |r_D - r_C|
        = (a_C - a_D) . e_DC         (identical: swapping C and D changes nothing)

The Fortran program computes  accel = (acc1 - acc2) . e12  with e12 read from
the orbit file. That equals the LGD above when e12 points from satellite 2 to
satellite 1. If your Fortran input e12 pointed from 1 to 2, the Fortran output
has the opposite sign.

Love numbers
------------
The Fortran code uses none: SH coefficients of the geopotential (GFC files, L2
products, AOD1B, ocean tides) already contain the potential of the load-induced
deformation, so the acceleration follows from them directly.

Load Love numbers k_l ARE needed when the coefficients describe surface mass
(equivalent water height or surface mass density) instead of the potential,
e.g. EWH mass-change models. Those are converted to Stokes coefficients with
(Wahr et al., 1998)

    dC_lm = 3 rho_w / (rho_e R) * (1 + k_l) / (2l + 1) * EWH_lm        [EWH_lm in m]

before synthesis. Use `surface_mass_to_stokes` with `read_love_numbers`.

Usage (Python)
--------------
>>> from lgd_synthesis import *
>>> sh   = read_gfc("model.gfc")                       # Stokes coefficients
>>> orbC = read_orbit_parquet("orbit_C.parquet", "C")
>>> orbD = read_orbit_parquet("orbit_D.parquet", "D")
>>> out  = compute_lgd(sh, orbC, orbD, lmin=2, lmax=96, position_type="geodetic")

EWH coefficients instead:
>>> ewh  = read_gfc("ewh_model.txt")
>>> k    = read_love_numbers("love_numbers.txt")
>>> sh   = surface_mass_to_stokes(ewh, k, units="ewh")
>>> out  = compute_lgd(sh, orbC, orbD, lmin=2)

Usage (command line; mirrors the Fortran arguments)
---------------------------------------------------
python lgd_synthesis.py model.gfc orbit_C.parquet orbit_D.parquet lgd_out.csv \
       --lmin 2 --lmax 96 [--gm 3.986004415e14 --radius 6378136.3] \
       [--units ewh --love love_numbers.txt] [--position-type geodetic]

Requires numpy and pandas (+ pyarrow or fastparquet for parquet files).
Optional: numba (pip install numba) -> compiled, multi-threaded kernel, much faster.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SHCoefficients",
    "read_gfc",
    "read_love_numbers",
    "surface_mass_to_stokes",
    "read_orbit_parquet",
    "prepare_orbit",
    "merge_orbits",
    "geodetic_to_ecef",
    "orbit_to_ecef",
    "ecef_to_geodetic",
    "gravity_acceleration_ecef",
    "compute_lgd",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
GM_EARTH = 3.986004415e14       # m^3/s^2   (GRACE / EGM2008 convention)
R_EARTH = 6378136.3             # m
G_NEWTON = 6.67430e-11          # m^3/(kg s^2), CODATA 2018
RHO_WATER = 1000.0              # kg/m^3
# GRS80 ellipsoid (geodetic -> ECEF). WGS84 differs by < 0.1 mm in this use.
GRS80_A = 6378137.0
GRS80_F = 1.0 / 298.257222101


# ==========================================================================
# Spherical-harmonic coefficients
# ==========================================================================
@dataclass
class SHCoefficients:
    """Fully normalised SH coefficients.

    C[l, m], S[l, m] are (lmax+1, lmax+1) arrays; entries with m > l are zero.
    """

    C: np.ndarray
    S: np.ndarray
    gm: float = GM_EARTH
    radius: float = R_EARTH
    tide_system: str | None = None
    name: str | None = None
    header: dict = field(default_factory=dict)

    @property
    def lmax(self) -> int:
        return self.C.shape[0] - 1

    def copy(self) -> "SHCoefficients":
        return copy.deepcopy(self)

    def truncate(self, lmin: int = 0, lmax: int | None = None) -> "SHCoefficients":
        """Keep only degrees lmin..lmax (others set to zero / removed)."""
        lmax = self.lmax if lmax is None else lmax
        if lmax > self.lmax:
            C = np.zeros((lmax + 1, lmax + 1)); S = np.zeros_like(C)
            C[: self.lmax + 1, : self.lmax + 1] = self.C
            S[: self.lmax + 1, : self.lmax + 1] = self.S
        else:
            C = self.C[: lmax + 1, : lmax + 1].copy()
            S = self.S[: lmax + 1, : lmax + 1].copy()
        C[:lmin, :] = 0.0
        S[:lmin, :] = 0.0
        out = self.copy(); out.C, out.S = C, S
        return out

    def rescale(self, gm: float, radius: float) -> "SHCoefficients":
        """Express the same field with a different GM and reference radius."""
        l = np.arange(self.lmax + 1)[:, None]
        f = (self.gm / gm) * (self.radius / radius) ** l
        out = self.copy()
        out.C, out.S, out.gm, out.radius = self.C * f, self.S * f, gm, radius
        return out

    def __sub__(self, other: "SHCoefficients") -> "SHCoefficients":
        """Difference of two fields, e.g. a monthly field minus a mean field."""
        if not (math.isclose(self.gm, other.gm) and math.isclose(self.radius, other.radius)):
            other = other.rescale(self.gm, self.radius)
        L = max(self.lmax, other.lmax)
        a, b = self.truncate(0, L), other.truncate(0, L)
        out = self.copy(); out.C, out.S = a.C - b.C, a.S - b.S
        return out

    def __add__(self, other: "SHCoefficients") -> "SHCoefficients":
        neg = other.copy(); neg.C, neg.S = -other.C, -other.S
        return self - neg


def _to_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def read_gfc(path: str, lmax: int | None = None) -> SHCoefficients:
    """Read an ICGEM .gfc file (or a plain 'L M C S [...]' text file).

    * ICGEM format: header up to 'end_of_head'; keywords earth_gravity_constant,
      radius, max_degree, tide_system and norm are read. Lines starting with
      'gfc' are used. 'gfct' lines are taken as the static value; 'trnd',
      'acos', 'asin' (time-variable terms) are ignored with a warning.
    * Plain format (as read by the Fortran code): every line with at least four
      numbers is 'L M C S ...'. GM and R then default to GM_EARTH and R_EARTH
      unless overridden in compute_lgd.
    Fortran 'D' exponents are accepted.
    """
    header: dict[str, str] = {}
    rows: list[tuple[int, int, float, float]] = []
    in_header = False
    has_icgem_header = False
    skipped: set[str] = set()

    with open(path, "r", errors="replace") as fh:
        lines = fh.readlines()

    # Detect ICGEM header.
    for ln in lines[:2000]:
        if ln.strip().lower().startswith("end_of_head"):
            has_icgem_header = True
            break
    in_header = has_icgem_header

    for ln in lines:
        s = ln.strip()
        if not s or s[0] in "#%!":
            continue
        if in_header:
            if s.lower().startswith("end_of_head"):
                in_header = False
                continue
            parts = s.split(None, 1)
            if len(parts) == 2:
                header[parts[0].lower()] = parts[1].strip()
            continue
        parts = s.split()
        key = parts[0].lower()
        if key in ("gfc", "gfct"):
            if key == "gfct":
                skipped.add("gfct (used as static value)")
            rows.append((int(parts[1]), int(parts[2]), _to_float(parts[3]), _to_float(parts[4])))
        elif key in ("trnd", "acos", "asin", "dot"):
            skipped.add(key)
        elif re.match(r"^[+-]?\d+$", parts[0]) and len(parts) >= 4:
            rows.append((int(parts[0]), int(parts[1]), _to_float(parts[2]), _to_float(parts[3])))

    if skipped:
        warnings.warn(f"{path}: time-variable ICGEM terms not evaluated: {sorted(skipped)}")
    if not rows:
        raise ValueError(f"No coefficients found in {path}")

    norm = header.get("norm", "fully_normalized").lower()
    if "fully" not in norm:
        raise ValueError(f"{path}: only fully normalised coefficients are supported (norm = {norm})")

    arr = np.array(rows, dtype=float)
    ls, ms = arr[:, 0].astype(int), arr[:, 1].astype(int)
    L = int(ls.max()) if lmax is None else int(lmax)
    keep = ls <= L
    C = np.zeros((L + 1, L + 1)); S = np.zeros((L + 1, L + 1))
    C[ls[keep], ms[keep]] = arr[keep, 2]
    S[ls[keep], ms[keep]] = arr[keep, 3] # ...

    gm = _to_float(header["earth_gravity_constant"]) if "earth_gravity_constant" in header else GM_EARTH
    radius = _to_float(header["radius"]) if "radius" in header else R_EARTH
    return SHCoefficients(C=C, S=S, gm=gm, radius=radius,
                          tide_system=header.get("tide_system"),
                          name=header.get("modelname", str(path)), header=header)


# ==========================================================================
# Love numbers and surface-mass -> Stokes conversion
# ==========================================================================
def read_love_numbers(path: str, first_degree: int = 0, k_column: int | None = None) -> np.ndarray:
    """Read load Love numbers k_l; return array k with k[l] for l = 0..lmax.

    Accepted layouts (comment lines starting with #, % or ! are skipped):
      * two (or more) columns: degree, k_l [, ...]   (k_column defaults to 1)
      * one column: k_l only, the first row being degree `first_degree`.
    Degrees missing from the file are NaN; missing k_0 is set to 0.
    """
    data = []
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s[0] in "#%!":
                continue
            toks = re.split(r"[,\s;]+", s)
            try:
                data.append([_to_float(t) for t in toks if t])
            except ValueError:
                continue  # text header line
    if not data:
        raise ValueError(f"No Love numbers found in {path}")
    ncol = min(len(r) for r in data)
    arr = np.array([r[:ncol] for r in data])

    if ncol == 1:
        k_vals = arr[:, 0]
        degrees = np.arange(first_degree, first_degree + len(k_vals))
    else:
        degrees = arr[:, 0]
        if not np.allclose(degrees, np.round(degrees)):
            raise ValueError("First column of the Love-number file is not an integer degree")
        degrees = degrees.astype(int)
        k_vals = arr[:, 1 if k_column is None else k_column]

    k = np.full(int(degrees.max()) + 1, np.nan)
    k[degrees] = k_vals
    if np.isnan(k[0]):
        k[0] = 0.0
    return k


def surface_mass_to_stokes(sh: SHCoefficients, love: np.ndarray, units: str = "ewh",
                           rho_water: float = RHO_WATER, rho_earth: float | None = None,
                           gm: float | None = None, radius: float | None = None) -> SHCoefficients:
    """Convert surface-mass SH coefficients to dimensionless Stokes coefficients.

        dC_lm = 3 rho_w / (rho_e R) * (1 + k_l) / (2l + 1) * EWH_lm

    units : 'ewh' (metres of equivalent water height), 'ewh_mm', 'ewh_cm'
            or 'smd' (surface mass density, kg/m^2).
    rho_earth defaults to the mean density implied by GM and R:
            3 GM / (4 pi G R^3)  (about 5514 kg/m^3).
    Note: k_1 depends on the reference frame (CM vs CF); use the value that
    matches how your degree-1 coefficients are defined.
    """
    gm = sh.gm if gm is None else gm
    radius = sh.radius if radius is None else radius
    if rho_earth is None:
        rho_earth = 3.0 * gm / (4.0 * math.pi * G_NEWTON * radius**3)
    scale = {"ewh": 1.0, "ewh_m": 1.0, "ewh_cm": 1e-2, "ewh_mm": 1e-3, "smd": 1.0 / rho_water}
    if units.lower() not in scale:
        raise ValueError(f"units must be one of {list(scale)}")

    L = sh.lmax
    if len(love) < L + 1:
        raise ValueError(f"Love numbers given up to degree {len(love) - 1}, coefficients go to {L}")
    k = np.asarray(love[: L + 1], dtype=float)
    l = np.arange(L + 1)
    used = np.any(sh.C != 0, axis=1) | np.any(sh.S != 0, axis=1)
    if np.any(np.isnan(k) & used):
        raise ValueError(f"Missing Love numbers for degrees {l[np.isnan(k) & used].tolist()}")
    k = np.nan_to_num(k)

    fac = (3.0 * rho_water / (rho_earth * radius)) * (1.0 + k) / (2 * l + 1) * scale[units.lower()]
    out = sh.copy()
    out.C, out.S = sh.C * fac[:, None], sh.S * fac[:, None]
    out.gm, out.radius = gm, radius
    return out


# ==========================================================================
# Orbits
# ==========================================================================
def prepare_orbit(df: pd.DataFrame, sat: str | None = None) -> pd.DataFrame:
    """Standardise an orbit table to columns gps_time, lat, lon, alt.

    Accepts latitude_X / longitude_X / altitude_X (X = sat suffix, e.g. 'C')
    or the same names without a suffix.
    """
    cols = {c.lower(): c for c in df.columns}
    out = {}
    for std, base in (("lat", "latitude"), ("lon", "longitude"), ("alt", "altitude")):
        cands = ([f"{base}_{sat}".lower()] if sat else []) + [base]
        cands += [c for c in cols if c.startswith(base + "_")]
        for c in cands:
            if c in cols:
                out[std] = df[cols[c]].to_numpy(dtype=float)
                break
        else:
            raise KeyError(f"No column for {base}" + (f"_{sat}" if sat else "") + f" in {list(df.columns)}")
    if "gps_time" not in cols:
        raise KeyError(f"No gps_time column in {list(df.columns)}")
    res = pd.DataFrame({"gps_time": df[cols["gps_time"]].to_numpy(), **out})
    return res.sort_values("gps_time").drop_duplicates("gps_time").reset_index(drop=True)


def read_orbit_parquet(path: str, sat: str | None = None) -> pd.DataFrame:
    """Read a parquet file with gps_time, latitude_X, longitude_X, altitude_X."""
    return prepare_orbit(pd.read_parquet(path), sat)


def merge_orbits(orb_c: pd.DataFrame, orb_d: pd.DataFrame) -> pd.DataFrame:
    """Inner-join both satellites on gps_time; columns *_C and *_D."""
    m = pd.merge(orb_c, orb_d, on="gps_time", suffixes=("_C", "_D"), how="inner")
    dropped = max(len(orb_c), len(orb_d)) - len(m)
    if len(m) == 0:
        raise ValueError("The two orbit files have no common gps_time epochs")
    if dropped:
        warnings.warn(f"{dropped} epochs without a match in the other satellite were dropped")
    return m


def geodetic_to_ecef(lat_deg, lon_deg, h, a: float = GRS80_A, f: float = GRS80_F) -> np.ndarray:
    """Geodetic latitude/longitude [deg] and ellipsoidal height [m] -> ECEF xyz [m]."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    e2 = f * (2.0 - f)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.column_stack([(N + h) * np.cos(lat) * np.cos(lon),
                            (N + h) * np.cos(lat) * np.sin(lon),
                            (N * (1.0 - e2) + h) * np.sin(lat)])


def ecef_to_geodetic(xyz, a: float = GRS80_A, f: float = GRS80_F):
    """ECEF xyz [m] (n, 3) -> geodetic lat [deg], lon [deg], ellipsoidal height [m].

    Bowring start + 3 fixed-point iterations: sub-micrometre at LEO heights.
    """
    xyz = np.atleast_2d(xyz)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    e2 = f * (2.0 - f)
    b = a * (1.0 - f)
    ep2 = e2 / (1.0 - e2)
    p = np.hypot(x, y)
    th = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)
    for _ in range(3):
        N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + h)))
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), h


def _check_alt_units(alt: np.ndarray, position_type: str) -> np.ndarray:
    med = float(np.nanmedian(np.abs(alt)))
    if position_type == "radius":
        if med < 1e6:
            raise ValueError(f"position_type='radius' but median value is {med:.0f}; "
                             "expected geocentric radius in metres (~6.8e6)")
        return alt
    if med > 3e6:
        raise ValueError(f"Median altitude is {med:.0f} m: this looks like a geocentric radius. "
                         "Use position_type='radius'.")
    if 50 < med < 5000:
        warnings.warn(f"Median altitude {med:.1f} looks like kilometres; converting to metres")
        return alt * 1e3
    return alt


def orbit_to_ecef(lat, lon, alt, position_type: str = "geodetic",
                  reference_radius: float = R_EARTH) -> np.ndarray:
    """Convert orbit columns to ECEF xyz [m].

    position_type:
      'geodetic'   : geodetic lat/lon on GRS80, altitude = ellipsoidal height
      'geocentric' : geocentric (spherical) lat/lon, altitude = r - reference_radius
      'radius'     : geocentric lat/lon, 'altitude' column holds the radius r
    """
    alt = _check_alt_units(np.asarray(alt, float), position_type)
    if position_type == "geodetic":
        return geodetic_to_ecef(lat, lon, alt)
    r = alt + reference_radius if position_type == "geocentric" else alt
    if position_type not in ("geocentric", "radius"):
        raise ValueError("position_type must be 'geodetic', 'geocentric' or 'radius'")
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)])


# ==========================================================================
# Gravitational acceleration from SH coefficients  (GET_ACC_VEC_SH + LGDR2)
# ==========================================================================
def _recursion_coeffs(L: int):
    """Coefficients of the fully normalised Legendre recursions (Jekeli 1996;
    Tscherning et al. 1983), as used in LGDR2."""
    l = np.arange(L + 1, dtype=float)[:, None]
    m = np.arange(L + 1, dtype=float)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.sqrt((2 * l - 1) * (2 * l + 1) / ((l - m) * (l + m)))
        b = np.sqrt((2 * l + 1) * (l + m - 1) * (l - m - 1) / ((l - m) * (l + m) * (2 * l - 3)))
        e = np.sqrt((l**2 - m**2) * (2 * l + 1) / (2 * l - 1))  # derivative coefficient
    for arr in (a, b, e):
        arr[~np.isfinite(arr)] = 0.0
    return a, b, e


try:                                        # optional compiled kernel
    import numba as _nb
    HAVE_NUMBA = True
except ImportError:                         # pragma: no cover
    HAVE_NUMBA = False


def _sh_ned_numpy(r, t, u, lon, C, S, a_rec, b_rec, e_rec, R0, lmin, chunk=1024,
                  n_threads=None):
    """Pure-NumPy kernel (used when numba is not installed).

    The radial factor is absorbed into the Legendre functions,
    Pt_lm = (R0/r)^(l+1) Pbar_lm, whose degree recursion is vectorised over m
    (L in-place NumPy steps per chunk). All sums over degree l are then matrix
    products K[m] (6, L+1) @ Pt[:, m, :] done by BLAS, and the sums over order
    m are a few reductions. Chunks run in parallel threads (NumPy releases the
    GIL inside its array operations).
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    n, L = len(r), C.shape[0] - 1
    M = L + 1
    lvec = np.arange(M, dtype=float)
    mvec = lvec[:, None]
    # K[m] rows: C, S, (l+1)C, (l+1)S, e_{l+1} C_{l+1}, e_{l+1} S_{l+1}
    K = np.zeros((M, 6, M))
    Ct, St = C.T, S.T                                   # [m, l]
    K[:, 0], K[:, 1] = Ct, St
    K[:, 2], K[:, 3] = Ct * (lvec + 1), St * (lvec + 1)
    eT = e_rec.T
    K[:, 4, :-1] = (eT * Ct)[:, 1:]
    K[:, 5, :-1] = (eT * St)[:, 1:]
    sfac = np.ones(M)
    if M > 1:
        sfac[1] = math.sqrt(3.0)
    for m in range(2, M):
        sfac[m] = math.sqrt((2 * m + 1) / (2 * m))
    a3 = a_rec[:, :, None]; b3 = b_rec[:, :, None]
    gn = np.empty(n); ge = np.empty(n); gd = np.empty(n)
    tls = threading.local()

    def work(i0):
        sl = slice(i0, min(i0 + chunk, n))
        tt, uu, lo = t[sl], u[sl], lon[sl]
        k = len(tt)
        if getattr(tls, "k", None) != k:                # per-thread buffers
            tls.k = k
            tls.Pt = np.zeros((M, M, k))                # [l, m, point]; m > l stays 0
            tls.w = np.empty((M, k)); tls.R = np.empty((M, 6, k))
        Pt, w, R = tls.Pt, tls.w, tls.R
        q = R0 / r[sl]
        tq, q2, uq = tt * q, q * q, uu * q
        Pt[0, 0] = q                                    # sectoral q^(m+1) Pbar_mm
        for m in range(1, M):
            np.multiply(Pt[m - 1, m - 1], uq, out=Pt[m, m]); Pt[m, m] *= sfac[m]
        for l in range(1, M):
            o = Pt[l, :l]
            np.multiply(a3[l, :l], Pt[l - 1, :l], out=o); o *= tq
            if l >= 2:
                ww = w[:l]
                np.multiply(b3[l, :l], Pt[l - 2, :l], out=ww); ww *= q2; o -= ww
        for m in range(M):
            np.matmul(K[m], Pt[:, m, :], out=R[m])
        cosm = np.cos(mvec * lo); sinm = np.sin(mvec * lo)
        X0, Y0, X1, Y1, X3, Y3 = (R[:, j] for j in range(6))
        dot = lambda A, B: np.einsum("mk,mk->k", A, B)
        gd[sl] = dot(cosm, X1) + dot(sinm, Y1)
        ge[sl] = (dot(cosm, mvec * Y0) - dot(sinm, mvec * X0)) / uu
        sA = dot(cosm, X1 - X0) + dot(sinm, Y1 - Y0)
        sB = dot(cosm, X3) + dot(sinm, Y3)
        gn[sl] = -(tt * sA - q * sB) / uu

    starts = range(0, n, chunk)
    n_threads = n_threads or os.cpu_count() or 1
    if n_threads == 1 or len(starts) == 1:
        for i0 in starts:
            work(i0)
    else:
        with ThreadPoolExecutor(n_threads) as ex:
            list(ex.map(work, starts))
    return gn, ge, gd


if HAVE_NUMBA:
    @_nb.njit(parallel=True, fastmath=False, cache=True)
    def _sh_ned_numba(r, t, u, lon, C, S, a_rec, b_rec, e_rec, R0, lmin):
        n = r.shape[0]
        M = C.shape[0]
        gn = np.empty(n); ge = np.empty(n); gd = np.empty(n)
        for i in _nb.prange(n):
            tt = t[i]; uu = u[i]; q = R0 / r[i]
            cm = np.empty(M); sm = np.empty(M)
            c1 = math.cos(lon[i]); s1 = math.sin(lon[i])
            cm[0] = 1.0; sm[0] = 0.0
            for m in range(1, M):
                cm[m] = cm[m - 1] * c1 - sm[m - 1] * s1
                sm[m] = sm[m - 1] * c1 + cm[m - 1] * s1
            P1 = np.zeros(M); P2 = np.zeros(M); P = np.zeros(M)
            pmm = 1.0
            Ql = q
            sn = 0.0; se = 0.0; sd = 0.0
            for l in range(M):
                for m in range(l):
                    P[m] = a_rec[l, m] * tt * P1[m] - b_rec[l, m] * P2[m]
                if l == 1:
                    pmm = math.sqrt(3.0) * uu
                elif l >= 2:
                    pmm = pmm * uu * math.sqrt((2 * l + 1) / (2.0 * l))
                P[l] = pmm
                if l >= lmin:
                    sP = 0.0; sE = 0.0; sQ = 0.0
                    for m in range(l + 1):
                        A = C[l, m] * cm[m] + S[l, m] * sm[m]
                        sP += P[m] * A
                        sE += m * P[m] * (S[l, m] * cm[m] - C[l, m] * sm[m])
                        sQ += e_rec[l, m] * P1[m] * A
                    sd += (l + 1) * Ql * sP
                    sn -= Ql * (l * tt * sP - sQ)
                    se += Ql * sE
                for m in range(l + 1):
                    P2[m] = P1[m]; P1[m] = P[m]
                Ql *= q
            gn[i] = sn / uu; ge[i] = se / uu; gd[i] = sd
        return gn, ge, gd


def gravity_acceleration_ecef(sh: SHCoefficients, xyz: np.ndarray, lmin: int = 0,
                              lmax: int | None = None, chunk: int = 1024,
                              return_ned: bool = False, engine: str = "auto") -> np.ndarray:
    """Gradient of the SH potential at ECEF positions xyz (n, 3) [m].

    Returns accelerations in ECEF [m/s^2], shape (n, 3)  (or local
    north/east/down components if return_ned=True). Equivalent to the Fortran
    GET_ACC_VEC_SH (ISELECT=1) followed by the FILLMAT rotations and GM/R0.

    engine : 'auto' (numba if installed, else numpy), 'numba' or 'numpy'.
    """
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    sh = sh.truncate(lmin, sh.lmax if lmax is None else lmax)
    C = np.ascontiguousarray(sh.C); S = np.ascontiguousarray(sh.S)
    R0 = float(sh.radius)
    a_rec, b_rec, e_rec = _recursion_coeffs(sh.lmax)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    t = z / r                                  # cos(colatitude) = sin(lat)
    u = np.sqrt(x * x + y * y) / r             # sin(colatitude) = cos(lat)
    lon = np.arctan2(y, x)

    if engine == "auto":
        engine = "numba" if HAVE_NUMBA else "numpy"
    if engine == "numba":
        if not HAVE_NUMBA:
            raise ImportError("engine='numba' requested but numba is not installed")
        gn, ge, gd = _sh_ned_numba(r, t, u, lon, C, S, a_rec, b_rec, e_rec, R0, int(lmin))
    else:
        gn, ge, gd = _sh_ned_numpy(r, t, u, lon, C, S, a_rec, b_rec, e_rec, R0, int(lmin), chunk)

    scale = sh.gm / R0 / r
    gn *= scale; ge *= scale; gd *= scale
    if return_ned:
        return np.column_stack([gn, ge, gd])
    # NED -> ECEF with geocentric latitude (FILLMAT: R3(-lon) R2(pi - colat))
    cl, sl = np.cos(lon), np.sin(lon)
    return np.column_stack([
        -t * cl * gn - sl * ge - u * cl * gd,
        -t * sl * gn + cl * ge - u * sl * gd,
        u * gn - t * gd,
    ])


# ==========================================================================
# LGD
# ==========================================================================
def compute_lgd(sh: SHCoefficients, orbit_c: pd.DataFrame, orbit_d: pd.DataFrame,
                lmin: int = 0, lmax: int | None = None, gm: float | None = None,
                radius: float | None = None, position_type: str = "geodetic",
                reference_radius: float = R_EARTH, chunk: int = 1024,
                engine: str = "auto") -> pd.DataFrame:
    """Line-of-sight gravity difference between GRACE-FO C and D.

    Parameters
    ----------
    sh : SHCoefficients (Stokes, dimensionless). For EWH input convert first
         with surface_mass_to_stokes.
    orbit_c, orbit_d : DataFrames from read_orbit_parquet / prepare_orbit.
    lmin, lmax : degree range (Fortran L1 and LMAX).
    gm, radius : override GM and R0 of the model (Fortran reads them from
         stdin). Overriding does NOT rescale the coefficients -- like Fortran.
    position_type : how latitude/longitude/altitude are defined
         ('geodetic', 'geocentric' or 'radius'; see orbit_to_ecef).

    Returns
    -------
    DataFrame with gps_time, lgd [m/s^2], and da_x, da_y, da_z = a_C - a_D in
    ECEF [m/s^2] (the Fortran 'acc1 - acc2'), the geodetic coordinates of the
    ECEF midpoint between C and D (mid_lat, mid_lon [deg], mid_alt [m],
    GRS80), and the inter-satellite range [m].
    """
    sh = sh.copy()
    if gm is not None:
        sh.gm = gm
    if radius is not None:
        sh.radius = radius
    if sh.C.shape[0] > 0 and lmin == 0 and abs(sh.C[0, 0]) > 0.5:
        warnings.warn("C00 ~ 1 is included (lmin=0): the LGD contains the central-field term")

    m = merge_orbits(orbit_c, orbit_d)
    r_c = orbit_to_ecef(m.lat_C, m.lon_C, m.alt_C, position_type, reference_radius)
    r_d = orbit_to_ecef(m.lat_D, m.lon_D, m.alt_D, position_type, reference_radius)

    mid_lat, mid_lon, mid_alt = ecef_to_geodetic(0.5 * (r_c + r_d))

    # both satellites in one call: one kernel launch, better load balance
    a_cd = gravity_acceleration_ecef(sh, np.vstack([r_c, r_d]), lmin, lmax, chunk, engine=engine)
    a_c, a_d = a_cd[:len(r_c)], a_cd[len(r_c):]

    dr = r_c - r_d
    rho = np.linalg.norm(dr, axis=1)
    e = dr / rho[:, None]                       # unit LOS, D -> C
    da = a_c - a_d
    lgd = np.einsum("ij,ij->i", da, e)          # = (a_D - a_C) . e_CD

    return pd.DataFrame({"gps_time": m.gps_time.to_numpy(),
                         "lgd": lgd,
                         "da_x": da[:, 0], "da_y": da[:, 1], "da_z": da[:, 2],
                         "mid_lat": mid_lat, "mid_lon": mid_lon, "mid_alt": mid_alt,
                         "range": rho})


def run(coef, orbit_c, orbit_d, output=None, *,
        lmin=0, lmax=None, gm=None, radius=None,
        reference=None, units="stokes", love=None, love_first_degree=0,
        position_type="geodetic", reference_radius=R_EARTH,
        engine="auto", save=True, verbose=True):
    """Synthesise LGDs and return them as a DataFrame.

    You can call this from the PyCharm Python console, e.g.
        out = run("model.gfc", "orbit_C.parquet", "orbit_D.parquet", lmax=96, save=False)
    """
    units = units.lower()
    valid_units = ("stokes", "ewh", "ewh_cm", "ewh_mm", "smd")
    if units not in valid_units:
        raise ValueError(f"units must be one of {valid_units}, got {units!r}")
    if position_type not in ("geodetic", "geocentric", "radius"):
        raise ValueError(f"invalid position_type {position_type!r}")
    if units != "stokes" and not love:
        raise ValueError("love (Love-number file) is required when units != 'stokes'")

    # Check every input file up front, so a typo fails immediately and clearly
    inputs = {"coef": coef, "orbit_c": orbit_c, "orbit_d": orbit_d,
              "reference": reference, "love": love}
    missing = [f"  {k}: {Path(v).resolve()}" for k, v in inputs.items()
               if v is not None and not Path(v).is_file()]
    if missing:
        raise FileNotFoundError("Input file(s) not found:\n" + "\n".join(missing))

    t0 = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)

    log(f"Reading coefficients: {coef}")
    sh = read_gfc(coef)
    if reference:
        log(f"Subtracting reference: {reference}")
        sh = sh - read_gfc(reference)
    if units != "stokes":
        log(f"Converting {units} -> Stokes with Love numbers from {love}")
        sh = surface_mass_to_stokes(sh, read_love_numbers(love, love_first_degree), units,
                                    gm=gm, radius=radius)

    log(f"Reading orbits: {orbit_c}, {orbit_d}")
    orb_c = read_orbit_parquet(orbit_c, "C")
    orb_d = read_orbit_parquet(orbit_d, "D")

    eng = ("numba" if HAVE_NUMBA else "numpy") if engine == "auto" else engine
    log(f"Computing LGD for {min(len(orb_c), len(orb_d))} epochs "
        f"(lmin={lmin}, lmax={lmax if lmax is not None else 'file max'}, engine={eng})")
    t1 = time.perf_counter()
    out = compute_lgd(sh, orb_c, orb_d, lmin=lmin, lmax=lmax, gm=gm, radius=radius,
                      position_type=position_type, reference_radius=reference_radius,
                      engine=engine)
    log(f"  synthesis: {time.perf_counter() - t1:.1f} s")

    if save and output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".parquet":
            out.to_parquet(output, index=False)
        else:
            out.to_csv(output, index=False, float_format="%.15e")
        log(f"{len(out)} epochs written to {output.resolve()}")

    log(f"Done in {time.perf_counter() - t0:.1f} s")
    return out


def main(argv=None):
    """Terminal interface (optional). Same options as run()."""
    import argparse
    p = argparse.ArgumentParser(description="Synthesise GRACE-FO LGDs from SH coefficients.")
    p.add_argument("coef"); p.add_argument("orbit_c"); p.add_argument("orbit_d"); p.add_argument("output")
    p.add_argument("--lmin", type=int, default=0)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--gm", type=float, default=None)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--reference", default=None)
    p.add_argument("--units", default="stokes", choices=["stokes", "ewh", "ewh_cm", "ewh_mm", "smd"])
    p.add_argument("--love", default=None)
    p.add_argument("--love-first-degree", type=int, default=0)
    p.add_argument("--position-type", default="geodetic", choices=["geodetic", "geocentric", "radius"])
    p.add_argument("--reference-radius", type=float, default=R_EARTH)
    a = p.parse_args(argv)
    return run(a.coef, a.orbit_c, a.orbit_d, a.output, lmin=a.lmin, lmax=a.lmax, gm=a.gm,
               radius=a.radius, reference=a.reference, units=a.units, love=a.love,
               love_first_degree=a.love_first_degree, position_type=a.position_type,
               reference_radius=a.reference_radius)


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # RUN SETTINGS: edit these and press Run (paths are relative to the
    # PyCharm working directory unless absolute)
    # ----------------------------------------------------------------------
    DATA = Path(r"../../orbit_data")          # <- your data folder
    DATA_SHC = Path(r"../../../solutions/CSR_CSR-Release-06_60x60_unfiltered")  # <- your data folder
    MODEL_NAME = "GSM-2_2020001-2020031_GRFO_UTCSR_BA01_0603.gfc"
    REF_path = "../../../Level3_EWH/models/GGM05C.gfc"

    SETTINGS = dict(
        coef=DATA_SHC / MODEL_NAME,
        orbit_c=DATA / "orbit_2020_01.parquet",
        orbit_d=DATA / "orbit_2020_01.parquet",
        output=DATA / "out" / "lgd_2020_01.parquet",  # .parquet or .csv
        lmin=2,
        lmax=60,
        reference=REF_path,          # e.g. DATA / "mean_field.gfc"
        units="stokes",          # "stokes", "ewh", "ewh_cm", "ewh_mm", "smd"
        love=None,               # required if units != "stokes"
        love_first_degree=0,
        position_type="geodetic",
        gm=None,                 # None = take from coefficient file
        radius=None,
        engine="auto",           # "auto" = numba if installed, else numpy
        save=True,
    )
    # ----------------------------------------------------------------------

    if len(sys.argv) > 1:        # arguments given -> use the terminal interface
        out = main()
    else:                        # pressed Run in PyCharm -> use SETTINGS
        out = run(**SETTINGS)