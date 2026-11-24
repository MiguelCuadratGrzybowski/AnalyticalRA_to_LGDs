from scipy.stats import binned_statistic_2d
from scipy.signal import csd, welch, coherence, sosfiltfilt, butter
import numpy as np

# orbit
n_mean = 1.1170e-03         # [rad/s]
dt = 5                      # [s]
f_cpr = n_mean / (2 * np.pi)

# Welch segment
nperseg_spec = 2 * int(2 ** np.ceil(np.log2(2 / f_cpr / dt)))
average_input_str = "median"

# ...
SCALE = 1.0                 # 1e9 if arrays are in m/s^2 rather than nm/s^2
NM = 1e9                    # m/s^2 -> nm/s^2 (applied after filtering)
F_CROSS_LO = 0.9e-3         # [Hz]
F_CROSS_HI = 0.95e-3        # [Hz]

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

# CW closure
KAPPA = "spherical"         # or a float, e.g. 1.0
HILBERT_SIGN = 1            # flip to -1 if overlap band (~0.9-1.5 mHz) disagrees with TF
# =============================================================================
# Statistics
# =============================================================================
def rmeds(x, scale=1.0):
    """Root median square sqrt(median(x^2)), NaN-safe."""
    x = np.asarray(x, dtype=float) * scale
    return np.sqrt(np.nanmedian(x**2))


def rms(x, scale=1.0):
    """Root mean square sqrt(mean(x^2)), NaN-safe."""
    x = np.asarray(x, dtype=float) * scale
    return np.sqrt(np.nanmean(x**2))


def tag(x):
    return f" — RMedS = {rmeds(x, SCALE):.3f} nm/s²"


# =============================================================================
# Empirical admittance (Ghobadi-Far et al., 2018)
# =============================================================================
def empirical_admittance(residual_rho_ddot, residual_lgd, dt, nperseg=nperseg_spec, window="hann"):
    """Z(f) = S_xy / S_xx, coherence = |S_xy|^2 / (S_xx S_yy).
    x = residual range acceleration, y = residual LGD. Welch averaging."""
    x = np.asarray(residual_rho_ddot, float)
    y = np.asarray(residual_lgd, float)
    kw = dict(fs=1.0 / dt, window=window, nperseg=nperseg)
    f, Sxx = welch(x, average=average_input_str, **kw)
    _, Sxy = csd(x, y, average=average_input_str, **kw)  # conj(X) Y
    _, gamma2 = coherence(x, y, **kw)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = Sxy / Sxx
    return f, Z.real, np.abs(Z), np.degrees(np.angle(Z)), gamma2


def fill_admittance(f_e, Z_GF, gamma2, freqs, coh_min=0.9, mode="interp",
                    H_fill=None, f_max=9e-3):
    """
    Replace the empirical admittance where gamma2 < coh_min, only for 0 < f <= f_max.
    mode = "zero"   : set unreliable in-band bins to 0
           "interp" : interpolate Re and Im linearly between reliable bins
           "cw"     : use H_fill (e.g. CW admittance on `freqs`) there
    Returns complex admittance on `freqs` and the reliability mask on `freqs`.
    """
    Z = np.nan_to_num(Z_GF, nan=0.0)
    g = np.nan_to_num(gamma2, nan=0.0)
    ok = (g >= coh_min) & (f_e > 0) & (f_e < f_max)
    if not ok.any():
        raise ValueError("no bin reaches coh_min")

    ok_full = np.interp(freqs, f_e, ok.astype(float)) > 0.5
    Z_full = np.interp(freqs, f_e, Z.real) + 1j * np.interp(freqs, f_e, Z.imag)

    in_band = freqs > 0 if f_max is None else (freqs > 0) & (freqs <= f_max)
    fill = ~ok_full & in_band

    if mode == "zero":
        Z_full[fill] = 0.0
    elif mode == "interp":
        Zi = np.interp(freqs, f_e[ok], Z[ok].real) + 1j * np.interp(freqs, f_e[ok], Z[ok].imag)
        Z_full[fill] = Zi[fill]
    elif mode == "cw":
        if H_fill is None:
            raise ValueError("mode='cw' needs H_fill on `freqs`")
        Z_full[fill] = np.nan_to_num(H_fill, nan=0.0, posinf=0.0, neginf=0.0)[fill]
    else:
        raise ValueError(mode)

    Z_full[freqs == 0] = 0.0
    return Z_full, ok_full


# =============================================================================
# kappa(omega)
# =============================================================================
def kappa_lk(l, k, alpha=0):
    return k * (l + 2) / (k**2 + l + 1)


def kappa_lk_exact(l, k, alpha):
    return (((l + 1) * np.sin(alpha * k) + k * alpha * np.cos(alpha * k))
            / ((l + 1) * alpha * np.cos(alpha * k) + k * np.sin(alpha * k)))


def sigma_kaula(l, A=1e-5):
    """Kaula's rule sigma_l = A / l^2 (amplitude cancels in kappa)."""
    return A / np.asarray(l, dtype=float) ** 2


def weight(l, k, rho, h, R=6371e3, sigma_l=sigma_kaula, gradient_limit=True):
    """Weight of the (l, k) term in kappa(omega) = sum_l kappa_lk w_lk / sum_l w_lk.

        w_lk = (2l+1) sigma_l^2 / (l+1) * (R/r)^(2l) * |delta g_y,lk|^2

    with alpha = rho/(2r):
        exact:    (l+1) sin(alpha) cos(k alpha) + k cos(alpha) sin(k alpha)
        gradient: alpha (k^2 + l + 1)   (= rho T_yy / 2)
    Constants common to all l cancel in kappa.
    """
    l = np.asarray(l, dtype=float)
    r = R + h
    alpha = rho / (2.0 * r)

    coeff_power = (2.0 * l + 1.0) * sigma_l(l) ** 2 / (l + 1.0)
    upward = (R / r) ** (2.0 * l)
    if gradient_limit:
        gy = alpha * (k**2 + l + 1.0)
    else:
        gy = (l + 1.0) * np.sin(alpha) * np.cos(k * alpha) + k * np.cos(alpha) * np.sin(k * alpha)
    return coeff_power * upward * gy**2


def kappa_eff(k, kfun, rho, h, R=6371e3, Lmax=60):
    alpha = rho / (2.0 * (h + R))
    l = np.arange(k, Lmax + 1, 2).astype(float)
    w = weight(l, k, rho, h, R=R)
    return np.sum(kfun(l, k, alpha) * w) / np.sum(w)


# =============================================================================
# CW closure
# =============================================================================
def lgd_closure_spectrum(res_rho_ddot, dt, n, kappa="spherical", hilbert_sign=+1,
                         h=450e3, rho=220e3, Lmax=60):
    """rFFT spectrum of the closure-based residual LGD, its freqs and Z(omega)."""
    N = len(res_rho_ddot)
    freqs = np.fft.rfftfreq(N, dt)
    omega = 2.0 * np.pi * freqs
    Rdd = np.fft.rfft(res_rho_ddot)

    if kappa == "spherical":
        n_const = n if np.isscalar(n) else np.median(n)
        k = omega / n_const
        k_int = np.arange(2, Lmax + 1)
        kappa_int = np.array([kappa_eff(ki, kappa_lk, rho, h, R=6371e3, Lmax=Lmax) for ki in k_int])
        kappa = np.interp(k, k_int, kappa_int)
    else:
        kappa = float(kappa) * np.ones_like(omega)

    kappa = hilbert_sign * kappa
    D = omega**2 - 2.0 * n * kappa * omega + 3.0 * n**2
    Z = (omega**2 - n**2) / D
    return Z * Rdd, freqs, Z

def asd(x, fs, nperseg):
    f, P = welch(x, fs=fs, nperseg=nperseg, window="hann", detrend="constant")
    return f[1:], np.sqrt(P[1:])  # drop DC for log axes

def grid_median(lon, lat, vals, lon_edges, lat_edges):
    """Median per (lon, lat) bin -> shape (nlon, nlat)."""
    return binned_statistic_2d(
        lon, lat, vals, statistic=np.nanmedian, bins=[lon_edges, lat_edges]
    ).statistic

