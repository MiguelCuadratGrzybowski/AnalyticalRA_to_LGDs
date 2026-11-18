import os
import glob
from utilities.map_plotter import *
from scipy.stats import binned_statistic_2d
from scipy.signal import csd, welch, coherence, welch, sosfiltfilt
from utilities.Utility_functions_TimeFilt_new import *
from utilities.Utility_functions_DIFF_TIME import derivative

# Choose solution
sol = "csr"
year, month = 2007, 9
year_month = f"{year}_{int(month):02d}"
# Define grid resolution
dlat, dlon = 1, 1   # 0.5° x 0.5° bins
lat_edges = np.arange(-90, 90 + dlat, dlat)
lon_edges = np.arange(-180, 180 + dlon, dlon)

# Load residuals file
main_dir = "../solutions/orbit_data"
data_static = np.load(f"{main_dir}/orbit_ggm05c.npy")
# Extract fields
lat = data_static[:, 0]*180/np.pi
lon = data_static[:, 1]*180/np.pi
dt = 5.0
# Sampling frequency (Hz)
fs = 1 / dt   # 5-second sampling → 0.2 Hz
# High-pass cutoff frequency (Hz;)
fc = 0.36 *1e-3    # 0.9 mHz
fc_low = 11.*1e-3
sos = butter(N=4, fs=fs, btype='bandpass', output='sos', Wn = [fc, fc_low])


# # retrieve for static
# rho_static = data_static[:, 2]
# rho_dot_static = data_static[:, 5]
# xAB_norm = data_static[:, 6]
# # mean motion from the static orbit: (|v_AB|^2 - rho_dot^2)/rho^2 ~ n^2
# n_mean = np.sqrt(np.median((xAB_norm ** 2 - rho_dot_static ** 2) / rho_static ** 2))
# n_static = np.sqrt((xAB_norm ** 2 - rho_dot_static ** 2) / rho_static ** 2)

# Orbital frequency
n_mean = 1.1170e-03  # rad/s
f_cpr = n_mean / (2 * np.pi)
# Welch segment
nperseg_spec = int(2 ** np.ceil(np.log2(2 / f_cpr / dt)))
nperseg_spec = int(4*nperseg_spec)
average_input_str = "median"
print(f"n = {n_mean:.4e} rad/s  (1 cpr = {f_cpr * 1e3:.4f} mHz)")
print(f"nperseg = {nperseg_spec} [-]")


# --- settings ---------------------------------------------------------------
KAPPA = "spherical"  # or a float, e.g. 1.0
# KAPPA = 1.0
HILBERT_SIGN = 1  # flip to -1 if overlap band (~0.9-1.5 mHz) disagrees with TF
F_CROSS_LO = 0.9e-3  # [Hz] below: closure only
F_CROSS_HI = 0.95e-3  # [Hz] above: Ghobadi-Far TF only (cosine blend in between)

# map gridding
LON_COL, LAT_COL = 1, 0
GRID_RES = 2.0  # [deg]

# map grid
lon_edges = np.arange(-180, 180 + GRID_RES, GRID_RES)
lat_edges = np.arange(-90, 90 + GRID_RES, GRID_RES)
lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])

OUTDIR_MAPS = "plots/maps"
OUTDIR_SPEC = "plots/spectra"


def empirical_admittance(residual_rho_ddot, residual_lgd, dt, nperseg=nperseg_spec, window="hann", detrend="constant"):
    """Z(f) = S_xy / S_xx,  coherence = |S_xy|^2 / (S_xx S_yy),
    x = residual range acceleration, y = residual LGD (forward-modelled). Welch averaging."""
    x = np.asarray(residual_rho_ddot, float); y = np.asarray(residual_lgd, float)
    kw = dict(fs=1.0 / dt, window=window, nperseg=nperseg,
              # detrend=detrend
              )
    f, Sxx = welch(x, average=average_input_str, **kw)
    _, Sxy = csd(x, y, average=average_input_str, **kw)            # conj(X) Y
    _, gamma2 = coherence(x, y, **kw)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = Sxy / Sxx
    return f, Z.real, np.abs(Z), np.degrees(np.angle(Z)), gamma2

def fill_admittance(f_e, Z_GF, gamma2, freqs, coh_min=0.9, mode="interp",
                    H_fill=None, f_max=9e-3):
    """
    Replace the empirical admittance where gamma2 < coh_min, but ONLY for
    0 < f <= f_max. Above f_max the empirical admittance is kept unchanged
    (no interpolation, no zeroing, no filling).
    mode = "zero"   : set unreliable in-band bins to 0
           "interp" : interpolate Re and Im linearly in f between reliable bins
                      (constant value below the lowest reliable bin)
           "cw"     : use H_fill (e.g. your CW admittance on `freqs`) there
    Returns admittance on `freqs` (complex) and the reliability mask on `freqs`.
    """
    Z = np.nan_to_num(Z_GF, nan=0.0)
    g = np.nan_to_num(gamma2, nan=0.0)
    ok = (g >= coh_min) & (f_e > 0) & (f_e < f_max)
    if not ok.any():
        raise ValueError("no bin reaches coh_min")

    # reliability decided on the estimation grid, then mapped to the full grid
    ok_full = np.interp(freqs, f_e, ok.astype(float)) > 0.5
    Z_full = np.interp(freqs, f_e, Z.real) + 1j * np.interp(freqs, f_e, Z.imag)

    in_band = freqs > 0 if f_max is None else (freqs > 0) & (freqs <= f_max)
    fill = ~ok_full & in_band            # unreliable bins below f_max only

    if mode == "zero":
        Z_full[fill] = 0.0
    elif mode == "interp":
        Zi = (np.interp(freqs, f_e[ok], Z[ok].real)
              + 1j * np.interp(freqs, f_e[ok], Z[ok].imag))
        Z_full[fill] = Zi[fill]
    elif mode == "cw":
        if H_fill is None:
            raise ValueError("mode='cw' needs H_fill on `freqs`")
        Z_full[fill] = np.nan_to_num(H_fill, nan=0.0, posinf=0.0, neginf=0.0)[fill]
    else:
        raise ValueError(mode)

    Z_full[freqs == 0] = 0.0             # only the DC bin; above f_max untouched
    return Z_full, ok_full

def rmeds(x, scale=1.0):
    """Root median square sqrt(median(x^2)), NaN-safe; scale e.g. 1e9 if x is in m/s^2."""
    x = np.asarray(x, dtype=float) * scale
    return np.sqrt(np.nanmedian(x**2))

SCALE = 1.0          # set to 1e9 if the arrays are in m/s^2 rather than nm/s^2
def tag(x):
    return f" — RMedS = {rmeds(x, SCALE):.3f} nm/s²"


def kappa_lk(l, k, alpha=0):
    return  k * (l + 2) / (k ** 2 + l + 1)

def kappa_lk_exact(l, k, alpha):
    return  ((l+1)*np.sin(alpha*k) + k*alpha*np.cos(alpha*k)) / ((l+1)*alpha*np.cos(alpha*k) + k*np.sin(alpha*k))

def sigma_kaula(l, A=1e-5):
    """Kaula's rule: RMS of one normalised coefficient of degree l, sigma_l = A / l^2.
    Only the shape l^-2 matters for kappa(omega); the amplitude A cancels."""
    return A / np.asarray(l, dtype=float) ** 2


def weight(l, k, rho, h, R=6371e3, sigma_l=sigma_kaula, gradient_limit=True):
    """Weight of the (l, k) term in kappa(omega) = sum_l kappa_lk w_lk / sum_l w_lk.

    w_lk = expected power of delta g_y = Delta g . y_hat in the Hill frame of O
    (virtual reference point midway between A and B), for degree l and order k:

        w_lk = (2l+1) sigma_l^2 / (l+1)  *  (R/r)^(2l)  *  |delta g_y,lk|^2

    1. (2l+1) sigma_l^2 / (l+1): degree-l power, spread evenly over the l+1 along-orbit
       orders k = l, l-2, ..., -l (approximates Kaula's F_lmp(I) for a near-polar orbit).
    2. (R/r)^(2l): upward continuation from R to r = R + h.
    3. |delta g_y,lk| per unit potential term, with alpha = rho/(2r):
         exact:    (l+1) sin(alpha) cos(k alpha) + k cos(alpha) sin(k alpha)
         gradient: alpha (k^2 + l + 1)   (= rho T_yy / 2)
       (l+1) sin(alpha) cos(k alpha): radial gravity at A, B projected on O's along-track
       axis (frame rotation); k cos(alpha) sin(k alpha): differenced along-track gravity.

    Constants common to all l (GM/r^2, factor 2, A) are dropped; they cancel in kappa.
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
    alpha = rho / (2.0 * (h+R))
    l = np.arange(k, Lmax + 1, 2).astype(float)
    w = weight(l, k, rho, h, R=R)
    return np.sum(kfun(l, k, alpha) * w) / np.sum(w)

# =============================================================================
# Closure + blending
# =============================================================================
def lgd_closure_spectrum(res_rho_ddot, dt, n,
                         kappa="spherical", hilbert_sign=+1, h=450e3, rho=220e3, Lmax=60):
    """Return rFFT spectrum of the closure-based residual LGD and its freqs."""
    N = len(res_rho_ddot)
    freqs = np.fft.rfftfreq(N, dt)
    omega = 2.0 * np.pi * freqs

    Rdd = np.fft.rfft(res_rho_ddot)

    if kappa == "spherical":
        n_constant = n if np.isscalar(n) else np.median(n)
        k = omega / n_constant
        # kappa on integer orders 2..Lmax
        k_int = np.arange(2, Lmax + 1)
        kappa_func_to_be_used = kappa_lk
        kappa_int = np.array([kappa_eff(k, kappa_func_to_be_used, rho, h, R=6371e3, Lmax=Lmax) for k in k_int])
        kappa = np.interp(k, k_int, kappa_int)  # kappa(omega)
    else:
        kappa = float(kappa) * np.ones_like(omega)

    kappa = hilbert_sign * kappa

    D = (omega ** 2 - 2.0 * n * kappa * omega + 3.0 * n ** 2)
    Z = (omega ** 2 - n ** 2) / D
    G = Z * Rdd

    return G, freqs, Z

# =============================================================================
# Plotting
# =============================================================================
def plot_and_save(stat, lon_c, lat_c, vmin, vmax, cmap, label, title, fname, outdir="plots"):
    Lon, Lat = np.meshgrid(lon_c, lat_c)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    plot_cartopy_map(ax)

    pc = ax.pcolormesh(
        Lon, Lat, stat.T,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    cbar = plt.colorbar(pc, orientation="horizontal", pad=0.05, fraction=0.05)
    cbar.set_label(label, fontsize=13)
    plt.title(title, fontsize=17)

    # --- save figure ---
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, fname)
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def asd(x, fs, nperseg):
    f, P = welch(x, fs=fs, nperseg=nperseg, window="hann", detrend="constant")
    return f[1:], np.sqrt(P[1:])  # drop DC for log axes


def plot_spectra(signals, errors, fs, nperseg, f_cpr, title, fname, outdir="plots"):
    """Two panels: ASD of signals/estimates (top), ASD of errors (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for name, (x, style) in signals.items():
        f, A = asd(x, fs, nperseg)
        ax1.loglog(f * 1e3, A, label=name, **style)
    for name, (x, style) in errors.items():
        f, A = asd(x, fs, nperseg)
        ax2.loglog(f * 1e3, A, label=name, **style)

    for ax in (ax1, ax2):
        ax.axvline(f_cpr * 1e3, color="0.4", ls=":", lw=1)
        ax.axvspan(F_CROSS_LO * 1e3, F_CROSS_HI * 1e3, color="0.85", alpha=0.6, lw=0)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10, loc="lower left")
        ax.set_ylabel(r"ASD [nm/s$^2$/$\sqrt{\rm Hz}$]", fontsize=12)

    # ax1.text(f_cpr * 1e3, ax1.get_ylim()[1], " 1 cpr", va="top", fontsize=9, color="0.3")
    ax1.set_title(title, fontsize=14)
    ax1.set_ylim(1e-3, 5e2)
    ax2.set_ylim(1e-3, 5e2)

    ax1.set_xlim(0.25, 20)
    ax2.set_xlim(0.25, 20)
    ax2.set_xlabel("Frequency [mHz]", fontsize=12)

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, fname)
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


# =============================================================================
# Loop over files
# =============================================================================
for filepath in sorted(glob.glob(f"{main_dir}/orbit_*_????_??.npy")):
    fname = os.path.basename(filepath)
    year_month = "_".join(fname.split("_")[2:4]).replace(".npy", "")
    sol = fname.split("_")[1]

    print(f"Processing {fname}")
    data_month = np.load(filepath)

    # residual signals
    t = np.arange(len(data_month[:, 3])) * dt  # or data_month[:, 0]
    residual_lgd = data_month[:, 3] - data_static[:, 3]
    residual_rho = data_month[:, 2] - data_static[:, 2]
    # treat both signals identically before comparing spectra
    # residual_lgd, _ = remove_1cpr(t, residual_lgd, period=2*np.pi/n_mean,
    #                               drift=True, linear_cpr=True, n_rev=2)
    residual_rho_ddot = derivative(residual_rho, dt, deriv=2, n_points=9, t=t)
    # residual_rho_ddot , _ = remove_1cpr(t, residual_rho_ddot , period=2*np.pi/n_mean,
    #                                     drift=True, linear_cpr=True, n_rev=2)

    # =================================================================
    # ========================= NEW CW METHOD =========================
    # =================================================================
    G_low, freqs, Z_CW = lgd_closure_spectrum(residual_rho_ddot, dt, n_mean,
        kappa=KAPPA, hilbert_sign=HILBERT_SIGN,
    )
    residual_lgdnewTF = np.fft.irfft(G_low, n=len(residual_rho)) - n_mean**2 * residual_rho

    # =================================================================
    # ========================= OLD GF METHOD =========================
    # =================================================================
    f_e, Z_GF, _, _, gamma2 = empirical_admittance(residual_rho_ddot, residual_lgd, dt, nperseg=nperseg_spec)
    N = len(residual_rho_ddot)
    # f_full = np.fft.rfftfreq(N, dt)  # ...
    Z_GF_c = np.nan_to_num(Z_GF, nan=0.0)  # f = 0 bin is NaN (0/0)
    Z_full = np.interp(freqs, f_e, Z_GF_c)  # onto the full grid

    CPR = 1 * n_mean / (2*np.pi)  # [Hz]; better: from the orbit
    coh_min = 0.9  # coherence threshold for "reliable"
    corr_min = np.sqrt(coh_min)  # coherence threshold for "reliable"
    gamma2_c = np.nan_to_num(gamma2, nan=0.0)  # f = 0 bin is NaN (0/0)
    gamma2_full = np.interp(freqs, f_e, gamma2_c)  # onto the full grid
    # Z_full, _ = fill_admittance(freqs, Z_full, gamma2_full, freqs, coh_min, mode="interp")

    # ---------------------------------------------------------------- figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 11), height_ratios=[3, 2])
    fm = 1e3  # Hz -> mHz

    # (a) admittance
    ax = axs[0]
    low = np.nan_to_num(np.sqrt(gamma2_full), nan=0) >= corr_min
    ax.fill_between(freqs*fm, -1, 5, where=low, color="0.9", step="mid",
                    label=rf"$\gamma(f)>{corr_min:.2f}$")
    ax.plot(freqs*fm, np.real(Z_full), color="0.25", lw=1.2,
            label=r"Empirical (GF method): Re$(S_{xy}(f)/S_{xx}(f))$")
    ax.plot(freqs * fm, np.real(Z_CW + (n_mean/(freqs*2*np.pi))**2), color="tab:green", lw=1.5,
            label=r"Analytical CW: $Z_{\rm CW}(f)+n^2/(2 \pi f)^2$")
    for k in range(1, 6):
        ax.axvline(k * CPR * fm, color="0.6", lw=0.6, ls=":")
    ax.set(xscale="log", xlim=(0.15, 20), ylim=(-1, 5), ylabel="Admittance [-]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    # (b) coherence
    ax = axs[1]
    ax.plot(freqs * fm, np.sqrt(gamma2_full), color="0.25", lw=1.2)
    ax.axhline(corr_min, color="tab:red", lw=1, ls="--", label=rf"$\gamma(f)>{corr_min:.2f}$)")
    for k in range(1, 6):
        ax.axvline(k * CPR * fm, color="0.6", lw=0.6, ls=":")
    ax.set(xscale="log", xlim=(0.15, 20), ylim=(0, 1.05),
           ylabel=r"Correlation spectrum $\gamma(f)$", xlabel="Frequency [mHz]")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    plt.show()


    # GF EXACT (empirical ratio from simulations).
    residual_lgdTF_exact = np.fft.irfft(Z_full * np.fft.rfft(residual_rho_ddot), n=N)
    # FIT
    Z_GF_fit = transfer_function(freqs)
    residual_lgdTF = compute_lgd_new(residual_rho_ddot)

    # ===================
    # ==== FILTERING ====
    # ===================
    residual_lgdnewTF = sosfiltfilt(sos, residual_lgdnewTF) * 1e9

    residual_lgdTF = sosfiltfilt(sos, residual_lgdTF) * 1e9
    residual_lgdTF_exact = sosfiltfilt(sos, residual_lgdTF_exact) * 1e9

    # FILTERED TRUE - REFERENCE
    residual_rho_ddot = sosfiltfilt(sos, residual_rho_ddot) * 1e9
    residual_lgd = sosfiltfilt(sos, residual_lgd) * 1e9


    # ---------------- admittance comparison ----------------
    freqs_mHz = np.asarray(freqs) * 1e3
    Z_CW_real = np.real(np.asarray(Z_CW))  # hilbert_sign / complex safety
    omega_cw = 2 * np.pi * np.asarray(freqs)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z_CW_total = Z_CW_real + n_mean ** 2 / omega_cw ** 2  # incl. -n^2 drho term (total residual LGD)

    m_e = (freqs >= 0.25e-3) & (freqs <= 20e-3)
    m_c = (freqs >= 0.25e-3) & (freqs <= 20e-3)
    m_fit = (freqs >= 0.9e-3) & (freqs <= 20e-3)
    m_fit_l = (freqs >= 0.25e-3) & (freqs <= 0.9e-3)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs_mHz[m_e], Z_full[m_e],  color="k", alpha=0.7,
            label=r"Ghobadi-Far et al. (2018): ${\rm Re}(S_{xy}(f)/S_{xx}(f))$")
    ax.plot(freqs_mHz[m_fit], Z_GF_fit[m_fit], color="tab:red", lw=2.0,
            label=r"Ghobadi-Far et al. (2018) fit: $1+3.45\times10^{-4}f^{-1.04}$")
    ax.plot(freqs_mHz[m_fit_l], Z_GF_fit[m_fit_l], color="tab:red", lw=2.0, alpha=0.35)
    # ax.plot(freqs_mHz[m_c], Z_CW_real[m_c], color="tab:green", lw=1.8,
    #         label=r"Analytical (CW): $Z(\omega)$, $\kappa(\omega)$")
    ax.plot(freqs_mHz[m_c], Z_CW_total[m_c], color="tab:green", lw=1.2,
            label=r"Analytical (CW): $Z(\omega) + n^2/\omega^2$")
    ax.axvspan(0.1, 0.9, color="0.93", zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(-1, 5)
    ticks = [0.3, 0.5, 0.7, 0.9, 2, 5, 10, 20]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(0.25, 20)
    ax.set_xlabel("Frequency [mHz]", fontsize=14)
    ax.set_ylabel("Admittance [-]", fontsize=14)
    ax.set_title(f"Admittance range acceleration → LGD — {sol} {year_month}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR_SPEC, f"admittance_{sol}_{year_month}.png"), dpi=200)
    plt.show()

    # errors
    residual_gc = residual_lgd - residual_rho_ddot
    error_TF = residual_lgd - residual_lgdTF
    error_TF_exact = residual_lgd - residual_lgdTF_exact
    error_newTF = residual_lgd - residual_lgdnewTF

    print("====== Errors =======")

    # ---------------- spectra ----------------
    signals = {
        r"$\delta g^{\rm LoS}_{\rm AB}$ (true)" + tag(residual_lgd): (residual_lgd, dict(color="k", lw=1.6)),
        r"$\delta\ddot\rho$" + tag(residual_rho_ddot): (residual_rho_ddot, dict(color="0.55", lw=1.0)),
        r"TF (Ghobadi-Far Z(f) = Sxy(f)/Sxx(f))" + tag(residual_lgdTF_exact): (
        residual_lgdTF_exact, dict(color="tab:blue", lw=0.9, ls="--")),
        r"TF (Ghobadi-Far Z(f) fit)" + tag(residual_lgdTF): (residual_lgdTF, dict(color="tab:blue", lw=1.1)),
        r"Analytical TF (TF-CW)" + tag(residual_lgdnewTF): (
        residual_lgdnewTF, dict(color="tab:green", lw=1.0, ls="--")),
    }
    errors = {
        r"$\epsilon_{\rm TF-fit}$" + tag(error_TF): (error_TF, dict(color="tab:blue", lw=1.1, ls="-")),
        r"$\epsilon_{\rm TF}$" + tag(error_TF_exact): (error_TF_exact, dict(color="tab:blue", lw=0.9, ls="--")),
        r"$\epsilon_{\rm TF-CW}$" + tag(error_newTF): (error_newTF, dict(color="tab:green", lw=1.0, ls="--")),
        r"$\delta g_c$ (centrifugal)" + tag(residual_gc): (residual_gc, dict(color="0.55", lw=1.0)),
    }
    plot_spectra(
        signals, errors, fs, nperseg_spec, f_cpr,
        title=f"Residual LGD spectra — {sol} {year_month}",
        fname=f"spectra_{sol}_{year_month}.png",
        outdir=OUTDIR_SPEC,
    )

    # ---------------- maps ----------------
    fields = {
        "res_LGD": (
            residual_lgd, -15, 15, "seismic", "[nm/s²]",
            r"$\delta g^{\rm LoS}_{\rm AB}$ (res. LGD)" + tag(residual_lgd)
        ),
        "res_rhoddot": (
            residual_rho_ddot, -15, 15, "seismic", "[nm/s²]",
            r"$\delta \ddot{\rho}$" + tag(residual_rho_ddot)
        ),
        "res_LGD_tf": (
            residual_lgdTF, -15, 15, "seismic", "[nm/s²]",
            r"$(\delta g^{\rm LoS}_{\rm AB})_{\rm TF-fit}$ (res. LGD estimate)" + tag(residual_lgdTF)
        ),
        "res_LGD_tf_exact": (
            residual_lgdTF_exact, -15, 15, "seismic", "[nm/s²]",
            r"$(\delta g^{\rm LoS}_{\rm AB})_{\rm TF}$ (res. LGD estimate)" + tag(residual_lgdTF_exact)
        ),
        "res_LGD_approx": (
            residual_lgdnewTF, -15, 15, "seismic", "[nm/s²]",
            r"$(\delta g^{\rm LoS}_{\rm AB})_{\rm TF-CW}$ (res. LGD estimate)" + tag(residual_lgdnewTF)
        ),
        "eps_TF": (
            error_TF, -1, 1, "seismic", "[nm/s²]",
            r"$\epsilon_{\rm TF-fit}$ (transfer function error)" + tag(error_TF)
        ),
        "eps_TF_exact": (
            error_TF_exact, -1, 1, "seismic", "[nm/s²]",
            r"$\epsilon_{\rm TF}$ (transfer function error)" + tag(error_TF_exact)
        ),
        "eps_approx": (
            error_newTF, -1, 1, "seismic", "[nm/s²]",
            r"$\epsilon_{\rm TF-CW}$" + tag(error_newTF)
        ),
        "res_gc": (
            residual_gc, -5, 5, "seismic", "[nm/s²]",
            r"$\delta g_c$ (res. centrifugal acceleration)" + tag(residual_gc)
        ),
    }

    lon = (data_month[:, LON_COL]*180/np.pi + 180.0) % 360.0 - 180.0
    lat = data_month[:, LAT_COL]*180/np.pi

    for key, (vals, vmin, vmax, cmap, unit, title) in fields.items():
        stat = binned_statistic_2d(
            lon, lat, vals, statistic=lambda x: np.nanmedian(x), bins=[lon_edges, lat_edges]
        ).statistic  # shape (nlon, nlat)
        plot_and_save(
            stat, lon_c, lat_c, vmin, vmax, cmap,
            label=unit,
            title=f"{title} — {year_month}",
            fname=f"{key}_{sol}_{year_month}.png",
            outdir=OUTDIR_MAPS,
        )
