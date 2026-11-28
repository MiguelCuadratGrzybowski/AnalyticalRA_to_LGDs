from utilities.map_plotter import *                      # plot_cartopy_map, ccrs, ...
import numpy as np
import os
from utilities.Utility_analytical_equations import asd
from utilities.Utility_functions_TimeFilt_new import transfer_function   # transfer_function, compute_lgd_new, ...

# =============================================================================
# Plotting
# =============================================================================
def plot_and_save(stat, lon_c, lat_c, vmin, vmax, cmap, label, title, fname, outdir="plots", save=True, show=False):
    """stat has shape (nlon, nlat)."""
    Lon, Lat = np.meshgrid(lon_c, lat_c)
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    plot_cartopy_map(ax)
    pc = ax.pcolormesh(Lon, Lat, stat.T, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree())
    cbar = plt.colorbar(pc, orientation="horizontal", pad=0.05, fraction=0.05)
    cbar.set_label(label, fontsize=13)
    plt.title(title, fontsize=17)
    if save:
        outfile = os.path.join(outdir, fname)
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    if not show:
        plt.close(fig)
    print(f"Saved {outfile}")

def plot_spectra(signals, errors, fs, nperseg, f_cpr, title, fname, outdir="plots"):
    """ASD of signals/estimates (top) and of errors (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for ax, group in ((ax1, signals), (ax2, errors)):
        for name, (x, style) in group.items():
            f, A = asd(x, fs, nperseg)
            ax.loglog(f * 1e3, A, label=name, **style)
        ax.axvline(f_cpr * 1e3, color="0.4", ls=":", lw=1)
        ax.axvspan(0.36e-3 * 1e3, 0.9e-3 * 1e3, color="0.85", alpha=0.6, lw=0)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10, loc="lower left")
        ax.set_ylabel(r"ASD [nm/s$^2$/$\sqrt{\rm Hz}$]", fontsize=12)
        ax.set(xlim=(0.25, 20), ylim=(1e-3, 5e2))
    ax1.set_title(title, fontsize=14)
    ax2.set_xlabel("Frequency [mHz]", fontsize=12)
    outfile = os.path.join(outdir, fname)
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_admittance(freqs, Z_full, Z_CW, title, fname, n_mean, option_bool=True):
    """Empirical GF vs GF fit vs analytical CW admittance."""
    freqs_mHz = freqs * 1e3
    omega = 2 * np.pi * freqs
    with np.errstate(divide="ignore", invalid="ignore"):
        Z_CW_total = np.real(Z_CW) + n_mean**2 / omega**2  # incl. -n^2 drho term
    Z_GF_fit = transfer_function(freqs)

    m_c = (freqs >= 0.25e-3) & (freqs <= 20e-3)
    m_fit = (freqs >= 0.9e-3) & (freqs <= 20e-3)
    m_fit_l = (freqs >= 0.25e-3) & (freqs <= 0.9e-3)

    fig, ax = plt.subplots(figsize=(12, 5))
    if option_bool:
        ax.plot(freqs_mHz[m_c], np.real(Z_full[m_c]), color="k", alpha=0.7,
                label=r"Ghobadi-Far et al. (2018): ${\rm Re}(S_{xy}(f)/S_{xx}(f))$")
    ax.plot(freqs_mHz[m_fit], Z_GF_fit[m_fit], color="tab:red", lw=2.0,
            label=r"Ghobadi-Far et al. (2018) fit: $1+3.45\times10^{-4}f^{-1.04}$")
    ax.plot(freqs_mHz[m_fit_l], Z_GF_fit[m_fit_l], color="tab:red", lw=2.0, alpha=0.35)
    ax.plot(freqs_mHz[m_c], Z_CW_total[m_c], color="tab:green", lw=1.2,
            label=r"Analytical (CW): $Z(\omega) + n^2/\omega^2$")
    ax.axvspan(0.1, 0.9, color="0.93", zorder=0)
    ticks = [0.3, 0.5, 0.7, 0.9, 2, 5, 10, 20]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set(xlim=(0.25, 20), ylim=(-1, 5))
    ax.set_xlabel("Frequency [mHz]", fontsize=14)
    ax.set_ylabel("Admittance [-]", fontsize=14)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
