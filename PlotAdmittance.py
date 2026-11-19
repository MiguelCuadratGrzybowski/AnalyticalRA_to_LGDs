"""Admittance between residual range acceleration and along-track LGD for GRACE(-FO).

Compares
  1. kappa = 1                    (rectilinear motion),
  2. kappa(omega)                 (frequency dependent, Kaula-weighted, gradient limit),
  3. Ghobadi-Far et al. (2018)    empirical fit 1 + 3.45e-4 f^-1.04, f in Hz.

Z(omega) = (omega^2 - n^2) / (omega^2 - 2 n omega kappa(omega) + 3 n^2)

kappa(omega) = sum_l w_lk kappa_lk / sum_l w_lk,   l = k, k+2, ..., Lmax,   k = omega / n
  kappa_lk = k (l+2) / (k^2 + l + 1)
  w_lk     = (R/r)^(2l) (2l+1) sigma_l^2 / (l+1) (k^2 + l + 1)^2,   sigma_l = 1e-5 / l^2 (Kaula)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gammaln
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from utilities.utilities_timeFilt import transfer_function

# ---------------- parameters ----------------
GM = 3.986004418e14      # m^3/s^2
R = 6378137.0            # reference radius of the expansion [m]
h = 450e3                # orbit altitude [m]
Lmax = 120               # maximum degree
f_min, f_max = 0.3e-3, 20e-3     # plotted frequency range [Hz]
f_gf_valid = 0.9e-3              # lower validity limit of the Ghobadi-Far fit [Hz]
f_gf_light = 0.7e-3              # extension shown lightly [Hz]

r = R + h
n = np.sqrt(GM / r**3)           # mean motion [rad/s]
f_orb = n / (2 * np.pi)          # orbital frequency [Hz]


# ---------------- model ----------------
def sigma_kaula(l, A=1e-5):
    """Kaula's rule: RMS of one normalised coefficient of degree l."""
    return A / np.asarray(l, dtype=float) ** 2


def kappa_lk(l, k):
    """Per-term ratio delta g_x = i kappa_lk delta g_y in the Hill frame of O (gradient limit)."""
    return k * (l + 2.0) / (k**2 + l + 1.0)


# ---------------- along-track power fractions ----------------
def f_iso(l, k):
    """Exact isotropic fraction (l-|k|)!/(l+|k|)! P_l^|k|(0)^2 for k = -l..l (sums to 1)."""
    a = np.abs(k)
    la, lb = (l + a) // 2, (l - a) // 2
    lnP = gammaln(l + a + 1) - (la * np.log(2) + gammaln(la + 1)) - (lb * np.log(2) + gammaln(lb + 1))
    out = np.exp(gammaln(l - a + 1) - gammaln(l + a + 1) + 2 * lnP)
    return out

def f_even(l, k):
    """Even split: 1/(l+1) on the l+1 orders k = l, l-2, ..., -l."""
    return np.where((l - np.abs(k)) % 2 == 0, 1.0 / (l + 1), 0.0)


def weight_lk(l, k, r=r, R=R, flk_func=f_iso):
    """Expected along-track gravity-difference power <|delta g_y,lk|^2> (common constants dropped)."""
    return (R / r) ** (2 * l) * (2 * l + 1.0) * sigma_kaula(l) ** 2 * flk_func(l, k) * (k**2 + l + 1.0) ** 2


def kappa_eff(k, r=r):
    """Effective kappa at integer along-orbit order k (regression of G_x on G_y)."""
    l = np.arange(k, Lmax + 1, 2).astype(float)          # l >= k, l = k mod 2
    w = weight_lk(l, k, r=r)
    return np.sum(kappa_lk(l, k) * w) / np.sum(w)


def admittance(omega, kappa, r=r):
    n = np.sqrt(GM / r**3)
    return (omega**2 - n**2) / (omega**2 - 2 * n * omega * kappa + 3 * n**2)  + n**2/omega**2

def ghobadi_far(f_hz):
    return transfer_function(f_hz)


# ---------------- evaluation ----------------
OUTDIR = "../RESULTS/AnalyticalRA_to_LGDs"
f = np.linspace(f_min, f_max, 10000)                      # Hz
omega = 2 * np.pi * f
fm = f * 1e3                                              # mHz

heights_km = np.array([300, 350, 400, 450, 500, 550])
# discrete colormap: viridis truncated before the pale yellow end (stays visible on white / grey band, distinct from red GF curve)
cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(heights_km))))
dh = np.diff(heights_km).mean() / 2
bounds = np.r_[heights_km - dh, heights_km[-1] + dh]
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(12, 5))
results = {}
for h_km in heights_km:
    h = h_km * 1e3
    r = h + R
    n = np.sqrt(GM / r**3)
    k = omega / n                                          # continuous along-orbit order

    k_int = np.arange(2, Lmax + 1, 2)
    kappa_int = np.array([kappa_eff(kk, r=r) for kk in k_int])
    kappa_f = np.interp(k, k_int, kappa_int)               # kappa(omega)

    Z_rect = admittance(omega, 1.0, r=r)
    Z_freq = admittance(omega, kappa_f, r=r)
    results[h_km] = (k, kappa_f, Z_rect, Z_freq)

    c = cmap(norm(h_km))
    ax.plot(fm, Z_rect, color=c, lw=1.4, ls="--")
    ax.plot(fm, Z_freq, color=c, lw=1.8, ls="-")

# ---------------- Ghobadi-Far ----------------
m_valid = f >= f_gf_valid
m_light = (f >= f_gf_light) & (f <= f_gf_valid)
ax.plot(fm[m_valid], ghobadi_far(f[m_valid]), color="C3", lw=2.0)
ax.plot(fm[m_light], ghobadi_far(f[m_light]), color="C3", lw=2.0, alpha=0.35)
ax.axvspan(f_min * 1e3, f_gf_valid * 1e3, color="0.93", zorder=0)

# ---------------- axes ----------------
ax.set_xscale("log")
ax.set_xlim(f_min * 1e3, f_max * 1e3)
ax.set_ylim(0.5, 2.5)
ticks = [0.3, 0.5, 0.7, 0.9, 2, 5, 10, 20]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())  # avoid stray minor labels on log axis
ax.tick_params(axis="both", which="major", labelsize=13)
ax.set_xlabel("Frequency [mHz]", fontsize=14)
ax.set_ylabel(r"Transfer function $Z(f)$ [-]", fontsize=14)
ax.grid(True, which="both", alpha=0.3)

# ---------------- legend (line style only, one entry each) ----------------
handles = [
    Line2D([], [], color="0.3", lw=1.4, ls="--", label=r"$\kappa=1$ (rectilinear motion)"),
    Line2D([], [], color="0.3", lw=1.8, ls="-",  label=r"$\kappa(\omega)$ (frequency dependent)"),
    Line2D([], [], color="C3",  lw=2.0,          label=r"Ghobadi-Far et al. (2018): $1+3.45\times10^{-4}f^{-1.04}$"),
]
ax.legend(handles=handles, fontsize=12, loc="upper right")

# ---------------- discrete colorbar for altitude ----------------
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, ticks=heights_km, pad=0.02, aspect=25)
cbar.set_label("Altitude $h$ [km]", fontsize=14)
cbar.ax.tick_params(labelsize=13)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/admittance_kappa.png", dpi=200)

# ---------------- table ----------------
for h_km, (k, kappa_f, Z_rect, Z_freq) in results.items():
    print(f"--- h = {h_km} km ---")
    for fq in [0.7e-3, 0.9e-3, 2e-3, 5e-3, 10e-3, 20e-3]:
        i = np.argmin(np.abs(f - fq))
        print(f"f={fq*1e3:5.2f} mHz  k={k[i]:6.2f}  kappa={kappa_f[i]:.4f}  "
              f"Z(kappa=1)={Z_rect[i]:.4f}  Z(kappa(w))={Z_freq[i]:.4f}  GF={ghobadi_far(f[i]):.4f}")
plt.show()