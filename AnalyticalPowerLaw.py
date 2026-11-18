"""Admittance Z(omega) between residual range acceleration and LGD: model comparison.

Curves (CW closure G_x = i kappa(omega) G_y, Kaula sigma_l = 1e-5/l^2):
  1. kappa = 1                      (rectilinear motion)
  2. even split kappa(omega)        degree power spread evenly over the l+1 orders k
  3. isotropic kappa(omega)         exact isotropic along-track fraction f_lk
  4. even split + Earth rotation    lines at f = k f_orb - m f_E, binned at df
RMS differences with Ghobadi-Far et al. (2018), Z = 1 + 3.45e-4 f^-1.04, over 0.9-20 mHz.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

# ---------------- constants ----------------
GM, R, J2 = 3.986004418e14, 6378137.0, 1.0826359e-3
h, inc, ecc = 450e3, np.deg2rad(89.0), 0.002
wE = 7.2921150e-5                      # Earth rotation [rad/s]
Lmax = 200                             # (R/r)^(2l) makes higher degrees negligible
df = 0.025e-3                          # frequency resolution [Hz]
f_lo, f_hi = 0.1e-3, 20e-3
f_rms_lo, f_rms_hi = 0.9e-3, 20e-3
SHOW_J2E = False                       # J2 (Schweighart-Sedwick) + eccentricity curve

r0 = R + h
n0 = np.sqrt(GM / r0**3)
f_orb = n0 / (2 * np.pi)
sig2 = lambda l: (1e-5 / l**2) ** 2
kappa_lk = lambda l, k: k * (l + 2.0) / (k**2 + l + 1.0)


# ---------------- along-track power fractions ----------------
def f_iso(l):
    """Exact isotropic fraction (l-|k|)!/(l+|k|)! P_l^|k|(0)^2 for k = -l..l (sums to 1)."""
    k = np.arange(-l, l + 1)
    a = np.abs(k)
    out = np.zeros(2 * l + 1)
    ok = (l - a) % 2 == 0
    la, lb = (l + a[ok]) // 2, (l - a[ok]) // 2
    lnP = gammaln(l + a[ok] + 1) - (la * np.log(2) + gammaln(la + 1)) - (lb * np.log(2) + gammaln(lb + 1))
    out[ok] = np.exp(gammaln(l - a[ok] + 1) - gammaln(l + a[ok] + 1) + 2 * lnP)
    return k, out


def f_even(l):
    """Even split: 1/(l+1) on the l+1 orders k = l, l-2, ..., -l."""
    k = np.arange(-l, l + 1)
    return k, np.where((l - np.abs(k)) % 2 == 0, 1.0 / (l + 1), 0.0)


def wigner_d2(l, beta):
    """|d^l_mk(beta)|^2, rows m, columns k (from exp(-i beta J_y))."""
    m = np.arange(-l, l + 1).astype(float)
    off = 0.5 * np.sqrt(l * (l + 1) - m[:-1] * (m[:-1] + 1))
    Jy = np.diag(-1j * off, 1) + np.diag(1j * off, -1)
    w, V = np.linalg.eigh(Jy)
    return np.abs((V * np.exp(-1j * beta * w)) @ V.conj().T) ** 2


# ---------------- kappa without Earth rotation (integer k, interpolated) ----------------
def kappa_table(frac, r):
    ks = np.arange(1, Lmax + 1)
    K = np.empty(ks.size)
    for j, k in enumerate(ks):
        num = den = 0.0
        for l in range(max(k, 2), Lmax + 1):
            _, fr = frac(l)
            w = (R / r) ** (2 * l) * (2 * l + 1) * sig2(l) * fr[k + l] * (k**2 + l + 1) ** 2
            num += kappa_lk(l, k) * w
            den += w
        K[j] = num / den if den > 0 else np.nan
    return ks, K


def admittance(omega, n, kappa, c=1.0):
    return (omega**2 - (2 - c**2) * n**2) / (omega**2 - 2 * n * c * omega * kappa + (5 * c**2 - 2) * n**2)


f = np.arange(f_lo, f_hi + df / 2, df)
omega = 2 * np.pi * f
k_cont = omega / n0

ks, K_even = kappa_table(f_even, r0)
_, K_iso = kappa_table(f_iso, r0)
Z_cw = admittance(omega, n0, 1.0)
Z_even = admittance(omega, n0, np.interp(k_cont, ks, K_even))
Z_iso = admittance(omega, n0, np.interp(k_cont, ks, K_iso))

# ---------------- J2 + eccentricity (kept, off by default) ----------------
if SHOW_J2E:
    s = 3 * J2 * R**2 / (8 * r0**2) * (1 + 3 * np.cos(2 * inc))
    c_ss = np.sqrt(1 + s)
    M = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    Z_j2e = np.zeros_like(f)
    for Mi in M:
        th_dot = n0 * (1 + 2 * ecc * np.cos(Mi))
        ks_r, K_r = kappa_table(f_even, r0 * (1 - ecc * np.cos(Mi)))
        Z_j2e += admittance(omega, th_dot, np.interp(omega / th_dot, ks_r, K_r), c=c_ss)
    Z_j2e /= M.size

# ---------------- Earth rotation: (l,k,m) lines binned at df (even split) ----------------
edges = np.concatenate([f - df / 2, [f[-1] + df / 2]])
num = np.zeros(f.size)
den = np.zeros(f.size)
for l in range(2, Lmax + 1):
    k, fr = f_even(l)
    m = np.arange(-l, l + 1)
    F = k[None, :] * f_orb - m[:, None] * wE / (2 * np.pi)              # line frequency [Hz]
    W = (R / r0) ** (2 * l) * (2 * l + 1) * sig2(l) * fr[None, :] * wigner_d2(l, inc) * (k[None, :] ** 2 + l + 1) ** 2
    Kp = np.broadcast_to(kappa_lk(l, k)[None, :], W.shape)
    sel = (F > 0) & (W > 0)
    idx = np.searchsorted(edges, F[sel]) - 1
    ok = (idx >= 0) & (idx < f.size)
    np.add.at(num, idx[ok], (W[sel] * Kp[sel])[ok])
    np.add.at(den, idx[ok], W[sel][ok])
Z_rot = admittance(omega, n0, np.where(den > 0, num / np.where(den > 0, den, 1), np.nan))

# ---------------- Ghobadi-Far and RMS ----------------
Z_gf = 1 + 3.45e-4 * f ** -1.04
m_rms = (f >= f_rms_lo) & (f <= f_rms_hi)


def rms(Z):
    d = (Z - Z_gf)[m_rms]
    d = d[np.isfinite(d)]
    return np.sqrt(np.mean(d**2))


curves = [
    (r"$\kappa=1$ (rectilinear)", Z_cw, dict(color="0.45", ls="--", lw=1.3)),
    (r"even split $\kappa(\omega)$", Z_even, dict(color="C0", lw=1.8)),
    (r"isotropic $\kappa(\omega)$", Z_iso, dict(color="C2", lw=1.8)),
    (r"even split + Earth rotation", Z_rot, dict(color="C1", lw=1.0)),
]
if SHOW_J2E:
    curves.append((r"$J_2$ + eccentricity (even split)", Z_j2e, dict(color="C4", ls=":", lw=1.8)))

print(f"df = {df*1e3:.3f} mHz, {f.size} frequencies")
print(f"RMS difference with Ghobadi-Far, {f_rms_lo*1e3:.1f}-{f_rms_hi*1e3:.0f} mHz:")
for name, Z, _ in curves:
    print(f"  {name:35s} {rms(Z):.4f}")

# ---------------- plot ----------------
fm = f * 1e3
fig, ax = plt.subplots(figsize=(12, 5.5))
for name, Z, st in curves:
    ax.plot(fm, Z, label=f"{name}  (RMS {rms(Z):.4f})", **st)
m_gf = f >= 0.9e-3
m_gf_l = (f >= 0.7e-3) & (f <= 0.9e-3)
ax.plot(fm[m_gf], Z_gf[m_gf], color="C3", lw=2.2, label=r"Ghobadi-Far et al. (2018): $1+3.45\times10^{-4}f^{-1.04}$")
ax.plot(fm[m_gf_l], Z_gf[m_gf_l], color="C3", lw=2.2, alpha=0.3)
ax.axvspan(f_lo * 1e3, 0.9, color="0.93", zorder=0)
ax.set_xscale("log")
ax.set_xlim(f_lo * 1e3, f_hi * 1e3)
ax.set_ylim(0.5, 2.5)
ticks = [0.1, 0.2, 0.5, 0.9, 2, 5, 10, 20]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.set_xlabel("Frequency [mHz]", fontsize=14)
ax.set_ylabel(r"Admittance $Z(\omega)$ [-]", fontsize=14)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=11, loc="upper right")
plt.tight_layout()
plt.savefig("admittance_models.png", dpi=200)
plt.show()