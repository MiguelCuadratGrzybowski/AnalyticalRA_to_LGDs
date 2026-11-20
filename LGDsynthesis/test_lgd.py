"""Verification of lgd_synthesis.py against a literal port of Compute_LGD.f90."""
import math
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.special import lpmv, gammaln

from lgd_synthesis import (SHCoefficients, compute_lgd, geodetic_to_ecef, gravity_acceleration_ecef,
                           prepare_orbit, read_gfc, read_love_numbers, surface_mass_to_stokes)

PI = 3.14159265358979  # Fortran's PI constant
D2R = PI / 180.0
rng = np.random.default_rng(1)


# ---------------- literal Fortran port (1-based logic kept) ----------------
def LGDR2(T, NMAX, M):
    n = NMAX - M + 1
    PBAR = np.zeros(n); PBAR1 = np.zeros(n); PBAR2 = np.zeros(n)
    P00 = 1.0; P11 = math.sqrt(3 * (1 - T**2))
    if M >= 1:
        PBAR[0] = P11; PBAR1[0] = math.sqrt(3.0) * T
        for I in range(2, M + 1):
            PBAR1[0] = math.sqrt((2 * I + 1) / (2 * I)) * (math.sqrt(1 - T**2) * PBAR1[0] + T * PBAR[0])
            PBAR[0] = math.sqrt((2 * I + 1) / (2 * I) * (1 - T**2)) * PBAR[0]
    else:
        PBAR[0] = P00; PBAR1[0] = 0.0
    if n >= 2:
        PBAR[1] = math.sqrt(2 * M + 3) * T * PBAR[0]
    for I in range(3, n + 1):
        C = ((2 * M + 2 * I - 3) * (2 * M + 2 * I - 1)) / ((I - 1) * (2 * M + I - 1))
        D = ((2 * M + 2 * I - 1) * (2 * M + I - 2) * (I - 2)) / ((2 * M + 2 * I - 5) * (I - 1) * (2 * M + I - 1))
        PBAR[I - 1] = math.sqrt(C) * T * PBAR[I - 2] - math.sqrt(D) * PBAR[I - 3]
    for I in range(2, n + 1):
        C = 1 / math.sqrt(1 - T**2) * T * (M + I - 1)
        D = 1 / math.sqrt(1 - T**2) * math.sqrt((((M + I - 1) ** 2 - M**2) * (2 * (M + I - 1) + 1)) / (2 * (M + I - 1) - 1))
        PBAR1[I - 1] = C * PBAR[I - 1] - D * PBAR[I - 2]
    return PBAR, PBAR1


def GET_ACC_VEC_SH(NMAX, R0, R, LAT, LON):
    AVEC = np.zeros((3, (NMAX + 1) ** 2))
    COS_CLAT = math.cos((90 - LAT) * D2R); SIN_CLAT = math.sin((90 - LAT) * D2R)
    R0_R = R0 / R; R0_R_M = 1.0
    M = 0; R_FACTOR = 1.0; L = NMAX + 1
    PBAR, PBAR1 = LGDR2(COS_CLAT, NMAX, M)
    for K in range(1, L + 1):
        R_FACTOR *= R0_R
        AVEC[0, K - 1] = R_FACTOR * PBAR1[K - 1] * (-1 / R)
        AVEC[1, K - 1] = 0
        AVEC[2, K - 1] = R_FACTOR * PBAR[K - 1] * K / R
    ID1 = NMAX + 1; ID2 = (NMAX + 1) * (NMAX + 2) // 2
    for M in range(1, NMAX + 1):
        COS_MLON = math.cos(M * LON * D2R); SIN_MLON = math.sin(M * LON * D2R)
        R0_R_M *= R0_R; R_FACTOR = R0_R_M; L = NMAX - M + 1
        PBAR, PBAR1 = LGDR2(COS_CLAT, NMAX, M)
        for K in range(1, L + 1):
            ID1 += 1; ID2 += 1
            R_FACTOR *= R0_R
            F = R_FACTOR * PBAR1[K - 1] * (-1 / R)
            AVEC[0, ID1 - 1] = F * COS_MLON; AVEC[0, ID2 - 1] = F * SIN_MLON
            F = R_FACTOR * PBAR[K - 1] * (1 / R / SIN_CLAT)
            AVEC[1, ID1 - 1] = F * (-M) * SIN_MLON; AVEC[1, ID2 - 1] = F * M * COS_MLON
            F = R_FACTOR * PBAR[K - 1] * (M + K) / R
            AVEC[2, ID1 - 1] = F * COS_MLON; AVEC[2, ID2 - 1] = F * SIN_MLON
    return AVEC


def FILLMAT(AX, C, S):
    M = np.zeros((3, 3))
    if AX == 2:
        M[1, 1] = 1; M[0, 0] = C; M[2, 2] = C; M[0, 2] = -S; M[2, 0] = S
    if AX == 3:
        M[2, 2] = 1; M[0, 0] = C; M[1, 1] = C; M[0, 1] = S; M[1, 0] = -S
    return M


def fortran_coe(C, S, LMAX):
    COE = np.zeros((LMAX + 1) ** 2); ID1 = 1; ID3 = (LMAX + 1) * (LMAX + 2) // 2 + 1
    for M in range(LMAX + 1):
        ID2 = ID1 + LMAX - M; COE[ID1 - 1:ID2] = C[M:LMAX + 1, M]; ID1 = ID2 + 1
        if M > 0:
            ID4 = ID3 + LMAX - M; COE[ID3 - 1:ID4] = S[M:LMAX + 1, M]; ID3 = ID4 + 1
    return COE


def fortran_acc(C, S, GM, R0, LMAX, lat, lon, r):
    acc = GET_ACC_VEC_SH(LMAX, R0, r, lat, lon) @ fortran_coe(C, S, LMAX)
    R2 = FILLMAT(2, math.cos(PI - (90 - lat) * D2R), math.sin(PI - (90 - lat) * D2R))
    R3 = FILLMAT(3, math.cos(-lon * D2R), math.sin(-lon * D2R))
    return (R3 @ R2 @ acc) * GM / R0


# ---------------- independent potential via scipy (for finite differences) --
def potential(C, S, GM, R0, xyz):
    x, y, z = xyz; r = math.sqrt(x * x + y * y + z * z); t = z / r; lon = math.atan2(y, x)
    V = 0.0
    for l in range(C.shape[0]):
        for m in range(l + 1):
            nrm = math.sqrt((2 - (m == 0)) * (2 * l + 1) * math.exp(gammaln(l - m + 1) - gammaln(l + m + 1)))
            P = nrm * lpmv(m, l, t) * (-1) ** m  # remove Condon-Shortley phase
            V += (R0 / r) ** (l + 1) * P * (C[l, m] * math.cos(m * lon) + S[l, m] * math.sin(m * lon))
    return GM / R0 * V


def random_field(L, decay=True):
    C = np.tril(rng.standard_normal((L + 1, L + 1))); S = np.tril(rng.standard_normal((L + 1, L + 1)))
    S[:, 0] = 0
    if decay:
        l = np.arange(L + 1)[:, None]; C *= 1e-6 / (l + 1) ** 2; S *= 1e-6 / (l + 1) ** 2
    C[0, 0] = 1.0
    return C, S


GM, R0 = 3.986004415e14, 6378136.3

# 1) vectorised vs literal Fortran port
L = 60
C, S = random_field(L)
sh = SHCoefficients(C, S, GM, R0)
lat = rng.uniform(-89, 89, 30); lon = rng.uniform(-180, 180, 30); r = R0 + rng.uniform(3e5, 6e5, 30)
xyz = np.column_stack([r * np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                       r * np.cos(np.radians(lat)) * np.sin(np.radians(lon)), r * np.sin(np.radians(lat))])
for lmin in (0, 2):
    Ct, St = C.copy(), S.copy(); Ct[:lmin] = 0; St[:lmin] = 0
    ref = np.array([fortran_acc(Ct, St, GM, R0, L, la, lo, rr) for la, lo, rr in zip(lat, lon, r)])
    new = gravity_acceleration_ecef(sh, xyz, lmin=lmin)
    rel = np.abs(new - ref).max() / np.abs(ref).max()
    print(f"[1] vs literal Fortran port, L={L}, lmin={lmin}: max rel diff {rel:.1e}")
    assert rel < 1e-10

# 2) point mass: a = -GM r / r^3
C0 = np.zeros((1, 1)); C0[0, 0] = 1
a = gravity_acceleration_ecef(SHCoefficients(C0, np.zeros((1, 1)), GM, R0), xyz)
exact = -GM * xyz / np.linalg.norm(xyz, axis=1)[:, None] ** 3
print(f"[2] point mass: max rel err {np.abs(a - exact).max() / np.abs(exact).max():.1e}")
assert np.allclose(a, exact, rtol=1e-13, atol=0)

# 3) finite-difference gradient of an independently computed potential (no C00)
L = 12
C, S = random_field(L); C[0, 0] = 0
sh = SHCoefficients(C, S, GM, R0)
h = 1.0
worst = 0
for p in xyz[:5]:
    g = gravity_acceleration_ecef(sh, p[None])[0]
    fd = np.array([(potential(C, S, GM, R0, p + h * ei) - potential(C, S, GM, R0, p - h * ei)) / (2 * h)
                   for ei in np.eye(3)])
    worst = max(worst, np.abs(g - fd).max() / np.abs(g).max())
print(f"[3] vs finite-difference gradient of scipy-based potential: max rel diff {worst:.1e}")
assert worst < 1e-6

# 4) high degree runs / numerically sane (L=180, near-polar points)
L = 180
C, S = random_field(L); C[0, 0] = 0
lat2 = np.array([-88.9, -45, 0, 45, 88.9]); lon2 = np.array([10, -170, 0, 90, 179.9]); r2 = np.full(5, R0 + 4.9e5)
xyz2 = np.column_stack([r2 * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2)),
                        r2 * np.cos(np.radians(lat2)) * np.sin(np.radians(lon2)), r2 * np.sin(np.radians(lat2))])
new = gravity_acceleration_ecef(SHCoefficients(C, S, GM, R0), xyz2)
ref = np.array([fortran_acc(C, S, GM, R0, L, la, lo, rr) for la, lo, rr in zip(lat2, lon2, r2)])
rel = np.abs(new - ref).max() / np.abs(ref).max()
print(f"[4] L=180 incl. |lat|=88.9: finite={np.isfinite(new).all()}, max rel diff vs literal port {rel:.1e}")
assert rel < 1e-9

# 5) full LGD vs Fortran formula with e12 = unit(r1 - r2); swap symmetry
L = 40
C, S = random_field(L); C[0, 0] = 0
sh = SHCoefficients(C, S, GM, R0)
n = 50
t = np.arange(n) * 5.0
latC = np.linspace(-80, 80, n); lonC = np.linspace(-170, 170, n)
oc = pd.DataFrame({"gps_time": t, "latitude_C": latC, "longitude_C": lonC, "altitude_C": 4.9e5 + 0 * t})
od = pd.DataFrame({"gps_time": t, "latitude_D": latC - 1.8, "longitude_D": lonC + 0.1, "altitude_D": 4.9e5 + 50 + 0 * t})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = compute_lgd(sh, prepare_orbit(oc, "C"), prepare_orbit(od, "D"), lmin=2, position_type="geocentric")
    res_sw = compute_lgd(sh, prepare_orbit(od.rename(columns=lambda c: c.replace("_D", "_C")), "C"),
                         prepare_orbit(oc.rename(columns=lambda c: c.replace("_C", "_D")), "D"),
                         lmin=2, position_type="geocentric")
Ct, St = C.copy(), S.copy(); Ct[:2] = 0; St[:2] = 0
f_lgd = []
for i in range(n):
    rC, rD = R0 + 4.9e5, R0 + 4.9e5 + 50
    a1 = fortran_acc(Ct, St, GM, R0, L, latC[i], lonC[i], rC)
    a2 = fortran_acc(Ct, St, GM, R0, L, latC[i] - 1.8, lonC[i] + 0.1, rD)
    p1 = rC * np.array([math.cos(math.radians(latC[i])) * math.cos(math.radians(lonC[i])),
                        math.cos(math.radians(latC[i])) * math.sin(math.radians(lonC[i])), math.sin(math.radians(latC[i]))])
    la, lo = math.radians(latC[i] - 1.8), math.radians(lonC[i] + 0.1)
    p2 = rD * np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    e12 = (p1 - p2) / np.linalg.norm(p1 - p2)
    f_lgd.append(np.dot(a1 - a2, e12))
f_lgd = np.array(f_lgd)
print(f"[5] LGD vs literal Fortran: max |diff| {np.abs(res.lgd - f_lgd).max():.1e} m/s^2 "
      f"(LGD rms {np.sqrt(np.mean(f_lgd**2)):.1e}); C/D swap max |diff| {np.abs(res.lgd - res_sw.lgd).max():.1e}")
assert np.allclose(res.lgd, f_lgd, rtol=1e-9, atol=1e-20)
assert np.allclose(res.lgd, res_sw.lgd, rtol=1e-12, atol=1e-22)

# 6) readers: gfc with header + D exponents, plain L M C S, Love numbers (2 cols / 1 col)
d = tempfile.mkdtemp()
gfc = os.path.join(d, "t.gfc")
with open(gfc, "w") as f:
    f.write("product_type gravity_field\nmodelname TEST\nearth_gravity_constant 3.986004415E+14\n"
            "radius 6.3781363E+06\nmax_degree 3\nnorm fully_normalized\ntide_system tide_free\n"
            "key L M C S sigC sigS\nend_of_head =========\n")
    f.write("gfc 0 0 1.0D+00 0.0 0 0\ngfc 2 0 -4.84165D-04 0.0 0 0\ngfc 2 2 2.4D-06 -1.4D-06 0 0\n"
            "gfc 3 1 2.0E-06 2.5E-07 0 0\n")
g = read_gfc(gfc)
assert g.lmax == 3 and g.C[2, 0] == -4.84165e-4 and g.S[2, 2] == -1.4e-6 and g.radius == 6378136.3
plain = os.path.join(d, "p.txt")
with open(plain, "w") as f:
    f.write("2 0 -4.84165E-04 0\n2 1 1e-10 2e-10\n")
assert read_gfc(plain).C[2, 1] == 1e-10
love2 = os.path.join(d, "l2.txt"); love1 = os.path.join(d, "l1.txt")
kvals = [0.0, 0.027, -0.303, -0.194, -0.132]
with open(love2, "w") as f:
    f.write("# l k_l\n" + "".join(f"{l} {k}\n" for l, k in enumerate(kvals)))
with open(love1, "w") as f:
    f.write("".join(f"{k}\n" for k in kvals[1:]))
assert np.allclose(read_love_numbers(love2), kvals)
assert np.allclose(read_love_numbers(love1, first_degree=1), kvals)
print("[6] gfc / plain / Love-number readers OK")

# 7) EWH -> Stokes: uniform 1 m EWH on degree 0 must give total mass 4 pi R^2 * 1000 kg
ew = SHCoefficients(np.zeros((5, 5)), np.zeros((5, 5)), GM, R0); ew.C[0, 0] = 1.0
st = surface_mass_to_stokes(ew, np.array(kvals), "ewh")
mass_from_stokes = st.C[0, 0] * GM / 6.67430e-11
print(f"[7] EWH->Stokes degree-0 mass check: {mass_from_stokes / (4 * math.pi * R0**2 * 1000):.12f} (expect 1)")
assert abs(mass_from_stokes / (4 * math.pi * R0**2 * 1000) - 1) < 1e-12

# 8) geodetic -> ECEF sanity (equator / pole)
assert np.allclose(geodetic_to_ecef(0, 0, 0), [[6378137.0, 0, 0]])
assert np.allclose(geodetic_to_ecef(90, 0, 0)[0, 2], 6356752.314140, atol=1e-5)
print("[8] geodetic -> ECEF OK")
print("ALL TESTS PASSED")
