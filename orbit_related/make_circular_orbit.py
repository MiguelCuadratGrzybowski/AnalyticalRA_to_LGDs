"""
Synthetic ("fake") GRACE-like circular orbit pair for LGD tests.

Two satellites on the same circular orbit (mean altitude ~450 km, i = 89 deg),
separated along-track by an inter-satellite chord distance rho = 220 km.
Earth rotation is included (ECI -> ECEF via GMST at the epoch + omega_E * t),
and positions are written as geodetic latitude/longitude/height (WGS84).

Output: a single file orbit_circular.parquet with columns
    time, latitude_X, longitude_X, altitude_X, latitude_Y, longitude_Y, altitude_Y
where (X, Y) = (A, B) if YEAR <= 2017 else (C, D).
Satellite X leads, satellite Y trails.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
YEAR = "2005"
MONTH = "12"
DATA = Path(r"../../solutions/orbits")
OUTPUT = DATA / "orbit_circular.parquet"

MEAN_ALTITUDE = 450e3        # [m]
ALTITUDE_REFERENCE = "ellipsoid"  # "ellipsoid": mean geodetic height = MEAN_ALTITUDE
                                  # "sphere":    r = R_MEAN + MEAN_ALTITUDE
INCLINATION = 89.0           # [deg]
RHO = 220e3                  # [m] inter-satellite chord distance
RAAN0 = 0.0                  # [deg] right ascension of ascending node at epoch
U0 = 0.0                     # [deg] argument of latitude of the leading sat at epoch
DT = 5.0                     # [s] sampling
DURATION_DAYS = 30.0         # [days]
EPOCH = pd.Timestamp(f"{YEAR}-{MONTH}-01 00:00:00")   # UTC (treated as UT1)
INCLUDE_XYZ = False          # also write ECEF x/y/z [m] per satellite

# Constants
GM = 3.986004415e14          # [m^3/s^2] (GGM05C / EGM value)
R_MEAN = 6371008.8           # [m] IUGG mean Earth radius
OMEGA_E = 7.2921150e-5       # [rad/s] Earth rotation rate
A_WGS84 = 6378137.0          # [m]
F_WGS84 = 1.0 / 298.257223563
# ----------------------------------------------------------------------


def gmst_rad(t: pd.Timestamp) -> float:
    """Greenwich mean sidereal time [rad] (IAU 1982, simple form)."""
    jd = t.to_julian_date()
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
    return np.deg2rad(gmst_deg % 360.0)


def ecef_to_geodetic(x, y, z, a=A_WGS84, f=F_WGS84, n_iter=6):
    """ECEF [m] -> geodetic lat [deg], lon [deg], height [m] (iterative)."""
    e2 = f * (2.0 - f)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(n_iter):
        N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + h)))
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h


def circular_ecef(t_sec, r, inc, raan, u0, n, theta0):
    """ECEF positions [m] of a circular orbit at times t_sec [s] since epoch."""
    u = u0 + n * t_sec
    cu, su = np.cos(u), np.sin(u)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    # inertial (ECI) position
    xi = r * (cO * cu - sO * su * ci)
    yi = r * (sO * cu + cO * su * ci)
    zi = r * (su * si)
    # rotate to Earth-fixed
    th = theta0 + OMEGA_E * t_sec
    ct, st = np.cos(th), np.sin(th)
    return ct * xi + st * yi, -st * xi + ct * yi, zi


def make_circular_orbit():
    sfx = ("_A", "_B") if int(YEAR) <= 2017 else ("_C", "_D")

    r = R_MEAN + MEAN_ALTITUDE
    if ALTITUDE_REFERENCE == "ellipsoid":
        # tune r so that the mean WGS84 height over one revolution = target
        u = np.linspace(0.0, 2.0 * np.pi, 3600, endpoint=False)
        inc_ = np.deg2rad(INCLINATION)
        for _ in range(3):
            x, y, z = r * np.cos(u), r * np.sin(u) * np.cos(inc_), r * np.sin(u) * np.sin(inc_)
            r -= ecef_to_geodetic(x, y, z)[2].mean() - MEAN_ALTITUDE
    elif ALTITUDE_REFERENCE != "sphere":
        raise ValueError("ALTITUDE_REFERENCE must be 'ellipsoid' or 'sphere'")
    n = np.sqrt(GM / r**3)                       # mean motion [rad/s]
    dtheta = 2.0 * np.arcsin(RHO / (2.0 * r))    # along-track angle for chord rho
    inc, raan, u0 = np.deg2rad([INCLINATION, RAAN0, U0])
    theta0 = gmst_rad(EPOCH)

    t_sec = np.arange(0.0, DURATION_DAYS * 86400.0, DT)
    time = EPOCH + pd.to_timedelta(t_sec, unit="s")

    cols = {"gps_time": time}
    for s, du in zip(sfx, (0.0, -dtheta)):      # leader, trailer
        x, y, z = circular_ecef(t_sec, r, inc, raan, u0 + du, n, theta0)
        lat, lon, h = ecef_to_geodetic(x, y, z)
        cols[f"latitude{s}"] = lat
        cols[f"longitude{s}"] = lon
        cols[f"altitude{s}"] = h
        if INCLUDE_XYZ:
            cols[f"x{s}"], cols[f"y{s}"], cols[f"z{s}"] = x, y, z

    df = pd.DataFrame(cols)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    # --- sanity report ----------------------------------------------------
    print(f"wrote {OUTPUT}  ({len(df)} epochs, suffixes {sfx})")
    print(f"radius            : {r/1e3:.3f} km   period: {2*np.pi/n/60:.2f} min")
    print(f"along-track angle : {np.rad2deg(dtheta):.4f} deg "
          f"(arc {r*dtheta/1e3:.3f} km, chord {RHO/1e3:.1f} km)")
    for s in sfx:
        h = df[f"altitude{s}"]
        print(f"geodetic height{s}: mean {h.mean()/1e3:.2f} km, "
              f"min {h.min()/1e3:.2f}, max {h.max()/1e3:.2f}")
    print(f"max |latitude|    : {df[f'latitude{sfx[0]}'].abs().max():.3f} deg")
    return df


if __name__ == "__main__":
    df = make_circular_orbit()