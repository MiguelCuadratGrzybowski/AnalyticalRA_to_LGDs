"""
plot_lgd.py -- scatter and median-gridded maps of LGD at the C-D midpoints.

Input: the LGD file written by lgd_synthesis.py (.parquet or .csv) with
columns mid_lat, mid_lon, lgd [m/s^2].
Edit SETTINGS at the bottom and press Run in PyCharm.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:                                   # coastlines if cartopy is installed
    import cartopy.crs as ccrs
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False

NMS2 = 1e9                             # m/s^2 -> nm/s^2


def read_lgd(path):
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = {"mid_lat", "mid_lon", "lgd"} - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    df = df.dropna(subset=["mid_lat", "mid_lon", "lgd"]).copy()
    df["mid_lon"] = (df["mid_lon"] + 180.0) % 360.0 - 180.0      # wrap to [-180, 180)
    return df


def grid_median(lat, lon, val, dlat, dlon, min_count=1):
    """Median of val in dlat x dlon [deg] cells. Returns lat_edges, lon_edges,
    median grid (nlat, nlon) and count grid; cells with < min_count points are NaN."""
    lat_edges = np.arange(-90.0, 90.0 + dlat / 2, dlat)
    lon_edges = np.arange(-180.0, 180.0 + dlon / 2, dlon)
    nlat, nlon = len(lat_edges) - 1, len(lon_edges) - 1
    i = np.clip(np.floor((lat + 90.0) / dlat).astype(int), 0, nlat - 1)
    j = np.clip(np.floor((lon + 180.0) / dlon).astype(int), 0, nlon - 1)
    s = pd.Series(val).groupby([i, j])
    med, cnt = s.median(), s.size()
    grid = np.full((nlat, nlon), np.nan)
    count = np.zeros((nlat, nlon), dtype=int)
    ii, jj = med.index.get_level_values(0), med.index.get_level_values(1)
    grid[ii, jj] = med.to_numpy()
    count[ii, jj] = cnt.to_numpy()
    grid[count < min_count] = np.nan
    return lat_edges, lon_edges, grid, count


def _axes(fig, pos):
    if HAVE_CARTOPY:
        ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        ax.set_global()
        return ax, dict(transform=ccrs.PlateCarree())
    ax = fig.add_subplot(pos)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_aspect("equal")
    return ax, {}


def plot_lgd(lgd_file, dlat=1.0, dlon=1.0, min_count=1, clim=None,
             cmap="RdBu_r", point_size=1.0, save=None):
    df = read_lgd(lgd_file)
    lat, lon, val = df["mid_lat"].to_numpy(), df["mid_lon"].to_numpy(), df["lgd"].to_numpy() * NMS2
    lat_e, lon_e, grid, count = grid_median(lat, lon, val, dlat, dlon, min_count)

    if clim is None:                   # symmetric, robust colour range
        c = np.nanpercentile(np.abs(val), 99)
        clim = (-c, c)

    fig = plt.figure(figsize=(11, 9))
    ax1, kw = _axes(fig, 211)
    sc = ax1.scatter(lon, lat, c=val, s=point_size, cmap=cmap, vmin=clim[0], vmax=clim[1],
                     linewidths=0, rasterized=True, **kw)
    ax1.set_title(f"LGD at C-D midpoints ({len(val)} epochs)")
    fig.colorbar(sc, ax=ax1, label="LGD [nm/s$^2$]", shrink=0.8)

    ax2, kw = _axes(fig, 212)
    pm = ax2.pcolormesh(lon_e, lat_e, grid, cmap=cmap, vmin=clim[0], vmax=clim[1], **kw)
    filled = np.isfinite(grid).sum()
    ax2.set_title(f"Median LGD, {dlat}° x {dlon}° cells "
                  f"({filled} filled, median {np.median(count[count > 0]):.0f} pts/cell)")
    fig.colorbar(pm, ax=ax2, label="LGD [nm/s$^2$]", shrink=0.8)

    fig.suptitle(Path(lgd_file).name)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200)
        print(f"Figure saved to {Path(save).resolve()}")
    plt.show()
    return lat_e, lon_e, grid, count


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # RUN SETTINGS
    # ----------------------------------------------------------------------
    SETTINGS = dict(
        lgd_file=Path("../orbit_data/out/lgd_2020_01.parquet"),   # INPUT: LGD file
        dlat=2.0,            # grid cell size in latitude [deg]
        dlon=2.0,            # grid cell size in longitude [deg]
        min_count=3,         # cells with fewer points are left empty
        clim=(-15, 15),           # e.g. (-20, 20) nm/s^2; None = automatic
        point_size=1.0,      # scatter marker size
        save=None,           # e.g. "lgd_2020_01.png"
    )
    # ----------------------------------------------------------------------
    lat_edges, lon_edges, grid, count = plot_lgd(**SETTINGS)
