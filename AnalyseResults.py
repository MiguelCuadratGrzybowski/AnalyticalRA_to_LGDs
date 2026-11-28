"""
Generic plotting of a NetCDF file, dispatched on each variable's dimensions.

  (time,)            -> time series (+ metrics in the title)
  (lat, lon)         -> map
  (lat, lon, time)   -> (any order)
        1. maps of every metric computed OVER TIME      (mean, std, rms, ... per grid cell)
        2. time series of every metric computed OVER SPACE (cos(lat)-weighted global stats)
        3. latitude-time (Hovmoeller) plot of a zonal metric (default: zonal RMS)

Metrics: mean, median, std, rms, min, max, absmean (mean |x|), p05, p95
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs



# ---------------------------------------------------------------------------
# key -> (vmin, vmax, label).  vmin/vmax are used for signed quantities
# (values, mean, median, min, max, p05, p95). Spread metrics (std, rms,
# absmean) use [0, vmax].
# ---------------------------------------------------------------------------
FIELD_META = {
    "eps_TF":      (-5, 5, r"$\epsilon_{\rm TF-GF_{\rm ext}}$ (transfer function error)"),
    "eps_TF_freq": (-5, 5, r"$\epsilon_{\rm TF-GF}$ (transfer function error)"),
    "eps_approx":  (-5, 5, r"$\epsilon_{\rm TF-CW}$"),
    "diff_eps":    (-2, 2, r"$|\epsilon_{\rm TF-CW}| - |\epsilon_{\rm TF-GF}|$"),
}

ALL_METRICS = ("mean", "median", "std", "rms", "min", "max", "absmean", "p05", "p95")
SPREAD_METRICS = {"std", "rms", "absmean"}

_LAT_NAMES = ("lat", "latitude", "Lat", "Latitude", "y")
_LON_NAMES = ("lon", "longitude", "Lon", "Longitude", "x")
_TIME_NAMES = ("time", "t", "Time", "epoch", "date")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pick(ds, candidates, given):
    if given is not None:
        return given
    for c in candidates:
        if c in ds.dims:
            return c
    return None


def _metric(da, name, dims, weights=None):
    """Compute one metric of `da` reduced over `dims` (optionally weighted)."""
    w = da.weighted(weights) if weights is not None else None
    if name == "mean":
        return w.mean(dims) if w is not None else da.mean(dims)
    if name == "std":
        return w.std(dims) if w is not None else da.std(dims)
    if name == "rms":
        sq = da ** 2
        return np.sqrt(sq.weighted(weights).mean(dims) if w is not None else sq.mean(dims))
    if name == "absmean":
        a = abs(da)
        return a.weighted(weights).mean(dims) if w is not None else a.mean(dims)
    if name in ("median", "p05", "p95"):
        q = {"median": 0.5, "p05": 0.05, "p95": 0.95}[name]
        out = w.quantile(q, dim=dims) if w is not None else da.quantile(q, dim=dims)
        return out.drop_vars("quantile", errors="ignore")
    if name == "min":
        return da.min(dims)
    if name == "max":
        return da.max(dims)
    raise ValueError(f"Unknown metric '{name}'. Choose from {ALL_METRICS}")


def _limits(name, metric, data, field_meta, symmetric=True):
    """Return (vmin, vmax, cmap) for a given variable/metric."""
    spread = metric in SPREAD_METRICS
    if name in field_meta:
        vmin, vmax, _ = field_meta[name]
        m = max(abs(vmin), abs(vmax))
        return (0, m, "viridis") if spread else (vmin, vmax, "RdBu_r")
    v = np.asarray(data)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0, 1, "viridis")
    lo, hi = np.percentile(v, [2, 98])
    if spread:
        return (0, hi, "viridis")
    if symmetric:
        m = max(abs(lo), abs(hi))
        return (-m, m, "RdBu_r")
    return (lo, hi, "RdBu_r")


def _label(name, da, field_meta):
    lab = field_meta[name][2] if name in field_meta else da.attrs.get("long_name", name)
    units = da.attrs.get("units")
    return lab, (rf" [$\rm {units}$]" if units else "")


def _new_map_ax(fig, spec, use_cartopy):
    ax = fig.add_subplot(spec, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(lw=0.4)
    return ax, dict(transform=ccrs.PlateCarree())


def _wrap_lon(da, lon):
    """Shift longitudes to [-180, 180) and sort."""
    da = da.assign_coords({lon: ((da[lon] + 180) % 360) - 180})
    return da.sortby(lon)


def _save(fig, save_dir, fname, dpi):
    if save_dir is not None:
        fig.savefig(Path(save_dir) / fname, dpi=dpi, bbox_inches="tight")


def _stats_str(v, units):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "all NaN"
    u = units.strip(" []")
    u = f" {u}" if u else ""
    return (f"mean={v.mean():.3g}  med={np.median(v):.3g}  std={v.std():.3g}  "
            f"rms={np.sqrt(np.mean(v**2)):.3g}  min={v.min():.3g}  max={v.max():.3g}{u}")


# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------
def plot_netcdf(
    source,
    variables=None,
    metrics=ALL_METRICS,
    field_meta=None,
    time_range=None,
    area_weighted=True,
    hovmoller_metric="rms",
    lat_name=None,
    lon_name=None,
    time_name=None,
    use_cartopy=True,
    ncols=3,
    save_dir=None,
    dpi=200,
    show=True,
):
    """
    Plot every (or selected) variable in a NetCDF file according to its dims.

    Parameters
    ----------
    source : str | Path | xr.Dataset
    variables : list[str] | None
        None -> every numeric data variable.
    metrics : sequence of str
        Subset of ALL_METRICS used for the (lat, lon, time) plots.
    field_meta : dict | None
        Extra/overriding {name: (vmin, vmax, label)} entries.
    time_range : (start, end) | None
        e.g. ("2008-01-01", "2008-12-31").
    area_weighted : bool
        Weight spatial statistics by cos(lat).
    hovmoller_metric : str | None
        Zonal metric for the latitude-time plot; None to skip.
    use_cartopy : bool
        Robinson projection + coastlines if cartopy is installed.
    save_dir : str | Path | None
        Saves <var>_timeseries.png, <var>_maps_over_time.png,
        <var>_global_timeseries.png, <var>_hovmoller.png, <var>_map.png.

    Returns
    -------
    dict[str, dict[str, Figure]]  : figures per variable and plot type.
    """
    ds = source if isinstance(source, xr.Dataset) else xr.open_dataset(source)
    fm = {**FIELD_META, **(field_meta or {})}

    lat = _pick(ds, _LAT_NAMES, lat_name)
    lon = _pick(ds, _LON_NAMES, lon_name)
    tim = _pick(ds, _TIME_NAMES, time_name)

    if time_range is not None and tim is not None:
        ds = ds.sel({tim: slice(*time_range)})

    if variables is None:
        variables = [v for v, da in ds.data_vars.items()
                     if np.issubdtype(da.dtype, np.number)]

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    w_lat = None
    if area_weighted and lat is not None:
        w_lat = np.cos(np.deg2rad(ds[lat])).clip(min=0)

    figs = {}
    for name in variables:
        if name not in ds:
            print(f"[skip] {name}: not in dataset")
            continue
        da = ds[name].squeeze(drop=True)
        dims = set(da.dims)
        lab, units = _label(name, da, fm)
        figs[name] = {}

        # ------------------------------------------------ (time,)
        if dims == {tim}:
            v = da.values.astype(float)
            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(da[tim].values, v, lw=0.5, color="0.15")
            ax.axhline(0, color="0.5", lw=0.5)
            if name in fm:
                ax.set_ylim(fm[name][0], fm[name][1])
            ax.set_ylabel(lab + units)
            ax.set_title(f"{name}\n{_stats_str(v, units)}", fontsize=9)
            ax.grid(lw=0.3, alpha=0.5)
            fig.tight_layout()
            _save(fig, save_dir, f"{name}_timeseries.png", dpi)
            figs[name]["timeseries"] = fig

        # ------------------------------------------------ (lat, lon)
        elif dims == {lat, lon}:
            d = _wrap_lon(da.transpose(lat, lon), lon)
            vmin, vmax, cmap = _limits(name, "value", d.values, fm)
            fig = plt.figure(figsize=(10, 5.5))
            ax, kw = _new_map_ax(fig, fig.add_gridspec(1, 1)[0], use_cartopy)
            pm = ax.pcolormesh(d[lon], d[lat], d, vmin=vmin, vmax=vmax,
                               cmap=cmap, shading="auto", **kw)
            fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.08,
                         shrink=0.7, extend="both").set_label(lab + units)
            ax.set_title(f"{name}\n{_stats_str(d.values, units)}", fontsize=9)
            _save(fig, save_dir, f"{name}_map.png", dpi)
            figs[name]["map"] = fig

        # ------------------------------------------------ (lat, lon, time)
        elif dims == {lat, lon, tim}:
            d = _wrap_lon(da.transpose(tim, lat, lon), lon)

            # 1) maps of metrics over time
            n = len(metrics)
            nr = int(np.ceil(n / ncols))
            fig = plt.figure(figsize=(5.2 * min(n, ncols), 3.6 * nr))
            gs = fig.add_gridspec(nr, min(n, ncols), hspace=0.35, wspace=0.12)
            for i, m in enumerate(metrics):
                field = _metric(d, m, tim).transpose(lat, lon)
                vmin, vmax, cmap = _limits(name, m, field.values, fm)
                ax, kw = _new_map_ax(fig, gs[i // ncols, i % ncols], use_cartopy)
                pm = ax.pcolormesh(d[lon], d[lat], field, vmin=vmin, vmax=vmax,
                                   cmap=cmap, shading="auto", **kw)
                cb = fig.colorbar(pm, ax=ax, orientation="horizontal",
                                  pad=0.06 if kw else 0.18, shrink=0.85, extend="both")
                cb.ax.tick_params(labelsize=7)
                cb.set_label(f"{m}{units}", fontsize=8)
                gm = float(_metric(field, "mean", [lat, lon],
                                   w_lat if w_lat is not None else None))
                ax.set_title(f"{m} over time  (global mean {gm:.3g})", fontsize=9)
                if not kw:
                    ax.tick_params(labelsize=7)
                    ax.xaxis.label.set_size(7)
                    ax.yaxis.label.set_size(7)
            t0, t1 = d[tim].values[[0, -1]]
            fig.suptitle(f"{lab}: statistics over time "
                         f"({np.datetime_as_string(t0, 'D') if np.issubdtype(d[tim].dtype, np.datetime64) else t0}"
                         f" – {np.datetime_as_string(t1, 'D') if np.issubdtype(d[tim].dtype, np.datetime64) else t1})")
            _save(fig, save_dir, f"{name}_maps_over_time.png", dpi)
            figs[name]["maps_over_time"] = fig

            # 2) time series of spatial metrics
            fig, axs = plt.subplots(n, 1, figsize=(11, 1.9 * n), sharex=True)
            axs = np.atleast_1d(axs)
            for ax, m in zip(axs, metrics):
                ts = _metric(d, m, [lat, lon], w_lat)
                ax.plot(ts[tim].values, ts.values, lw=0.9, color="C0", marker=".", ms=3)
                if m in ("mean", "median"):
                    ax.axhline(0, color="0.5", lw=0.5)
                ax.set_ylabel(f"{m}{units}", fontsize=8)
                ax.grid(lw=0.3, alpha=0.5)
                v = ts.values[np.isfinite(ts.values)]
                if v.size:
                    ax.text(0.995, 0.92, f"avg {v.mean():.3g}  |  range [{v.min():.3g}, {v.max():.3g}]",
                            transform=ax.transAxes, ha="right", va="top", fontsize=7,
                            bbox=dict(fc="white", ec="none", alpha=0.7))
            axs[0].set_title(f"{lab}: global{' (cos-lat weighted)' if w_lat is not None else ''} "
                             f"statistics in time")
            fig.tight_layout()
            _save(fig, save_dir, f"{name}_global_timeseries.png", dpi)
            figs[name]["global_timeseries"] = fig

            # 3) latitude-time (zonal metric)
            if hovmoller_metric:
                z = _metric(d, hovmoller_metric, lon).transpose(lat, tim)
                vmin, vmax, cmap = _limits(name, hovmoller_metric, z.values, fm)
                fig, ax = plt.subplots(figsize=(11, 4))
                pm = ax.pcolormesh(z[tim].values, z[lat].values, z.values,
                                   vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
                fig.colorbar(pm, ax=ax, pad=0.01, extend="both").set_label(
                    f"zonal {hovmoller_metric}{units}")
                ax.set_ylabel("Latitude [deg]")
                ax.set_title(f"{lab}: zonal {hovmoller_metric} vs time")
                fig.tight_layout()
                _save(fig, save_dir, f"{name}_hovmoller.png", dpi)
                figs[name]["hovmoller"] = fig

        else:
            print(f"[skip] {name}: dims {da.dims} not handled "
                  f"(lat={lat}, lon={lon}, time={tim})")
            del figs[name]

    if show:
        plt.show()
    return figs


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    path = Path(r"..\RESULTS\AnalyticalRA_to_LGDs\geofitData\netcdf\lgd_residuals_CSR_RL06_real.nc")
    ds_ = xr.open_dataset(path)
    ds_["diff_eps"] = np.abs(ds_["eps_approx"]) - np.abs(ds_["eps_TF_freq"])
    ds_["diff_eps_ext"] = np.abs(ds_["eps_approx"]) - np.abs(ds_["eps_TF"])

    # all variables, all metrics
    plot_netcdf(ds_, save_dir=r"..\RESULTS\AnalyticalRA_to_LGDs\geofitData\netcdf\plots", show=False)

    # only some variables / metrics, one year:
    # plot_netcdf(ds_, variables=["eps_TF_freq", "eps_approx", "diff_eps"],
    #             metrics=("mean", "std", "rms"), time_range=("2008-01-01", "2008-12-31"))