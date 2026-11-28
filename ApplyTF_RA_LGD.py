import os
import glob
from collections import defaultdict
import xarray as xr
import cmcrameri.cm as cmc
from utilities.Utility_functions_TimeFilt_new import *   # transfer_function, compute_lgd_new, ...
from utilities.Utility_functions_DIFF_TIME import derivative
from utilities.Utility_analytical_equations import *
from utilities.plotters import *

# =============================================================================
# Settings
# =============================================================================
main_dir = "../solutions/orbit_data"
data_static = np.load(f"{main_dir}/orbit_ggm05c.npy")

# sampling / filtering
dt = 5.0                    # [s]
fs = 1.0 / dt               # [Hz]
fc = 0.4e-3                # [Hz] high-pass edge
fc_low = 11.0e-3            # [Hz] low-pass edge
sos = butter(N=4, Wn=[fc, fc_low], btype="bandpass", fs=fs, output="sos")

# orbit
n_mean = 1.1170e-03         # [rad/s]
f_cpr = n_mean / (2 * np.pi)

# Welch segment
nperseg_spec = 2 * int(2 ** np.ceil(np.log2(2 / f_cpr / dt)))
average_input_str = "median"
print(f"n = {n_mean:.4e} rad/s  (1 cpr = {f_cpr * 1e3:.4f} mHz)")
print(f"nperseg = {nperseg_spec} [-]")

# map grid
LON_COL, LAT_COL = 1, 0
GRID_RES = 2.0              # [deg]
lon_edges = np.arange(-180, 180 + GRID_RES, GRID_RES)
lat_edges = np.arange(-90, 90 + GRID_RES, GRID_RES)
lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])


OUTDIR_MAPS = "../RESULTS/AnalyticalRA_to_LGDs/maps"
OUTDIR_SPEC = "../RESULTS/AnalyticalRA_to_LGDs/spectra"
OUTDIR_NC = "../RESULTS/AnalyticalRA_to_LGDs/netcdf"
for d in (OUTDIR_MAPS, OUTDIR_SPEC, OUTDIR_NC):
    os.makedirs(d, exist_ok=True)


# =============================================================================
# Field definitions: key -> (vmin, vmax, description)
# =============================================================================
FIELD_META = {
    "res_LGD":          (-15, 15, r"$\delta g^{\rm LoS}_{\rm AB}$ (res. LGD)"),
    "res_rhoddot":      (-15, 15, r"$\delta \ddot{\rho}$"),
    "res_LGD_tf":       (-15, 15, r"$(\delta g^{\rm LoS}_{\rm AB})_{\rm TF-fit}$ (res. LGD estimate)"),
    "res_LGD_tf_exact": (-15, 15, r"$(\delta g^{\rm LoS}_{\rm AB})_{\rm TF}$ (res. LGD estimate)"),
    "res_LGD_approx":   (-15, 15, r"$(\delta g^{\rm LoS}_{\rm AB})_{\rm TF-CW}$ (res. LGD estimate)"),
    "eps_TF":           (-1, 1,   r"$\epsilon_{\rm TF-fit}$ (transfer function error)"),
    "eps_TF_exact":     (-1, 1,   r"$\epsilon_{\rm TF}$ (transfer function error)"),
    "eps_approx":       (-1, 1,   r"$\epsilon_{\rm TF-CW}$"),
    "res_gc":           (-5, 5,   r"$\delta g_c$ (res. centrifugal acceleration)"),
}

# one accumulator per solution -> one NetCDF per solution
store = defaultdict(lambda: {"time": [], "grids": defaultdict(list),
                             "rms": defaultdict(list), "rmeds": defaultdict(list)})

# =============================================================================
# Loop over monthly files: orbit_<sol>_<YYYY>_<MM>.npy
# =============================================================================
for filepath in sorted(glob.glob(f"{main_dir}/orbit_*_????_??.npy")):
    fname = os.path.basename(filepath)
    _, sol, yyyy, mm = fname.replace(".npy", "").split("_")

    year_month = f"{yyyy}_{mm}"
    t_month = pd.Timestamp(year=int(yyyy), month=int(mm), day=15)  # mid-month stamp

    print(f"Processing {fname}")
    data_month = np.load(filepath)

    # ------------------------------------------------------------ residuals
    t = np.arange(data_month.shape[0]) * dt
    residual_lgd = data_month[:, 3] - data_static[:, 3]
    residual_rho = data_month[:, 2] - data_static[:, 2]
    residual_rho_ddot = derivative(residual_rho, dt, deriv=2, n_points=9, t=t)
    N = len(residual_rho_ddot)

    # ------------------------------------------------------------ CW method
    G_low, freqs, Z_CW = lgd_closure_spectrum(
        residual_rho_ddot, dt, n_mean, kappa=KAPPA, hilbert_sign=HILBERT_SIGN,
    )
    residual_lgdnewTF = np.fft.irfft(G_low, n=N) - n_mean**2 * residual_rho

    # ------------------------------------------------------------ GF method
    f_e, Z_GF, _, _, gamma2 = empirical_admittance(
        residual_rho_ddot, residual_lgd, dt, nperseg=nperseg_spec
    )
    Z_full = np.interp(freqs, f_e, np.nan_to_num(Z_GF, nan=0.0))  # f=0 bin is NaN
    # Optional coherence-based filling:
    # Z_full, _ = fill_admittance(f_e, Z_GF, gamma2, freqs, coh_min=0.9, mode="interp")
    residual_lgdTF_exact = np.fft.irfft(Z_full * np.fft.rfft(residual_rho_ddot), n=N)
    residual_lgdTF = compute_lgd_new(residual_rho_ddot)

    plot_admittance(
        freqs, Z_full, Z_CW,
        title=f"Admittance range acceleration → LGD — {sol} {year_month}",
        fname=os.path.join(OUTDIR_SPEC, f"admittance_{sol}_{year_month}.png"),
        n_mean=n_mean
    )

    # ------------------------------------------------------------ filtering [nm/s^2]
    residual_lgdnewTF = sosfiltfilt(sos, residual_lgdnewTF) * NM
    residual_lgdTF = sosfiltfilt(sos, residual_lgdTF) * NM
    residual_lgdTF_exact = sosfiltfilt(sos, residual_lgdTF_exact) * NM
    residual_rho_ddot = sosfiltfilt(sos, residual_rho_ddot) * NM
    residual_lgd = sosfiltfilt(sos, residual_lgd) * NM

    # ------------------------------------------------------------ errors
    residual_gc = residual_lgd - residual_rho_ddot
    error_TF = residual_lgd - residual_lgdTF
    error_TF_exact = residual_lgd - residual_lgdTF_exact
    error_newTF = residual_lgd - residual_lgdnewTF

    series = {
        "res_LGD": residual_lgd,
        "res_rhoddot": residual_rho_ddot,
        "res_LGD_tf": residual_lgdTF,
        "res_LGD_tf_exact": residual_lgdTF_exact,
        "res_LGD_approx": residual_lgdnewTF,
        "eps_TF": error_TF,
        "eps_TF_exact": error_TF_exact,
        "eps_approx": error_newTF,
        "res_gc": residual_gc,
    }

    # ------------------------------------------------------------ spectra
    signals = {
        r"$\delta g^{\rm LoS}_{\rm AB}$ (true)" + tag(residual_lgd):
            (residual_lgd, dict(color="k", lw=1.6)),
        r"$\delta\ddot\rho$" + tag(residual_rho_ddot):
            (residual_rho_ddot, dict(color="0.55", lw=1.0)),
        r"TF (Ghobadi-Far Z(f) = Sxy(f)/Sxx(f))" + tag(residual_lgdTF_exact):
            (residual_lgdTF_exact, dict(color="tab:blue", lw=0.9, ls="--")),
        r"TF (Ghobadi-Far Z(f) fit)" + tag(residual_lgdTF):
            (residual_lgdTF, dict(color="tab:blue", lw=1.1)),
        r"Analytical TF (TF-CW)" + tag(residual_lgdnewTF):
            (residual_lgdnewTF, dict(color="tab:green", lw=1.0, ls="--")),
    }
    errors = {
        r"$\epsilon_{\rm TF-fit}$" + tag(error_TF):
            (error_TF, dict(color="tab:blue", lw=1.1, ls="-")),
        r"$\epsilon_{\rm TF}$" + tag(error_TF_exact):
            (error_TF_exact, dict(color="tab:blue", lw=0.9, ls="--")),
        r"$\epsilon_{\rm TF-CW}$" + tag(error_newTF):
            (error_newTF, dict(color="tab:green", lw=1.0, ls="--")),
        r"$\delta g_c$ (centrifugal)" + tag(residual_gc):
            (residual_gc, dict(color="0.55", lw=1.0)),
    }
    plot_spectra(
        signals, errors, fs, nperseg_spec, f_cpr,
        title=f"Residual LGD spectra — {sol} {year_month}",
        fname=f"spectra_{sol}_{year_month}.png",
        outdir=OUTDIR_SPEC,
    )

    # ------------------------------------------------------------ maps + storage
    lon = (data_month[:, LON_COL] * 180 / np.pi + 180.0) % 360.0 - 180.0
    lat = data_month[:, LAT_COL] * 180 / np.pi

    S = store[sol]
    S["time"].append(t_month)
    for key, vals in series.items():
        vmin, vmax, desc = FIELD_META[key]
        stat = grid_median(lon, lat, vals, lon_edges, lat_edges)  # (nlon, nlat)

        plot_and_save(
            stat, lon_c, lat_c, vmin, vmax, cmc.vik,
            label="[nm/s²]",
            title=f"{desc}{tag(vals)} — {year_month}",
            fname=f"{key}_{sol}_{year_month}.png",
            outdir=OUTDIR_MAPS,
        )

        S["grids"][key].append(stat.T.astype(np.float32))  # store as (lat, lon)
        S["rms"][key].append(rms(vals, SCALE))
        S["rmeds"][key].append(rmeds(vals, SCALE))

# =============================================================================
# NetCDF per solution: gridded fields (time, lat, lon) + RMS / RMedS (time)
# =============================================================================
for sol, S in store.items():
    order = np.argsort(S["time"])
    time = pd.DatetimeIndex(S["time"])[order]

    data_vars = {}
    for key, (_, _, desc) in FIELD_META.items():
        common = {"units": "nm/s^2", "description": desc}
        data_vars[key] = (
            ("time", "lat", "lon"), np.stack(S["grids"][key])[order],
            {**common, "long_name": f"{key}: monthly median per grid cell"},
        )
        data_vars[f"{key}_rms"] = (
            ("time",), np.asarray(S["rms"][key])[order],
            {**common, "long_name": f"{key}: RMS of the 5-s time series over the month"},
        )
        data_vars[f"{key}_rmeds"] = (
            ("time",), np.asarray(S["rmeds"][key])[order],
            {**common, "long_name": f"{key}: root-median-square of the 5-s time series over the month"},
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": ("time", time, {"long_name": "month (mid-month stamp)"}),
            "lat": ("lat", lat_c, {"units": "degrees_north"}),
            "lon": ("lon", lon_c, {"units": "degrees_east"}),
        },
        attrs={
            "title": f"Residual LGD / range-acceleration fields and errors — {sol}",
            "solution": sol,
            "static_reference": "orbit_ggm05c",
            "sampling_input": f"{dt} s",
            "bandpass": f"Butterworth N=4, {fc*1e3:.2f}-{fc_low*1e3:.1f} mHz (sosfiltfilt)",
            "grid_resolution_deg": GRID_RES,
            "kappa": str(KAPPA),
            "hilbert_sign": HILBERT_SIGN,
            "processing": "residual w.r.t. static run, band-pass filtered, binned median "
                          "on lon/lat grid; RMS and RMedS over all samples of the month",
        },
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    out_nc = os.path.join(OUTDIR_NC, f"lgd_residuals_{sol}.nc")
    ds.to_netcdf(out_nc, encoding=encoding)
    print(f"Saved {out_nc}  ({len(time)} months)")