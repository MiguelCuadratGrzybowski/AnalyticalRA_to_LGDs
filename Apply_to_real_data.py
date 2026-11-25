import os
import glob
from collections import defaultdict
import xarray as xr
import cmcrameri.cm as cmc
from utilities.Utility_functions_TimeFilt_new import *   # transfer_function, compute_lgd_new, ...
from utilities.Utility_functions_DIFF_TIME import derivative
from utilities.Utility_analytical_equations import *
from utilities.plotters import *
import re

# =============================================================================
# Settings
# =============================================================================
main_dir = rf"C:\Users\michalcuadratg\PycharmProjects\HighFreqMassModels\sub_datafiles\monthly_rrsPublic"
dir_with_lgds = rf"C:\Users\michalcuadratg\PycharmProjects\RESULTS\AnalyticalRA_to_LGDs\synthesis"

# sampling / filtering
dt = 5.0                    # [s]
fs = 1.0 / dt               # [Hz]
fc = 0.4e-3                 # [Hz] high-pass edge
fc_low = 11.0e-3            # [Hz] low-pass edge
sos = butter(N=4, Wn=[fc, fc_low], btype="bandpass", fs=fs, output="sos")
sos_low = butter(N=4, Wn=fc_low, btype="lowpass", fs=fs, output="sos")

# Inputs
inputs = {
    "n": 4,
    "fs": 0.2,
    "int_flag": True,
    "tol_gap": 20.0,           # NEW: use nicer number than 20 s
    "dx_interp": 5.0,          # regular 5 s grid
    "arc_seconds": 86400.0,    # 1 day arcs
    "outlier_N": np.inf,    # requested N*sigma_robust
    "outlier_maxrun": 5*12, # ~60 s if dx=5 s
    "min_piece_seconds": 250,
    # minimum-piece sizing derived from edge effects (in new module)
    # you can also override via:
    # "min_piece_samples": ...
}

# map grid
GRID_RES = 2.0              # [deg]
lon_edges = np.arange(-180, 180 + GRID_RES, GRID_RES)
lat_edges = np.arange(-90, 90 + GRID_RES, GRID_RES)
lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])

OUTDIR_MAPS = "../RESULTS/AnalyticalRA_to_LGDs/geofitData/maps"
OUTDIR_SPEC = "../RESULTS/AnalyticalRA_to_LGDs/geofitData/spectra"
OUTDIR_NC = "../RESULTS/AnalyticalRA_to_LGDs/geofitData/netcdf"
for d in (OUTDIR_MAPS, OUTDIR_SPEC, OUTDIR_NC):
    os.makedirs(d, exist_ok=True)


# =============================================================================
# Field definitions: key -> (vmin, vmax, description)
# =============================================================================
FIELD_META = {
    "res_LGD":          (-15, 15, r"$\delta g^{\rm LoS}_{\rm AB}$ (res. LGD)"),
    "res_rhoddot":      (-15, 15, r"$\delta \ddot{\rho}$"),
    "res_LGD_tf":       (-15, 15, r"$(\delta g^{\rm LoS}_{\rm AB})_{\rm TF-fit}$ (res. LGD estimate)"),
    "res_LGD_approx":   (-15, 15, r"$(\delta g^{\rm LoS}_{\rm AB})_{\rm TF-CW}$ (res. LGD estimate)"),
    "eps_TF":           (-1, 1,   r"$\epsilon_{\rm TF-fit}$ (transfer function error)"),
    "eps_approx":       (-1, 1,   r"$\epsilon_{\rm TF-CW}$"),
    "res_gc":           (-5, 5,   r"$\delta g_c$ (res. centrifugal acceleration)"),
}

# one accumulator per solution -> one NetCDF per solution
store = defaultdict(lambda: {"time": [], "grids": defaultdict(list),
                             "rms": defaultdict(list), "rmeds": defaultdict(list)})

# =============================================================================
# Settings (data / columns)
# =============================================================================
SOLUTION = "CSR_RL06"          # key for `store` (one NetCDF per solution)

# column names: check against your files
RR_TIME_COL  = "timestamp"     # rr file: seconds since RR_EPOCH
RR_COL       = "up_global"     # range-rate residual column
RR_LAT_COL   = "latitude_mid"  # rr mid-point latitude  (diagnostic only)
RR_LON_COL   = "longitude_mid" # rr mid-point longitude (diagnostic only)

LGD_TIME_COL = "gps_time"      # lgd file: seconds since GPS_EPOCH
LGD_COL      = "lgd"           # LGD column
H_COL, RHO_COL = "mid_alt", "range"
# positions for gridding: taken from the LGD synthesis (precise orbits), NOT from rr
LGD_LAT_COL  = "mid_lat"       # mid-point latitude  in lgd_YYYY_MM.parquet
LGD_LON_COL  = "mid_lon"       # mid-point longitude in lgd_YYYY_MM.parquet
LGD_LATLON_IN_RADIANS = False

RR_EPOCH  = pd.Timestamp("2000-01-01 00:00:00")
GPS_EPOCH = pd.Timestamp("2000-01-01 12:00:00")   # GRACE L1B convention; adjust if needed
_REF = pd.Timestamp("2000-01-01 00:00:00")

# =============================================================================
# Arc settings
# =============================================================================
MAX_GAP_SEC   = 20.0           # [s] larger gap -> new arc
MIN_ARC_SEC   = 250            # [s] drop arcs shorter
EDGE_TRIM_SEC = 5.0            # [s] drop arc ends after filtfilt
DEBUG_PLOTS   = False

# outlier screening of the REAL range-rate residuals only (LGD is synthetic -> untouched)
OUTLIER_SG_WIN   = 11          # [samples] Savitzky-Golay window (55 s at 5 s)
OUTLIER_SG_POLY  = 3           # SG polynomial order
OUTLIER_MAD_SEC  = 5 * 15      # [s] running window for the robust scale (MAD)
OUTLIER_K        = 5.0         # reject |res| > K * 1.4826 * MAD
OUTLIER_MAX_ITER = 3

# Welch segment
nperseg_spec = 1 * int(2 ** np.ceil(np.log2(2 / 0.1760e-3 / dt)))
average_input_str = "median"
print(f"nperseg = {nperseg_spec} [-]")


# =============================================================================
# Helpers
# =============================================================================
def _sec_since_2000(t, epoch):
    return np.asarray(t, dtype=float) + (epoch - _REF).total_seconds()


def great_circle_km(lat1, lon1, lat2, lon2, R=6371.0):
    """Haversine distance [km], inputs in degrees."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi, dlmb = p2 - p1, np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_and_match(rr_path, lgd_path):
    """Read both files, convert to a common time axis and match at the coarser rate.
    Range rate comes from rr; LGD, mid_alt, range and the gridding positions from lgd."""
    rr = pd.read_parquet(rr_path)
    lgd = pd.read_parquet(lgd_path)

    missing = [c for c in (LGD_TIME_COL, LGD_COL, H_COL, RHO_COL, LGD_LAT_COL, LGD_LON_COL)
               if c not in lgd.columns]
    if missing:
        raise RuntimeError(f"columns {missing} not in {os.path.basename(lgd_path)}; "
                           f"available: {list(lgd.columns)}")

    rr_cols = {"t": _sec_since_2000(rr[RR_TIME_COL], RR_EPOCH),
               "rr": rr[RR_COL].to_numpy()}
    if RR_LAT_COL in rr.columns and RR_LON_COL in rr.columns:   # diagnostic only
        rr_cols["rr_lat"] = rr[RR_LAT_COL].to_numpy()
        rr_cols["rr_lon"] = rr[RR_LON_COL].to_numpy()
    rr = pd.DataFrame(rr_cols).dropna().sort_values("t").drop_duplicates("t")

    to_deg = 180.0 / np.pi if LGD_LATLON_IN_RADIANS else 1.0
    lgd = (lgd.assign(t=_sec_since_2000(lgd[LGD_TIME_COL], GPS_EPOCH),
                      lat_grid=lgd[LGD_LAT_COL] * to_deg,
                      lon_grid=(lgd[LGD_LON_COL] * to_deg + 180.0) % 360.0 - 180.0)
              .dropna(subset=[LGD_COL]).sort_values("t").drop_duplicates("t"))

    dt_rr, dt_lgd = np.median(np.diff(rr.t)), np.median(np.diff(lgd.t))
    coarse, fine = (rr, lgd) if dt_rr >= dt_lgd else (lgd, rr)
    fine = fine.assign(t_fine=fine["t"])

    t0, t1 = max(rr.t.iloc[0], lgd.t.iloc[0]), min(rr.t.iloc[-1], lgd.t.iloc[-1])
    coarse = coarse[(coarse.t >= t0) & (coarse.t <= t1)]

    m = pd.merge_asof(coarse, fine, on="t", direction="nearest",
                      tolerance=0.5 * min(dt_rr, dt_lgd)).dropna(subset=["rr", LGD_COL])

    if len(m) < 0.5 * len(coarse):
        off = pd.merge_asof(coarse, fine, on="t", direction="nearest")
        raise RuntimeError(
            f"only {len(m)}/{len(coarse)} epochs matched; median offset to nearest "
            f"epoch = {np.median(off.t - off.t_fine):.1f} s -> check RR_EPOCH/GPS_EPOCH "
            f"(a 12 h or 13-18 s offset means epoch or leap-second mismatch)")

    print(f"  dt_rr = {dt_rr:g} s, dt_lgd = {dt_lgd:g} s -> matched {len(m)} epochs "
          f"at {max(dt_rr, dt_lgd):g} s (max |offset| {np.abs(m.t - m.t_fine).max():.3f} s)")

    if "rr_lat" in m.columns:   # how far off are the rr positions w.r.t. the synthesis?
        d_km = great_circle_km(m["rr_lat"], m["rr_lon"], m["lat_grid"], m["lon_grid"])
        print(f"  |pos_rr - pos_LGD|: median {np.median(d_km):.3f} km, "
              f"p99 {np.percentile(d_km, 99):.3f} km, max {np.max(d_km):.3f} km")
        m = m.drop(columns=["rr_lat", "rr_lon"])

    return m.drop(columns="t_fine"), max(dt_rr, dt_lgd)

def load_rr(rr_path):
    """Range-rate residuals on their own (native) epochs, time in s since 2000-01-01 00:00."""
    rr = pd.read_parquet(rr_path)
    cols = {"t": _sec_since_2000(rr[RR_TIME_COL], RR_EPOCH),
            "rr": rr[RR_COL].to_numpy()}
    if RR_LAT_COL in rr.columns and RR_LON_COL in rr.columns:   # diagnostic only
        cols["rr_lat"] = rr[RR_LAT_COL].to_numpy()
        cols["rr_lon"] = rr[RR_LON_COL].to_numpy()
    return (pd.DataFrame(cols).dropna(subset=["t", "rr"])
              .sort_values("t").drop_duplicates("t").reset_index(drop=True))


def match_rr_lgd(rr, lgd_path):
    """Match the (already differentiated, outlier-cleaned) rr DataFrame [t, rr, rho_ddot, ...]
    with the LGD synthesis at the coarser of the two samplings.
    LGD, mid_alt, range and the gridding positions come from the lgd file."""
    lgd = pd.read_parquet(lgd_path)

    missing = [c for c in (LGD_TIME_COL, LGD_COL, H_COL, RHO_COL, LGD_LAT_COL, LGD_LON_COL)
               if c not in lgd.columns]
    if missing:
        raise RuntimeError(f"columns {missing} not in {os.path.basename(lgd_path)}; "
                           f"available: {list(lgd.columns)}")

    rr = rr.dropna(subset=["rho_ddot"])

    to_deg = 180.0 / np.pi if LGD_LATLON_IN_RADIANS else 1.0
    lgd = (lgd.assign(t=_sec_since_2000(lgd[LGD_TIME_COL], GPS_EPOCH),
                      lat_grid=lgd[LGD_LAT_COL] * to_deg,
                      lon_grid=(lgd[LGD_LON_COL] * to_deg + 180.0) % 360.0 - 180.0)
              .dropna(subset=[LGD_COL]).sort_values("t").drop_duplicates("t"))

    dt_rr, dt_lgd = np.median(np.diff(rr.t)), np.median(np.diff(lgd.t))
    coarse, fine = (rr, lgd) if dt_rr >= dt_lgd else (lgd, rr)
    fine = fine.assign(t_fine=fine["t"])

    t0, t1 = max(rr.t.iloc[0], lgd.t.iloc[0]), min(rr.t.iloc[-1], lgd.t.iloc[-1])
    coarse = coarse[(coarse.t >= t0) & (coarse.t <= t1)]

    m = pd.merge_asof(coarse, fine, on="t", direction="nearest",
                      tolerance=0.5 * min(dt_rr, dt_lgd)).dropna(subset=["rho_ddot", LGD_COL])

    if len(m) < 0.5 * len(coarse):
        off = pd.merge_asof(coarse, fine, on="t", direction="nearest")
        raise RuntimeError(
            f"only {len(m)}/{len(coarse)} epochs matched; median offset to nearest "
            f"epoch = {np.median(off.t - off.t_fine):.1f} s -> check RR_EPOCH/GPS_EPOCH "
            f"(a 12 h or 13-18 s offset means epoch or leap-second mismatch)")

    print(f"  dt_rr = {dt_rr:g} s, dt_lgd = {dt_lgd:g} s -> matched {len(m)} epochs "
          f"at {max(dt_rr, dt_lgd):g} s (max |offset| {np.abs(m.t - m.t_fine).max():.3f} s)")

    if "rr_lat" in m.columns:   # how far off are the rr positions w.r.t. the synthesis?
        d_km = great_circle_km(m["rr_lat"], m["rr_lon"], m["lat_grid"], m["lon_grid"])
        print(f"  |pos_rr - pos_LGD|: median {np.median(d_km):.3f} km, "
              f"p99 {np.percentile(d_km, 99):.3f} km, max {np.max(d_km):.3f} km")
        m = m.drop(columns=["rr_lat", "rr_lon"])

    return m.drop(columns="t_fine"), max(dt_rr, dt_lgd)


def flag_rr_outliers(t, rr, dt=dt, win=OUTLIER_SG_WIN, poly=OUTLIER_SG_POLY,
                     mad_sec=OUTLIER_MAD_SEC, k=OUTLIER_K, max_iter=OUTLIER_MAX_ITER):
    """Outlier flags for range-rate residuals.
    1. per contiguous segment, subtract a local Savitzky-Golay fit (follows the
       signal, incl. in-band curvature, so only spikes remain);
    2. robust scale = running MAD (time-based, gap-aware) of that residual;
    3. flag |res| > K * 1.4826 * MAD, iterate with flagged samples interpolated.
    Returns a boolean mask: True = outlier."""
    from scipy.signal import savgol_filter

    t = np.asarray(t, float)
    x0 = np.asarray(rr, float)
    bad = ~np.isfinite(x0)
    seg_id = np.r_[0, np.cumsum(np.diff(t) > 1.5 * dt)]   # contiguous dt-sampled pieces
    seg_bounds = np.flatnonzero(np.r_[True, np.diff(seg_id) != 0, True])

    for _ in range(max_iter):
        res = np.full_like(x0, np.nan)
        for i0, i1 in zip(seg_bounds[:-1], seg_bounds[1:]):
            if i1 - i0 < win:
                continue                                    # too short to fit: leave unscreened
            xs = (pd.Series(np.where(bad[i0:i1], np.nan, x0[i0:i1]))
                    .interpolate(limit_direction="both").to_numpy())
            res[i0:i1] = x0[i0:i1] - savgol_filter(xs, win, poly)
        dev = pd.Series(np.abs(np.where(bad, np.nan, res)), index=pd.to_timedelta(t, unit="s"))
        mad = dev.rolling(f"{mad_sec:g}s", center=True, min_periods=10).median().to_numpy()
        new = (np.abs(res) > k * 1.4826 * mad) & ~bad
        if not new.any():
            break
        bad |= new
    return bad


def split_into_arcs(df, dt, max_gap_sec=MAX_GAP_SEC, min_arc_sec=MIN_ARC_SEC):
    """Split matched data into uninterrupted arcs (gaps <= max_gap_sec are bridged).
    Each arc is put on a regular dt grid; short gaps are linearly interpolated.
    Returns a list of (grid_df, ok) with ok = True for real (non-interpolated) samples."""
    t = df.t.to_numpy()
    breaks = np.flatnonzero(np.diff(t) > max_gap_sec + 0.5) + 1
    bounds = np.r_[0, breaks, len(df)]

    arcs, n_short = [], 0
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        seg = df.iloc[i0:i1]
        if seg.t.iloc[-1] - seg.t.iloc[0] < min_arc_sec:
            n_short += 1
            continue
        idx = np.round((seg.t - seg.t.iloc[0]) / dt).astype(int)
        g = (seg.select_dtypes("number").set_index(idx)
               .loc[lambda x: ~x.index.duplicated()]
               .reindex(np.arange(idx.iloc[-1] + 1)))
        ok = g["rr"].notna().to_numpy() & g[LGD_COL].notna().to_numpy()
        g = g.interpolate(limit_area="inside")
        g["t"] = seg.t.iloc[0] + g.index.to_numpy() * dt
        arcs.append((g.reset_index(drop=True), ok))

    lens = [len(g) * dt / 3600 for g, _ in arcs]
    print(f"  {len(arcs)} arcs kept (dropped {n_short} < {min_arc_sec:g} s); "
          f"lengths {min(lens, default=0):.2f}-{max(lens, default=0):.2f} h")
    return arcs


# =============================================================================
# Loop over monthly files: rr_YYYY_MM_DD.parquet  +  lgd_YYYY_MM.parquet
# =============================================================================
for rr_path in sorted(glob.glob(os.path.join(main_dir, "rr_????_??_??.parquet"))):
    fname = os.path.basename(rr_path)
    yyyy, mm, _ = re.match(r"rr_(\d{4})_(\d{2})_(\d{2})\.parquet$", fname).groups()
    lgd_path = os.path.join(dir_with_lgds, f"lgd_{yyyy}_{mm}.parquet")
    sol = SOLUTION

    year_month = f"{yyyy}_{mm}"
    t_month = pd.Timestamp(year=int(yyyy), month=int(mm), day=15)  # mid-month stamp

    if not os.path.exists(lgd_path):
        print(f"Skipping {fname}: no {os.path.basename(lgd_path)}")
        continue
    print(f"Processing {fname}  <->  {os.path.basename(lgd_path)}")

    # # ------------------------------------------------------------ load + match (OLD)
    # try:
    #     matched, dt_match = load_and_match(rr_path, lgd_path)
    # except RuntimeError as e:
    #     print(f"  {e}")
    #     continue
    # if not np.isclose(dt_match, dt):
    #     print(f"  matched sampling {dt_match:g} s != dt = {dt:g} s (filter/sos assume dt); skipping")
    #     continue
    # rr_bad = flag_rr_outliers(matched["t"].to_numpy(), matched["rr"].to_numpy())
    # print(f"  range-rate outliers: {rr_bad.sum()} / {len(rr_bad)} "
    #       f"({100 * rr_bad.mean():.3f} %) removed")

    # ------------------------------------------------------------ range acceleration (native rr epochs)
    rr_df = load_rr(rr_path)
    # ------------------------------------------------------------ outliers (real data only)
    rr_bad = flag_rr_outliers(rr_df["t"].to_numpy(), rr_df["rr"].to_numpy())
    print(f"  range-rate outliers: {rr_bad.sum()} / {len(rr_bad)} "
          f"({100 * rr_bad.mean():.3f} %) removed")
    rr_df = rr_df.loc[~rr_bad].reset_index(drop=True)
    # removed epochs become gaps: <= MAX_GAP_SEC -> interpolated inside the arc
    # (and excluded from maps via `ok`), longer -> arc break

    # ------------------------------------------------------------ range acceleration (own method)
    # output is typically SHORTER than the input -> use its own time axis `new_time`
    rho_ddot, new_time = perform_filtering_sequence_new(
        rr_df["rr"].to_numpy(), rr_df["t"].to_numpy(),
        methods=["diff"], Ns=[], btypes=[], cutoffs=[],
        **inputs)
    rho_ddot = np.asarray(rho_ddot, dtype=float).ravel()
    new_time = np.asarray(new_time, dtype=float).ravel()
    if len(rho_ddot) != len(new_time):
        print(f"  len(rho_ddot) = {len(rho_ddot)} != len(new_time) = {len(new_time)}; skipping")
        continue
    if abs(np.median(new_time) - np.median(rr_df["t"])) > 86400.0:
        print("  new_time is not on the rr time axis (s since 2000-01-01 00:00?); skipping")
        continue
    print(f"  diff: {len(rr_df)} -> {len(rho_ddot)} samples")

    # range acceleration on its own epochs; carry rr (and diagnostic rr positions) along
    acc_df = (pd.DataFrame({"t": new_time, "rho_ddot": rho_ddot})
              .dropna().sort_values("t").drop_duplicates("t"))
    acc_df = pd.merge_asof(acc_df, rr_df, on="t", direction="nearest", tolerance=0.5 * dt)

    # ------------------------------------------------------------ match with LGD synthesis
    try:
        matched, dt_match = match_rr_lgd(acc_df, lgd_path)
    except RuntimeError as e:
        print(f"  {e}")
        continue
    if not np.isclose(dt_match, dt):
        print(f"  matched sampling {dt_match:g} s != dt = {dt:g} s (filter/sos assume dt); skipping")
        continue

    # ------------------------------------------------------------ monthly orbit means
    n_mean = np.nanmean(np.sqrt(GM_EARTH / (R_EARTH + matched[H_COL]) ** 3))
    f_cpr = n_mean / (2 * np.pi)

    h_mean = matched[H_COL].mean()
    rho_mean = matched[RHO_COL].mean()
    print(f"  h = {h_mean / 1e3:.6g} km, rho = {rho_mean / 1e3:.6g} km  (monthly means)")
    print(f"  n = {n_mean:.6g}")

    # ------------------------------------------------------------ arcs
    arcs = split_into_arcs(matched, dt)
    if not arcs:
        print("  no usable arcs; skipping")
        continue

    # # range acceleration per arc
    # for g, ok in arcs:
    #     t_arc = g["t"].to_numpy() - g["t"].iloc[0]
    #
    #     g["rho_ddot"] = derivative(g["rr"].to_numpy(), dt, deriv=1, n_points=3, t=t_arc)

    # ------------------------------------------------------------ GF admittance (month-wide, plot only)
    # Welch needs >= nperseg samples: estimate per long-enough arc, then average
    # weighted by the number of Welch segments (50 % overlap) in each arc.
    Zs, ws = [], []
    for g, ok in arcs:
        if len(g) < nperseg_spec:
            continue
        f_e, Z_a, _, _, _ = empirical_admittance(
            g["rho_ddot"].to_numpy(), g[LGD_COL].to_numpy(), dt, nperseg=nperseg_spec
        )
        Zs.append(Z_a)
        ws.append((len(g) - nperseg_spec) // (nperseg_spec // 2) + 1)
    Z_GF = np.average(np.array(Zs), axis=0, weights=ws) if Zs else None
    print(f"  GF admittance from {len(Zs)} arcs ({sum(ws)} Welch segments)" if Zs
          else f"  no arc >= nperseg ({nperseg_spec * dt / 3600:.1f} h): no GF admittance")

    # ------------------------------------------------------------ per arc: TF methods + filtering
    keys = ["res_LGD", "res_rhoddot", "res_LGD_approx", "res_LGD_tf"]
    month = defaultdict(list)      # good samples of all arcs (for maps / RMS)
    n_trim = int(round(EDGE_TRIM_SEC / dt))
    longest = max(range(len(arcs)), key=lambda k: len(arcs[k][0]))

    for k, (g, ok) in enumerate(arcs):
        rho_ddot = g["rho_ddot"].to_numpy()
        lgd = g[LGD_COL].to_numpy()
        N = len(rho_ddot)

        # CW method
        G_low, freqs, Z_CW = lgd_closure_spectrum(
            rho_ddot, dt, n_mean, kappa=KAPPA, hilbert_sign=HILBERT_SIGN,
            h=h_mean, rho=rho_mean,
        )
        # -n^2 * rho  ==  + (n^2 / w^2) * rho_ddot   since rho(w) = -rho_ddot(w) / w^2
        Rddot = np.fft.rfft(rho_ddot, n=N)
        assert len(freqs) == len(Rddot), "freqs must be np.fft.rfftfreq(N, dt)"
        omega = 2.0 * np.pi * np.asarray(freqs)
        K = np.zeros_like(omega)
        band = freqs >= 1e-4                    # skip DC (n^2/w^2 blows up)
        K[band] = n_mean**2 / omega[band] ** 2
        lgd_newTF = np.fft.irfft(G_low + K * Rddot, n=N)

        # GF fit
        lgd_TF = compute_lgd_new(rho_ddot)

        if DEBUG_PLOTS and k == longest:
            plt.plot(rho_ddot, label=r"$\delta\ddot\rho$")
            plt.plot(lgd, label="LGD")
            plt.legend(); plt.title(f"{year_month}: longest arc (N={N})"); plt.show()

        # filtering [nm/s^2], per arc
        arc = {
            "res_LGD":        sosfiltfilt(sos, lgd) * NM,
            "res_rhoddot":    sosfiltfilt(sos, rho_ddot) * NM,
            "res_LGD_approx": sosfiltfilt(sos, lgd_newTF) * NM,
            "res_LGD_tf":     sosfiltfilt(sos, lgd_TF) * NM,
        }

        # keep the longest arc continuous (incl. interpolated samples) for spectra / admittance plot
        if k == longest:
            spec_arc = {key: arc[key][n_trim:N - n_trim] for key in keys}
            Z_full_plot = (np.interp(freqs, f_e, np.nan_to_num(Z_GF, nan=0.0))  # f=0 bin is NaN
                           if Z_GF is not None else np.full(len(freqs), np.nan, dtype=complex))
            freqs_plot, Z_CW_plot = freqs, Z_CW

        # maps / statistics: only real samples (no interpolated gaps, edge trim)
        good = ok.copy()
        if n_trim:
            good[:n_trim] = False
            good[-n_trim:] = False
        for key in keys:
            month[key].append(arc[key][good])
        month["lat"].append(g["lat_grid"].to_numpy()[good])   # LGD-synthesis positions
        month["lon"].append(g["lon_grid"].to_numpy()[good])

    month = {key: np.concatenate(v) for key, v in month.items()}

    # ------------------------------------------------------------ month series + errors
    residual_lgd      = month["res_LGD"]
    residual_rho_ddot = month["res_rhoddot"]
    residual_lgdnewTF = month["res_LGD_approx"]
    residual_lgdTF    = month["res_LGD_tf"]

    residual_gc = residual_lgd - residual_rho_ddot
    error_TF    = residual_lgd - residual_lgdTF
    error_newTF = residual_lgd - residual_lgdnewTF

    series = {
        "res_LGD": residual_lgd,
        "res_rhoddot": residual_rho_ddot,
        "res_LGD_tf": residual_lgdTF,
        "res_LGD_approx": residual_lgdnewTF,
        "eps_TF": error_TF,
        "eps_approx": error_newTF,
        "res_gc": residual_gc,
    }

    plot_admittance(
        freqs_plot, Z_full_plot, Z_CW_plot,
        title=f"Admittance range acceleration → LGD — {sol} {year_month}",
        fname=os.path.join(OUTDIR_SPEC, f"admittance_{sol}_{year_month}.png"),
        n_mean=n_mean,
        option_bool=False
    )

    # ------------------------------------------------------------ spectra (longest arc)
    # Welch needs a continuous series -> use the longest arc; labels show month-wide RMS tags
    s = spec_arc
    s_gc    = s["res_LGD"] - s["res_rhoddot"]
    s_TF    = s["res_LGD"] - s["res_LGD_tf"]
    s_newTF = s["res_LGD"] - s["res_LGD_approx"]

    signals = {
        r"$\delta g^{\rm LoS}_{\rm AB}$ (true)" + tag(residual_lgd):
            (s["res_LGD"], dict(color="k", lw=1.6)),
        r"$\delta\ddot\rho$" + tag(residual_rho_ddot):
            (s["res_rhoddot"], dict(color="0.55", lw=1.0)),
        r"TF (Ghobadi-Far Z(f) fit)" + tag(residual_lgdTF):
            (s["res_LGD_tf"], dict(color="tab:blue", lw=1.1)),
        r"Analytical TF (TF-CW)" + tag(residual_lgdnewTF):
            (s["res_LGD_approx"], dict(color="tab:green", lw=1.0, ls="--")),
    }
    errors = {
        r"$\epsilon_{\rm TF-fit}$" + tag(error_TF):
            (s_TF, dict(color="tab:blue", lw=1.1, ls="-")),
        r"$\epsilon_{\rm TF-CW}$" + tag(error_newTF):
            (s_newTF, dict(color="tab:green", lw=1.0, ls="--")),
        r"$\delta g_c$ (centrifugal)" + tag(residual_gc):
            (s_gc, dict(color="0.55", lw=1.0)),
    }
    plot_spectra(
        signals, errors, fs, nperseg_spec, f_cpr,
        title=f"Residual LGD spectra — {sol} {year_month} (longest arc, "
              f"{len(s['res_LGD']) * dt / 3600:.1f} h)",
        fname=f"spectra_{sol}_{year_month}.png",
        outdir=OUTDIR_SPEC,
    )

    # ------------------------------------------------------------ maps + storage
    lat = month["lat"]          # positions from the LGD synthesis (precise orbits)
    lon = month["lon"]          # already wrapped to [-180, 180)

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
            "bandpass": f"Butterworth N=4, {fc*1e3:.2f}-{fc_low*1e3:.1f} mHz (sosfiltfilt, per arc)",
            "grid_resolution_deg": GRID_RES,
            "kappa": str(KAPPA),
            "hilbert_sign": HILBERT_SIGN,
            "positions": "mid-point lat/lon from the LGD synthesis (precise orbits), "
                         "matched in time to the range-rate epochs",
            "processing": "range-rate outliers removed (SG + running MAD); arcs split at gaps "
                          f"> {MAX_GAP_SEC:g} s; band-pass filtered per arc; binned median on "
                          "lon/lat grid; RMS and RMedS over all real samples of the month",
        },
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    out_nc = os.path.join(OUTDIR_NC, f"lgd_residuals_{sol}_real.nc")
    ds.to_netcdf(out_nc, encoding=encoding)
    print(f"Saved {out_nc}  ({len(time)} months)")
