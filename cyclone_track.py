"""
Local AI tropical-cyclone forecast + track & intensity extraction with NVIDIA earth2studio.

Pipeline:
  1. Load one or more AI global forecast models (FCN, Pangu, ...)   -> px (prognostic)
  2. Load TCTrackerWuDuan cyclone tracker                           -> dx (diagnostic)
  3. Pull initial conditions from a data source (GFS live, or ARCO ERA5)
  4. Roll each model forward N steps, run the tracker each step
  5. (optional) also track the analysis "truth" over the same window
  6. Plot forecast track(s) vs truth  -> outputs/cyclone_tracks.jpg
  7. (optional) plot storm INTENSITY (min central pressure / max 10m wind)
                of the primary storm  -> outputs/intensity.png

The tracker returns, per track point, the 4 channels: [lat, lon, min_MSLP(Pa), max_10m_wind(m/s)],
so intensity comes for free -- we just plot channels 2 and 3.

Run:  python cyclone_track.py
Any config below can be overridden via env vars (handy for quick experiments), e.g.:
  WX_COMPARE=fcn,pangu WX_START=2024-10-06T00:00 WX_NSTEPS=16 WX_TRUTH=True \
  WX_EXTENT=-100,-78,17,32 python cyclone_track.py

Hardware notes (RTX 5080, 16 GB):
  - FCN / FourCastNet: fits and runs fast on the GPU (~1 GB VRAM).
  - Pangu: its 0.25 deg ONNX transformer OOMs sharing the 16 GB card, so it runs on the
    CPU here (see CPU_MODELS) -- slower but fine; the GPU still runs the tracker.
  - Aurora (0.25 deg): activation-bound, does NOT fit 16 GB even with weight offload (~24 GB+).
"""

import os
# Reduce CUDA fragmentation so the model (PyTorch) and tracker (cupy) can share 16 GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import contextlib
import importlib
from datetime import datetime, timedelta

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Configuration -- edit these (or override with WX_* env vars)
# ----------------------------------------------------------------------------
MODEL       = "fcn"           # single model: "fcn" | "pangu" | "sfno" | "aurora"
COMPARE     = []              # e.g. ["fcn", "pangu"] to overlay several models; [] -> just MODEL
INTENSITY   = True            # also plot storm intensity (min pressure / max wind)
USE_BF16    = True            # run torch models under bfloat16 autocast
DATA_SOURCE = "gfs"           # "gfs" (live NOAA, archive ~2021->now) or "arco" (ERA5)
START_TIME  = datetime(2026, 6, 5, 12)   # init time; GFS has 00/06/12/18Z cycles
NSTEPS      = 20              # forecast steps; 6h each -> 16 = 4 days, 40 = 10 days
TRACK_TRUTH = False           # overlay analysis "truth" (only for past windows)
MAP_EXTENT  = [-100, -78, 17, 32]    # [lon_min, lon_max, lat_min, lat_max] in [-180,180]
VORT_THRESHOLD = 1.0e-4       # detection sensitivity (850 hPa vorticity, s^-1): LOWER = lock
                              # onto weaker/earlier lows (more false alarms); HIGHER = only strong
STRENGTH    = True            # color track markers by Saffir-Simpson category
CATEGORY    = "estimate"      # "estimate" = ballpark Vmax from central-pressure depth (undoes
                              #   0.25deg smoothing); "wind" = raw model 10m wind (~2-3 cats low)
INTENSITY_GAIN = 2.0          # ["estimate" only] MSLP-deficit gain; ~2.0 lands Milton near Cat 4

# --- optional env overrides (don't edit) ---
MODEL       = os.environ.get("WX_MODEL", MODEL)
COMPARE     = os.environ["WX_COMPARE"].split(",") if os.environ.get("WX_COMPARE") else COMPARE
INTENSITY   = os.environ.get("WX_INTENSITY", str(INTENSITY)) == "True"
DATA_SOURCE = os.environ.get("WX_DATA", DATA_SOURCE)
NSTEPS      = int(os.environ.get("WX_NSTEPS", NSTEPS))
TRACK_TRUTH = os.environ.get("WX_TRUTH", str(TRACK_TRUTH)) == "True"
if os.environ.get("WX_START"):
    START_TIME = datetime.fromisoformat(os.environ["WX_START"])
if os.environ.get("WX_EXTENT"):
    MAP_EXTENT = [float(v) for v in os.environ["WX_EXTENT"].split(",")]
VORT_THRESHOLD = float(os.environ.get("WX_VORT", VORT_THRESHOLD))
STRENGTH    = os.environ.get("WX_STRENGTH", str(STRENGTH)) == "True"
CATEGORY    = os.environ.get("WX_CATEGORY", CATEGORY)
INTENSITY_GAIN = float(os.environ.get("WX_GAIN", INTENSITY_GAIN))
# ----------------------------------------------------------------------------

os.makedirs("outputs", exist_ok=True)

from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.data import fetch_data, prep_data_array
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import map_coords

try:
    import cupy
    _cupy_pool = cupy.get_default_memory_pool()
except Exception:
    _cupy_pool = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[setup] device = {device}")
if device.type == "cuda":
    print(f"[setup] gpu    = {torch.cuda.get_device_name(0)}")

# earth2studio class name for each model key
PX_CLASS = {"fcn": "FCN", "pangu": "Pangu6", "sfno": "SFNO", "aurora": "Aurora"}
MODEL_COLOR = {"fcn": "tab:blue", "pangu": "tab:red", "sfno": "tab:green", "aurora": "tab:purple"}

# Models too memory-heavy for a 16 GB GPU at 0.25 deg -> run on CPU (slower, uses RAM).
# Pangu's ONNX transformer wants a full ~16 GB to itself and OOMs sharing the card.
CPU_MODELS = {"pangu"}


def device_for(name):
    return torch.device("cpu") if name in CPU_MODELS else device


def _patch_ort_low_vram():
    """Rebuild earth2studio's ONNX sessions with (a) kSameAsRequested arena (avoids the
    kNextPowerOfTwo over-allocation that OOMs on smaller GPUs) and (b) multi-threaded CPU
    inference (earth2studio hardcodes 1 thread, which makes CPU runs very slow)."""
    try:
        import onnxruntime as ort
        import earth2studio.models.px.pangu as pmod
    except Exception:
        return

    def builder(onnx_file, dev=torch.device("cpu", 0)):
        o = ort.SessionOptions()
        o.enable_cpu_mem_arena = False
        o.enable_mem_pattern = False
        o.enable_mem_reuse = False
        o.intra_op_num_threads = os.cpu_count() or 1   # use all cores on CPU
        o.log_severity_level = 3
        os.stat(onnx_file)
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            providers = [("CUDAExecutionProvider",
                          {"device_id": idx, "arena_extend_strategy": "kSameAsRequested"}),
                         "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(onnx_file, sess_options=o, providers=providers)

    pmod.create_ort_session = builder


_patch_ort_low_vram()


class SensitiveTCTracker(TCTrackerWuDuan):
    """TCTrackerWuDuan with an adjustable 850 hPa vorticity detection threshold.
    The base class calls `_find_centers(...)` with no override, so its hardcoded 1e-4
    default always applies; here we expose it as a constructor arg to tune sensitivity.
    (Track-linking knobs `path_search_distance`/`path_search_window_size` are left at
    their defaults — pass them through **kw if you ever want to tune those too.)"""

    def __init__(self, vort_threshold=1.0e-4, **kw):
        super().__init__(**kw)
        self.vort_threshold = float(vort_threshold)

    def _find_centers(self, lat, lon, vort850, w10m, msl, vort850_threshold=None):
        if vort850_threshold is None:
            vort850_threshold = torch.tensor(self.vort_threshold)
        return super()._find_centers(lat, lon, vort850, w10m, msl, vort850_threshold)


# Saffir-Simpson scale: (min sustained 10 m wind in KNOTS, label, color). Strong -> weak.
SAFFIR = [
    (137, "Cat 5", "#a020f0"),
    (113, "Cat 4", "#ff3030"),
    (96,  "Cat 3", "#ff8c00"),
    (83,  "Cat 2", "#ffc000"),
    (64,  "Cat 1", "#ffe521"),
    (34,  "Trop. Storm", "#34c759"),
    (0,   "Trop. Depression", "#9aa0a6"),
]


def saffir(wind_kt):
    """(label, color) for a max-sustained-wind value in knots (NaN -> depression bucket)."""
    if wind_kt != wind_kt:  # NaN
        return SAFFIR[-1][1], SAFFIR[-1][2]
    for thr, label, color in SAFFIR:
        if wind_kt >= thr:
            return label, color
    return SAFFIR[-1][1], SAFFIR[-1][2]


def est_vmax_kt(mslp_hpa, wind_ms):
    """Wind (kt) used for categorization.
      CATEGORY='wind'     -> raw model 10 m wind. Honest, but a 0.25 deg grid can't resolve a
                             hurricane core, so it reads ~2-3 categories low (even for analyses).
      CATEGORY='estimate' -> ballpark Vmax from the storm's central-pressure DEPTH: inflate the
                             MSLP deficit by INTENSITY_GAIN (to undo the coarse-grid smoothing),
                             then apply the Atkinson-Holliday wind-pressure relation. Its 0.644
                             power keeps shallow lows weak and boosts deep storms more -- an
                             intensity-aware correction, NOT a calibrated intensity forecast."""
    if CATEGORY == "wind":
        return np.asarray(wind_ms) * 1.943844
    deficit = np.maximum(1010.0 - np.asarray(mslp_hpa), 0.0) * INTENSITY_GAIN
    return 6.7 * np.power(deficit, 0.644)


def make_data_source():
    if DATA_SOURCE == "gfs":
        from earth2studio.data import GFS
        print("[setup] data   = GFS (live NOAA analysis)")
        return GFS()
    from earth2studio.data import ARCO
    print("[setup] data   = ARCO (ERA5 reanalysis)")
    return ARCO()


def fit_lat(x, coords, target_lat):
    """Slice/pad the latitude dim so x matches target_lat (handles 720 vs 721 grids)."""
    n_have, n_want = x.shape[-2], len(target_lat)
    if n_have > n_want:
        x = x[..., :n_want, :].contiguous()
    elif n_have < n_want:
        pad = x[..., -1:, :].repeat_interleave(n_want - n_have, dim=-2)
        x = torch.cat([x, pad], dim=-2)
    coords = coords.copy()
    coords["lat"] = target_lat
    return x, coords


def _free_gpu():
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if _cupy_pool is not None:
        _cupy_pool.free_all_blocks()


def run_truth(tracker, tracker_lat, data):
    """Track the analysis 'truth' over the forecast window (past times only)."""
    print(f"\n=== Tracking analysis truth ({DATA_SOURCE.upper()}) ===")
    times = [START_TIME + timedelta(hours=6 * i) for i in range(NSTEPS + 1)]
    tracker.reset_path_buffer()
    output = None
    for step, time in enumerate(times):
        da = data(time, tracker.input_coords()["variable"])
        x, coords = prep_data_array(da, device=device)
        output, _ = tracker(x, coords)
        _free_gpu()
        print(f"  step {step:2d}  {time:%Y-%m-%d %HZ}  tracks {tuple(output.shape)}")
    return output.cpu().numpy()


def run_forecast(name, tracker, tracker_lat, data):
    """Load one model, roll it forward, track each step. Returns tracks ndarray [1,P,T,4]."""
    cls_name = PX_CLASS[name]
    mdev = device_for(name)
    print(f"\n=== Forecast: {name.upper()} ({cls_name}) on {mdev.type.upper()} ===")
    px_cls = getattr(importlib.import_module("earth2studio.models.px"), cls_name)
    prognostic = px_cls.load_model(px_cls.load_default_package()).to(mdev)
    model_lat = prognostic.input_coords()["lat"]

    tracker.reset_path_buffer()
    x, coords = fetch_data(
        source=data,
        time=to_time_array([START_TIME]),
        variable=prognostic.input_coords()["variable"],
        lead_time=prognostic.input_coords()["lead_time"],
        device=mdev,
    )
    x, coords = fit_lat(x, coords, model_lat)

    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if (USE_BF16 and mdev.type == "cuda") else contextlib.nullcontext())
    iterator = prognostic.create_iterator(x, coords)
    output = None
    with amp:
        for step, (xo, co) in enumerate(iterator):
            xo = xo.float().to(device)                      # tracker (cupy) runs on the GPU
            xm, cm = fit_lat(xo, co, tracker_lat)
            xm, cm = map_coords(xm, cm, tracker.input_coords())
            output, _ = tracker(xm, cm)
            output = output[:, 0]
            _free_gpu()
            tag = ("peakVRAM %.1fGB" % (torch.cuda.max_memory_allocated() / 1e9)) if device.type == "cuda" else ""
            print(f"  step {step:2d}  +{6*step:>4}h  tracks {tuple(output.shape)}  {tag}")
            if step == NSTEPS:
                break
    del prognostic
    _free_gpu()
    return output.cpu().numpy()


def wrap_lon(lon):
    return (lon + 180) % 360 - 180


def primary_track(paths, extent):
    """Pick the strongest (lowest-pressure) track whose mean position is inside the map extent.
    Returns (hours, mslp_hPa, wind_kt) full-length series with NaN gaps, or None."""
    if paths is None:
        return None
    lon0, lon1, lat0, lat1 = extent
    best, best_p = None, np.inf
    for p in range(paths.shape[1]):
        lat = paths[0, p, :, 0]
        lon = wrap_lon(paths[0, p, :, 1])
        valid = ~np.isnan(lat) & ~np.isnan(lon)
        if valid.sum() < 3:
            continue
        if not (lon0 <= np.nanmean(lon[valid]) <= lon1 and lat0 <= np.nanmean(lat[valid]) <= lat1):
            continue
        mslp = paths[0, p, :, 2] / 100.0          # Pa -> hPa
        pmin = np.nanmin(mslp[valid]) if valid.any() else np.inf
        if pmin < best_p:
            best_p = pmin
            hours = 6 * np.arange(paths.shape[2])
            wind_kt = paths[0, p, :, 3] * 1.943844  # m/s -> knots
            best = (hours, np.where(valid, mslp, np.nan), np.where(valid, wind_kt, np.nan))
    return best


def plot_tracks(truth, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end_time = START_TIME + timedelta(hours=6 * NSTEPS)
    plt.figure(figsize=(10, 8))
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.add_feature(cfeature.BORDERS, alpha=0.3)
        ax.gridlines(draw_labels=True, alpha=0.4)
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        tf = {"transform": ccrs.PlateCarree()}
    except Exception as e:
        print(f"[plot] cartopy unavailable ({e}); plain axes")
        ax = plt.axes()
        ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1]); ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.grid(alpha=0.3)
        tf = {}

    def draw(paths, color, style, label):
        if paths is None:
            return
        first = True
        for p in range(paths.shape[1]):
            lat = paths[0, p, :, 0]
            lon = wrap_lon(paths[0, p, :, 1])
            m = ~np.isnan(lat) & ~np.isnan(lon)
            if m.sum() <= 2:
                continue
            # the LINE encodes which model/source (thin when category markers are drawn on top)
            ax.plot(lon[m], lat[m], color=color, linestyle=style,
                    linewidth=1.3 if STRENGTH else 1.6, alpha=0.6 if STRENGTH else 1.0,
                    marker=None if STRENGTH else ("x" if style != "-" else None), markersize=3,
                    label=label if first else None, **tf)
            first = False
            if STRENGTH:
                # the MARKERS encode (estimated) Saffir-Simpson category -- see est_vmax_kt/CATEGORY
                vmax = est_vmax_kt(paths[0, p, m, 2] / 100.0, paths[0, p, m, 3])
                ax.scatter(lon[m], lat[m], c=[saffir(w)[1] for w in vmax],
                           s=24, edgecolors="black", linewidths=0.3, zorder=5, **tf)

    draw(truth, "black", "-.", f"{DATA_SOURCE.upper()} analysis (truth)")
    for name, paths in results.items():
        draw(paths, MODEL_COLOR.get(name, "tab:orange"), "-", f"{name.upper()} AI forecast")

    leg1 = ax.legend(loc="upper right", title="Track (line = source)", fontsize=9)
    ax.add_artist(leg1)
    if STRENGTH:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=c,
                          markeredgecolor="k", markersize=7, label=l)
                   for (_, l, c) in reversed(SAFFIR)]
        cat_title = (f"Marker = est. category (pressure ×{INTENSITY_GAIN:g})"
                     if CATEGORY == "estimate" else "Marker = category (model wind)")
        ax.legend(handles=handles, loc="lower left", fontsize=8, title=cat_title)
    title_models = " vs ".join(n.upper() for n in results)
    plt.title(f"AI Tropical Cyclone Tracks — {title_models}\n{START_TIME:%Y-%m-%d %HZ} → {end_time:%Y-%m-%d %HZ}")
    if STRENGTH and CATEGORY == "estimate":
        plt.figtext(0.5, 0.005, "Category = rough ballpark from central-pressure depth "
                    "(resolution-corrected), not a calibrated intensity forecast.",
                    ha="center", fontsize=7, style="italic")
    out = "outputs/cyclone_tracks.jpg"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"[plot] saved {out}")


def plot_intensity(truth, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    if truth is not None:
        t = primary_track(truth, MAP_EXTENT)
        if t:
            series["analysis (truth)"] = (t, "black", "-.")
    for name, paths in results.items():
        t = primary_track(paths, MAP_EXTENT)
        if t:
            series[f"{name.upper()} forecast"] = (t, MODEL_COLOR.get(name, "tab:orange"), "-")

    if not series:
        print("[intensity] no storm track found inside MAP_EXTENT — skipping intensity plot")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    for label, ((hours, mslp, wind), color, style) in series.items():
        ax1.plot(hours, mslp, color=color, linestyle=style, linewidth=2, label=f"{label} (pressure)")
        ax2.plot(hours, wind, color=color, linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.invert_yaxis()  # lower pressure (stronger) at top
    ax1.set_xlabel("forecast lead time (hours)")
    ax1.set_ylabel("min central pressure (hPa)  [solid]")
    ax2.set_ylabel("max 10 m wind (kt)  [dotted]")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)
    plt.title(f"Primary-storm intensity — {START_TIME:%Y-%m-%d %HZ} init\n"
              "(AI models tend to under-deepen; treat as relative guidance)")
    out = "outputs/intensity.png"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"[plot] saved {out}")


def main():
    print(f"[load] TC tracker (vort_threshold={VORT_THRESHOLD:.2e})...")
    tracker = SensitiveTCTracker(vort_threshold=VORT_THRESHOLD).to(device)
    tracker_lat = tracker.input_coords()["lat"]
    data = make_data_source()

    models = COMPARE if COMPARE else [MODEL]
    print(f"[setup] models = {models}  | steps = {NSTEPS} ({6*NSTEPS}h)  | truth = {TRACK_TRUTH}")

    truth = run_truth(tracker, tracker_lat, data) if TRACK_TRUTH else None
    if truth is not None:
        np.save("outputs/truth_paths.npy", truth)

    results = {}
    for name in models:
        results[name] = run_forecast(name, tracker, tracker_lat, data)
        np.save(f"outputs/{name}_paths.npy", results[name])

    plot_tracks(truth, results)
    if INTENSITY:
        plot_intensity(truth, results)
    print("\n[done] outputs/ written")


if __name__ == "__main__":
    main()
