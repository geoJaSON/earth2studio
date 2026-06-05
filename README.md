# Local AI Weather — Tropical Cyclone Forecasting (NVIDIA earth2studio)

Run global AI weather forecast models **locally** on an RTX 5080 (16 GB) and extract
**tropical-cyclone tracks**, comparing an AI forecast against analysis "truth".

Working demo: a 4-day **FourCastNet (FCN)** forecast initialized from **live NOAA GFS**
data, tracking **Typhoon Jangmi** (W. Pacific, late May 2026). See
[`outputs/cyclone_tracks.jpg`](outputs/cyclone_tracks.jpg).

## What's installed

- conda env **`earth2`** (Python 3.12) at `~/miniconda3/envs/earth2`
- **PyTorch 2.11.0 + CUDA 12.8** (required for Blackwell / RTX 5080 `sm_120`)
- **earth2studio** (NVIDIA) with the `cyclone` + `fcn` model stacks
- forecast model: **FCN / FourCastNet** (light, ~1 GB VRAM); tracker: **TCTrackerWuDuan**
- data sources: **GFS** (live, no account) and **ARCO ERA5** (historical, no account)

## Run it

```bash
~/miniconda3/envs/earth2/bin/python cyclone_track.py
# output -> outputs/cyclone_tracks.jpg  (+ ai_paths.pt / truth_paths.pt)
```

Edit the config block at the top of [`cyclone_track.py`](cyclone_track.py):

| Setting | Meaning |
|---|---|
| `MODEL` | `"fcn"` (fits 16 GB), `"sfno"` (needs `makani`), `"aurora"` (needs >16 GB) |
| `DATA_SOURCE` | `"gfs"` (live, recent dates) or `"arco"` (ERA5, any historical date) |
| `START_TIME` | init time; GFS only has 00/06/12/18Z cycles |
| `NSTEPS` | forecast steps (6 h each; 16 = 4 days) |
| `MAP_EXTENT` | `[lon_min, lon_max, lat_min, lat_max]` plot window |

To forecast a **different / current storm**: keep `DATA_SOURCE="gfs"`, set `START_TIME`
to a recent cycle, and set `MAP_EXTENT` over the basin of interest.

## Hard-won setup notes (Blackwell / 16 GB gotchas)

These were the non-obvious fixes — keep them if you rebuild the env:

1. **Blackwell needs CUDA 12.8 wheels.** Install torch from the cu128 index:
   `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.
   Verified: `torch.cuda.get_arch_list()` includes `sm_120` and a GPU matmul runs.
2. **torchvision must match.** The Aurora extra pulled a non-cu128 torchvision →
   `operator torchvision::nms does not exist`. Fix:
   `pip install torchvision==0.26.0 --index-url .../cu128 --force-reinstall --no-deps`.
3. **cupy/cucim must be CUDA-12, not 13.** The `cyclone` extra installs
   `cupy-cuda13x`/`cucim-cu13` → `libcublas.so.13: cannot open shared object file`.
   Fix: `pip uninstall -y cupy-cuda13x cucim-cu13 nvidia-nvimgcodec-cu13` then
   `pip install cupy-cuda12x cucim-cu12`. (cupy auto-finds torch's bundled cu12 libs.)
   The TC tracker hard-depends on cupy+cucim — there is no pure-CPU fallback.
4. **Tracker (cupy) and model (PyTorch) share 16 GB.** `cyclone_track.py` sets
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and frees both allocators each
   step (`torch.cuda.empty_cache()` + `cupy ... free_all_blocks()`).
5. **Lat-grid mismatch (720 vs 721).** Some models (Aurora) use 720 lats, while
   ERA5/GFS and the tracker use 721. `fit_lat()` slices the input / pads the polar
   row of the output. Irrelevant for tropical latitudes.
6. **Data-source variable gaps.** FCN needs relative humidity (`r500`, `r850`),
   which **ARCO ERA5 lacks** but **GFS has** — hence the GFS demo. Always check
   `model.input_coords()["variable"]` vs the source lexicon.

## Model options for this 16 GB card

| Model | Fits 16 GB? | Notes |
|---|---|---|
| **FCN / FourCastNet** | ✅ ~1 GB | used here; needs RH → use GFS (or CDS ERA5) |
| **SFNO / FourCastNet-v2** | ✅ (light) | needs `makani` from git (not on PyPI) |
| **Pangu-Weather** | ✅ | ONNX; strong TC tracks; GPU on Blackwell is uncertain |
| **Aurora** (0.25°) | ❌ OOM (~24 GB+) | best TC model; also bf16 breaks its lon check |
| **GraphCast / GenCast** | ⚠️ | JAX on CUDA-13; heavier |

## Ideas to extend

- Swap in **SFNO** (install `makani` from git) or **Pangu** for a model comparison.
- Forecast the **current** storm: set `START_TIME` to today's latest GFS cycle and
  remove/extend the analysis "truth" loop (no truth exists for the future).
- Save full forecast fields (not just tracks) and plot intensity (min MSLP / max wind).
