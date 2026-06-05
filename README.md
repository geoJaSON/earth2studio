# Local AI Weather — Tropical Cyclone Forecasting (NVIDIA earth2studio)

Run global AI weather forecast models **locally** on an RTX 5080 (16 GB) and extract
**tropical-cyclone tracks and intensity**, comparing one or more AI forecasts against
analysis "truth".

Working demos:
- a 4-day **FCN** forecast from **live GFS** tracking **Typhoon Jangmi** (W. Pacific);
- a 3-day **FCN vs Pangu** comparison of **Hurricane Milton** in the Gulf of Mexico,
  with an intensity (min-pressure / max-wind) panel.

See [`outputs/cyclone_tracks.jpg`](outputs/cyclone_tracks.jpg) and
[`outputs/intensity.png`](outputs/intensity.png).

## What's installed

- conda env **`earth2`** (Python 3.12) at `~/miniconda3/envs/earth2`
- **PyTorch 2.11.0 + CUDA 12.8** (required for Blackwell / RTX 5080 `sm_120`)
- **earth2studio** (NVIDIA) with the `cyclone`, `fcn`, and Pangu (`onnxruntime-gpu`) stacks
- models: **FCN / FourCastNet** (GPU, ~1 GB) and **Pangu-Weather** (CPU; see below);
  tracker: **TCTrackerWuDuan**
- data sources: **GFS** (live, no account) and **ARCO ERA5** (historical, no account)

## Replicate on another machine

One script rebuilds the whole environment (Linux + NVIDIA GPU, `conda` on PATH):

```bash
bash setup.sh            # creates conda env 'earth2'  (or: bash setup.sh myenv)
```

It pins the exact versions and bakes in the non-obvious fixes (cu128 wheels, the
torchvision re-pin, the cupy/cucim CUDA-13→12 swap — see notes below), then verifies
the GPU and all imports. For a non-Blackwell GPU you can pick a different CUDA build:

```bash
CUDA_IDX=https://download.pytorch.org/whl/cu124 bash setup.sh
```

[`requirements-freeze.txt`](requirements-freeze.txt) is an exact `pip freeze` of the
working env (reference lock / for debugging a version regression). Copy `setup.sh`,
`requirements-freeze.txt`, and `cyclone_track.py` to the new machine.

## Run it

```bash
~/miniconda3/envs/earth2/bin/python cyclone_track.py
# -> outputs/cyclone_tracks.jpg, outputs/intensity.png, outputs/<model>_paths.npy
```

Edit the config block at the top of [`cyclone_track.py`](cyclone_track.py):

| Setting | Meaning |
|---|---|
| `MODEL` | single model: `"fcn"` \| `"pangu"` \| `"sfno"` \| `"aurora"` |
| `COMPARE` | list to overlay several models, e.g. `["fcn", "pangu"]` (`[]` = just `MODEL`) |
| `INTENSITY` | `True` → also plot min-pressure / max-wind of the primary storm |
| `STRENGTH` | `True` → color track markers by Saffir-Simpson category |
| `CATEGORY` | `"estimate"` = ballpark Vmax from central-pressure depth (resolution-corrected) · `"wind"` = raw model 10 m wind (~2–3 cats low) |
| `INTENSITY_GAIN` | (estimate mode) MSLP-deficit gain; `~2.0` lands Milton near Cat 4 |
| `VORT_THRESHOLD` | detection sensitivity (850 hPa vorticity, default `1e-4`); **lower = lock onto weaker lows**, higher = only strong storms |
| `DATA_SOURCE` | `"gfs"` (live, ~2021→now) or `"arco"` (ERA5, any historical date) |
| `START_TIME` | init time; GFS only has 00/06/12/18Z cycles |
| `NSTEPS` | forecast steps (6 h each; 16 = 4 days, 40 = 10 days) |
| `TRACK_TRUTH` | overlay analysis "truth" (only for *past* windows) |
| `MAP_EXTENT` | `[lon_min, lon_max, lat_min, lat_max]` in `[-180,180]` |

Any setting is also overridable via `WX_*` env vars (no file edit needed), e.g. the
Milton comparison:

```bash
WX_COMPARE=fcn,pangu WX_START=2024-10-06T00:00 WX_NSTEPS=12 WX_TRUTH=True \
WX_EXTENT=-100,-78,17,32 ~/miniconda3/envs/earth2/bin/python cyclone_track.py
```

To forecast a **current** storm: `DATA_SOURCE="gfs"`, `START_TIME` = latest cycle,
`TRACK_TRUTH=False` (the future has no analysis yet), `MAP_EXTENT` over the basin.

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
7. **Pangu (ONNX) on a shared GPU.** earth2studio's ORT session uses the default
   `kNextPowerOfTwo` arena, which over-allocates and OOMs even with free VRAM; and it
   hardcodes 1 CPU thread. `_patch_ort_low_vram()` rebuilds the session with
   `kSameAsRequested` + all CPU cores. Even so, Pangu's 0.25° transformer won't share
   16 GB with the framework → it runs on **CPU** (`CPU_MODELS`); the GPU runs the tracker.

## Model options for this 16 GB card

| Model | On 16 GB | Notes |
|---|---|---|
| **FCN / FourCastNet** | ✅ GPU ~1 GB | fast; needs RH → use GFS (or CDS ERA5) |
| **Pangu-Weather** | ⚠️ CPU only | great TC tracks; ONNX OOMs sharing the GPU → CPU (~30 s/step) |
| **SFNO / FourCastNet-v2** | ✅ GPU (light) | needs `makani` from git (not on PyPI) |
| **Aurora** (0.25°) | ❌ ~24 GB+ | best TC model; activation-bound, offload doesn't help |
| **GraphCast / GenCast** | ⚠️ | JAX on CUDA-13; heavier |

## Note on intensity (and the `CATEGORY` estimate)

A 0.25° grid can't resolve a hurricane core, so **both the model *and* the GFS analysis
under-deepen storms**. For Hurricane Milton (real peak ~897 hPa / 160 kt, Cat 5) the
tracker recorded the *GFS analysis* at only **961 hPa / 85 kt** and the *FCN forecast* at
**996 hPa / 31 kt** — so raw winds read ~2–3 categories low *even for the "truth"*.

`CATEGORY="estimate"` (default) gives a ballpark by inflating the central-pressure deficit
(`INTENSITY_GAIN`) and applying the **Atkinson–Holliday** wind–pressure relation; its 0.644
power keeps shallow lows weak and boosts deep storms more. That puts Milton's analysis at
~**Cat 4** and FCN at ~**TS** (still honestly weaker). It's a rough estimate, **not a
calibrated forecast** — for real verification of a past storm, compare to NHC best track.
Use `CATEGORY="wind"` for the raw, honest-but-low model wind.

## Ideas to extend

- Add **SFNO** (install `makani` from git) to the `COMPARE` list for a 3-model panel.
- Save full forecast fields (not just tracks) and map wind/pressure at landfall.
- Drive a **live** daily run from the latest GFS cycle (`TRACK_TRUTH=False`).
