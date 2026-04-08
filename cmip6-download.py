"""
=================================================================
  CMIP6 RAINFALL DOWNLOAD + BIAS-CORRECTION — Google Colab
=================================================================

Run this in Google Colab (GPU not needed, but free tier works fine).
It downloads NEX-GDDP-CMIP6 precipitation for the Kabini basin,
bias-corrects against your CHIRPS observations, and saves everything
to Google Drive so you can use it locally with forecast.py.

Steps:
  1. Mount Google Drive
  2. Install dependencies
  3. Upload your master_dataset.csv (for bias-correction)
  4. Download CMIP6 from AWS S3
  5. Bias-correct with quantile mapping
  6. Save to Drive

Copy-paste this entire file into a Colab notebook or upload as .py
and run with: !python cmip6_colab.py
"""

# ============================================================
# CELL 1: Setup — run this cell first
# ============================================================
import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

print("📦 Installing dependencies...")
install('xarray')
install('s3fs')
install('h5netcdf')
install('netcdf4')
print("✅ Dependencies installed.\n")

# ============================================================
# CELL 2: Mount Drive + Upload observed data
# ============================================================
import os
from pathlib import Path

# --- Google Drive mount ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = Path('/content/drive/MyDrive/kabini_cmip6')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Drive mounted. Output dir: {DRIVE_DIR}")
    ON_COLAB = True
except ImportError:
    # Not on Colab — use local directory
    DRIVE_DIR = Path('./cmip6_data/processed')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ℹ️  Not on Colab. Output dir: {DRIVE_DIR}")
    ON_COLAB = False

# --- Upload observed data for bias-correction ---
# Option A: Upload from Colab file picker
# Option B: Copy from Drive if you already have it there

MASTER_CSV = None

# Try to find master_dataset.csv in common locations
search_paths = [
    Path('/content/drive/MyDrive/kabini_cmip6/master_dataset.csv'),
    Path('/content/drive/MyDrive/master_dataset.csv'),
    Path('/content/master_dataset.csv'),
    Path('./data/master_dataset.csv'),
    Path('./master_dataset.csv'),
]

for p in search_paths:
    if p.exists():
        MASTER_CSV = p
        print(f"✅ Found observed data: {MASTER_CSV}")
        break

if MASTER_CSV is None and ON_COLAB:
    print("\n📂 Upload your master_dataset.csv for bias-correction:")
    print("   (Contains columns: date, rainfall_max_mm, q_upstream_mk)")
    from google.colab import files
    uploaded = files.upload()
    if uploaded:
        fname = list(uploaded.keys())[0]
        MASTER_CSV = Path(f'/content/{fname}')
        print(f"✅ Uploaded: {MASTER_CSV}")
    else:
        print("⚠️  No file uploaded — will skip bias-correction (use raw GCM)")

if MASTER_CSV is None:
    print("⚠️  No observed data found — bias-correction will be skipped.")


# ============================================================
# CELL 3: Configuration
# ============================================================
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  CMIP6 DOWNLOAD — Kabini Basin")
print("=" * 65)

# Kabini upstream catchment bounding box
# Origin: Wayanad/Brahmagiri hills, Kerala (~11.6°N, 75.8°E)
# Dam: Kabini reservoir (~11.95°N, 76.3°E)
# We capture the full upstream catchment above the dam.
LAT_MIN = 11.60
LAT_MAX = 12.25
LON_MIN = 75.80
LON_MAX = 76.50

print(f"   Basin: {LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°E")

# GCMs — chosen for good monsoon representation
MODELS = {
    'ACCESS-CM2':     'r1i1p1f1',   # Australian, good tropical performance
    'MIROC6':         'r1i1p1f1',   # Japanese, strong Asian monsoon
    'MPI-ESM1-2-HR':  'r1i1p1f1',   # German, high resolution
}

SSPS = ['ssp245', 'ssp585']

# Years
HIST_YEARS   = list(range(2000, 2015))   # overlap with CHIRPS for bias-correction
FUTURE_YEARS = list(range(2025, 2031))   # forecast period

# S3 (public, no auth)
fs = s3fs.S3FileSystem(anon=True)
BUCKET = "nex-gddp-cmip6"
PREFIX = "NEX-GDDP-CMIP6"


# ============================================================
# CELL 4: Download function
# ============================================================
def read_cmip6_year(model, experiment, member, year, var='pr'):
    """Read one year from S3, subset to Kabini, return daily basin-average."""
    fname = f"{var}_day_{model}_{experiment}_{member}_gn_{year}.nc"
    s3path = f"{BUCKET}/{PREFIX}/{model}/{experiment}/{member}/{var}/{fname}"

    try:
        with fs.open(s3path) as f:
            ds = xr.open_dataset(f, engine='h5netcdf')
            ds_sub = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
            pr = ds_sub[var].mean(dim=['lat', 'lon'])
            pr_mm = pr.values * 86400.0  # kg/m²/s → mm/day
            dates = pd.to_datetime(pr.time.values)
            ds.close()
            return pd.DataFrame({'date': dates, 'pr_mm': pr_mm})
    except FileNotFoundError:
        # Try alternate grid label (some models use 'gr' instead of 'gn')
        fname_alt = f"{var}_day_{model}_{experiment}_{member}_gr_{year}.nc"
        s3path_alt = f"{BUCKET}/{PREFIX}/{model}/{experiment}/{member}/{var}/{fname_alt}"
        try:
            with fs.open(s3path_alt) as f:
                ds = xr.open_dataset(f, engine='h5netcdf')
                ds_sub = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
                pr = ds_sub[var].mean(dim=['lat', 'lon'])
                pr_mm = pr.values * 86400.0
                dates = pd.to_datetime(pr.time.values)
                ds.close()
                return pd.DataFrame({'date': dates, 'pr_mm': pr_mm})
        except Exception:
            return None
    except Exception as e:
        print(f"      ⚠️  {model}/{experiment}/{year}: {e}")
        return None


# ============================================================
# CELL 5: Download all data
# ============================================================
print("\n📥 Downloading from AWS S3 (this may take 5-15 minutes)...")
print("   Each file is ~5-15 MB. Total: ~100-300 MB.\n")

model_data = {}

for model, member in MODELS.items():
    print(f"📦 {model}:")

    # Historical (for bias-correction)
    hist_frames = []
    for year in HIST_YEARS:
        sys.stdout.write(f"   hist/{year}...")
        sys.stdout.flush()
        df_yr = read_cmip6_year(model, 'historical', member, year)
        if df_yr is not None:
            hist_frames.append(df_yr)
            print(f" ✓ ({len(df_yr)}d)", end='')
        else:
            print(f" ✗", end='')
        if year % 5 == 4:
            print()  # newline every 5 years

    print()
    if hist_frames:
        model_data[f"{model}_historical"] = pd.concat(hist_frames).sort_values('date').reset_index(drop=True)
        n = len(model_data[f"{model}_historical"])
        print(f"   → historical: {n} days ({n/365:.1f} years)")

    # Future SSPs
    for ssp in SSPS:
        fut_frames = []
        for year in FUTURE_YEARS:
            sys.stdout.write(f"   {ssp}/{year}...")
            sys.stdout.flush()
            df_yr = read_cmip6_year(model, ssp, member, year)
            if df_yr is not None:
                fut_frames.append(df_yr)
                print(f" ✓", end='')
            else:
                print(f" ✗", end='')
        print()

        if fut_frames:
            key = f"{model}_{ssp}"
            model_data[key] = pd.concat(fut_frames).sort_values('date').reset_index(drop=True)
            print(f"   → {ssp}: {len(model_data[key])} days")

    print()

print(f"\n✅ Download complete. {len(model_data)} datasets loaded.")


# ============================================================
# CELL 6: Bias-correction
# ============================================================
print("\n🔧 Bias-correcting with quantile mapping...")

df_obs = None
obs_col = 'rainfall_max_mm'

if MASTER_CSV is not None and MASTER_CSV.exists():
    df_obs = pd.read_csv(MASTER_CSV)
    df_obs['date'] = pd.to_datetime(df_obs['date'])
    # Find rainfall column
    if obs_col not in df_obs.columns:
        rain_cols = [c for c in df_obs.columns if 'rain' in c.lower() and 'max' in c.lower()]
        if rain_cols:
            obs_col = rain_cols[0]
        else:
            rain_cols = [c for c in df_obs.columns if 'rain' in c.lower()]
            obs_col = rain_cols[0] if rain_cols else None

    if obs_col:
        print(f"   Observed: {len(df_obs)} days, column: '{obs_col}'")
    else:
        print("   ⚠️  Cannot find rainfall column in observed data")
        df_obs = None


def quantile_mapping(gcm_hist, obs_df, gcm_future, obs_column, n_quantiles=100):
    """Monthly quantile mapping."""
    corrected = gcm_future.copy()
    corrected['pr_corrected'] = np.nan

    gh = gcm_hist.copy()
    ob = obs_df.copy()
    gh['month'] = gh['date'].dt.month
    ob['month'] = ob['date'].dt.month
    corrected['month'] = corrected['date'].dt.month

    quantiles = np.linspace(0, 1, n_quantiles + 1)

    for month in range(1, 13):
        g_vals = gh.loc[gh['month'] == month, 'pr_mm'].dropna().values
        o_vals = ob.loc[ob['month'] == month, obs_column].dropna().values
        mask = corrected['month'] == month

        if len(g_vals) < 30 or len(o_vals) < 30:
            ratio = o_vals.mean() / max(g_vals.mean(), 0.01) if len(g_vals) > 0 and len(o_vals) > 0 else 1.0
            corrected.loc[mask, 'pr_corrected'] = corrected.loc[mask, 'pr_mm'] * ratio
        else:
            g_q = np.quantile(g_vals, quantiles)
            o_q = np.quantile(o_vals, quantiles)
            corrected.loc[mask, 'pr_corrected'] = np.maximum(
                np.interp(corrected.loc[mask, 'pr_mm'].values, g_q, o_q), 0.0
            )

    corrected.drop(columns=['month'], inplace=True)
    return corrected


corrected_data = {}

for model in MODELS:
    hist_key = f"{model}_historical"
    if hist_key not in model_data:
        continue

    for ssp in SSPS:
        fut_key = f"{model}_{ssp}"
        if fut_key not in model_data:
            continue

        if df_obs is not None and obs_col is not None:
            print(f"   {model} {ssp}...", end=' ')
            corrected = quantile_mapping(model_data[hist_key], df_obs, model_data[fut_key], obs_col)
            corrected_data[fut_key] = corrected
            mean_r = corrected['pr_corrected'].mean()
            print(f"✓ (mean: {mean_r:.2f} mm/day)")
        else:
            model_data[fut_key]['pr_corrected'] = model_data[fut_key]['pr_mm']
            corrected_data[fut_key] = model_data[fut_key]
            print(f"   {model} {ssp}: raw (no bias-correction)")


# ============================================================
# CELL 7: Build ensemble & save
# ============================================================
print("\n💾 Building multi-model ensemble and saving...")

import matplotlib.pyplot as plt

for ssp in SSPS:
    frames = []
    used = []

    for model in MODELS:
        key = f"{model}_{ssp}"
        if key in corrected_data:
            df_m = corrected_data[key][['date', 'pr_corrected']].copy()
            df_m.columns = ['date', f'pr_{model}']
            frames.append(df_m)
            used.append(model)

    if not frames:
        print(f"   ⚠️  No data for {ssp}")
        continue

    ens = frames[0]
    for df_m in frames[1:]:
        ens = ens.merge(df_m, on='date', how='outer')

    pr_cols = [c for c in ens.columns if c.startswith('pr_')]
    ens['rainfall_mm'] = ens[pr_cols].mean(axis=1)
    ens['rainfall_std'] = ens[pr_cols].std(axis=1).fillna(0)
    ens = ens.sort_values('date').reset_index(drop=True)

    # Full ensemble CSV
    ens.to_csv(DRIVE_DIR / f"cmip6_{ssp}_daily_rainfall.csv", index=False)

    # Forecast-ready CSV (matches master_dataset.csv column names)
    fcast = ens[['date', 'rainfall_mm']].copy()
    fcast.columns = ['date', 'rainfall_max_mm']
    fcast['rainfall_std_mm'] = ens['rainfall_std']
    fcast.to_csv(DRIVE_DIR / f"forecast_input_{ssp}.csv", index=False)

    print(f"   ✅ {ssp.upper()}: {len(ens)} days, {len(used)} models → saved to Drive")

    # Annual totals
    ens['year'] = pd.to_datetime(ens['date']).dt.year
    annual = ens.groupby('year')['rainfall_mm'].sum()
    for yr, total in annual.items():
        print(f"      {yr}: {total:.0f} mm/year")


# ============================================================
# CELL 8: Plots
# ============================================================
print("\n📸 Generating plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, ssp in enumerate(SSPS):
    ax = axes[i]
    src = DRIVE_DIR / f"cmip6_{ssp}_daily_rainfall.csv"
    if not src.exists():
        continue

    df_ssp = pd.read_csv(src)
    df_ssp['date'] = pd.to_datetime(df_ssp['date'])
    df_ssp['month'] = df_ssp['date'].dt.month
    cmip_monthly = df_ssp.groupby('month')['rainfall_mm'].mean()

    if df_obs is not None and obs_col:
        obs_tmp = df_obs.copy()
        obs_tmp['month'] = obs_tmp['date'].dt.month
        obs_monthly = obs_tmp.groupby('month')[obs_col].mean()
        ax.bar(obs_monthly.index - 0.2, obs_monthly.values, 0.4,
               color='#3B8BD4', alpha=0.7, label='CHIRPS observed')

    ax.bar(cmip_monthly.index + 0.2, cmip_monthly.values, 0.4,
           color='#E74C3C', alpha=0.7, label=f'CMIP6 {ssp.upper()}')

    ax.set_title(f"Monthly Rainfall — {ssp.upper()}")
    ax.set_xlabel("Month"); ax.set_ylabel("mm/day")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15, axis='y')

plt.suptitle("CMIP6 vs CHIRPS — Kabini Basin", fontsize=13)
plt.tight_layout()
plt.savefig(DRIVE_DIR / "cmip6_vs_observed.png", dpi=150)
plt.show()
print(f"   📸 Saved: cmip6_vs_observed.png")

# Time series
fig, ax = plt.subplots(figsize=(14, 4))
for ssp in SSPS:
    src = DRIVE_DIR / f"cmip6_{ssp}_daily_rainfall.csv"
    if not src.exists():
        continue
    df_ssp = pd.read_csv(src)
    df_ssp['date'] = pd.to_datetime(df_ssp['date'])
    rolling = df_ssp['rainfall_mm'].rolling(30, min_periods=1).mean()
    color = '#8E44AD' if '245' in ssp else '#C0392B'
    ax.plot(df_ssp['date'], rolling, color=color, linewidth=0.8, alpha=0.8,
            label=f'{ssp.upper()} (30-day avg)')

ax.set_title("CMIP6 Projected Rainfall — Kabini (2025–2030)")
ax.set_xlabel("Date"); ax.set_ylabel("mm/day"); ax.legend(); ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(DRIVE_DIR / "cmip6_timeseries.png", dpi=150)
plt.show()
print(f"   📸 Saved: cmip6_timeseries.png")


# ============================================================
# CELL 9: Download to local machine
# ============================================================
print(f"""
{'='*65}
  ✅ DONE!

  Files saved to: {DRIVE_DIR}
    forecast_input_ssp245.csv   ← plug into forecast.py
    forecast_input_ssp585.csv   ← plug into forecast.py
    cmip6_ssp245_daily_rainfall.csv  (full ensemble with individual models)
    cmip6_ssp585_daily_rainfall.csv  (full ensemble with individual models)
    cmip6_vs_observed.png
    cmip6_timeseries.png

  NEXT STEPS:
    1. Download forecast_input_ssp245.csv and forecast_input_ssp585.csv
    2. Place them in: your_project/cmip6_data/processed/
    3. Run: python forecast.py
       → It auto-detects CMIP6 files and adds SSP scenarios

  Basin:  {LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°E
  Models: {', '.join(MODELS.keys())}
  SSPs:   SSP2-4.5 (moderate), SSP5-8.5 (high emissions)
{'='*65}
""")

# Offer download if on Colab
if ON_COLAB:
    print("📥 Downloading files to your local machine...")
    from google.colab import files as colab_files
    for ssp in SSPS:
        fpath = DRIVE_DIR / f"forecast_input_{ssp}.csv"
        if fpath.exists():
            colab_files.download(str(fpath))
            print(f"   ↓ {fpath.name}")