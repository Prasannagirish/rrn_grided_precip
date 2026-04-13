import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 55)
print("  PHASE 3: FEATURE ENGINEERING")
print("=" * 55)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
FEAT_DIR    = BASE_DIR / "feature_outputs"
FEAT_DIR.mkdir(exist_ok=True)

chirps_path = BASE_DIR / "data/chirps_kabini_daily.csv"
master_path = BASE_DIR / "data/master_dataset.csv"
output_path = BASE_DIR / "data/features_dataset.csv"

# --------------------------------------------------
# LOAD
# --------------------------------------------------
df_chirps = pd.read_csv(chirps_path)
df_chirps.columns = df_chirps.columns.str.strip().str.lower()
df_chirps['date'] = pd.to_datetime(df_chirps['date'])

df = pd.read_csv(master_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Master dataset: {df.shape[0]} rows, {df.shape[1]} cols")

# --------------------------------------------------
# BRING IN EXTRA CHIRPS BANDS
# Only rainfall_std_mm carries meaningful signal (r=0.36).
# rainfall_min_mm (~0.00) and rainfall_mean_mm (0.22,
# redundant with rolling means) are dropped.
# --------------------------------------------------
if 'rainfall_std_mm' in df_chirps.columns:
    df = df.merge(df_chirps[['date', 'rainfall_std_mm']], on='date', how='left')
    print("✅ Merged extra CHIRPS band: rainfall_std_mm")
else:
    print("⚠️  rainfall_std_mm not found in CHIRPS — skipping.")

# --------------------------------------------------
# FEATURE 1: LAGGED RAINFALL (1–7 days)
# --------------------------------------------------
print("\n⚙️  Building lag features...")
for lag in range(1, 8):
    df[f'rain_lag_{lag}d'] = df['rainfall_max_mm'].shift(lag)

# --------------------------------------------------
# FEATURE 2: ROLLING RAINFALL SUMS
# Shift by 1 to prevent leakage of today's rain.
# rain_rollmean_Xd dropped — identical signal to roll sum,
# just scaled by 1/window.
# --------------------------------------------------
print("⚙️  Building rolling sum features...")
for window in [3, 7, 14, 30]:
    df[f'rain_roll_{window}d'] = (
        df['rainfall_max_mm']
        .shift(1)
        .rolling(window, min_periods=1)
        .sum()
    )

# --------------------------------------------------
# FEATURE 3: ROLLING STD (rainfall variability / burst intensity)
# --------------------------------------------------
print("⚙️  Building rolling std features...")
for window in [7, 14]:
    df[f'rain_rollstd_{window}d'] = (
        df['rainfall_max_mm']
        .shift(1)
        .rolling(window, min_periods=1)
        .std()
        .fillna(0)
    )

# --------------------------------------------------
# FEATURE 4: LAGGED DISCHARGE (1–3 days)
# q_lag_1d is the strongest single predictor (r=0.94).
# --------------------------------------------------
print("⚙️  Building discharge lag features...")
for lag in range(1, 4):
    df[f'q_lag_{lag}d'] = df['q_upstream_mk'].shift(lag)

# --------------------------------------------------
# FEATURE 5: CALENDAR / SEASONALITY
# sin/cos encoding avoids discontinuity at year boundary.
# Raw month and dayofyear dropped — redundant with sin/cos.
# --------------------------------------------------
print("⚙️  Building calendar features...")
month     = df['date'].dt.month
dayofyear = df['date'].dt.dayofyear

df['month_sin'] = np.sin(2 * np.pi * month     / 12)
df['month_cos'] = np.cos(2 * np.pi * month     / 12)
df['doy_sin']   = np.sin(2 * np.pi * dayofyear / 365)
df['doy_cos']   = np.cos(2 * np.pi * dayofyear / 365)

# --------------------------------------------------
# FEATURE 6: MONSOON FLAG (June–September)
# --------------------------------------------------
df['is_monsoon'] = month.between(6, 9).astype(int)

# --------------------------------------------------
# FEATURE 6b: ANTECEDENT PRECIPITATION INDEX (API)
#
# API is the classic hydrological soil-moisture proxy:
#   API(t) = k × API(t-1) + P(t)
# where k = decay constant (related to recession).
#
# It gives the model temporal "state" awareness — the
# cumulative effect of recent rainfall with exponential
# memory — without needing observed discharge values.
# This is the rainfall-only equivalent of discharge lags.
#
# Multiple decay rates capture different timescales:
#   fast (k=0.85) — ~4-day half-life  (surface runoff)
#   med  (k=0.92) — ~8-day half-life  (interflow)
#   slow (k=0.97) — ~23-day half-life (baseflow)
#
# Shifted by 1 day to prevent leakage.
# --------------------------------------------------
print("⚙️  Building Antecedent Precipitation Index (API) features...")
rain_vals = df['rainfall_max_mm'].values.astype(float)

for k_val, k_name in [(0.85, 'fast'), (0.92, 'med'), (0.97, 'slow')]:
    api_raw = np.zeros(len(df))
    api_raw[0] = rain_vals[0]
    for t in range(1, len(df)):
        api_raw[t] = k_val * api_raw[t-1] + rain_vals[t]
    # Shift by 1 to prevent leakage (use yesterday's API)
    df[f'api_{k_name}'] = pd.Series(api_raw).shift(1).fillna(0).values

# --------------------------------------------------
# FEATURE 6c: DRY SPELL LENGTH
# Number of consecutive days with rainfall < 1mm.
# Captures basin drying — longer dry spell = lower base flow.
# --------------------------------------------------
print("⚙️  Building dry spell length feature...")
dry_spell = np.zeros(len(df))
for t in range(1, len(df)):
    if rain_vals[t] < 1.0:
        dry_spell[t] = dry_spell[t-1] + 1
    else:
        dry_spell[t] = 0
df['dry_spell_days'] = dry_spell

# --------------------------------------------------
# FEATURE 6d: CUMULATIVE MONSOON RAINFALL
# Running sum of rainfall since June 1 each year.
# Resets at start of each monsoon season.
# Captures seasonal soil saturation buildup.
# --------------------------------------------------
print("⚙️  Building cumulative monsoon rainfall feature...")
cum_monsoon = np.zeros(len(df))
for t in range(1, len(df)):
    m = df['date'].iloc[t].month
    d = df['date'].iloc[t].day
    # Reset on June 1
    if m == 6 and d == 1:
        cum_monsoon[t] = rain_vals[t]
    elif 6 <= m <= 11:  # Accumulate Jun–Nov
        cum_monsoon[t] = cum_monsoon[t-1] + rain_vals[t]
    else:
        cum_monsoon[t] = 0.0
df['cum_monsoon_rain'] = pd.Series(cum_monsoon).shift(1).fillna(0).values

# --------------------------------------------------
# FEATURE 7: LOG TRANSFORMS
# Both rainfall and discharge are heavily right-skewed.
# log1p = log(x + 1), safely handles zeros.
# log_q is the MODEL TARGET — reverse with np.expm1()
# after prediction to recover real discharge values.
#
# FIX: Also log-transform rainfall_std_mm and rolling std
#      features so every raw feature has a log counterpart.
#      log_rainfall (today's rain) is kept for reference
#      but must be DROPPED in training to avoid leakage.
# --------------------------------------------------
print("⚙️  Applying log transforms...")
df['log_rainfall'] = np.log1p(df['rainfall_max_mm'])
df['log_q']        = np.log1p(df['q_upstream_mk'])

# FIX: log-transform rainfall_std_mm
if 'rainfall_std_mm' in df.columns:
    df['log_rainfall_std'] = np.log1p(df['rainfall_std_mm'])

for lag in range(1, 8):
    df[f'log_rain_lag_{lag}d'] = np.log1p(df[f'rain_lag_{lag}d'])

for window in [3, 7, 14, 30]:
    df[f'log_rain_roll_{window}d'] = np.log1p(df[f'rain_roll_{window}d'])

# FIX: log-transform rolling std features (were previously
#      dropped in training with no log counterpart, losing
#      the rainfall variability signal entirely)
for window in [7, 14]:
    df[f'log_rain_rollstd_{window}d'] = np.log1p(df[f'rain_rollstd_{window}d'])

for lag in range(1, 4):
    df[f'log_q_lag_{lag}d'] = np.log1p(df[f'q_lag_{lag}d'])

# Log-transform API features
for k_name in ['fast', 'med', 'slow']:
    df[f'log_api_{k_name}'] = np.log1p(df[f'api_{k_name}'])

# Log-transform dry spell and cumulative monsoon rain
df['log_dry_spell'] = np.log1p(df['dry_spell_days'])
df['log_cum_monsoon'] = np.log1p(df['cum_monsoon_rain'])

# --------------------------------------------------
# FEATURE 8: INTERACTION FEATURES
#
# These help tree models find the key nonlinear thresholds:
#   - Wet soil + heavy rain → big runoff (API × rain)
#   - Monsoon context changes the rainfall-runoff relationship
#   - Saturated soil + continued rain → extreme peaks
# --------------------------------------------------
print("⚙️  Building interaction features...")

# API × recent rainfall — "wet soil + rain event" signal
df['api_slow_x_rain7d'] = df['log_api_slow'] * df['log_rain_roll_7d']
df['api_med_x_rain3d']  = df['log_api_med']  * df['log_rain_roll_3d']

# API × monsoon flag — monsoon API behaves differently from dry-season API
df['api_slow_x_monsoon'] = df['log_api_slow'] * df['is_monsoon']

# Cumulative monsoon × recent rain — "saturated basin + more rain"
df['cum_monsoon_x_rain7d'] = df['log_cum_monsoon'] * df['log_rain_roll_7d']

# --------------------------------------------------
# DROP WARM-UP ROWS
# --------------------------------------------------
warmup = 31
df = df.iloc[warmup:].reset_index(drop=True)
print(f"\n✅ Dropped {warmup} warm-up rows. Remaining: {len(df)}")

# --------------------------------------------------
# MISSING VALUE CHECK
# --------------------------------------------------
missing = df.isna().sum()
missing = missing[missing > 0]
if missing.empty:
    print("✅ No missing values in feature set.")
else:
    print("\n⚠️  Missing values detected:")
    print(missing)

# --------------------------------------------------
# CORRELATION CHECK (log-space features → log_q)
# --------------------------------------------------
target = 'log_q'
feat_cols = [c for c in df.columns if c not in ('date', 'q_upstream_mk', target)]
feat_corr = (
    df[feat_cols + [target]]
    .corr()[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print(f"\n🔗 Feature correlations with '{target}' (log-space):")
print(feat_corr.round(4).to_string())
feat_corr.to_csv(FEAT_DIR / "feature_target_correlation.csv", header=True)

# --------------------------------------------------
# PLOT: Top-20 feature correlations
# --------------------------------------------------
top20 = feat_corr.abs().nlargest(20).index
plt.figure(figsize=(8, 6))
feat_corr[top20].sort_values().plot(kind='barh')
plt.title(f"Top 20 Feature Correlations → {target}")
plt.xlabel("Pearson r")
plt.tight_layout()
plt.savefig(FEAT_DIR / "feature_correlations.png")
plt.close()
print(f"📸 Saved: {FEAT_DIR / 'feature_correlations.png'}")

# --------------------------------------------------
# PLOT: Log-transformed distributions
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df['log_rainfall'].dropna(), bins=50)
axes[0].set_title("log(1 + rainfall_max_mm)")
axes[0].set_xlabel("log-rainfall")
axes[0].set_ylabel("Frequency")

axes[1].hist(df['log_q'].dropna(), bins=50)
axes[1].set_title("log(1 + q_upstream_mk)")
axes[1].set_xlabel("log-discharge")
axes[1].set_ylabel("Frequency")

plt.suptitle("Log-Transformed Distributions", fontsize=13)
plt.tight_layout()
plt.savefig(FEAT_DIR / "log_distributions.png")
plt.close()
print(f"📸 Saved: {FEAT_DIR / 'log_distributions.png'}")

# --------------------------------------------------
# SAVE
# --------------------------------------------------
df.to_csv(output_path, index=False)
print(f"\n✅ Feature dataset saved → {output_path}")
print(f"   Shape  : {df.shape}")
print(f"   Columns: {list(df.columns)}")
print("=" * 55)