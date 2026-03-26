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
# FEATURE 7: LOG TRANSFORMS
# Both rainfall and discharge are heavily right-skewed.
# log1p = log(x + 1), safely handles zeros.
# log_q is the MODEL TARGET — reverse with np.expm1()
# after prediction to recover real discharge values.
# --------------------------------------------------
print("⚙️  Applying log transforms...")
df['log_rainfall'] = np.log1p(df['rainfall_max_mm'])
df['log_q']        = np.log1p(df['q_upstream_mk'])

for lag in range(1, 8):
    df[f'log_rain_lag_{lag}d'] = np.log1p(df[f'rain_lag_{lag}d'])

for window in [3, 7, 14, 30]:
    df[f'log_rain_roll_{window}d'] = np.log1p(df[f'rain_roll_{window}d'])

for lag in range(1, 4):
    df[f'log_q_lag_{lag}d'] = np.log1p(df[f'q_lag_{lag}d'])

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