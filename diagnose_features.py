"""
Diagnostic: compare precompute_rain_features() output against
features_dataset.csv for each rolling convention, find the best match,
then print the exact formula to use in forecast.py.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
feat_path = BASE_DIR / "data/features_dataset.csv"
mast_path = BASE_DIR / "data/master_dataset.csv"

df_feat   = pd.read_csv(feat_path,  parse_dates=['date'])
df_master = pd.read_csv(mast_path,  parse_dates=['date'])
df_feat   = df_feat.sort_values('date').reset_index(drop=True)
df_master = df_master.sort_values('date').reset_index(drop=True)

# Use a stable interior window (avoid edge effects at series boundaries)
DIAG_START = pd.Timestamp('2005-01-01')
DIAG_END   = pd.Timestamp('2015-12-31')

ftest  = df_feat[(df_feat['date'] >= DIAG_START) & (df_feat['date'] <= DIAG_END)].reset_index(drop=True)
mtest  = df_master[(df_master['date'] >= DIAG_START) & (df_master['date'] <= DIAG_END)].reset_index(drop=True)
# align on common dates
mtest  = mtest[mtest['date'].isin(ftest['date'])].reset_index(drop=True)
ftest  = ftest[ftest['date'].isin(mtest['date'])].reset_index(drop=True)

print(f"Diagnostic period: {DIAG_START.date()} – {DIAG_END.date()}  ({len(ftest)} days)\n")

rain = mtest['rainfall_max_mm'].astype(float)

# ── Build all candidate conventions ──────────────────────────────────
candidates = {}

for lag_shift in [1]:                          # lags are unambiguous
    for roll_shift in [0, 1]:                  # 0 = include today, 1 = exclude today
        for std_shift in [0, 1]:
            key = f"roll_shift={roll_shift}_std_shift={std_shift}"
            df_c = pd.DataFrame()

            # Lags
            for lag in range(1, 8):
                df_c[f'log_rain_lag_{lag}d'] = np.log1p(rain.shift(lag).fillna(0).clip(lower=0))

            # Rolling sum
            r_shifted = rain.shift(roll_shift) if roll_shift else rain
            for w in [3, 7, 14, 30]:
                df_c[f'log_rain_roll_{w}d'] = np.log1p(
                    r_shifted.rolling(w, min_periods=1).sum().fillna(0).clip(lower=0)
                )

            # Rolling std
            r_std_shifted = rain.shift(std_shift) if std_shift else rain
            for w in [7, 14]:
                df_c[f'log_rain_rollstd_{w}d'] = np.log1p(
                    r_std_shifted.rolling(w, min_periods=1).std().fillna(0).clip(lower=0)
                )

            # log_rainfall_std  (spatial std column from master if present)
            if 'log_rainfall_std' in ftest.columns:
                std_col = None
                for c in ['rainfall_std_mm', 'rain_std_mm']:
                    if c in mtest.columns:
                        std_col = c; break
                if std_col:
                    df_c['log_rainfall_std'] = np.log1p(mtest[std_col].fillna(0).clip(lower=0))

            candidates[key] = df_c

# ── Per-feature correlation report ───────────────────────────────────
target_cols = [c for c in ftest.columns
               if c.startswith('log_rain') and c in candidates[list(candidates.keys())[0]].columns]

print(f"{'Feature':<30}  " + "  ".join(f"{k[-20:]}" for k in candidates))
print("-" * (32 + 24 * len(candidates)))

col_scores = {k: [] for k in candidates}

for col in sorted(target_cols):
    actual = ftest[col].values.astype(float)
    row = f"{col:<30}"
    for k, df_c in candidates.items():
        pred = df_c[col].values.astype(float)
        corr = float(np.corrcoef(actual, pred)[0, 1]) if np.std(pred) > 0 else 0.0
        row += f"  r={corr:.4f}"
        col_scores[k].append(corr)
    print(row)

print("\n" + "─" * 60)
mean_scores = {k: np.mean(v) for k, v in col_scores.items()}
best_key    = max(mean_scores, key=mean_scores.get)
print("Mean correlation by convention:")
for k, s in sorted(mean_scores.items(), key=lambda x: -x[1]):
    marker = " ← BEST" if k == best_key else ""
    print(f"  {k:<40}  mean_r={s:.4f}{marker}")

# ── Show exact per-feature errors for best convention ────────────────
print(f"\nPer-feature MAE for best convention ({best_key}):")
df_best = candidates[best_key]
for col in sorted(target_cols):
    actual = ftest[col].values.astype(float)
    pred   = df_best[col].values.astype(float)
    mae    = np.mean(np.abs(actual - pred))
    bias   = np.mean(pred - actual)
    print(f"  {col:<35}  MAE={mae:.4f}  bias={bias:+.4f}")

print(f"\n→ Use roll_shift={best_key.split('roll_shift=')[1][0]}, "
      f"std_shift={best_key.split('std_shift=')[1][0]} in precompute_rain_features()")