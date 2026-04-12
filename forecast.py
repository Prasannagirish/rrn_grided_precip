import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  PHASE 5: CMIP6 SCENARIO FORECASTING (2025–2030)")
print("=" * 60)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
MODEL_DIR    = BASE_DIR / "model_outputs"
FORECAST_DIR = BASE_DIR / "forecast_outputs"
FORECAST_DIR.mkdir(exist_ok=True)

features_path = BASE_DIR / "data/features_dataset.csv"
master_path   = BASE_DIR / "data/master_dataset.csv"
CMIP6_DIR     = BASE_DIR / "cmip6_data" / "processed"

# --------------------------------------------------
# LOAD SAVED MODEL ASSETS
# --------------------------------------------------
print("\n📦 Loading saved model assets...")

model_pkl = MODEL_DIR / "ridge_forecast_model.pkl"
meta_json = MODEL_DIR / "forecast_meta.json"
cols_json = MODEL_DIR / "feature_cols.json"

if not (model_pkl.exists() and meta_json.exists() and cols_json.exists()):
    print("❌  Saved assets not found — run train_model.py first.")
    raise SystemExit(1)

ridge_final       = joblib.load(model_pkl)
forecast_meta     = json.load(open(meta_json))
feature_cols      = json.load(open(cols_json))
correction_factor = forecast_meta["correction_factor"]
p90_threshold     = forecast_meta["p90_threshold_m3s"]
best_nse          = forecast_meta.get("best_nse", "?")

print(f"   ✅ ridge_forecast_model.pkl  (NSE={best_nse} on test set)")
print(f"   ✅ {len(feature_cols)} features")
print(f"   ✅ Peak correction ×{correction_factor:.3f} above {p90_threshold:.0f} m³/s")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df_feat = pd.read_csv(features_path)
df_feat['date'] = pd.to_datetime(df_feat['date'])
df_feat = df_feat.sort_values('date').reset_index(drop=True)

df_master = pd.read_csv(master_path)
df_master['date'] = pd.to_datetime(df_master['date'])
df_master = df_master.sort_values('date').reset_index(drop=True)

print(f"\nHistorical features: {len(df_feat)} rows "
      f"({df_feat['date'].min().date()} → {df_feat['date'].max().date()})")

# --------------------------------------------------
# CLIMATOLOGICAL MONTHLY DISCHARGE
# Used to reset q_history at January 1 of each forecast
# year, preventing multi-year cascade error accumulation.
# --------------------------------------------------
df_master['month'] = df_master['date'].dt.month
df_master['doy']   = df_master['date'].dt.dayofyear
clim_monthly_q = df_master.groupby('month')['q_upstream_mk'].median()
# January (month=1): deepest dry-season baseline — used for annual reset
ANNUAL_RESET_Q = float(clim_monthly_q[1])
print(f"\n   Climatological January Q (reset value): {ANNUAL_RESET_Q:.1f} m³/s")


# ══════════════════════════════════════════════════════════
#  ROLLING CONVENTION DETECTION
#  Confirmed by diagnose_features.py: roll_shift=1, std_shift=1
#  gives mean_r=0.9989, which is the correct match to training.
#  We run the detection here anyway so the script is self-contained.
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 1: Detecting rolling-window convention")
print("=" * 60)

RAIN_STD_COL = next((c for c in ['rainfall_std_mm', 'rain_std_mm']
                     if c in df_master.columns), None)

DIAG_START = pd.Timestamp('2005-01-01')
DIAG_END   = pd.Timestamp('2011-12-31')
fdiag = df_feat[(df_feat['date'] >= DIAG_START) & (df_feat['date'] <= DIAG_END)].reset_index(drop=True)
mdiag = df_master[(df_master['date'] >= DIAG_START) & (df_master['date'] <= DIAG_END)].reset_index(drop=True)
mdiag = mdiag[mdiag['date'].isin(fdiag['date'])].reset_index(drop=True)
fdiag = fdiag[fdiag['date'].isin(mdiag['date'])].reset_index(drop=True)

check_cols = [c for c in fdiag.columns if c.startswith('log_rain_roll') and c in feature_cols]
diag_rain  = mdiag['rainfall_max_mm'].astype(float)

best_rs, best_ss, best_r = 1, 1, -1.0

for rs in [0, 1]:
    for ss in [0, 1]:
        r_roll = diag_rain.shift(rs) if rs else diag_rain
        r_std  = diag_rain.shift(ss) if ss else diag_rain
        corrs  = []
        for col in check_cols:
            w = int(col.split('_roll_')[1].replace('d', ''))
            pred = np.log1p(r_roll.rolling(w, min_periods=1).sum().fillna(0).clip(lower=0))
            actual = fdiag[col].values.astype(float)
            if np.std(pred) > 0:
                corrs.append(float(np.corrcoef(actual, pred.values)[0, 1]))
        mean_r = np.mean(corrs) if corrs else 0.0
        marker = ""
        if mean_r > best_r:
            best_r, best_rs, best_ss = mean_r, rs, ss
            marker = " ← best"
        print(f"   roll_shift={rs}  std_shift={ss}  mean_r={mean_r:.4f}{marker}")

print(f"\n   ✅ roll_shift={best_rs}, std_shift={best_ss}  (mean_r={best_r:.4f})")


# ══════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def precompute_rain_features(dates_series, rain_series, rain_std_series=None):
    rain  = pd.Series(rain_series.values.astype(float))
    dates = pd.to_datetime(dates_series.values)

    df_c  = pd.DataFrame()
    for lag in range(1, 8):
        df_c[f'log_rain_lag_{lag}d'] = np.log1p(rain.shift(lag).fillna(0).clip(lower=0))

    r_roll = rain.shift(best_rs) if best_rs else rain
    for w in [3, 7, 14, 30]:
        df_c[f'log_rain_roll_{w}d'] = np.log1p(
            r_roll.rolling(w, min_periods=1).sum().fillna(0).clip(lower=0))

    r_std = rain.shift(best_ss) if best_ss else rain
    for w in [7, 14]:
        df_c[f'log_rain_rollstd_{w}d'] = np.log1p(
            r_std.rolling(w, min_periods=1).std().fillna(0).clip(lower=0))

    if rain_std_series is not None:
        df_c['log_rainfall_std'] = np.log1p(
            pd.Series(rain_std_series.values.astype(float)).fillna(0).clip(lower=0).values)
    else:
        df_c['log_rainfall_std'] = 0.0

    df_c['month_sin']  = np.sin(2 * np.pi * pd.Series(dates).dt.month     / 12).values
    df_c['month_cos']  = np.cos(2 * np.pi * pd.Series(dates).dt.month     / 12).values
    df_c['doy_sin']    = np.sin(2 * np.pi * pd.Series(dates).dt.dayofyear / 365).values
    df_c['doy_cos']    = np.cos(2 * np.pi * pd.Series(dates).dt.dayofyear / 365).values
    df_c['is_monsoon'] = (pd.Series(dates).dt.month.between(6, 9)).astype(int).values

    return df_c.reset_index(drop=True)


def build_feat_with_seed(seed_df, target_df):
    s_rain = seed_df['rainfall_max_mm'].reset_index(drop=True) \
             if 'rainfall_max_mm' in seed_df.columns else pd.Series([0.0]*len(seed_df))
    t_rain = target_df['rainfall_max_mm'].reset_index(drop=True)
    full_rain  = pd.concat([s_rain, t_rain], ignore_index=True)
    full_dates = pd.concat([seed_df['date'].reset_index(drop=True),
                            target_df['date'].reset_index(drop=True)], ignore_index=True)
    s_std = seed_df[RAIN_STD_COL].reset_index(drop=True) \
            if RAIN_STD_COL and RAIN_STD_COL in seed_df.columns else None
    t_std = target_df[RAIN_STD_COL].reset_index(drop=True) \
            if RAIN_STD_COL and RAIN_STD_COL in target_df.columns else None
    full_std = pd.concat([s_std, t_std], ignore_index=True) \
               if (s_std is not None and t_std is not None) else None

    feat_full = precompute_rain_features(full_dates, full_rain, full_std)
    return feat_full.iloc[len(seed_df):].reset_index(drop=True)


# ══════════════════════════════════════════════════════════
#  AUTOREGRESSIVE DISCHARGE LOOP — WITH TWO KEY FIXES
#
#  FIX 1 — Annual reset:
#    On January 1 of each year, reset q_history to the
#    climatological January discharge. This prevents cascade
#    errors from accumulating across multiple years.
#    Without this, any prediction error from the previous
#    year's dry season corrupts the entire next year.
#
#  FIX 2 — Correct correction-factor feedback:
#    The peak-correction (×1.097) adjusts the OUTPUT for
#    users, but must NOT feed back into q_history.  Storing
#    the corrected (inflated) value in q_history would make
#    log_q_lag_1d on the next day artificially high, causing
#    a runaway upward bias during the monsoon.
#    We store pred_q_raw (uncorrected) in q_history.
# ══════════════════════════════════════════════════════════

def ar_discharge_loop(rain_feat_df, model, feature_cols,
                      seed_q_history, correction_factor, p90_threshold,
                      dates=None, annual_reset_q=None):
    q_history = list(seed_q_history)
    predicted = np.zeros(len(rain_feat_df))

    for i in range(len(rain_feat_df)):

        # FIX 1: Annual reset at January 1
        if annual_reset_q is not None and dates is not None:
            d = pd.Timestamp(dates[i])
            if d.month == 1 and d.day == 1:
                # Reset the most recent 30 values to a realistic
                # dry-season recession starting from annual_reset_q.
                # Each step back gets a small multiplier to approximate
                # a gentle pre-January recession.
                for k in range(min(30, len(q_history))):
                    q_history[-(k+1)] = annual_reset_q * (1.0 + k * 0.015)

        row = rain_feat_df.iloc[i].to_dict()

        # Update discharge lags from running q_history
        for lag in range(1, 4):
            val = q_history[-lag] if len(q_history) >= lag else 0.0
            row[f'log_q_lag_{lag}d'] = np.log1p(max(val, 0.0))

        X_row     = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])
        pred_q_raw = max(np.expm1(float(model.predict(X_row)[0])), 0.0)

        # FIX 2: apply correction to OUTPUT only; store raw in feedback
        pred_q_out = pred_q_raw * correction_factor \
                     if pred_q_raw > p90_threshold else pred_q_raw

        predicted[i] = pred_q_out
        q_history.append(pred_q_raw)   # ← raw, NOT corrected

    return predicted


# ══════════════════════════════════════════════════════════
#  STEP 2: HINDCAST VALIDATION (2015–2020)
#
#  We run three AR configurations to isolate error sources:
#  (a) Direct Ridge on pre-computed features  — benchmark
#  (b) AR + perfect features, NO reset, OLD buggy feedback
#      (reproduces the NSE=0.14 failure we already observed)
#  (c) AR + perfect features, annual reset + fixed feedback
#      (tests whether the fixes work)
#  (d) AR + reconstructed features, annual reset + fixed
#      (full pipeline as it will run for CMIP6)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: Hindcast validation (2015–2020)")
print("=" * 60)

TEST_START  = pd.Timestamp('2015-01-01')
df_test     = df_feat[df_feat['date'] >= TEST_START].copy().reset_index(drop=True)
y_obs       = np.expm1(df_test['log_q'].values)
test_dates  = pd.DatetimeIndex(df_test['date'])

# Direct Ridge baseline (upper bound)
direct_pred = np.expm1(ridge_final.predict(df_test[feature_cols]))
direct_corr = np.where(direct_pred > p90_threshold, direct_pred * correction_factor, direct_pred)
nse_direct  = 1 - np.sum((y_obs - direct_corr)**2) / np.sum((y_obs - y_obs.mean())**2)
rmse_direct = np.sqrt(mean_squared_error(y_obs, direct_corr))
print(f"\n   (a) Direct Ridge:              NSE={nse_direct:.4f}  RMSE={rmse_direct:.2f}  [upper bound]")

seed_hc   = df_master[df_master['date'] < TEST_START].tail(30)
seed_q_hc = list(seed_hc['q_upstream_mk'].fillna(ANNUAL_RESET_Q).values.astype(float))

test_master = df_master[(df_master['date'] >= TEST_START) &
                        (df_master['date'].isin(df_test['date']))].reset_index(drop=True)

# Perfect rainfall features from features_dataset.csv
perfect_feat_df = df_test[feature_cols].copy()
for lag in range(1, 4):
    perfect_feat_df[f'log_q_lag_{lag}d'] = 0.0

# (b) Buggy AR — no reset, correction fed back (reproduces old failure)
def ar_buggy(rain_feat_df, model, feature_cols, seed_q, cf, p90):
    q_h = list(seed_q)
    out = np.zeros(len(rain_feat_df))
    for i in range(len(rain_feat_df)):
        row = rain_feat_df.iloc[i].to_dict()
        for lag in range(1, 4):
            row[f'log_q_lag_{lag}d'] = np.log1p(max(q_h[-lag] if len(q_h)>=lag else 0.0, 0.0))
        X  = pd.DataFrame([{c: row.get(c, 0.0) for c in feature_cols}])
        pq = max(np.expm1(float(model.predict(X)[0])), 0.0)
        if pq > p90: pq *= cf   # BUG: corrected value fed back
        out[i] = pq
        q_h.append(pq)
    return out

ar_b = ar_buggy(perfect_feat_df, ridge_final, feature_cols, seed_q_hc, correction_factor, p90_threshold)
nse_b = 1 - np.sum((y_obs-ar_b)**2)/np.sum((y_obs-y_obs.mean())**2)
print(f"   (b) AR + perfect feat, OLD bugs:  NSE={nse_b:.4f}  [expected ~0.14]")

# (c) Fixed AR + perfect features — annual reset + correct feedback
ar_c = ar_discharge_loop(
    perfect_feat_df, ridge_final, feature_cols,
    seed_q_hc, correction_factor, p90_threshold,
    dates=test_dates, annual_reset_q=ANNUAL_RESET_Q
)
nse_c  = 1 - np.sum((y_obs-ar_c)**2)/np.sum((y_obs-y_obs.mean())**2)
rmse_c = np.sqrt(mean_squared_error(y_obs, ar_c))
print(f"   (c) AR + perfect feat, FIXED:     NSE={nse_c:.4f}  RMSE={rmse_c:.2f}")

# (d) Fixed AR + reconstructed features — what CMIP6 will actually use
rain_feat_hc = build_feat_with_seed(seed_hc, test_master)
ar_d = ar_discharge_loop(
    rain_feat_hc, ridge_final, feature_cols,
    seed_q_hc, correction_factor, p90_threshold,
    dates=test_dates, annual_reset_q=ANNUAL_RESET_Q
)
nse_d  = 1 - np.sum((y_obs-ar_d)**2)/np.sum((y_obs-y_obs.mean())**2)
rmse_d = np.sqrt(mean_squared_error(y_obs, ar_d))
corr_d = float(np.corrcoef(ar_d, direct_corr)[0,1])
print(f"   (d) AR + reconstructed feat, FIXED: NSE={nse_d:.4f}  RMSE={rmse_d:.2f}  r={corr_d:.4f}")

print(f"\n   Gap analysis:")
print(f"   (b)→(c) fix: NSE {nse_b:.4f} → {nse_c:.4f}  [{nse_c-nse_b:+.4f}]  (annual reset + feedback fix)")
print(f"   (c)→(d) gap: NSE {nse_c:.4f} → {nse_d:.4f}  [{nse_d-nse_c:+.4f}]  (rainfall reconstruction)")
print(f"   (d)→(a) gap: NSE {nse_d:.4f} → {nse_direct:.4f}  [{nse_direct-nse_d:+.4f}]  (AR vs direct prediction)")

THRESHOLD = 0.70
if nse_d >= THRESHOLD:
    print(f"\n   ✅ Hindcast NSE={nse_d:.4f} ≥ {THRESHOLD} — pipeline validated for CMIP6 forecasting.")
else:
    print(f"\n   ⚠️  Hindcast NSE={nse_d:.4f} < {THRESHOLD}")
    print(f"      Forecasts will be generated but interpret as directional projections only.")

# Hindcast validation plot
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
td = df_test['date'].values
axes[0].plot(td, y_obs,       label='Observed',
             color='#1a1a1a', lw=0.8, alpha=0.8)
axes[0].plot(td, direct_corr, label=f'Direct Ridge  NSE={nse_direct:.3f}',
             color='#F39C12', lw=1.2, alpha=0.9)
axes[0].plot(td, ar_d,        label=f'AR Fixed (reconstructed)  NSE={nse_d:.3f}',
             color='#2E86C1', lw=0.9, alpha=0.8, linestyle='--')
axes[0].plot(td, ar_b,        label=f'AR Buggy (no reset)  NSE={nse_b:.3f}',
             color='#E74C3C', lw=0.7, alpha=0.6, linestyle=':')
axes[0].set_ylabel("Discharge (m³/s)")
axes[0].set_title("Hindcast Validation (2015–2020) — Bug Analysis")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.15)

axes[1].plot(td, ar_d - y_obs, color='#2E86C1', lw=0.6, alpha=0.7)
axes[1].axhline(0, color='gray', lw=0.8, linestyle='--')
axes[1].set_ylabel("AR Fixed − Obs (m³/s)")
axes[1].set_title(f"AR Fixed Residuals  (mean={np.mean(ar_d-y_obs):.1f}  std={np.std(ar_d-y_obs):.1f})")
axes[1].grid(True, alpha=0.15)

# Show annual NSE to verify reset is working
annual_nse = []
years = sorted(set(df_test['date'].dt.year))
for yr in years:
    mask = (df_test['date'].dt.year == yr).values
    if mask.sum() < 30: continue
    obs_yr, pred_yr = y_obs[mask], ar_d[mask]
    nse_yr = 1 - np.sum((obs_yr-pred_yr)**2)/np.sum((obs_yr-obs_yr.mean())**2)
    annual_nse.append((yr, nse_yr))

yr_vals, nse_vals = zip(*annual_nse)
axes[2].bar(yr_vals, nse_vals, color=['#2ECC71' if n >= 0.6 else '#E74C3C' for n in nse_vals],
            alpha=0.8, edgecolor='white')
axes[2].axhline(0.6, color='gray', linestyle='--', lw=0.8)
axes[2].set_ylabel("NSE"); axes[2].set_xlabel("Year")
axes[2].set_title("Per-Year NSE — AR Fixed Pipeline"); axes[2].grid(True, alpha=0.15, axis='y')

plt.tight_layout()
plt.savefig(FORECAST_DIR / "hindcast_validation.png", dpi=150)
plt.close()
print(f"\n   📸 hindcast_validation.png")

pd.DataFrame({
    'date':        df_test['date'].values,
    'observed':    y_obs,
    'direct_ridge': direct_corr,
    'ar_fixed':    ar_d,
    'ar_buggy':    ar_b,
}).to_csv(FORECAST_DIR / "hindcast_validation.csv", index=False)
print(f"   💾 hindcast_validation.csv")


# ══════════════════════════════════════════════════════════
#  STEP 3: CMIP6 SCENARIO FORECASTS
# ══════════════════════════════════════════════════════════
SSPS       = ['ssp245', 'ssp585']
SSP_LABELS = {'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}
SSP_COLORS = {'ssp245': '#8E44AD',  'ssp585': '#C0392B'}

cmip6_available = {}
for ssp in SSPS:
    fpath = CMIP6_DIR / f"forecast_input_{ssp}.csv"
    if fpath.exists():
        cmip6_available[ssp] = fpath
        print(f"   ✅ {fpath.name}")
    else:
        print(f"   ❌ Missing: {fpath.name}")

if not cmip6_available:
    raise SystemExit(1)

print("\n" + "=" * 60)
print(f"  STEP 3: CMIP6 forecasts  [hindcast NSE={nse_d:.3f}]")
print("=" * 60)

seed_fc   = df_master.tail(30)
seed_q_fc = list(seed_fc['q_upstream_mk'].fillna(ANNUAL_RESET_Q).values.astype(float))

scenario_results = {}

for ssp, fpath in cmip6_available.items():
    label = SSP_LABELS[ssp]
    print(f"\n   🌍 {label}...")

    df_cmip = pd.read_csv(fpath)
    df_cmip['date'] = pd.to_datetime(df_cmip['date'])
    df_cmip = df_cmip.sort_values('date').reset_index(drop=True)
    cmip_dates = pd.DatetimeIndex(df_cmip['date'])
    print(f"      {cmip_dates[0].date()} → {cmip_dates[-1].date()} ({len(cmip_dates)} days)")

    rain_feat_cmip = build_feat_with_seed(seed_fc, df_cmip)
    q_pred = ar_discharge_loop(
        rain_feat_cmip, ridge_final, feature_cols,
        seed_q_fc, correction_factor, p90_threshold,
        dates=cmip_dates, annual_reset_q=ANNUAL_RESET_Q
    )

    future_df = pd.DataFrame({'date': cmip_dates, 'Q': q_pred,
                              'P': df_cmip['rainfall_max_mm'].values})
    future_df['year'] = future_df['date'].dt.year
    annual = future_df.groupby('year').agg(
        mean_Q=('Q','mean'), max_Q=('Q','max'), total_P=('P','sum'))

    scenario_results[label] = {
        'ssp': ssp, 'dates': cmip_dates,
        'rainfall': df_cmip['rainfall_max_mm'].values,
        'discharge': q_pred, 'annual': annual,
    }

    print(f"      {'Year':>6}  {'Mean Q':>10}  {'Peak Q':>10}  {'Rain mm':>10}")
    for yr, r in annual.iterrows():
        print(f"      {yr:>6}  {r['mean_Q']:>10.1f}  {r['max_Q']:>10.1f}  {r['total_P']:>10.0f}")


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
print("\n📸 Generating plots...")
reliability = f"Hindcast NSE={nse_d:.3f} | Direct NSE={nse_direct:.3f}"

fig, ax = plt.subplots(figsize=(16, 6))
ht = df_master.tail(1095)
ax.plot(ht['date'], ht['q_upstream_mk'],
        color='#1a1a1a', lw=0.7, alpha=0.5, label='Historical observed')
for label, data in scenario_results.items():
    ax.plot(data['dates'], data['discharge'],
            color=SSP_COLORS[data['ssp']], lw=0.8, alpha=0.85, label=label)
fs = min(data['dates'][0] for data in scenario_results.values())
ax.axvline(fs, color='gray', linestyle='--', lw=0.8, alpha=0.5)
ax.text(fs, ax.get_ylim()[1]*0.92, '  Forecast →', fontsize=9, color='gray')
ax.set_title(f"Kabini Discharge — CMIP6 Forecasts  [{reliability}]")
ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_forecast_full.png", dpi=150)
plt.close()
print(f"   📸 cmip6_forecast_full.png")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2.5]})
z0, z1 = pd.Timestamp('2026-04-01'), pd.Timestamp('2026-11-30')
fl = list(scenario_results.keys())[0]; fd = scenario_results[fl]
zm = (fd['dates'] >= z0) & (fd['dates'] <= z1)
axes[0].bar(fd['dates'][zm], fd['rainfall'][zm], color='#3B8BD4', alpha=0.6, width=1)
axes[0].invert_yaxis(); axes[0].set_ylabel("Rainfall (mm)")
axes[0].set_title("2026 Monsoon Zoom"); axes[0].set_xlim(z0, z1)
for label, data in scenario_results.items():
    zm2 = (data['dates'] >= z0) & (data['dates'] <= z1)
    axes[1].plot(data['dates'][zm2], data['discharge'][zm2],
                 color=SSP_COLORS[data['ssp']], lw=1.3, alpha=0.85, label=label)
axes[1].set_xlabel("Date"); axes[1].set_ylabel("Discharge (m³/s)")
axes[1].legend(fontsize=10); axes[1].set_xlim(z0, z1); axes[1].grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_monsoon_2026.png", dpi=150)
plt.close()
print(f"   📸 cmip6_monsoon_2026.png")

fig, ax = plt.subplots(figsize=(12, 5))
ham = df_master.groupby(df_master['date'].dt.year)['q_upstream_mk'].max()
ax.bar(ham.index, ham.values, color='#BDC3C7', alpha=0.6, edgecolor='white', lw=0.5, label='Historical')
bw = 0.35
for i, (label, data) in enumerate(scenario_results.items()):
    yrs = sorted(data['annual'].index)
    ax.bar([y+(i-0.5)*bw for y in yrs],
           [data['annual'].loc[y,'max_Q'] for y in yrs],
           bw, color=SSP_COLORS[data['ssp']], alpha=0.85, edgecolor='white', lw=0.5, label=label)
ax.set_title("Annual Peak Discharge — Historical vs CMIP6")
ax.set_xlabel("Year"); ax.set_ylabel("Peak Discharge (m³/s)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.15, axis='y')
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_annual_peaks.png", dpi=150)
plt.close()
print(f"   📸 cmip6_annual_peaks.png")

if len(scenario_results) >= 2:
    fig, ax = plt.subplots(figsize=(14, 5))
    lbls  = list(scenario_results.keys())
    cd    = scenario_results[lbls[0]]['dates']
    all_q = np.column_stack([scenario_results[l]['discharge'] for l in lbls])
    ax.fill_between(cd,
                    pd.Series(all_q.min(axis=1)).rolling(30, min_periods=1).mean(),
                    pd.Series(all_q.max(axis=1)).rolling(30, min_periods=1).mean(),
                    alpha=0.25, color='#8E44AD', label='SSP range (30-day smooth)')
    ax.plot(cd, pd.Series(all_q.mean(axis=1)).rolling(30, min_periods=1).mean(),
            color='#8E44AD', lw=1.5, label='Ensemble mean')
    ax.set_title("CMIP6 Forecast Envelope — SSP2-4.5 to SSP5-8.5")
    ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(FORECAST_DIR / "cmip6_forecast_envelope.png", dpi=150)
    plt.close()
    print(f"   📸 cmip6_forecast_envelope.png")

fig, ax = plt.subplots(figsize=(10, 5))
bw = 0.35
for i, (label, data) in enumerate(scenario_results.items()):
    yrs = sorted(data['annual'].index)
    ax.bar([y+(i-0.5)*bw for y in yrs],
           [data['annual'].loc[y,'mean_Q'] for y in yrs],
           bw, color=SSP_COLORS[data['ssp']], alpha=0.85, edgecolor='white', lw=0.5, label=label)
ax.set_title("Projected Annual Mean Discharge (2025–2030)")
ax.set_xlabel("Year"); ax.set_ylabel("Mean Discharge (m³/s)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.15, axis='y')
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_annual_mean.png", dpi=150)
plt.close()
print(f"   📸 cmip6_annual_mean.png")

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------
print("\n💾 Saving forecast data...")
for label, data in scenario_results.items():
    pd.DataFrame({'date': data['dates'], 'rainfall_mm': data['rainfall'],
                  'discharge_m3s': data['discharge'],
                  }).to_csv(FORECAST_DIR / f"forecast_{data['ssp']}.csv", index=False)
    print(f"   💾 forecast_{data['ssp']}.csv")

summary = []
for label, data in scenario_results.items():
    for yr, r in data['annual'].iterrows():
        summary.append({'scenario': label, 'year': yr,
                        'mean_discharge_m3s': round(r['mean_Q'], 2),
                        'peak_discharge_m3s': round(r['max_Q'],  2),
                        'total_rainfall_mm':  round(r['total_P'], 1)})
pd.DataFrame(summary).to_csv(FORECAST_DIR / "forecast_annual_summary.csv", index=False)
print(f"   💾 forecast_annual_summary.csv")

print(f"\n{'='*60}")
print(f"  FORECAST RELIABILITY SUMMARY")
print(f"{'='*60}")
print(f"  Direct Ridge NSE (benchmark):        {nse_direct:.4f}")
print(f"  AR buggy (old bugs — NSE collapse):  {nse_b:.4f}")
print(f"  AR fixed, perfect features:          {nse_c:.4f}")
print(f"  AR fixed, reconstructed features:    {nse_d:.4f}  ← CMIP6 uses this")
print(f"  Rolling convention: roll_shift={best_rs}, std_shift={best_ss}  mean_r={best_r:.4f}")
verdict = "✅ Reliable for scenario projections." if nse_d >= THRESHOLD \
          else "⚠️  Directional projections only — interpret relative differences, not absolute values."
print(f"  Verdict: {verdict}")
print(f"\n✅ Forecasting complete. Outputs in: {FORECAST_DIR}")
print("=" * 60)