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
print("  ─── Rainfall-Only Model + Physical Post-Processing ───")
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

# Full model (with discharge lags) — for hindcast comparison only
full_model_pkl = MODEL_DIR / "ridge_forecast_model.pkl"
full_cols_json = MODEL_DIR / "feature_cols.json"

# Rainfall-only model — for CMIP6 projections
ro_model_pkl   = MODEL_DIR / "ridge_rainfall_only_model.pkl"
ro_cols_json   = MODEL_DIR / "rainfall_only_feature_cols.json"

# Base XGBoost model (for damped-AR base predictions)
xgb_ro_pkl     = MODEL_DIR / "xgb_rainfall_only_model.pkl"
ro_base_json   = MODEL_DIR / "rainfall_only_base_cols.json"
ro_meta_json   = MODEL_DIR / "rainfall_only_meta.json"

# Shared metadata
meta_json      = MODEL_DIR / "forecast_meta.json"
hydro_json     = MODEL_DIR / "hydro_params.json"

# Verify core assets exist
required = [full_model_pkl, full_cols_json, ro_model_pkl, ro_cols_json, meta_json]
missing = [p for p in required if not p.exists()]
if missing:
    print("❌  Missing model assets — run train_model.py first:")
    for p in missing:
        print(f"     {p.name}")
    raise SystemExit(1)

full_model        = joblib.load(full_model_pkl)
full_feature_cols = json.load(open(full_cols_json))
ro_model          = joblib.load(ro_model_pkl)
ro_feature_cols   = json.load(open(ro_cols_json))
forecast_meta     = json.load(open(meta_json))

correction_factor = forecast_meta["correction_factor"]
p90_threshold     = forecast_meta["p90_threshold_m3s"]
best_nse          = forecast_meta.get("best_nse", "?")
ro_nse_train      = forecast_meta.get("rainfall_only_nse", "?")
ro_model_type     = forecast_meta.get("rainfall_only_type", "ridge")

# Load damped-AR metadata
DAMP_ALPHA    = 0.6   # default
clim_q_monthly = None
xgb_ro_model   = None
ro_base_cols   = None

if ro_meta_json.exists():
    ro_meta = json.load(open(ro_meta_json))
    ro_model_type = ro_meta.get("ro_model_type", ro_model_type)
    if ro_meta.get("damp_alpha") is not None:
        DAMP_ALPHA = ro_meta["damp_alpha"]
    if ro_meta.get("clim_q_monthly"):
        clim_q_monthly = {int(k): v for k, v in ro_meta["clim_q_monthly"].items()}

if ro_model_type == "damped_ar" and xgb_ro_pkl.exists() and ro_base_json.exists():
    xgb_ro_model = joblib.load(xgb_ro_pkl)
    ro_base_cols = json.load(open(ro_base_json))
    print(f"   ✅ Damped-AR model (α={DAMP_ALPHA}, base: XGBoost {len(ro_base_cols)} features)")

# Load hydrological parameters for post-processing
if hydro_json.exists():
    hydro_params      = json.load(open(hydro_json))
    recession_k       = hydro_params["recession_constant"]
    clim_monthly_mean = {int(k): v for k, v in hydro_params["clim_monthly_mean"].items()}
    clim_monthly_max  = {int(k): v for k, v in hydro_params["clim_monthly_max"].items()}
    mean_annual_q     = hydro_params["mean_annual_q"]
    print(f"   ✅ hydro_params.json (recession k={recession_k:.4f})")
    # Use hydro clim as fallback for damped-AR if not in ro_meta
    if clim_q_monthly is None:
        clim_q_monthly = {int(k): v for k, v in hydro_params["clim_monthly_median"].items()}
else:
    print("   ⚠️  hydro_params.json not found — estimating from data...")
    recession_k       = 0.975
    clim_monthly_mean = None
    clim_monthly_max  = None
    mean_annual_q     = None

print(f"   ✅ Full Ridge model        ({len(full_feature_cols)} features, NSE={best_nse})")
print(f"   ✅ Rainfall-Only model     ({len(ro_feature_cols)} features, NSE={ro_nse_train}, type={ro_model_type})")
print(f"   ✅ Peak correction ×{correction_factor:.3f} above {p90_threshold:.0f} m³/s")
print(f"   ✅ Recession constant k={recession_k:.4f}")

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

# Estimate climatology from master if not loaded from JSON
if clim_monthly_mean is None:
    df_master['month'] = df_master['date'].dt.month
    clim_monthly_mean = df_master.groupby('month')['q_upstream_mk'].mean().to_dict()
    clim_monthly_max  = df_master.groupby('month')['q_upstream_mk'].max().to_dict()
    mean_annual_q     = float(df_master['q_upstream_mk'].mean())


# ══════════════════════════════════════════════════════════
#  ROLLING CONVENTION DETECTION
#  (needed for feature reconstruction)
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

check_cols = [c for c in fdiag.columns
             if c.startswith('log_rain_roll_') and 'rollstd' not in c and c in ro_feature_cols]
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
#  FEATURE ENGINEERING (rainfall features only)
# ══════════════════════════════════════════════════════════

def precompute_rain_features(dates_series, rain_series, rain_std_series=None):
    """Build all rainfall-derived features (no discharge lags)."""
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

    # Antecedent Precipitation Index (API) — soil moisture proxy
    rain_vals = rain.values.astype(float)
    for k_val, k_name in [(0.85, 'fast'), (0.92, 'med'), (0.97, 'slow')]:
        api_raw = np.zeros(len(rain_vals))
        api_raw[0] = rain_vals[0] if not np.isnan(rain_vals[0]) else 0.0
        for t in range(1, len(rain_vals)):
            rv = rain_vals[t] if not np.isnan(rain_vals[t]) else 0.0
            api_raw[t] = k_val * api_raw[t-1] + rv
        # Shift by 1 to prevent leakage
        api_shifted = pd.Series(api_raw).shift(1).fillna(0).clip(lower=0).values
        df_c[f'log_api_{k_name}'] = np.log1p(api_shifted)

    # Dry spell length
    dry_spell = np.zeros(len(rain_vals))
    for t in range(1, len(rain_vals)):
        rv = rain_vals[t] if not np.isnan(rain_vals[t]) else 0.0
        dry_spell[t] = dry_spell[t-1] + 1 if rv < 1.0 else 0
    df_c['log_dry_spell'] = np.log1p(dry_spell)

    # Cumulative monsoon rainfall
    cum_monsoon = np.zeros(len(rain_vals))
    for t in range(1, len(rain_vals)):
        d = dates[t]
        m, day = d.month, d.day
        rv = rain_vals[t] if not np.isnan(rain_vals[t]) else 0.0
        if m == 6 and day == 1:
            cum_monsoon[t] = rv
        elif 6 <= m <= 11:
            cum_monsoon[t] = cum_monsoon[t-1] + rv
        else:
            cum_monsoon[t] = 0.0
    df_c['log_cum_monsoon'] = np.log1p(
        pd.Series(cum_monsoon).shift(1).fillna(0).clip(lower=0).values)

    df_c['month_sin']  = np.sin(2 * np.pi * pd.Series(dates).dt.month     / 12).values
    df_c['month_cos']  = np.cos(2 * np.pi * pd.Series(dates).dt.month     / 12).values
    df_c['doy_sin']    = np.sin(2 * np.pi * pd.Series(dates).dt.dayofyear / 365).values
    df_c['doy_cos']    = np.cos(2 * np.pi * pd.Series(dates).dt.dayofyear / 365).values
    df_c['is_monsoon'] = (pd.Series(dates).dt.month.between(6, 9)).astype(int).values

    return df_c.reset_index(drop=True)


def build_feat_with_seed(seed_df, target_df):
    """Build rainfall features with historical seed for warm-up."""
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
#  PHYSICAL POST-PROCESSOR
#
#  The rainfall-only model predicts daily discharge from
#  rainfall features alone. This gives the right MAGNITUDE
#  (seasonal pattern, monsoon peaks) but misses the smooth
#  day-to-day transitions that discharge lags captured.
#
#  The post-processor applies physical constraints:
#
#  1. RECESSION FILTER: During dry spells (low rainfall),
#     discharge decays exponentially following the recession
#     constant k estimated from the historical record.
#     This prevents unphysical jumps during dry periods.
#
#  2. ADAPTIVE SMOOTHING: EWM with monsoon-aware span.
#     Short span during monsoon (responsive to events),
#     longer span during dry season (smoother recession).
#
#  3. CLIMATOLOGICAL BOUNDS: Cap predictions at 1.5×
#     historical monthly max to prevent runaway values.
# ══════════════════════════════════════════════════════════

def physical_postprocess(raw_predictions, dates, rainfall,
                         recession_k, clim_monthly_mean,
                         clim_monthly_max, correction_factor,
                         p90_threshold, seed_q=None):
    """
    Apply physical constraints to rainfall-only predictions.
    """
    n = len(raw_predictions)
    processed = np.zeros(n)
    dates = pd.DatetimeIndex(dates)

    if seed_q is None:
        first_month = dates[0].month
        seed_q = clim_monthly_mean.get(first_month, np.mean(list(clim_monthly_mean.values())))

    prev_q = seed_q

    for i in range(n):
        month = dates[i].month
        pred_q = max(raw_predictions[i], 0.0)

        rain_today = rainfall[i] if i < len(rainfall) else 0.0
        rain_recent = np.mean(rainfall[max(0, i-2):i+1]) if i < len(rainfall) else 0.0

        recession_q = prev_q * recession_k

        if rain_recent < 2.0:
            pred_q = min(pred_q, max(recession_q, pred_q * 0.3))
        else:
            pred_q = max(pred_q, recession_q * 0.8)

        month_max = clim_monthly_max.get(month, max(clim_monthly_mean.values()) * 3)
        pred_q = min(pred_q, month_max * 1.5)

        if pred_q > p90_threshold:
            pred_q *= correction_factor

        pred_q = max(pred_q, 0.0)
        processed[i] = pred_q
        prev_q = pred_q

    result = pd.Series(processed)
    months = dates.month
    monsoon_mask = months.isin([6, 7, 8, 9])

    smooth = result.copy()
    for i in range(1, n):
        if monsoon_mask[i]:
            alpha = 2.0 / (3 + 1)
        else:
            alpha = 2.0 / (7 + 1)
        smooth.iloc[i] = alpha * result.iloc[i] + (1 - alpha) * smooth.iloc[i-1]

    return smooth.values


def predict_with_ro_model(rain_feat_df, dates, ro_model, ro_feature_cols,
                          ro_model_type, xgb_ro_model=None, ro_base_cols=None,
                          damp_alpha=0.6, clim_q_monthly=None):
    """
    Predict discharge using the best rainfall-only approach.
    Handles Ridge, XGBoost, and Damped-AR model types.
    """
    if ro_model_type == 'damped_ar' and xgb_ro_model is not None and ro_base_cols is not None:
        # Step 1: Get base predictions from XGBoost rainfall-only
        X_base = pd.DataFrame([{col: rain_feat_df.iloc[i].get(col, 0.0) for col in ro_base_cols}
                                for i in range(len(rain_feat_df))])
        base_pred_log = xgb_ro_model.predict(X_base)
        base_pred_real = np.expm1(base_pred_log)
        base_pred_real = np.maximum(base_pred_real, 0.0)

        # Step 2: Build pseudo-Q lags with climatological damping
        dates_arr = pd.DatetimeIndex(dates)
        pseudo_q = np.zeros(len(rain_feat_df))
        for i in range(len(pseudo_q)):
            month_i = dates_arr[i].month
            clim_q  = clim_q_monthly.get(month_i, 30.0) if clim_q_monthly else 30.0
            pseudo_q[i] = damp_alpha * base_pred_real[i] + (1 - damp_alpha) * clim_q

        X_dar = pd.DataFrame([{col: rain_feat_df.iloc[i].get(col, 0.0) for col in ro_feature_cols
                                if not col.startswith('log_pseudo_q_lag')}
                               for i in range(len(rain_feat_df))])

        for lag in range(1, 4):
            col_name = f'log_pseudo_q_lag_{lag}d'
            if col_name in ro_feature_cols:
                X_dar[col_name] = np.log1p(pd.Series(pseudo_q).shift(lag).fillna(0).clip(lower=0).values)

        pred_log = ro_model.predict(X_dar[ro_feature_cols])

    else:
        # Simple Ridge or XGBoost — direct prediction
        X_ro = pd.DataFrame([{col: rain_feat_df.iloc[i].get(col, 0.0) for col in ro_feature_cols}
                              for i in range(len(rain_feat_df))])
        pred_log = ro_model.predict(X_ro)

    raw_pred = np.expm1(pred_log)
    return np.maximum(raw_pred, 0.0)


# ══════════════════════════════════════════════════════════
#  AR DISCHARGE LOOP (kept for hindcast comparison only)
# ══════════════════════════════════════════════════════════

def ar_discharge_loop(rain_feat_df, model, feature_cols,
                      seed_q_history, correction_factor, p90_threshold,
                      dates=None, annual_reset_q=None):
    """Autoregressive loop — feeds predicted Q back as lags.
    Used ONLY for hindcast comparison, NOT for CMIP6 forecasting."""
    q_history = list(seed_q_history)
    predicted = np.zeros(len(rain_feat_df))

    for i in range(len(rain_feat_df)):
        if annual_reset_q is not None and dates is not None:
            d = pd.Timestamp(dates[i])
            if d.month == 1 and d.day == 1:
                for k in range(min(30, len(q_history))):
                    q_history[-(k+1)] = annual_reset_q * (1.0 + k * 0.015)

        row = rain_feat_df.iloc[i].to_dict()
        for lag in range(1, 4):
            val = q_history[-lag] if len(q_history) >= lag else 0.0
            row[f'log_q_lag_{lag}d'] = np.log1p(max(val, 0.0))

        X_row      = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])
        pred_q_raw = max(np.expm1(float(model.predict(X_row)[0])), 0.0)

        pred_q_out = pred_q_raw * correction_factor \
                     if pred_q_raw > p90_threshold else pred_q_raw

        predicted[i] = pred_q_out
        q_history.append(pred_q_raw)

    return predicted


# ══════════════════════════════════════════════════════════
#  STEP 2: HINDCAST VALIDATION (2015–2020)
#
#  Compare three approaches:
#  (a) Direct Ridge (with Q lags, pre-computed) — upper bound
#  (b) AR loop with full model — old approach (error accumulation)
#  (c) Rainfall-only Ridge + post-processing — new CMIP6 approach
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: Hindcast validation (2015–2020)")
print("=" * 60)

TEST_START  = pd.Timestamp('2015-01-01')
df_test     = df_feat[df_feat['date'] >= TEST_START].copy().reset_index(drop=True)
y_obs       = np.expm1(df_test['log_q'].values)
test_dates  = pd.DatetimeIndex(df_test['date'])

# Climatological January Q for AR reset
df_master['month'] = df_master['date'].dt.month
clim_monthly_q = df_master.groupby('month')['q_upstream_mk'].median()
ANNUAL_RESET_Q = float(clim_monthly_q[1])

# ── (a) Direct Ridge (upper bound) ──
direct_pred = np.expm1(full_model.predict(df_test[full_feature_cols]))
direct_corr = np.where(direct_pred > p90_threshold, direct_pred * correction_factor, direct_pred)
nse_direct  = 1 - np.sum((y_obs - direct_corr)**2) / np.sum((y_obs - y_obs.mean())**2)
rmse_direct = np.sqrt(mean_squared_error(y_obs, direct_corr))
print(f"\n   (a) Direct Ridge (with Q lags):     NSE={nse_direct:.4f}  RMSE={rmse_direct:.2f}  [upper bound]")

# ── (b) AR loop — old approach for comparison ──
seed_hc   = df_master[df_master['date'] < TEST_START].tail(30)
seed_q_hc = list(seed_hc['q_upstream_mk'].fillna(ANNUAL_RESET_Q).values.astype(float))

test_master = df_master[(df_master['date'] >= TEST_START) &
                        (df_master['date'].isin(df_test['date']))].reset_index(drop=True)

rain_feat_hc = build_feat_with_seed(seed_hc, test_master)
ar_pred = ar_discharge_loop(
    rain_feat_hc, full_model, full_feature_cols,
    seed_q_hc, correction_factor, p90_threshold,
    dates=test_dates, annual_reset_q=ANNUAL_RESET_Q
)
nse_ar  = 1 - np.sum((y_obs-ar_pred)**2)/np.sum((y_obs-y_obs.mean())**2)
rmse_ar = np.sqrt(mean_squared_error(y_obs, ar_pred))
print(f"   (b) AR loop (full model + reset):   NSE={nse_ar:.4f}  RMSE={rmse_ar:.2f}  [old approach]")

# ── (c) Rainfall-only + post-processing — NEW CMIP6 approach ──
ro_feat_hc = build_feat_with_seed(seed_hc, test_master)

# Get raw rainfall-only predictions using the best model type
raw_ro_pred = predict_with_ro_model(
    ro_feat_hc, test_dates, ro_model, ro_feature_cols,
    ro_model_type, xgb_ro_model, ro_base_cols,
    DAMP_ALPHA, clim_q_monthly
)

# Raw rainfall-only (no post-processing)
nse_ro_raw = 1 - np.sum((y_obs-raw_ro_pred)**2)/np.sum((y_obs-y_obs.mean())**2)
rmse_ro_raw = np.sqrt(mean_squared_error(y_obs, raw_ro_pred))
print(f"   (c1) Rainfall-only (raw):           NSE={nse_ro_raw:.4f}  RMSE={rmse_ro_raw:.2f}  [no post-processing]")

# Seed Q from last observed value before test period
seed_q_val = float(seed_hc['q_upstream_mk'].iloc[-1])

ro_post = physical_postprocess(
    raw_ro_pred, test_dates,
    test_master['rainfall_max_mm'].values,
    recession_k, clim_monthly_mean, clim_monthly_max,
    correction_factor, p90_threshold,
    seed_q=seed_q_val
)
nse_ro  = 1 - np.sum((y_obs-ro_post)**2)/np.sum((y_obs-y_obs.mean())**2)
rmse_ro = np.sqrt(mean_squared_error(y_obs, ro_post))
corr_ro = float(np.corrcoef(ro_post, y_obs)[0,1])
print(f"   (c2) Rainfall-only + post-process:  NSE={nse_ro:.4f}  RMSE={rmse_ro:.2f}  r={corr_ro:.4f}  ← CMIP6 uses this")

print(f"\n   Gap analysis:")
print(f"   (a) Direct Ridge (Q lags):   NSE = {nse_direct:.4f}  [theoretical max]")
print(f"   (b) AR loop (old approach):  NSE = {nse_ar:.4f}  [{nse_ar-nse_direct:+.4f} vs direct]")
print(f"   (c) Rainfall-only + PP:      NSE = {nse_ro:.4f}  [{nse_ro-nse_direct:+.4f} vs direct]")
print(f"\n   WHY rainfall-only is better for CMIP6:")
print(f"   • AR loop compounds prediction errors over 5+ years")
print(f"   • Rainfall-only uses ONLY inputs CMIP6 actually provides (no future Q)")
print(f"   • Post-processor adds physical constraints (recession, smoothing)")
print(f"   • Lower hindcast NSE is the honest cost of not using unavailable future data")

THRESHOLD = 0.40
if nse_ro >= THRESHOLD:
    print(f"\n   ✅ Hindcast NSE={nse_ro:.4f} ≥ {THRESHOLD} — pipeline validated for CMIP6 forecasting.")
else:
    print(f"\n   ⚠️  Hindcast NSE={nse_ro:.4f} < {THRESHOLD}")
    print(f"      Forecasts will be generated but interpret as directional projections only.")

# Per-year NSE breakdown
annual_nse = []
years = sorted(set(df_test['date'].dt.year))
for yr in years:
    mask = (df_test['date'].dt.year == yr).values
    if mask.sum() < 30: continue
    obs_yr, pred_yr = y_obs[mask], ro_post[mask]
    nse_yr = 1 - np.sum((obs_yr-pred_yr)**2)/np.sum((obs_yr-obs_yr.mean())**2)
    annual_nse.append((yr, nse_yr))

print(f"\n   Per-year NSE (rainfall-only + PP):")
for yr, nse_yr in annual_nse:
    status = "✅" if nse_yr >= 0.3 else "⚠️"
    print(f"     {status} {yr}: NSE={nse_yr:.4f}")

# ── Hindcast validation plot ──
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
td = df_test['date'].values

axes[0].plot(td, y_obs,       label='Observed',
             color='#1a1a1a', lw=0.8, alpha=0.8)
axes[0].plot(td, direct_corr, label=f'Direct Ridge (Q lags)  NSE={nse_direct:.3f}',
             color='#F39C12', lw=1.2, alpha=0.9)
axes[0].plot(td, ro_post,     label=f'Rainfall-Only + PP  NSE={nse_ro:.3f}',
             color='#2E86C1', lw=0.9, alpha=0.8, linestyle='--')
axes[0].plot(td, ar_pred,     label=f'AR Loop old  NSE={nse_ar:.3f}',
             color='#E74C3C', lw=0.7, alpha=0.6, linestyle=':')
axes[0].set_ylabel("Discharge (m³/s)")
axes[0].set_title("Hindcast Validation — Model Comparison")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.15)

axes[1].plot(td, ro_post - y_obs, color='#2E86C1', lw=0.6, alpha=0.7)
axes[1].axhline(0, color='gray', lw=0.8, linestyle='--')
axes[1].set_ylabel("Rainfall-Only − Obs (m³/s)")
axes[1].set_title(f"Rainfall-Only Residuals  (mean={np.mean(ro_post-y_obs):.1f}  std={np.std(ro_post-y_obs):.1f})")
axes[1].grid(True, alpha=0.15)

yr_vals, nse_vals = zip(*annual_nse)
axes[2].bar(yr_vals, nse_vals, color=['#2ECC71' if n >= 0.3 else '#E74C3C' for n in nse_vals],
            alpha=0.8, edgecolor='white')
axes[2].axhline(0.3, color='gray', linestyle='--', lw=0.8)
axes[2].set_ylabel("NSE"); axes[2].set_xlabel("Year")
axes[2].set_title("Per-Year NSE — Rainfall-Only + Post-Processing"); axes[2].grid(True, alpha=0.15, axis='y')

plt.tight_layout()
plt.savefig(FORECAST_DIR / "hindcast_validation.png", dpi=150)
plt.close()
print(f"\n   📸 hindcast_validation.png")

pd.DataFrame({
    'date':          df_test['date'].values,
    'observed':      y_obs,
    'direct_ridge':  direct_corr,
    'rainfall_only': ro_post,
    'ar_loop_old':   ar_pred,
}).to_csv(FORECAST_DIR / "hindcast_validation.csv", index=False)
print(f"   💾 hindcast_validation.csv")


# ══════════════════════════════════════════════════════════
#  STEP 3: CMIP6 SCENARIO FORECASTS
#  Using rainfall-only model + physical post-processing
# ══════════════════════════════════════════════════════════
SSPS       = ['ssp245', 'ssp585']
SSP_LABELS = {'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}
SSP_COLORS = {'ssp245': '#8E44AD',  'ssp585': '#C0392B'}

print("\n" + "=" * 60)
print("  STEP 3: Checking for CMIP6 data")
print("=" * 60)

cmip6_available = {}
for ssp in SSPS:
    fpath = CMIP6_DIR / f"forecast_input_{ssp}.csv"
    if fpath.exists():
        cmip6_available[ssp] = fpath
        print(f"   ✅ {fpath.name}")
    else:
        print(f"   ❌ Missing: {fpath.name}")

if not cmip6_available:
    print("\n   No CMIP6 data found. Run cmip6-download.py first.")
    print(f"   Expected location: {CMIP6_DIR}/forecast_input_ssp*.csv")
    raise SystemExit(1)

print("\n" + "=" * 60)
print(f"  STEP 4: CMIP6 forecasts  [hindcast NSE={nse_ro:.3f}]")
print("=" * 60)

seed_fc = df_master.tail(30)
seed_q_fc = float(df_master['q_upstream_mk'].iloc[-1])

scenario_results = {}

for ssp, fpath in cmip6_available.items():
    label = SSP_LABELS[ssp]
    print(f"\n   🌍 {label}...")

    df_cmip = pd.read_csv(fpath)
    df_cmip['date'] = pd.to_datetime(df_cmip['date'])
    df_cmip = df_cmip.sort_values('date').reset_index(drop=True)
    cmip_dates = pd.DatetimeIndex(df_cmip['date'])
    print(f"      {cmip_dates[0].date()} → {cmip_dates[-1].date()} ({len(cmip_dates)} days)")

    # Build rainfall features with historical seed
    rain_feat_cmip = build_feat_with_seed(seed_fc, df_cmip)

    # Predict with rainfall-only model (handles all model types)
    raw_pred = predict_with_ro_model(
        rain_feat_cmip, cmip_dates, ro_model, ro_feature_cols,
        ro_model_type, xgb_ro_model, ro_base_cols,
        DAMP_ALPHA, clim_q_monthly
    )

    # Apply physical post-processing
    q_pred = physical_postprocess(
        raw_pred, cmip_dates,
        df_cmip['rainfall_max_mm'].values,
        recession_k, clim_monthly_mean, clim_monthly_max,
        correction_factor, p90_threshold,
        seed_q=seed_q_fc
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
reliability = f"Hindcast NSE={nse_ro:.3f} (rainfall-only)"

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

# Methodology comparison plot — 2017 monsoon zoom
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
zoom_start = pd.Timestamp('2017-04-01')
zoom_end   = pd.Timestamp('2017-11-30')
zm_hc = (df_test['date'] >= zoom_start) & (df_test['date'] <= zoom_end)

if zm_hc.sum() > 30:
    axes[0].plot(df_test.loc[zm_hc, 'date'], y_obs[zm_hc.values],
                 color='#1a1a1a', lw=1.2, label='Observed')
    axes[0].plot(df_test.loc[zm_hc, 'date'], direct_corr[zm_hc.values],
                 color='#F39C12', lw=1.0, alpha=0.8, label=f'Direct Ridge (NSE={nse_direct:.3f})')
    axes[0].plot(df_test.loc[zm_hc, 'date'], ro_post[zm_hc.values],
                 color='#2E86C1', lw=1.0, alpha=0.8, linestyle='--',
                 label=f'Rainfall-Only + PP (NSE={nse_ro:.3f})')
    axes[0].plot(df_test.loc[zm_hc, 'date'], ar_pred[zm_hc.values],
                 color='#E74C3C', lw=0.8, alpha=0.6, linestyle=':',
                 label=f'AR Loop old (NSE={nse_ar:.3f})')
    axes[0].set_ylabel("Discharge (m³/s)")
    axes[0].set_title("2017 Monsoon — Method Comparison")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.15)

    zm_rain = test_master[(test_master['date'] >= zoom_start) & (test_master['date'] <= zoom_end)]
    axes[1].bar(zm_rain['date'], zm_rain['rainfall_max_mm'], color='#3B8BD4', alpha=0.5, width=1)
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Rainfall (mm)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Observed Rainfall")
    axes[1].grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(FORECAST_DIR / "methodology_comparison.png", dpi=150)
plt.close()
print(f"   📸 methodology_comparison.png")

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
print(f"  Direct Ridge (Q lags, upper bound):  NSE={nse_direct:.4f}")
print(f"  AR loop (old approach):              NSE={nse_ar:.4f}  ⚠️ Error accumulation")
print(f"  Rainfall-only (raw):                 NSE={nse_ro_raw:.4f}")
print(f"  Rainfall-only + post-processing:     NSE={nse_ro:.4f}  ← CMIP6 uses this")
print(f"  Rolling convention: roll_shift={best_rs}, std_shift={best_ss}")
print(f"")
print(f"  METHODOLOGY NOTE:")
print(f"  The rainfall-only model has lower hindcast NSE than the full Ridge")
print(f"  because it cannot use discharge autocorrelation (q_lag_1d, r=0.94).")
print(f"  This is the CORRECT approach for CMIP6 projections because:")
print(f"    • Future discharge values don't exist to feed as lags")
print(f"    • AR loop compounds prediction errors over 5+ years")
print(f"    • Physical post-processing (recession + smoothing) provides")
print(f"      hydrological realism without requiring future Q values")
print(f"")
verdict = "✅ Validated for scenario projections." if nse_ro >= THRESHOLD \
          else "⚠️  Directional projections only — interpret relative differences, not absolute values."
print(f"  Verdict: {verdict}")
print(f"\n✅ Forecasting complete. Outputs in: {FORECAST_DIR}")
print("=" * 60)