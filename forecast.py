import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  PHASE 5: CMIP6 SCENARIO FORECASTING (2025–2030)")
print("=" * 60)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
FORECAST_DIR  = BASE_DIR / "forecast_outputs"
FORECAST_DIR.mkdir(exist_ok=True)
features_path = BASE_DIR / "data/features_dataset.csv"
master_path   = BASE_DIR / "data/master_dataset.csv"

CMIP6_DIR = BASE_DIR / "cmip6_data" / "processed"

# --------------------------------------------------
# CHECK CMIP6 DATA EXISTS
# --------------------------------------------------
SSPS = ['ssp245', 'ssp585']
SSP_LABELS = {'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}
SSP_COLORS = {'ssp245': '#8E44AD', 'ssp585': '#C0392B'}

cmip6_available = {}
for ssp in SSPS:
    fpath = CMIP6_DIR / f"forecast_input_{ssp}.csv"
    if fpath.exists():
        cmip6_available[ssp] = fpath
        print(f"   ✅ Found: {fpath.name}")
    else:
        print(f"   ❌ Missing: {fpath.name}")

if not cmip6_available:
    print("\n❌ No CMIP6 data found. Run cmip6_colab.py first, then place")
    print(f"   forecast_input_ssp245.csv and forecast_input_ssp585.csv in:")
    print(f"   {CMIP6_DIR}")
    raise SystemExit(1)

# --------------------------------------------------
# LOAD HISTORICAL DATA
# --------------------------------------------------
df_feat = pd.read_csv(features_path)
df_feat['date'] = pd.to_datetime(df_feat['date'])
df_feat = df_feat.sort_values('date').reset_index(drop=True)

df_master = pd.read_csv(master_path)
df_master['date'] = pd.to_datetime(df_master['date'])
df_master = df_master.sort_values('date').reset_index(drop=True)

hist_end = df_feat['date'].max()
print(f"\nHistorical data ends: {hist_end.date()}")

# --------------------------------------------------
# RETRAIN MODEL ON ALL HISTORICAL DATA
# --------------------------------------------------
print("\n🚀 Training final model on ALL historical data...")

TARGET = 'log_q'
DROP_COLS = [
    'date', 'q_upstream_mk', 'log_q', 'log_rainfall',
    'rainfall_max_mm', 'rainfall_std_mm',
    'rain_lag_1d', 'rain_lag_2d', 'rain_lag_3d',
    'rain_lag_4d', 'rain_lag_5d', 'rain_lag_6d', 'rain_lag_7d',
    'rain_roll_3d', 'rain_roll_7d', 'rain_roll_14d', 'rain_roll_30d',
    'rain_rollstd_7d', 'rain_rollstd_14d',
    'q_lag_1d', 'q_lag_2d', 'q_lag_3d',
    'rain_ewm_3d', 'rain_ewm_7d', 'rain_ewm_14d',
    'rain_x_qlag1', 'rain7d_x_qlag1',
    'q_rollmean_7d', 'q_rollmean_14d',
    'q_rollstd_7d', 'q_rollstd_14d',
    'dry_spell_days',
]

drop_existing = [c for c in DROP_COLS if c in df_feat.columns]
feature_cols  = [c for c in df_feat.columns if c not in drop_existing + [TARGET]]

X_all = df_feat[feature_cols]
y_all = df_feat[TARGET]

# Ridge was the best model — use it for forecasting
ridge_final = Ridge(alpha=1.0)
ridge_final.fit(X_all, y_all)
print(f"   Ridge model trained on {len(X_all)} samples, {len(feature_cols)} features")


# --------------------------------------------------
# AUTO-REGRESSIVE FORECASTING ENGINE
# --------------------------------------------------
def forecast_scenario(dates, rainfall, rain_std_vals, model, feature_cols, hist_df):
    """Day-by-day auto-regressive forecast."""
    n_days = len(dates)

    seed_rows = hist_df.tail(30).copy()
    seed_q = seed_rows['q_upstream_mk'].values

    predicted_q_raw = np.zeros(n_days)
    q_history = list(seed_q)
    rain_history = list(seed_rows['rainfall_max_mm'].values) if 'rainfall_max_mm' in seed_rows.columns else list(np.zeros(30))

    for i in range(n_days):
        date_i = dates[i]
        rain_i = rainfall[i]
        rain_history.append(rain_i)

        row = {}

        # Log rain lags
        for lag in range(1, 8):
            idx = -(lag)
            row[f'log_rain_lag_{lag}d'] = np.log1p(rain_history[idx]) if abs(idx) <= len(rain_history) else 0.0

        # Log rolling rain sums
        for window in [3, 7, 14, 30]:
            end_idx = len(rain_history) - 1
            start_idx = max(0, end_idx - window)
            row[f'log_rain_roll_{window}d'] = np.log1p(sum(rain_history[start_idx:end_idx]))

        # Log rolling rain std
        for window in [7, 14]:
            end_idx = len(rain_history) - 1
            start_idx = max(0, end_idx - window)
            segment = rain_history[start_idx:end_idx]
            row[f'log_rain_rollstd_{window}d'] = np.log1p(np.std(segment) if len(segment) > 1 else 0.0)

        # Log discharge lags
        for lag in range(1, 4):
            row[f'log_q_lag_{lag}d'] = np.log1p(q_history[-lag]) if lag <= len(q_history) else 0.0

        # Discharge rate of change
        if len(q_history) >= 3:
            row['delta_q_1d'] = q_history[-1] - q_history[-2]
            row['delta_q_2d'] = q_history[-2] - q_history[-3]
        else:
            row['delta_q_1d'] = 0.0
            row['delta_q_2d'] = 0.0

        # EWM rainfall
        for span in [3, 7, 14]:
            alpha = 2.0 / (span + 1)
            ewm_val = sum(rain_history[-(j+2)] * ((1 - alpha) ** j) for j in range(min(span * 3, len(rain_history) - 1)))
            row[f'log_rain_ewm_{span}d'] = np.log1p(ewm_val * alpha)

        # Rain × discharge interaction
        rain_yesterday = rain_history[-2] if len(rain_history) >= 2 else 0
        q_yesterday    = q_history[-1] if len(q_history) >= 1 else 0
        row['log_rain_x_qlag1']   = np.log1p(max(0, rain_yesterday * q_yesterday))
        rain_7d = sum(rain_history[max(0, len(rain_history)-8):len(rain_history)-1])
        row['log_rain7d_x_qlag1'] = np.log1p(max(0, rain_7d * q_yesterday))

        # Rolling discharge stats
        for window in [7, 14]:
            q_window = q_history[-window:] if len(q_history) >= window else q_history
            row[f'log_q_rollmean_{window}d'] = np.log1p(np.mean(q_window))
            row[f'log_q_rollstd_{window}d']  = np.log1p(np.std(q_window) if len(q_window) > 1 else 0)

        # Dry spell
        dry_count = 0
        for j in range(1, min(60, len(rain_history))):
            if rain_history[-j] < 1.0:
                dry_count += 1
            else:
                break
        row['log_dry_spell'] = np.log1p(dry_count)

        # Calendar
        doy = date_i.dayofyear
        month = date_i.month
        row['month_sin'] = np.sin(2 * np.pi * month / 12)
        row['month_cos'] = np.cos(2 * np.pi * month / 12)
        row['doy_sin']   = np.sin(2 * np.pi * doy / 365)
        row['doy_cos']   = np.cos(2 * np.pi * doy / 365)
        row['is_monsoon'] = 1 if 6 <= month <= 9 else 0

        # Rainfall std
        if 'log_rainfall_std' in feature_cols and rain_std_vals is not None:
            row['log_rainfall_std'] = np.log1p(rain_std_vals[i] if i < len(rain_std_vals) else 0)

        # Predict
        X_row = pd.DataFrame([row])
        for col in feature_cols:
            if col not in X_row.columns:
                X_row[col] = 0.0
        X_row = X_row[feature_cols]

        pred_log = model.predict(X_row)[0]
        pred_raw = max(np.expm1(pred_log), 0.0)

        predicted_q_raw[i] = pred_raw
        q_history.append(pred_raw)

    return predicted_q_raw


# --------------------------------------------------
# RUN CMIP6 SCENARIOS
# --------------------------------------------------
print("\n" + "=" * 60)
print("  Running CMIP6 scenario forecasts...")
print("=" * 60)

scenario_results = {}

for ssp, fpath in cmip6_available.items():
    label = SSP_LABELS[ssp]
    print(f"\n   🌍 {label} ({fpath.name})...")

    df_cmip = pd.read_csv(fpath)
    df_cmip['date'] = pd.to_datetime(df_cmip['date'])
    df_cmip = df_cmip.sort_values('date').reset_index(drop=True)

    cmip_dates = pd.DatetimeIndex(df_cmip['date'])
    cmip_rain  = df_cmip['rainfall_max_mm'].values
    cmip_rain_std = df_cmip['rainfall_std_mm'].values if 'rainfall_std_mm' in df_cmip.columns else None

    print(f"      Period: {cmip_dates[0].date()} → {cmip_dates[-1].date()} ({len(cmip_dates)} days)")

    q_pred = forecast_scenario(
        cmip_dates, cmip_rain, cmip_rain_std,
        ridge_final, feature_cols, df_master
    )

    scenario_results[label] = {
        'ssp':       ssp,
        'dates':     cmip_dates,
        'rainfall':  cmip_rain,
        'discharge': q_pred,
    }

    # Annual summary
    future_df = pd.DataFrame({'date': cmip_dates, 'Q': q_pred, 'P': cmip_rain})
    future_df['year'] = future_df['date'].dt.year
    annual = future_df.groupby('year').agg(
        mean_Q=('Q', 'mean'), max_Q=('Q', 'max'),
        total_P=('P', 'sum'),
    )
    scenario_results[label]['annual'] = annual

    print(f"      Annual summary:")
    print(f"      {'Year':>6}  {'Mean Q':>10}  {'Peak Q':>10}  {'Rain':>10}")
    for yr, r in annual.iterrows():
        print(f"      {yr:>6}  {r['mean_Q']:>10.1f}  {r['max_Q']:>10.1f}  {r['total_P']:>10.0f}")


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
print("\n📸 Generating plots...")

# --- PLOT 1: Full forecast time-series ---
fig, ax = plt.subplots(figsize=(16, 6))

# Historical tail (last 3 years)
hist_tail = df_master.tail(1095).copy()
ax.plot(hist_tail['date'], hist_tail['q_upstream_mk'],
        color='#1a1a1a', linewidth=0.7, alpha=0.5, label='Historical observed')

for label, data in scenario_results.items():
    color = SSP_COLORS[data['ssp']]
    ax.plot(data['dates'], data['discharge'],
            color=color, linewidth=0.8, alpha=0.85, label=label)

# Forecast boundary
forecast_start = min(data['dates'][0] for data in scenario_results.values())
ax.axvline(forecast_start, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(forecast_start, ax.get_ylim()[1] * 0.95, '  Forecast →', fontsize=9, color='gray', va='top')

ax.set_title("Kabini River Discharge — CMIP6 Scenario Forecasts")
ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_forecast_full.png", dpi=150)
plt.close()
print(f"   📸 cmip6_forecast_full.png")


# --- PLOT 2: Monsoon zoom (2026) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2.5]})

zoom_start = pd.Timestamp('2026-04-01')
zoom_end   = pd.Timestamp('2026-11-30')

# Pick first available scenario for rainfall hyetograph
first_label = list(scenario_results.keys())[0]
first_data  = scenario_results[first_label]

ax_rain = axes[0]
zm = (first_data['dates'] >= zoom_start) & (first_data['dates'] <= zoom_end)
ax_rain.bar(first_data['dates'][zm], first_data['rainfall'][zm],
            color='#3B8BD4', alpha=0.6, width=1)
ax_rain.invert_yaxis()
ax_rain.set_ylabel("Rainfall (mm)")
ax_rain.set_title(f"2026 Monsoon — Rainfall ({first_label}) & Discharge Forecasts")
ax_rain.set_xlim(zoom_start, zoom_end)

ax_q = axes[1]
for label, data in scenario_results.items():
    zm = (data['dates'] >= zoom_start) & (data['dates'] <= zoom_end)
    ax_q.plot(data['dates'][zm], data['discharge'][zm],
              color=SSP_COLORS[data['ssp']], linewidth=1.3, alpha=0.85, label=label)

ax_q.set_xlabel("Date"); ax_q.set_ylabel("Discharge (m³/s)")
ax_q.legend(fontsize=10); ax_q.set_xlim(zoom_start, zoom_end)
ax_q.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_monsoon_2026.png", dpi=150)
plt.close()
print(f"   📸 cmip6_monsoon_2026.png")


# --- PLOT 3: Annual peak discharge comparison ---
fig, ax = plt.subplots(figsize=(12, 5))

# Historical annual peaks
hist_annual_max = df_master.groupby(df_master['date'].dt.year)['q_upstream_mk'].max()
ax.bar(hist_annual_max.index, hist_annual_max.values, color='#BDC3C7', alpha=0.6,
       edgecolor='white', linewidth=0.5, label='Historical observed')

# Scenario peaks side by side
bar_width = 0.35
for i, (label, data) in enumerate(scenario_results.items()):
    years = sorted(data['annual'].index)
    peaks = [data['annual'].loc[yr, 'max_Q'] for yr in years]
    offsets = [yr + (i - 0.5) * bar_width for yr in years]
    ax.bar(offsets, peaks, bar_width, color=SSP_COLORS[data['ssp']],
           alpha=0.85, edgecolor='white', linewidth=0.5, label=label)

ax.set_title("Annual Peak Discharge — Historical vs CMIP6 Projections")
ax.set_xlabel("Year"); ax.set_ylabel("Peak Discharge (m³/s)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.15, axis='y')
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_annual_peaks.png", dpi=150)
plt.close()
print(f"   📸 cmip6_annual_peaks.png")


# --- PLOT 4: Envelope (SSP range) ---
fig, ax = plt.subplots(figsize=(14, 5))

if len(scenario_results) >= 2:
    labels = list(scenario_results.keys())
    # Align on common dates
    common_dates = scenario_results[labels[0]]['dates']
    all_q = np.column_stack([scenario_results[l]['discharge'] for l in labels])
    q_min = all_q.min(axis=1)
    q_max = all_q.max(axis=1)
    q_mean = all_q.mean(axis=1)

    window = 30
    q_min_s = pd.Series(q_min).rolling(window, min_periods=1).mean().values
    q_max_s = pd.Series(q_max).rolling(window, min_periods=1).mean().values
    q_mean_s = pd.Series(q_mean).rolling(window, min_periods=1).mean().values

    ax.fill_between(common_dates, q_min_s, q_max_s,
                    alpha=0.25, color='#8E44AD', label='SSP range (30-day smooth)')
    ax.plot(common_dates, q_mean_s,
            color='#8E44AD', linewidth=1.5, label='Ensemble mean')

    ax.set_title("CMIP6 Forecast Envelope — SSP2-4.5 to SSP5-8.5")
    ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(FORECAST_DIR / "cmip6_forecast_envelope.png", dpi=150)
    plt.close()
    print(f"   📸 cmip6_forecast_envelope.png")


# --- PLOT 5: Annual mean discharge bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))

bar_width = 0.35
for i, (label, data) in enumerate(scenario_results.items()):
    years = sorted(data['annual'].index)
    means = [data['annual'].loc[yr, 'mean_Q'] for yr in years]
    offsets = [yr + (i - 0.5) * bar_width for yr in years]
    ax.bar(offsets, means, bar_width, color=SSP_COLORS[data['ssp']],
           alpha=0.85, edgecolor='white', linewidth=0.5, label=label)

ax.set_title("Projected Annual Mean Discharge (2025–2030)")
ax.set_xlabel("Year"); ax.set_ylabel("Mean Discharge (m³/s)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.15, axis='y')
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_annual_mean.png", dpi=150)
plt.close()
print(f"   📸 cmip6_annual_mean.png")


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------
print("\n💾 Saving forecast data...")

for label, data in scenario_results.items():
    out_df = pd.DataFrame({
        'date': data['dates'],
        'rainfall_mm': data['rainfall'],
        'discharge_m3s': data['discharge'],
    })
    safe = data['ssp']
    out_df.to_csv(FORECAST_DIR / f"forecast_{safe}.csv", index=False)

# Combined annual summary
summary_rows = []
for label, data in scenario_results.items():
    for yr, r in data['annual'].iterrows():
        summary_rows.append({
            'scenario': label,
            'year': yr,
            'mean_discharge_m3s': round(r['mean_Q'], 2),
            'peak_discharge_m3s': round(r['max_Q'], 2),
            'total_rainfall_mm': round(r['total_P'], 1),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(FORECAST_DIR / "forecast_annual_summary.csv", index=False)
print(f"   💾 forecast_annual_summary.csv")

print(f"\n✅ Forecasting complete. Outputs in {FORECAST_DIR}")
print("=" * 60)