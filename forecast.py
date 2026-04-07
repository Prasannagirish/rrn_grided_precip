import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

print("=" * 60)
print("  PHASE 5: FUTURE SCENARIO FORECASTING (2025–2030)")
print("=" * 60)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
MODEL_DIR     = BASE_DIR / "model_outputs"
FORECAST_DIR  = BASE_DIR / "forecast_outputs"
FORECAST_DIR.mkdir(exist_ok=True)
features_path = BASE_DIR / "data/features_dataset.csv"
master_path   = BASE_DIR / "data/master_dataset.csv"

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
print(f"Historical data ends: {hist_end.date()}")

# --------------------------------------------------
# BUILD RAINFALL CLIMATOLOGY FROM HISTORICAL DATA
#
# For each day-of-year, compute statistics from the
# historical record. This gives us a "typical" rainfall
# pattern to project forward.
# --------------------------------------------------
print("\n📊 Building daily rainfall climatology...")
df_master['doy'] = df_master['date'].dt.dayofyear

rain_col = 'rainfall_max_mm'
clim = df_master.groupby('doy')[rain_col].agg(
    mean='mean', median='median', std='std',
    p25=lambda x: np.percentile(x, 25),
    p75=lambda x: np.percentile(x, 75),
    p90=lambda x: np.percentile(x, 90),
).reset_index()
clim['std'] = clim['std'].fillna(0)

print(f"   Climatology built from {df_master['date'].dt.year.nunique()} years")
print(f"   Annual mean rainfall: {clim['mean'].sum():.0f} mm")

# Also build rainfall_std_mm climatology if available
has_rain_std = 'rainfall_std_mm' in df_feat.columns
if has_rain_std:
    if 'rainfall_std_mm' in df_master.columns:
        clim_std = df_master.groupby('doy')['rainfall_std_mm'].mean().reset_index()
        clim_std.columns = ['doy', 'rainfall_std_mean']
    else:
        # Approximate from the features dataset
        clim_std_src = df_feat.copy()
        clim_std_src['doy'] = clim_std_src['date'].dt.dayofyear
        if 'rainfall_std_mm' in clim_std_src.columns:
            clim_std = clim_std_src.groupby('doy')['rainfall_std_mm'].mean().reset_index()
            clim_std.columns = ['doy', 'rainfall_std_mean']
        else:
            has_rain_std = False


# --------------------------------------------------
# DEFINE SCENARIOS
#
# Each scenario is a multiplier on the climatological
# mean rainfall. This is the standard approach in
# hydrology for "what-if" analysis.
# --------------------------------------------------
SCENARIOS = {
    'Average':       1.0,    # Normal monsoon (climatological mean)
    'Wet (+20%)':    1.20,   # Above-average monsoon
    'Dry (-20%)':    0.80,   # Below-average / drought year
    'Extreme (+40%)': 1.40,  # Severe flooding scenario
}

FORECAST_START = pd.Timestamp('2025-01-01')
FORECAST_END   = pd.Timestamp('2030-12-31')

future_dates = pd.date_range(FORECAST_START, FORECAST_END, freq='D')
print(f"\nForecast period: {FORECAST_START.date()} → {FORECAST_END.date()} ({len(future_dates)} days)")
print(f"Scenarios: {list(SCENARIOS.keys())}")


# --------------------------------------------------
# RETRAIN MODEL ON ALL HISTORICAL DATA
#
# For actual forecasting, we use ALL available data
# (no held-out test set — we already validated the
# model in the training phase).
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

# Use best params from model_comparison if available, else defaults
best_params = {
    'n_estimators': 1000, 'learning_rate': 0.04, 'max_depth': 5,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
}

xgb_final = XGBRegressor(**best_params)
xgb_final.fit(X_all, y_all, verbose=50)
print(f"   Model trained on {len(X_all)} samples, {len(feature_cols)} features")


# --------------------------------------------------
# AUTO-REGRESSIVE FORECASTING ENGINE
#
# The key challenge: discharge lag features (log_q_lag_1d,
# log_q_lag_2d, log_q_lag_3d) require yesterday's predicted
# discharge as input for today's prediction. So we must
# predict day-by-day, feeding each prediction back as
# the next day's lag input.
#
# Rainfall features come from the scenario climatology.
# --------------------------------------------------
def generate_scenario_rainfall(dates, clim_df, multiplier, seed=42):
    """Generate daily rainfall for a scenario using climatology + noise."""
    rng = np.random.RandomState(seed)
    doys = dates.dayofyear.values
    rain = np.zeros(len(dates))

    for i, doy in enumerate(doys):
        # Handle leap year day 366 → use day 365 climatology
        doy_lookup = min(doy, 365)
        row = clim_df[clim_df['doy'] == doy_lookup]
        if len(row) == 0:
            row = clim_df[clim_df['doy'] == 365]

        mean_rain = row['mean'].values[0] * multiplier
        std_rain  = row['std'].values[0] * multiplier

        # Sample from a gamma-like distribution (non-negative, right-skewed)
        if mean_rain > 0.5:
            # Shape parameter from mean/std
            shape = (mean_rain / max(std_rain, 0.1)) ** 2
            scale = max(std_rain, 0.1) ** 2 / max(mean_rain, 0.01)
            rain[i] = rng.gamma(shape, scale)
        else:
            # Dry day with small probability of light rain
            rain[i] = rng.exponential(0.3) if rng.random() < 0.15 else 0.0

    return rain


def forecast_scenario(dates, rainfall, rain_std_vals, model, feature_cols, hist_df):
    """
    Day-by-day auto-regressive forecast.

    Uses the last rows of historical data to seed the lag features,
    then steps forward one day at a time, feeding each prediction
    back as the next day's discharge lag.
    """
    n_days = len(dates)

    # Seed from the last 30 days of historical data
    seed_rows = hist_df.tail(30).copy()
    seed_q = seed_rows['q_upstream_mk'].values  # raw discharge for lags

    # Initialise output
    predicted_q_log = np.zeros(n_days)
    predicted_q_raw = np.zeros(n_days)

    # Running state: last N discharge values (raw space) for lag computation
    q_history = list(seed_q)  # most recent at the end

    # Running rainfall history for rolling features
    rain_history = list(seed_rows['rainfall_max_mm'].values) if 'rainfall_max_mm' in seed_rows.columns else list(np.zeros(30))

    for i in range(n_days):
        date_i = dates[i]
        rain_i = rainfall[i]

        # Append today's rain to history
        rain_history.append(rain_i)

        # Build feature vector for this day
        row = {}

        # --- Log rain lags (shifted: lag_1d = yesterday's rain) ---
        for lag in range(1, 8):
            idx = -(lag)  # -1 = yesterday, -2 = day before, etc.
            if abs(idx) <= len(rain_history):
                row[f'log_rain_lag_{lag}d'] = np.log1p(rain_history[idx])
            else:
                row[f'log_rain_lag_{lag}d'] = 0.0

        # --- Log rolling rain sums (shifted by 1) ---
        for window in [3, 7, 14, 30]:
            end_idx = len(rain_history) - 1  # exclude today (shifted)
            start_idx = max(0, end_idx - window)
            roll_sum = sum(rain_history[start_idx:end_idx])
            row[f'log_rain_roll_{window}d'] = np.log1p(roll_sum)

        # --- Log rolling rain std ---
        for window in [7, 14]:
            end_idx = len(rain_history) - 1
            start_idx = max(0, end_idx - window)
            segment = rain_history[start_idx:end_idx]
            roll_std = np.std(segment) if len(segment) > 1 else 0.0
            row[f'log_rain_rollstd_{window}d'] = np.log1p(roll_std)

        # --- Log discharge lags ---
        for lag in range(1, 4):
            if lag <= len(q_history):
                row[f'log_q_lag_{lag}d'] = np.log1p(q_history[-lag])
            else:
                row[f'log_q_lag_{lag}d'] = 0.0

        # --- Discharge rate of change ---
        if len(q_history) >= 3:
            row['delta_q_1d'] = q_history[-1] - q_history[-2]
            row['delta_q_2d'] = q_history[-2] - q_history[-3]
        else:
            row['delta_q_1d'] = 0.0
            row['delta_q_2d'] = 0.0

        # --- EWM rainfall (approximate with simple exponential decay) ---
        for span in [3, 7, 14]:
            alpha = 2.0 / (span + 1)
            ewm_val = 0.0
            for j in range(min(span * 3, len(rain_history) - 1)):
                ewm_val += rain_history[-(j+2)] * ((1 - alpha) ** j)
            ewm_val *= alpha
            row[f'log_rain_ewm_{span}d'] = np.log1p(ewm_val)

        # --- Rain × discharge interaction ---
        rain_yesterday = rain_history[-2] if len(rain_history) >= 2 else 0
        q_yesterday    = q_history[-1] if len(q_history) >= 1 else 0
        row['log_rain_x_qlag1']   = np.log1p(max(0, rain_yesterday * q_yesterday))

        rain_7d = sum(rain_history[max(0, len(rain_history)-8):len(rain_history)-1])
        row['log_rain7d_x_qlag1'] = np.log1p(max(0, rain_7d * q_yesterday))

        # --- Rolling discharge stats ---
        for window in [7, 14]:
            q_window = q_history[-window:] if len(q_history) >= window else q_history
            row[f'log_q_rollmean_{window}d'] = np.log1p(np.mean(q_window))
            row[f'log_q_rollstd_{window}d']  = np.log1p(np.std(q_window) if len(q_window) > 1 else 0)

        # --- Dry spell ---
        dry_count = 0
        for j in range(1, min(60, len(rain_history))):
            if rain_history[-j] < 1.0:
                dry_count += 1
            else:
                break
        row['log_dry_spell'] = np.log1p(dry_count)

        # --- Calendar features ---
        doy = date_i.dayofyear
        month = date_i.month
        row['month_sin'] = np.sin(2 * np.pi * month / 12)
        row['month_cos'] = np.cos(2 * np.pi * month / 12)
        row['doy_sin']   = np.sin(2 * np.pi * doy / 365)
        row['doy_cos']   = np.cos(2 * np.pi * doy / 365)
        row['is_monsoon'] = 1 if 6 <= month <= 9 else 0

        # --- Rainfall std (from climatology) ---
        if 'log_rainfall_std' in feature_cols and rain_std_vals is not None:
            row['log_rainfall_std'] = np.log1p(rain_std_vals[i] if i < len(rain_std_vals) else 0)

        # Build feature DataFrame in the correct column order
        X_row = pd.DataFrame([row])

        # Add any missing columns as 0
        for col in feature_cols:
            if col not in X_row.columns:
                X_row[col] = 0.0

        X_row = X_row[feature_cols]

        # Predict
        pred_log = model.predict(X_row)[0]
        pred_raw = np.expm1(pred_log)
        pred_raw = max(pred_raw, 0.0)  # discharge can't be negative

        predicted_q_log[i] = pred_log
        predicted_q_raw[i] = pred_raw

        # Feed prediction back into history for next day's lags
        q_history.append(pred_raw)

    return predicted_q_raw


# --------------------------------------------------
# RUN ALL SCENARIOS
# --------------------------------------------------
print("\n" + "=" * 60)
print("  Running scenario forecasts...")
print("=" * 60)

scenario_results = {}

# --- A) Climatology-based scenarios ---
for name, multiplier in SCENARIOS.items():
    print(f"\n   📈 Scenario: {name} (rainfall × {multiplier:.2f})...")

    rain_scenario = generate_scenario_rainfall(future_dates, clim, multiplier, seed=42)

    if has_rain_std:
        rain_std_vals = np.zeros(len(future_dates))
        for i, d in enumerate(future_dates):
            doy = min(d.dayofyear, 365)
            row = clim_std[clim_std['doy'] == doy]
            rain_std_vals[i] = row['rainfall_std_mean'].values[0] * multiplier if len(row) > 0 else 0
    else:
        rain_std_vals = None

    q_pred = forecast_scenario(
        future_dates, rain_scenario, rain_std_vals,
        xgb_final, feature_cols, df_master
    )

    scenario_results[name] = {
        'dates':    future_dates,
        'rainfall': rain_scenario,
        'discharge': q_pred,
    }

    future_df = pd.DataFrame({'date': future_dates, 'Q': q_pred, 'P': rain_scenario})
    future_df['year'] = future_df['date'].dt.year
    annual = future_df.groupby('year').agg(
        mean_Q=('Q', 'mean'), max_Q=('Q', 'max'),
        total_P=('P', 'sum'),
    )
    scenario_results[name]['annual'] = annual

    print(f"   Annual summary:")
    for yr, row in annual.iterrows():
        print(f"     {yr}: mean Q={row['mean_Q']:.1f} m³/s, peak Q={row['max_Q']:.1f} m³/s, rain={row['total_P']:.0f} mm")


# --- B) CMIP6-driven scenarios (if data available) ---
CMIP6_PROCESSED = BASE_DIR / "cmip6_data" / "processed"
cmip6_ssps = ['ssp245', 'ssp585']
cmip6_labels = {'ssp245': 'CMIP6 SSP2-4.5', 'ssp585': 'CMIP6 SSP5-8.5'}

for ssp in cmip6_ssps:
    cmip6_file = CMIP6_PROCESSED / f"forecast_input_{ssp}.csv"
    if not cmip6_file.exists():
        continue

    name = cmip6_labels[ssp]
    print(f"\n   🌍 Scenario: {name} (NEX-GDDP-CMIP6, bias-corrected)...")

    df_cmip = pd.read_csv(cmip6_file)
    df_cmip['date'] = pd.to_datetime(df_cmip['date'])

    # Filter to forecast period
    mask = (df_cmip['date'] >= FORECAST_START) & (df_cmip['date'] <= FORECAST_END)
    df_cmip = df_cmip[mask].sort_values('date').reset_index(drop=True)

    if len(df_cmip) < 365:
        print(f"      ⚠️  Only {len(df_cmip)} days — skipping (need at least 1 year)")
        continue

    cmip_dates = pd.DatetimeIndex(df_cmip['date'])
    cmip_rain  = df_cmip['rainfall_max_mm'].values

    # Rainfall std if available
    if 'rainfall_std_mm' in df_cmip.columns:
        cmip_rain_std = df_cmip['rainfall_std_mm'].values
    else:
        cmip_rain_std = None

    q_pred = forecast_scenario(
        cmip_dates, cmip_rain, cmip_rain_std,
        xgb_final, feature_cols, df_master
    )

    scenario_results[name] = {
        'dates':    cmip_dates,
        'rainfall': cmip_rain,
        'discharge': q_pred,
    }

    future_df = pd.DataFrame({'date': cmip_dates, 'Q': q_pred, 'P': cmip_rain})
    future_df['year'] = future_df['date'].dt.year
    annual = future_df.groupby('year').agg(
        mean_Q=('Q', 'mean'), max_Q=('Q', 'max'),
        total_P=('P', 'sum'),
    )
    scenario_results[name]['annual'] = annual

    print(f"   Annual summary:")
    for yr, row in annual.iterrows():
        print(f"     {yr}: mean Q={row['mean_Q']:.1f} m³/s, peak Q={row['max_Q']:.1f} m³/s, rain={row['total_P']:.0f} mm")

n_cmip = sum(1 for k in scenario_results if 'CMIP6' in k)
if n_cmip > 0:
    print(f"\n   ✅ {n_cmip} CMIP6 scenarios loaded and forecasted")
else:
    print(f"\n   ℹ️  No CMIP6 data found in {CMIP6_PROCESSED}")
    print(f"      Run cmip6_download.py first to add climate-model-driven scenarios")
    print(f"   Annual summary:")
    for yr, row in annual.iterrows():
        print(f"     {yr}: mean Q={row['mean_Q']:.1f} m³/s, peak Q={row['max_Q']:.1f} m³/s, rain={row['total_P']:.0f} mm")

    scenario_results[name]['annual'] = annual


# --------------------------------------------------
# PLOT 1: Full forecast — all scenarios
# --------------------------------------------------
print("\n📸 Generating plots...")

colors = {
    'Average': '#2E86C1', 'Wet (+20%)': '#27AE60',
    'Dry (-20%)': '#E67E22', 'Extreme (+40%)': '#E74C3C',
    'CMIP6 SSP2-4.5': '#8E44AD', 'CMIP6 SSP5-8.5': '#C0392B',
}

fig, ax = plt.subplots(figsize=(16, 6))

# Historical context (last 2 years)
hist_tail = df_master.tail(730).copy()
ax.plot(hist_tail['date'], hist_tail['q_upstream_mk'],
        color='#1a1a1a', linewidth=0.7, alpha=0.6, label='Historical observed')

for name, data in scenario_results.items():
    ax.plot(data['dates'], data['discharge'],
            color=colors.get(name, 'gray'), linewidth=0.8, alpha=0.8, label=name)

ax.axvline(FORECAST_START, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(FORECAST_START, ax.get_ylim()[1] * 0.95, ' Forecast →',
        fontsize=9, color='gray', va='top')

ax.set_title("Kabini River Discharge — Scenario Forecasts (2025–2030)")
ax.set_xlabel("Date")
ax.set_ylabel("Discharge (m³/s)")
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FORECAST_DIR / "scenario_forecast_full.png", dpi=150)
plt.close()
print(f"   📸 scenario_forecast_full.png")

# --------------------------------------------------
# PLOT 2: Single year zoom (2026 monsoon)
# --------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2.5]})

zoom_start = pd.Timestamp('2026-04-01')
zoom_end   = pd.Timestamp('2026-11-30')

# Rainfall panel (inverted, like hyetograph)
ax_rain = axes[0]
avg_data = scenario_results['Average']
zoom_mask = (avg_data['dates'] >= zoom_start) & (avg_data['dates'] <= zoom_end)
ax_rain.bar(avg_data['dates'][zoom_mask], avg_data['rainfall'][zoom_mask],
            color='#3B8BD4', alpha=0.6, width=1)
ax_rain.invert_yaxis()
ax_rain.set_ylabel("Rainfall (mm)")
ax_rain.set_title("2026 Monsoon Season — Rainfall Input (Average Scenario) & Discharge Forecasts")
ax_rain.set_xlim(zoom_start, zoom_end)

# Discharge panel
ax_q = axes[1]
for name, data in scenario_results.items():
    zm = (data['dates'] >= zoom_start) & (data['dates'] <= zoom_end)
    ax_q.plot(data['dates'][zm], data['discharge'][zm],
              color=colors.get(name, 'gray'), linewidth=1.2, alpha=0.85, label=name)

ax_q.set_xlabel("Date")
ax_q.set_ylabel("Discharge (m³/s)")
ax_q.legend(fontsize=9)
ax_q.set_xlim(zoom_start, zoom_end)
ax_q.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(FORECAST_DIR / "scenario_monsoon_2026.png", dpi=150)
plt.close()
print(f"   📸 scenario_monsoon_2026.png")

# --------------------------------------------------
# PLOT 3: Annual peak discharge comparison
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

# Historical annual peaks
hist_annual_max = df_master.groupby(df_master['date'].dt.year)['q_upstream_mk'].max()
ax.bar(hist_annual_max.index, hist_annual_max.values, color='#95A5A6', alpha=0.6, label='Historical')

# Scenario annual peaks
bar_width = 0.18
years = sorted(scenario_results['Average']['annual'].index)
for i, (name, data) in enumerate(scenario_results.items()):
    peaks = [data['annual'].loc[yr, 'max_Q'] if yr in data['annual'].index else 0 for yr in years]
    offsets = [yr + (i - len(SCENARIOS)/2) * bar_width for yr in years]
    ax.bar(offsets, peaks, bar_width, color=colors.get(name, 'gray'), alpha=0.8, label=name)

ax.set_title("Annual Peak Discharge — Historical vs Forecast Scenarios")
ax.set_xlabel("Year")
ax.set_ylabel("Peak Discharge (m³/s)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15, axis='y')
plt.tight_layout()
plt.savefig(FORECAST_DIR / "annual_peak_comparison.png", dpi=150)
plt.close()
print(f"   📸 annual_peak_comparison.png")

# --------------------------------------------------
# PLOT 4: Envelope plot (uncertainty band)
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

# Monthly means across scenarios for the envelope
envelope_df = pd.DataFrame(index=future_dates)

for name, data in scenario_results.items():
    # Map the discharge values to their specific dates
    envelope_df[name] = pd.Series(data['discharge'], index=data['dates'])
q_min = envelope_df.min(axis=1).values
q_max = envelope_df.max(axis=1).values
q_avg = envelope_df['Average'].values

# Smooth with 30-day rolling for readability
window = 30
q_min_smooth = pd.Series(q_min).rolling(window, min_periods=1).mean().values
q_max_smooth = pd.Series(q_max).rolling(window, min_periods=1).mean().values
q_avg_smooth = pd.Series(q_avg).rolling(window, min_periods=1).mean().values

ax.fill_between(future_dates, q_min_smooth, q_max_smooth,
                alpha=0.25, color='#2E86C1', label='Scenario range (Dry → Extreme)')
ax.plot(future_dates, q_avg_smooth,
        color='#2E86C1', linewidth=1.5, label='Average scenario (30-day smooth)')
ax.set_title("Forecast Uncertainty Envelope (2025–2030)")
ax.set_xlabel("Date")
ax.set_ylabel("Discharge (m³/s)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FORECAST_DIR / "forecast_envelope.png", dpi=150)
plt.close()
print(f"   📸 forecast_envelope.png")


# --------------------------------------------------
# SAVE FORECAST DATA
# --------------------------------------------------
print("\n💾 Saving forecast data...")

for name, data in scenario_results.items():
    out_df = pd.DataFrame({
        'date': data['dates'],
        'rainfall_mm': data['rainfall'],
        'discharge_m3s': data['discharge'],
    })
    safe_name = name.replace(' ', '_').replace('+', 'plus').replace('-', 'minus').replace('(', '').replace(')', '').replace('%', 'pct')
    out_df.to_csv(FORECAST_DIR / f"forecast_{safe_name}.csv", index=False)

# Combined summary
summary_rows = []
for name, data in scenario_results.items():
    for yr, row in data['annual'].iterrows():
        summary_rows.append({
            'scenario': name,
            'year': yr,
            'mean_discharge_m3s': round(row['mean_Q'], 2),
            'peak_discharge_m3s': round(row['max_Q'], 2),
            'total_rainfall_mm': round(row['total_P'], 1),
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(FORECAST_DIR / "forecast_annual_summary.csv", index=False)
print(f"   💾 forecast_annual_summary.csv")

print(f"\n✅ Forecasting complete. Outputs in {FORECAST_DIR}")
print("=" * 60)