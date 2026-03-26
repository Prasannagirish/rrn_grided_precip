import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 55)
print("  PHASE 2.5: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 55)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
EDA_DIR  = BASE_DIR / "eda_outputs"
EDA_DIR.mkdir(exist_ok=True)

chirps_path = BASE_DIR / "data/chirps_kabini_daily.csv"
master_path = BASE_DIR / "data/master_dataset.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df_chirps = pd.read_csv(chirps_path)
df_master = pd.read_csv(master_path)

df_chirps['date'] = pd.to_datetime(df_chirps['date'])
df_master['date'] = pd.to_datetime(df_master['date'])

# --------------------------------------------------
# Single, consistent rainfall column detection.
# Phase 2 renames the raw column to 'rainfall_max_mm',
# so we look for that first. If running EDA independently,
# fall back to keyword detection — but NEVER re-hardcode
# 'rainfall_max_mm' mid-script after detecting a different name.
# --------------------------------------------------
AGREED_RAIN_COL = 'rainfall_max_mm'

if AGREED_RAIN_COL in df_master.columns:
    rain_col = AGREED_RAIN_COL
    print(f"✅ Rain column: '{rain_col}'")
else:
    _candidates = [
        c for c in df_master.columns
        if any(kw in c for kw in ('precip', 'rain', 'chirps', 'prcp', 'rf'))
        and c != 'date'
    ]
    if not _candidates:
        raise ValueError(
            f"❌ No rainfall column found in master dataset. "
            f"Columns present: {list(df_master.columns)}"
        )
    rain_col = _candidates[0]
    print(f"⚠️  'rainfall_max_mm' not found — using '{rain_col}' instead. "
          f"Re-run phase2_clean.py to standardise column names.")

# --------------------------------------------------
# BASIC INFO
# --------------------------------------------------
def basic_info(df, name):
    print(f"\n📊 DATASET: {name}")
    print("-" * 40)
    print(df.shape)
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isna().sum())

basic_info(df_chirps, "CHIRPS")
basic_info(df_master, "MASTER DATASET")

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def save_plot(filename):
    filepath = EDA_DIR / filename
    plt.savefig(filepath)
    print(f"📸 Saved: {filepath}")
    plt.close()

def plot_timeseries(df, cols, title, filename):
    plt.figure()
    for col in cols:
        plt.plot(df['date'], df[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    save_plot(filename)

def plot_hist(df, col, title, filename):
    plt.figure()
    plt.hist(df[col].dropna(), bins=50)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    save_plot(filename)

# --------------------------------------------------
# TIME SERIES PLOTS
# --------------------------------------------------
plot_timeseries(df_master, [rain_col],
                "Rainfall Time Series", "rainfall_timeseries.png")

plot_timeseries(df_master, ['q_upstream_mk'],
                "River Discharge Time Series", "discharge_timeseries.png")

# --------------------------------------------------
# DISTRIBUTION
# --------------------------------------------------
plot_hist(df_master, rain_col,
          "Rainfall Distribution", "rainfall_distribution.png")

plot_hist(df_master, 'q_upstream_mk',
          "Muthankera Flow Distribution", "flow_distribution.png")

# --------------------------------------------------
# CORRELATION
# --------------------------------------------------
numeric_df = df_master.select_dtypes(include='number')
corr = numeric_df.corr()

print("\n🔗 CORRELATION MATRIX:")
print(corr)

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
save_plot("correlation_heatmap.png")

corr.to_csv(EDA_DIR / "correlation_matrix.csv")

# --------------------------------------------------
# SEASONALITY
# --------------------------------------------------
df_master['month'] = df_master['date'].dt.month
monthly_avg = df_master.groupby('month').mean(numeric_only=True)

plt.figure()
plt.plot(monthly_avg.index, monthly_avg[rain_col], marker='o')
plt.title("Monthly Avg Rainfall")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.tight_layout()
save_plot("monthly_rainfall.png")

plt.figure()
plt.plot(monthly_avg.index, monthly_avg['q_upstream_mk'], marker='o')
plt.title("Monthly Avg Discharge (Muthankera)")
plt.xlabel("Month")
plt.ylabel("Flow")
plt.tight_layout()
save_plot("monthly_discharge.png")

# --------------------------------------------------
# EXTREME EVENTS
# --------------------------------------------------
print("\n⚠ EXTREME EVENTS")

top_rain = df_master.nlargest(5, rain_col)[['date', rain_col]]
top_flow = df_master.nlargest(5, 'q_upstream_mk')[['date', 'q_upstream_mk']]

print("\nTop 5 Rainfall Days:")
print(top_rain)

print("\nTop 5 Flood Days (Muthankera):")
print(top_flow)

top_rain.to_csv(EDA_DIR / "top_rainfall_events.csv", index=False)
top_flow.to_csv(EDA_DIR / "top_flood_events.csv", index=False)

# --------------------------------------------------
# Lag correlation — use the single `rain_col` resolved above.
# --------------------------------------------------
print("\n🔁 LAG CORRELATION ANALYSIS")
print(f"   Using rain column: '{rain_col}'")

lag_results = []

for lag in range(1, 15):
    shifted  = df_master[rain_col].shift(lag)
    corr_val = shifted.corr(df_master['q_upstream_mk'])
    lag_results.append((lag, corr_val))
    print(f"Lag {lag} days → Correlation: {corr_val:.4f}")

lag_df = pd.DataFrame(lag_results, columns=['lag_days', 'correlation'])
lag_df.to_csv(EDA_DIR / "lag_correlation.csv", index=False)

plt.figure()
plt.plot(lag_df['lag_days'], lag_df['correlation'], marker='o')
plt.title("Lag vs Correlation (Rainfall → Discharge)")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.grid()
plt.tight_layout()
save_plot("lag_correlation.png")

# --------------------------------------------------
# SUMMARY STATS
# --------------------------------------------------
df_master.describe().to_csv(EDA_DIR / "summary_stats.csv")

print("\n✅ All EDA outputs saved in:", EDA_DIR)
print("=" * 55)