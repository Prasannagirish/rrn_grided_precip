import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 55)
print("  PHASE 4: MODEL TRAINING")
print("=" * 55)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
MODEL_DIR   = BASE_DIR / "model_outputs"
MODEL_DIR.mkdir(exist_ok=True)

features_path = BASE_DIR / "data/features_dataset.csv"

# --------------------------------------------------
# LOAD
# --------------------------------------------------
df = pd.read_csv(features_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Feature dataset: {df.shape[0]} rows, {df.shape[1]} cols")

# --------------------------------------------------
# DEFINE FEATURES AND TARGET
#
# TARGET  : log_q  (log-transformed discharge)
# FEATURES: log-space rainfall lags + rolling sums,
#           discharge lags, seasonality, monsoon flag.
#
# Dropped from X:
#   - date            (not a numeric feature)
#   - q_upstream_mk   (raw target, would leak)
#   - log_q           (target itself)
#   - log_rainfall    (today's rain — leaks future info;
#                      at prediction time we don't have
#                      today's total rainfall yet)
#   - raw rainfall_max_mm / rainfall_std_mm / rain_lag_Xd
#     / rain_roll_Xd / rain_rollstd_Xd
#     (replaced by their log versions)
#   - raw q_lag_Xd    (replaced by log_q_lag_Xd)
# --------------------------------------------------
DROP_COLS = [
    'date', 'q_upstream_mk', 'log_q',
    # FIX: drop log_rainfall — it uses today's rain, causing
    #      data leakage (lags are shifted, this is not)
    'log_rainfall',
    # raw features (log counterparts are used instead)
    'rainfall_max_mm', 'rainfall_std_mm',
    'rain_lag_1d', 'rain_lag_2d', 'rain_lag_3d',
    'rain_lag_4d', 'rain_lag_5d', 'rain_lag_6d', 'rain_lag_7d',
    'rain_roll_3d', 'rain_roll_7d', 'rain_roll_14d', 'rain_roll_30d',
    'rain_rollstd_7d', 'rain_rollstd_14d',
    'q_lag_1d', 'q_lag_2d', 'q_lag_3d',
]

TARGET = 'log_q'

# Only drop columns that actually exist in the dataframe
drop_existing = [c for c in DROP_COLS if c in df.columns]
feature_cols = [c for c in df.columns if c not in drop_existing + [TARGET]]
print(f"\nFeatures used ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  {f}")

X = df[feature_cols]
y = df[TARGET]

# --------------------------------------------------
# TIME-BASED TRAIN / TEST SPLIT
# Never shuffle hydrology data — future must not
# leak into training. Use last 2 years as test set.
# --------------------------------------------------
split_date = pd.Timestamp('2018-01-01')
train_mask = df['date'] < split_date
test_mask  = df['date'] >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain: {train_mask.sum()} rows  ({df.loc[train_mask,'date'].min().date()} → {df.loc[train_mask,'date'].max().date()})")
print(f"Test : {test_mask.sum()}  rows  ({df.loc[test_mask, 'date'].min().date()} → {df.loc[test_mask, 'date'].max().date()})")

# --------------------------------------------------
# MODEL: XGBoost
# Good default for tabular hydrology data —
# handles non-linearity, feature interactions,
# and is robust to the remaining skew after log transform.
# --------------------------------------------------
model = XGBRegressor(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1,
)

print("\n🚀 Training XGBoost...")
model.fit(
    X_train, y_train,
    eval_set        = [(X_test, y_test)],
    verbose         = 50,
)

# --------------------------------------------------
# PREDICT & INVERSE TRANSFORM
# Model predicts log_q → expm1 to get real m³/s.
# --------------------------------------------------
y_pred_log  = model.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test.values)

# --------------------------------------------------
# METRICS (in real discharge space, not log)
# --------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae  = mean_absolute_error(y_test_real, y_pred_real)
r2   = r2_score(y_test_real, y_pred_real)

# NSE (Nash-Sutcliffe Efficiency) — standard in hydrology.
# NSE=1 is perfect; NSE=0 means model is no better than
# predicting the mean; NSE<0 means worse than the mean.
nse = 1 - (
    np.sum((y_test_real - y_pred_real) ** 2) /
    np.sum((y_test_real - y_test_real.mean()) ** 2)
)

print("\n📊 TEST SET METRICS (real discharge, m³/s):")
print(f"   RMSE : {rmse:.2f}")
print(f"   MAE  : {mae:.2f}")
print(f"   R²   : {r2:.4f}")
print(f"   NSE  : {nse:.4f}")

# --------------------------------------------------
# PLOT 1: Observed vs Predicted (full test period)
# --------------------------------------------------
test_dates = df.loc[test_mask, 'date'].values

plt.figure(figsize=(14, 5))
plt.plot(test_dates, y_test_real,  label='Observed',  alpha=0.8)
plt.plot(test_dates, y_pred_real,  label='Predicted', alpha=0.8)
plt.title(f"Observed vs Predicted Discharge — Test Set\nRMSE={rmse:.1f}  MAE={mae:.1f}  NSE={nse:.3f}")
plt.xlabel("Date")
plt.ylabel("Discharge (m³/s)")
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "obs_vs_pred_timeseries.png")
plt.close()
print(f"📸 Saved: {MODEL_DIR / 'obs_vs_pred_timeseries.png'}")

# --------------------------------------------------
# PLOT 2: Scatter — Observed vs Predicted
# --------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test_real, y_pred_real, alpha=0.3, s=5)
max_val = max(y_test_real.max(), y_pred_real.max())
plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
plt.title(f"Scatter: Observed vs Predicted\nR²={r2:.4f}")
plt.xlabel("Observed (m³/s)")
plt.ylabel("Predicted (m³/s)")
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "scatter_obs_vs_pred.png")
plt.close()
print(f"📸 Saved: {MODEL_DIR / 'scatter_obs_vs_pred.png'}")

# --------------------------------------------------
# PLOT 3: Feature Importance
# --------------------------------------------------
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(8, 6))
importance.head(15).sort_values().plot(kind='barh')
plt.title("Top 15 Feature Importances (XGBoost gain)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(MODEL_DIR / "feature_importance.png")
plt.close()
print(f"📸 Saved: {MODEL_DIR / 'feature_importance.png'}")

# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------
results_df = pd.DataFrame({
    'date'      : df.loc[test_mask, 'date'].values,
    'observed'  : y_test_real,
    'predicted' : y_pred_real,
    'residual'  : y_test_real - y_pred_real,
})
results_df.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
print(f"💾 Saved: {MODEL_DIR / 'test_predictions.csv'}")

print("\n✅ Training complete.")
print("=" * 55)