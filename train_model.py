import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  optuna not installed — using manual hyperparameters.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  lightgbm not installed — ensemble will use XGBoost only.")

try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
    tf.get_logger().setLevel('ERROR')
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  tensorflow not installed — LSTM comparison will be skipped.")
    print("   Install with: pip install tensorflow")

print("=" * 60)
print("  PHASE 4: MODEL TRAINING (XGBoost + LSTM Comparison)")
print("=" * 60)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
MODEL_DIR     = BASE_DIR / "model_outputs"
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
# --------------------------------------------------
TARGET = 'log_q'

DROP_COLS = [
    'date', 'q_upstream_mk', 'log_q',
    'log_rainfall',
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

drop_existing = [c for c in DROP_COLS if c in df.columns]
feature_cols  = [c for c in df.columns if c not in drop_existing + [TARGET]]

print(f"\nFeatures used ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  {f}")

X = df[feature_cols]
y = df[TARGET]

# --------------------------------------------------
# 3-WAY TEMPORAL SPLIT
#
# Train : everything before 2012-01-01
# Val   : 2012-01-01 → 2014-12-31 (tuning + early stopping)
# Test  : 2015-01-01 onwards      (final evaluation, untouched)
# --------------------------------------------------
val_start  = pd.Timestamp('2012-01-01')
test_start = pd.Timestamp('2015-01-01')

train_mask = df['date'] < val_start
val_mask   = (df['date'] >= val_start) & (df['date'] < test_start)
test_mask  = df['date'] >= test_start

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain: {train_mask.sum()} rows  ({df.loc[train_mask,'date'].min().date()} → {df.loc[train_mask,'date'].max().date()})")
print(f"Val  : {val_mask.sum()}  rows  ({df.loc[val_mask,  'date'].min().date()} → {df.loc[val_mask,  'date'].max().date()})")
print(f"Test : {test_mask.sum()}  rows  ({df.loc[test_mask, 'date'].min().date()} → {df.loc[test_mask, 'date'].max().date()})")


# --------------------------------------------------
# HELPER: Metrics in real (m³/s) space
# --------------------------------------------------
def compute_metrics(y_true_log, y_pred_log, label=""):
    y_true = np.expm1(np.array(y_true_log, dtype=float))
    y_pred = np.expm1(np.array(y_pred_log, dtype=float))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    if label:
        print(f"   [{label}]  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  NSE={nse:.4f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nse': nse}


# ══════════════════════════════════════════════════════════
#  PART A: XGBoost ENSEMBLE (same as before)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART A: XGBoost Ensemble")
print("=" * 60)

# --- Optuna tuning ---
if HAS_OPTUNA:
    print("\n🔍 Tuning XGBoost with Optuna (60 trials)...")
    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'random_state': 42, 'n_jobs': -1,
        }
        m = XGBRegressor(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        return mean_squared_error(y_val, m.predict(X_val))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    best_params = {**study.best_params, 'n_estimators': 1000, 'random_state': 42, 'n_jobs': -1}
    print(f"   Best val MSE: {study.best_value:.6f}")
else:
    best_params = {
        'n_estimators': 1000, 'learning_rate': 0.04, 'max_depth': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
    }

# --- Train-only models for ensemble weight optimisation ---
# FIX: We need val predictions from models that have NOT seen
# val data. Train lightweight copies on train-only first.
print("\n🔗 Training train-only models for ensemble weight selection...")

xgb_trainonly = XGBRegressor(**best_params)
xgb_trainonly.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
xgb_val_pred = xgb_trainonly.predict(X_val)

ridge_trainonly = Ridge(alpha=1.0)
ridge_trainonly.fit(X_train, y_train)
ridge_val_pred = ridge_trainonly.predict(X_val)

if HAS_LGB:
    lgb_trainonly = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=best_params.get('learning_rate', 0.04),
        max_depth=best_params.get('max_depth', 5), subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        min_child_samples=best_params.get('min_child_weight', 5),
        reg_alpha=best_params.get('reg_alpha', 0.1), reg_lambda=best_params.get('reg_lambda', 1.0),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_trainonly.fit(X_train, y_train)
    lgb_val_pred = lgb_trainonly.predict(X_val)

    best_w, best_val_mse = None, float('inf')
    for w_xgb in np.arange(0.3, 0.8, 0.05):
        for w_lgb in np.arange(0.1, 0.6, 0.05):
            w_ridge = 1.0 - w_xgb - w_lgb
            if w_ridge < 0: continue
            mse = mean_squared_error(y_val, w_xgb*xgb_val_pred + w_lgb*lgb_val_pred + w_ridge*ridge_val_pred)
            if mse < best_val_mse:
                best_val_mse = mse
                best_w = (w_xgb, w_lgb, w_ridge)
    print(f"   Ensemble weights: XGB={best_w[0]:.2f}  LGB={best_w[1]:.2f}  Ridge={best_w[2]:.2f}")
else:
    best_w, best_val_mse = None, float('inf')
    for w_xgb in np.arange(0.5, 1.0, 0.05):
        w_ridge = 1.0 - w_xgb
        mse = mean_squared_error(y_val, w_xgb*xgb_val_pred + w_ridge*ridge_val_pred)
        if mse < best_val_mse:
            best_val_mse = mse
            best_w = (w_xgb, w_ridge)
    print(f"   Ensemble weights: XGB={best_w[0]:.2f}  Ridge={best_w[1]:.2f}")

# --- Now retrain final models on train+val for maximum data ---
# FIX: eval_set uses val (not test) — test must stay invisible.
print("\n🚀 Training final models on train+val...")
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

xgb_model = XGBRegressor(**best_params)
xgb_model.fit(X_trainval, y_trainval, verbose=100)
xgb_pred = xgb_model.predict(X_test)
print("\n📊 XGBoost:")
xgb_metrics = compute_metrics(y_test.values, xgb_pred, "XGBoost")

if HAS_LGB:
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=best_params.get('learning_rate', 0.04),
        max_depth=best_params.get('max_depth', 5), subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        min_child_samples=best_params.get('min_child_weight', 5),
        reg_alpha=best_params.get('reg_alpha', 0.1), reg_lambda=best_params.get('reg_lambda', 1.0),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(X_trainval, y_trainval)
    lgb_pred = lgb_model.predict(X_test)
    print("📊 LightGBM:")
    lgb_metrics = compute_metrics(y_test.values, lgb_pred, "LightGBM")
else:
    lgb_pred = None

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_trainval, y_trainval)
ridge_pred = ridge_model.predict(X_test)
print("📊 Ridge:")
ridge_metrics = compute_metrics(y_test.values, ridge_pred, "Ridge")

# --- Apply locked ensemble weights to test predictions ---
if HAS_LGB:
    ensemble_pred = best_w[0]*xgb_pred + best_w[1]*lgb_pred + best_w[2]*ridge_pred
else:
    ensemble_pred = best_w[0]*xgb_pred + best_w[1]*ridge_pred

print("\n📊 XGB Ensemble:")
ens_metrics = compute_metrics(y_test.values, ensemble_pred, "Ensemble")


# ══════════════════════════════════════════════════════════
#  PART B: LSTM MODEL
#
#  Why LSTM for hydrology:
#  - River discharge is inherently sequential — today's
#    flow depends on the *sequence* of rain and flow over
#    the past days, not just individual lag values.
#  - LSTM can learn temporal dependencies that tree models
#    can only approximate through hand-crafted lag features.
#  - Bidirectional variant removed — backward pass would
#    see future context within the window, which is
#    conceptually invalid for a forecasting task.
#
#  Architecture:
#    Input  → [lookback × features] sliding windows
#    Layer1 → LSTM (64 units, return sequences)
#    Drop   → 0.3
#    Layer2 → LSTM (32 units)
#    Drop   → 0.2
#    Dense  → 16 → 1 (linear output = log_q)
# ══════════════════════════════════════════════════════════
if HAS_TF:
    print("\n" + "=" * 60)
    print("  PART B: LSTM Model")
    print("=" * 60)

    LOOKBACK = 14  # days of history as input sequence

    # --- Scale features (critical for LSTM) ---
    # FIX: fit scaler on TRAINING data only — fitting on the
    # full dataset leaks test-set distribution into the inputs.
    scaler = StandardScaler()
    scaler.fit(X[train_mask])              # learn mean/std from train only
    X_all_scaled = scaler.transform(X)     # apply to everything

    # --- Create sliding window sequences ---
    def create_sequences(X_arr, y_arr, lookback):
        Xs, ys, indices = [], [], []
        for i in range(lookback, len(X_arr)):
            Xs.append(X_arr[i - lookback:i])
            ys.append(y_arr[i])
            indices.append(i)
        return np.array(Xs), np.array(ys), np.array(indices)

    print(f"   Creating {LOOKBACK}-day sliding windows...")

    # We need to be careful with the temporal split:
    # The first LOOKBACK rows of each set use data from the
    # previous set, which is fine (it's past data), but we
    # need to build sequences from the full scaled array
    # and then split by date index.

    X_seq, y_seq, idx_seq = create_sequences(X_all_scaled, y.values, LOOKBACK)

    # Map back to date-based masks
    seq_dates = df['date'].values[idx_seq]
    seq_train = seq_dates < val_start
    seq_val   = (seq_dates >= val_start) & (seq_dates < test_start)
    seq_test  = seq_dates >= test_start

    X_lstm_train, y_lstm_train = X_seq[seq_train], y_seq[seq_train]
    X_lstm_val,   y_lstm_val   = X_seq[seq_val],   y_seq[seq_val]
    X_lstm_test,  y_lstm_test  = X_seq[seq_test],   y_seq[seq_test]

    print(f"   LSTM Train: {len(X_lstm_train)}  Val: {len(X_lstm_val)}  Test: {len(X_lstm_test)}")
    print(f"   Input shape: {X_lstm_train.shape}  (samples, timesteps, features)")

    # --- Build model ---
    n_features = X_lstm_train.shape[2]

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ])

    lstm_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
    )

    lstm_model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    print("\n🚀 Training LSTM...")
    history = lstm_model.fit(
        X_lstm_train, y_lstm_train,
        validation_data=(X_lstm_val, y_lstm_val),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Predict ---
    lstm_pred = lstm_model.predict(X_lstm_test, verbose=0).flatten()

    print("\n📊 LSTM:")
    lstm_metrics = compute_metrics(y_lstm_test, lstm_pred, "LSTM")

    # --- Training history plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], label='Train loss', linewidth=0.9)
    ax.plot(history.history['val_loss'], label='Val loss', linewidth=0.9)
    ax.set_title("LSTM Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "lstm_training_history.png", dpi=150)
    plt.close()
    print(f"📸 Saved: lstm_training_history.png")

    # --- Hybrid: LSTM + XGBoost average ---
    # Align predictions (LSTM test may be slightly shorter
    # due to lookback window consuming first LOOKBACK rows)
    lstm_test_dates = seq_dates[seq_test]

    # Find common test dates
    xgb_test_dates  = df.loc[test_mask, 'date'].values
    xgb_test_real   = np.expm1(y_test.values)
    xgb_ens_real    = np.expm1(ensemble_pred)

    # Build aligned arrays
    lstm_date_set = set(pd.to_datetime(lstm_test_dates).date)
    common_idx_xgb  = []
    common_idx_lstm = []
    xgb_dates_list  = [pd.Timestamp(d).date() for d in xgb_test_dates]
    lstm_dates_list = [pd.Timestamp(d).date() for d in lstm_test_dates]

    for i, d in enumerate(xgb_dates_list):
        if d in lstm_date_set:
            j = lstm_dates_list.index(d)
            common_idx_xgb.append(i)
            common_idx_lstm.append(j)

    common_idx_xgb  = np.array(common_idx_xgb)
    common_idx_lstm = np.array(common_idx_lstm)

    # Aligned arrays on common dates
    common_dates    = np.array(xgb_test_dates)[common_idx_xgb]
    common_obs      = xgb_test_real[common_idx_xgb]
    common_xgb      = xgb_ens_real[common_idx_xgb]
    common_lstm     = np.expm1(lstm_pred[common_idx_lstm])

    # Hybrid = weighted average (optimise on val if possible, else 50/50)
    hybrid_pred = 0.5 * common_xgb + 0.5 * common_lstm

    # Compute hybrid metrics
    hybrid_rmse = np.sqrt(mean_squared_error(common_obs, hybrid_pred))
    hybrid_mae  = mean_absolute_error(common_obs, hybrid_pred)
    hybrid_r2   = r2_score(common_obs, hybrid_pred)
    hybrid_nse  = 1 - np.sum((common_obs - hybrid_pred)**2) / np.sum((common_obs - common_obs.mean())**2)
    hybrid_metrics = {'rmse': hybrid_rmse, 'mae': hybrid_mae, 'r2': hybrid_r2, 'nse': hybrid_nse}

    print(f"\n📊 Hybrid (XGB+LSTM):")
    print(f"   [Hybrid]  RMSE={hybrid_rmse:.2f}  MAE={hybrid_mae:.2f}  R²={hybrid_r2:.4f}  NSE={hybrid_nse:.4f}")

else:
    lstm_metrics = None
    hybrid_metrics = None
    common_dates = None


# ══════════════════════════════════════════════════════════
#  PART C: COMPREHENSIVE COMPARISON & PLOTS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART C: Model Comparison")
print("=" * 60)

# --- Use ensemble as primary prediction ---
y_pred_log  = ensemble_pred
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test.values)
test_dates  = df.loc[test_mask, 'date']
test_months = test_dates.dt.month

rmse = ens_metrics['rmse']
mae  = ens_metrics['mae']
r2   = ens_metrics['r2']
nse  = ens_metrics['nse']

# --- Seasonal & peak diagnostics ---
monsoon_mask = test_months.between(6, 9).values
dry_mask     = ~monsoon_mask

print(f"\n📊 SEASONAL BREAKDOWN (XGB Ensemble):")
if monsoon_mask.sum() > 0:
    compute_metrics(y_test.values[monsoon_mask], y_pred_log[monsoon_mask], "Monsoon")
if dry_mask.sum() > 0:
    compute_metrics(y_test.values[dry_mask], y_pred_log[dry_mask], "Dry season")

p90 = np.percentile(y_test_real, 90)
peak_mask = y_test_real > p90
if peak_mask.sum() > 0:
    peak_obs  = y_test_real[peak_mask]
    peak_pred = y_pred_real[peak_mask]
    peak_bias = np.mean(peak_pred - peak_obs) / np.mean(peak_obs) * 100
    peak_rmse = np.sqrt(mean_squared_error(peak_obs, peak_pred))
    print(f"\n📊 PEAK FLOW (>{p90:.0f} m³/s, n={peak_mask.sum()}):")
    print(f"   Peak RMSE: {peak_rmse:.2f}   Bias: {peak_bias:+.1f}%")

# ===========================================================
#  PLOT 1: Observed vs Predicted time-series (XGB Ensemble)
# ===========================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates.values, y_test_real, label='Observed', alpha=0.8, linewidth=0.9)
ax.plot(test_dates.values, y_pred_real, label='XGB Ensemble', alpha=0.8, linewidth=0.9)
ax.fill_between(test_dates.values, y_test_real, y_pred_real, alpha=0.1, color='red')
ax.set_title(f"Observed vs Predicted — XGB Ensemble\nRMSE={rmse:.1f}  MAE={mae:.1f}  NSE={nse:.3f}")
ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)"); ax.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "obs_vs_pred_timeseries.png", dpi=150)
plt.close()
print(f"\n📸 Saved: obs_vs_pred_timeseries.png")

# ===========================================================
#  PLOT 2: Scatter
# ===========================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test_real, y_pred_real, alpha=0.3, s=5, c='steelblue')
if peak_mask.sum() > 0:
    ax.scatter(y_test_real[peak_mask], y_pred_real[peak_mask], alpha=0.6, s=12, c='coral', label=f'Peak (>{p90:.0f})')
max_val = max(y_test_real.max(), y_pred_real.max()) * 1.05
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
ax.set_title(f"Scatter — XGB Ensemble (R²={r2:.4f})")
ax.set_xlabel("Observed (m³/s)"); ax.set_ylabel("Predicted (m³/s)"); ax.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "scatter_obs_vs_pred.png", dpi=150)
plt.close()
print(f"📸 Saved: scatter_obs_vs_pred.png")

# ===========================================================
#  PLOT 3: Feature Importance
# ===========================================================
importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 7))
importance.head(20).sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title("Top 20 Feature Importances (XGBoost)"); ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
plt.close()
print(f"📸 Saved: feature_importance.png")

# ===========================================================
#  PLOT 4: Monthly NSE
# ===========================================================
fig, ax = plt.subplots(figsize=(10, 4))
monthly_nse = []
for m in range(1, 13):
    mask_m = (test_months == m).values
    if mask_m.sum() < 10:
        monthly_nse.append(np.nan); continue
    obs_m, pred_m = y_test_real[mask_m], y_pred_real[mask_m]
    ss_res = np.sum((obs_m - pred_m)**2)
    ss_tot = np.sum((obs_m - obs_m.mean())**2)
    monthly_nse.append(1 - ss_res/ss_tot if ss_tot > 0 else np.nan)

months_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
colors = ['#ef9f27' if 6 <= m <= 9 else '#3b8bd4' for m in range(1, 13)]
ax.bar(months_labels, monthly_nse, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_title("Monthly NSE — Monsoon (orange) vs Dry (blue)"); ax.set_ylabel("NSE")
min_nse = min((n for n in monthly_nse if n is not None and not np.isnan(n)), default=0)
ax.set_ylim(min(min_nse - 0.1, -0.3), 1.05)
plt.tight_layout()
plt.savefig(MODEL_DIR / "monthly_nse.png", dpi=150)
plt.close()
print(f"📸 Saved: monthly_nse.png")

# ===========================================================
#  PLOT 5: Residual Analysis
# ===========================================================
residuals = y_test_real - y_pred_real
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_real, residuals, alpha=0.2, s=5, c='steelblue')
axes[0].axhline(0, color='red', linewidth=0.8, linestyle='--')
axes[0].set_xlabel("Predicted (m³/s)"); axes[0].set_ylabel("Residual"); axes[0].set_title("Residuals vs Predicted")
axes[1].hist(residuals, bins=60, edgecolor='white', linewidth=0.3, color='steelblue')
axes[1].axvline(0, color='red', linewidth=0.8, linestyle='--')
axes[1].set_xlabel("Residual (m³/s)"); axes[1].set_ylabel("Freq"); axes[1].set_title(f"Residuals (mean={np.mean(residuals):.1f})")
plt.tight_layout()
plt.savefig(MODEL_DIR / "residual_analysis.png", dpi=150)
plt.close()
print(f"📸 Saved: residual_analysis.png")

# ===========================================================
#  PLOT 6 (NEW): XGB vs LSTM time-series overlay
# ===========================================================
if HAS_TF and common_dates is not None and len(common_dates) > 0:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(common_dates, common_obs,  label='Observed', color='#1a1a1a', linewidth=1, alpha=0.85)
    ax.plot(common_dates, common_xgb,  label=f'XGB Ensemble (NSE={ens_metrics["nse"]:.3f})', color='#2E86C1', linewidth=0.9, alpha=0.8)
    ax.plot(common_dates, common_lstm, label=f'LSTM (NSE={lstm_metrics["nse"]:.3f})', color='#E74C3C', linewidth=0.9, alpha=0.8)
    ax.plot(common_dates, hybrid_pred, label=f'Hybrid XGB+LSTM (NSE={hybrid_nse:.3f})', color='#27AE60', linewidth=1.1, alpha=0.85, linestyle='--')
    ax.set_title("Model Comparison — Observed vs Predictions (Test Period)")
    ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "comparison_timeseries.png", dpi=150)
    plt.close()
    print(f"📸 Saved: comparison_timeseries.png")

# ===========================================================
#  PLOT 7 (NEW): Scatter comparison — 3 panels
# ===========================================================
if HAS_TF and common_dates is not None and len(common_dates) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    max_v = max(common_obs.max(), common_xgb.max(), common_lstm.max()) * 1.05

    for ax, pred, name, color, metrics in [
        (axes[0], common_xgb,  "XGB Ensemble", '#2E86C1', ens_metrics),
        (axes[1], common_lstm, "LSTM",         '#E74C3C', lstm_metrics),
        (axes[2], hybrid_pred, "Hybrid",       '#27AE60', hybrid_metrics),
    ]:
        ax.scatter(common_obs, pred, alpha=0.3, s=8, c=color)
        ax.plot([0, max_v], [0, max_v], 'k--', alpha=0.4)
        ax.set_title(f'{name}\nNSE={metrics["nse"]:.4f}  RMSE={metrics["rmse"]:.1f}')
        ax.set_xlabel("Observed (m³/s)"); ax.set_ylabel("Predicted (m³/s)")
        ax.set_xlim(0, max_v); ax.set_ylim(0, max_v)
        ax.set_aspect('equal')

    plt.suptitle("Scatter Comparison: XGB vs LSTM vs Hybrid", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "comparison_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Saved: comparison_scatter.png")

# ===========================================================
#  PLOT 8 (NEW): Bar chart — metric comparison
# ===========================================================
all_models = {'XGBoost': xgb_metrics, 'Ridge': ridge_metrics, 'XGB Ensemble': ens_metrics}
if HAS_LGB:
    all_models['LightGBM'] = lgb_metrics
if HAS_TF and lstm_metrics:
    all_models['LSTM'] = lstm_metrics
if HAS_TF and hybrid_metrics:
    all_models['Hybrid'] = hybrid_metrics

model_names = list(all_models.keys())
metric_names = ['RMSE', 'MAE', 'NSE']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
bar_colors = ['#3498DB', '#95A5A6', '#2E86C1', '#1ABC9C', '#E74C3C', '#27AE60']

for ax, metric_key, metric_label in zip(axes, ['rmse', 'mae', 'nse'], metric_names):
    vals = [all_models[m][metric_key] for m in model_names]
    bars = ax.barh(model_names, vals, color=bar_colors[:len(model_names)], edgecolor='white', linewidth=0.5)
    ax.set_title(metric_label, fontweight='bold')
    ax.set_xlabel(metric_label)

    # Annotate values
    for bar, val in zip(bars, vals):
        fmt = f"{val:.4f}" if metric_key == 'nse' else f"{val:.1f}"
        ax.text(bar.get_width() + (max(vals) * 0.02), bar.get_y() + bar.get_height()/2,
                fmt, va='center', fontsize=9)

plt.suptitle("Model Comparison — Test Set Metrics (real m³/s)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(MODEL_DIR / "comparison_metrics_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"📸 Saved: comparison_metrics_bar.png")

# ===========================================================
#  PLOT 9 (NEW): Monsoon zoom — 1 peak season
# ===========================================================
if HAS_TF and common_dates is not None and len(common_dates) > 0:
    # Pick the first monsoon season in the test set
    cd = pd.to_datetime(common_dates)
    first_year = cd.year.min()
    monsoon_start = pd.Timestamp(f'{first_year}-06-01')
    monsoon_end   = pd.Timestamp(f'{first_year}-10-31')
    zoom_mask = (cd >= monsoon_start) & (cd <= monsoon_end)

    if zoom_mask.sum() > 30:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cd[zoom_mask], common_obs[zoom_mask],  label='Observed', color='#1a1a1a', linewidth=1.2)
        ax.plot(cd[zoom_mask], common_xgb[zoom_mask],  label='XGB Ensemble', color='#2E86C1', linewidth=0.9, alpha=0.85)
        ax.plot(cd[zoom_mask], common_lstm[zoom_mask], label='LSTM', color='#E74C3C', linewidth=0.9, alpha=0.85)
        ax.plot(cd[zoom_mask], hybrid_pred[zoom_mask], label='Hybrid', color='#27AE60', linewidth=1, linestyle='--', alpha=0.85)
        ax.set_title(f"Monsoon {first_year} — Peak Season Zoom")
        ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "comparison_monsoon_zoom.png", dpi=150)
        plt.close()
        print(f"📸 Saved: comparison_monsoon_zoom.png")


# ===========================================================
#  SAVE PREDICTIONS & COMPARISON TABLE
# ===========================================================
results_df = pd.DataFrame({
    'date':      df.loc[test_mask, 'date'].values,
    'observed':  y_test_real,
    'xgb_ensemble': y_pred_real,
    'residual':  residuals,
})
# Add LSTM column if available (aligned to common dates)
if HAS_TF and common_dates is not None:
    lstm_col = np.full(len(results_df), np.nan)
    hybrid_col = np.full(len(results_df), np.nan)
    for i, idx in enumerate(common_idx_xgb):
        lstm_col[idx] = common_lstm[i]
        hybrid_col[idx] = hybrid_pred[i]
    results_df['lstm'] = lstm_col
    results_df['hybrid'] = hybrid_col

results_df.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
print(f"\n💾 Saved: test_predictions.csv")

# --- Model comparison CSV ---
comp_rows = []
for name, m in all_models.items():
    comp_rows.append({'Model': name, 'RMSE': round(m['rmse'], 2), 'MAE': round(m['mae'], 2),
                      'R2': round(m['r2'], 4), 'NSE': round(m['nse'], 4)})
comparison_df = pd.DataFrame(comp_rows)
comparison_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
print(f"💾 Saved: model_comparison.csv")

print(f"\n{'='*60}")
print(f"  FINAL COMPARISON TABLE")
print(f"{'='*60}")
print(comparison_df.to_string(index=False))

print("\n✅ Training complete.")
print("=" * 60)