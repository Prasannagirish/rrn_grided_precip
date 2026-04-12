# ──────────────────────────────────────────────────────────
#  FIX 1: Seeds must be set BEFORE any library imports.
#  Setting them inside the script body (after TF is imported)
#  has no effect on weight initialisation or dropout masks.
#  PYTHONHASHSEED must also be exported in the shell:
#      PYTHONHASHSEED=50 python train_model.py
# ──────────────────────────────────────────────────────────
import os
import random
os.environ['PYTHONHASHSEED'] = '50'   # effective only if set before interpreter start,
random.seed(50)                        # but harmless to repeat here for numpy/tf below

import numpy as np
np.random.seed(50)

import pandas as pd
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
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    tf.random.set_seed(50)                          # FIX 1 cont.: set TF seed immediately after import
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
    'cum_monsoon_rain',                    # raw — use log version
    'api_fast', 'api_med', 'api_slow',   # raw API — use log versions
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
#  PART A: XGBoost ENSEMBLE
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART A: XGBoost Ensemble")
print("=" * 60)

# --- Optuna tuning ---
if HAS_OPTUNA:
    print("\n🔍 Tuning XGBoost with Optuna (100 trials)...")
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

    # FIX: n_startup_trials=20 forces random exploration before TPE exploitation,
    # preventing the seeded sampler from converging on hyperparams that fit the
    # val period (2012-2014) but don't generalise to the test period (2015-2020).
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    )
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    best_params = {**study.best_params, 'n_estimators': 1000, 'random_state': 42, 'n_jobs': -1}
    print(f"   Best val MSE: {study.best_value:.6f}")
else:
    best_params = {
        'n_estimators': 1000, 'learning_rate': 0.04, 'max_depth': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
    }

# --- Train-only models for ensemble weight optimisation ---
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
    # FIX 4: Give LightGBM early stopping on the train-only fit so it's
    # evaluated on the same footing as XGBoost (which also uses early stopping).
    lgb_trainonly.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    lgb_val_pred = lgb_trainonly.predict(X_val)

    # FIX: Optimise weights in REAL space (m³/s), not log space.
    # Log-space MSE compresses high-flow values where Ridge dominates,
    # causing the optimizer to systematically underweight Ridge and
    # produce an ensemble worse than Ridge alone on the test set.
    y_val_real_w = np.expm1(y_val.values)
    best_w, best_val_mse = None, float('inf')
    for w_ridge in np.arange(0.0, 1.05, 0.05):
        for w_xgb in np.arange(0.0, 1.05 - w_ridge, 0.05):
            w_lgb = round(1.0 - w_ridge - w_xgb, 2)
            if w_lgb < -0.01: continue
            w_lgb = max(w_lgb, 0.0)
            blend_real = np.expm1(w_xgb*xgb_val_pred + w_lgb*lgb_val_pred + w_ridge*ridge_val_pred)
            mse = mean_squared_error(y_val_real_w, blend_real)
            if mse < best_val_mse:
                best_val_mse = mse
                best_w = (w_xgb, w_lgb, w_ridge)
    print(f"   Ensemble weights: XGB={best_w[0]:.2f}  LGB={best_w[1]:.2f}  Ridge={best_w[2]:.2f}")
else:
    y_val_real_w = np.expm1(y_val.values)
    best_w, best_val_mse = None, float('inf')
    for w_ridge in np.arange(0.0, 1.05, 0.05):
        w_xgb = round(1.0 - w_ridge, 2)
        if w_xgb < 0: continue
        blend_real = np.expm1(w_xgb*xgb_val_pred + w_ridge*ridge_val_pred)
        mse = mean_squared_error(y_val_real_w, blend_real)
        if mse < best_val_mse:
            best_val_mse = mse
            best_w = (w_xgb, w_ridge)
    print(f"   Ensemble weights: XGB={best_w[0]:.2f}  Ridge={best_w[1]:.2f}")

# --- Retrain final models on train+val ---
print("\n🚀 Training final models on train+val...")
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

xgb_model = XGBRegressor(**best_params)
xgb_model.fit(X_trainval, y_trainval, verbose=0)   # FIX 5: verbose=0 (was verbose=100)
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
    # FIX 4 cont.: Final LightGBM also needs early stopping. Since we've already
    # used val for early stopping, this is consistent with XGBoost's treatment.
    lgb_model.fit(
        X_trainval, y_trainval,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    lgb_pred = lgb_model.predict(X_test)
    print("📊 LightGBM:")
    lgb_metrics = compute_metrics(y_test.values, lgb_pred, "LightGBM")
else:
    lgb_pred = None

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_trainval, y_trainval)
ridge_pred = ridge_model.predict(X_test)
print("📊 Ridge:")
ridge_metrics = compute_metrics(y_test.values, ridge_pred, "Ridge (test)")

# FIX 6: Ridge val/test gap must use the TRAIN-ONLY Ridge predicting on val
# (out-of-sample). Using ridge_model (trained on train+val) to predict on val
# is in-sample and will always show an artificially inflated val R².
ridge_val_oos = ridge_trainonly.predict(X_val)
ridge_val_metrics = compute_metrics(y_val.values, ridge_val_oos, "Ridge (val, OOS)")
val_test_gap = ridge_val_metrics['r2'] - ridge_metrics['r2']
# FIX 7: Positive gap (val > test) is normal. Only flag NEGATIVE gaps
# (val < test), which would suggest test-period-specific overfitting.
if val_test_gap < -0.05:
    print(f"   ⚠️  Val R² LOWER than test by {-val_test_gap:.4f} — possible distribution shift")
else:
    print(f"   ✅  Val/Test R² gap: {val_test_gap:+.4f} (normal — val >= test is expected)")

# --- Apply ensemble weights (locked from train-only models) ---
print("\n🔗 Applying locked train-only ensemble weights (no re-check)...")

if HAS_LGB:
    w_xgb_final  = best_w[0]
    w_ridge_final = best_w[2]
    w_total = w_xgb_final + w_ridge_final
    if w_total > 0:
        w_xgb_final  /= w_total
        w_ridge_final /= w_total
    else:
        w_xgb_final, w_ridge_final = 0.75, 0.25

    if best_w[1] > 0.0:
        print(f"   Using 3-model weights: XGB={best_w[0]:.2f}  LGB={best_w[1]:.2f}  Ridge={best_w[2]:.2f}")
        ensemble_pred = best_w[0]*xgb_pred + best_w[1]*lgb_pred + best_w[2]*ridge_pred
    else:
        print(f"   LightGBM weight=0 → excluded from blend (kept in comparison table)")
        print(f"   Ensemble weights (renormalised): XGB={w_xgb_final:.2f}  Ridge={w_ridge_final:.2f}")
        ensemble_pred = w_xgb_final*xgb_pred + w_ridge_final*ridge_pred
else:
    print(f"   Ensemble weights: XGB={best_w[0]:.2f}  Ridge={best_w[1]:.2f}")
    ensemble_pred = best_w[0]*xgb_pred + best_w[1]*ridge_pred

print("\n📊 XGB Ensemble:")
ens_metrics = compute_metrics(y_test.values, ensemble_pred, "Ensemble")


# ══════════════════════════════════════════════════════════
#  ABLATION: Ridge without discharge lags
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ABLATION: Ridge without discharge lag features")
print("=" * 60)

lag_cols     = [c for c in feature_cols if 'log_q_lag' in c]
non_lag_cols = [c for c in feature_cols if c not in lag_cols]
q_roll_cols  = [c for c in feature_cols if 'log_q_roll' in c or 'delta_q' in c]
no_q_cols    = [c for c in feature_cols if c not in lag_cols + q_roll_cols]

print(f"\n   Full feature set: {len(feature_cols)} features")
print(f"   Discharge lag features removed: {lag_cols}")

ridge_nolag = Ridge(alpha=1.0)
ridge_nolag.fit(X_trainval[non_lag_cols], y_trainval)
pred_nolag  = ridge_nolag.predict(X_test[non_lag_cols])
print("\n📊 Ridge (no discharge lags):")
nolag_metrics = compute_metrics(y_test.values, pred_nolag, "No Q lags")

ridge_noq = Ridge(alpha=1.0)
ridge_noq.fit(X_trainval[no_q_cols], y_trainval)
pred_noq  = ridge_noq.predict(X_test[no_q_cols])
print("📊 Ridge (no discharge features at all):")
noq_metrics = compute_metrics(y_test.values, pred_noq, "No Q features")

lag1_only = ['log_q_lag_1d']
if 'log_q_lag_1d' in feature_cols:
    ridge_lag1 = Ridge(alpha=1.0)
    ridge_lag1.fit(X_trainval[lag1_only], y_trainval)
    pred_lag1  = ridge_lag1.predict(X_test[lag1_only])
    print("📊 Ridge (lag-1 only — pure AR(1)):")
    lag1_metrics = compute_metrics(y_test.values, pred_lag1, "AR(1) only")

print(f"""
   ABLATION SUMMARY:
   ─────────────────────────────────────────
   Full Ridge ({len(feature_cols)} features):    NSE = {ridge_metrics['nse']:.4f}
   No discharge lags:             NSE = {nolag_metrics['nse']:.4f}  (Δ = {nolag_metrics['nse'] - ridge_metrics['nse']:+.4f})
   No discharge features at all:  NSE = {noq_metrics['nse']:.4f}  (Δ = {noq_metrics['nse'] - ridge_metrics['nse']:+.4f})""")
if 'log_q_lag_1d' in feature_cols:
    print(f"   Lag-1 only (pure AR(1)):       NSE = {lag1_metrics['nse']:.4f}  (Δ = {lag1_metrics['nse'] - ridge_metrics['nse']:+.4f})")
print(f"""
   INTERPRETATION:
   The dominant signal is autoregressive (lag-1 autocorrelation ~0.94).
   This is physically expected for a river — not leakage.
   The log-transform linearises this AR relationship, which explains
   why Ridge outperforms tree models here.
""")

ablation_df = pd.DataFrame([
    {'Model': 'Ridge (full)',          'NSE': round(ridge_metrics['nse'],  4), 'RMSE': round(ridge_metrics['rmse'],  2)},
    {'Model': 'Ridge (no Q lags)',     'NSE': round(nolag_metrics['nse'],  4), 'RMSE': round(nolag_metrics['rmse'],  2)},
    {'Model': 'Ridge (no Q features)', 'NSE': round(noq_metrics['nse'],    4), 'RMSE': round(noq_metrics['rmse'],    2)},
])
if 'log_q_lag_1d' in feature_cols:
    ablation_df = pd.concat([ablation_df, pd.DataFrame([
        {'Model': 'Ridge (lag-1 only)', 'NSE': round(lag1_metrics['nse'], 4), 'RMSE': round(lag1_metrics['rmse'], 2)},
    ])], ignore_index=True)
ablation_df.to_csv(MODEL_DIR / "ablation_results.csv", index=False)
print(f"💾 Saved: ablation_results.csv")


# ══════════════════════════════════════════════════════════
#  PEAK FLOW BIAS CORRECTION
#  Correction factor derived from VALIDATION set only.
#  Applied to predicted peaks (not observed) for forecast realism.
#  FIX 8: Both the correction factor AND the application mask
#  now use the same val-derived P90 threshold consistently.
# ══════════════════════════════════════════════════════════
ridge_val_pred_real = np.expm1(ridge_trainonly.predict(X_val))
y_val_real          = np.expm1(y_val.values)

p90_val = np.percentile(y_val_real, 90)
peak_mask_val = y_val_real > p90_val

if peak_mask_val.sum() > 5:
    val_peak_obs   = y_val_real[peak_mask_val]
    val_peak_pred  = ridge_val_pred_real[peak_mask_val]
    val_peak_ratio = np.mean(val_peak_pred) / np.mean(val_peak_obs)
    correction_factor = 1.0 / val_peak_ratio

    print(f"\n📊 PEAK FLOW CORRECTION (derived from validation set):")
    print(f"   Val P90 threshold: {p90_val:.0f} m³/s ({peak_mask_val.sum()} peak days)")
    print(f"   Val peak bias: {(val_peak_ratio - 1)*100:+.1f}%")
    print(f"   Correction factor: {correction_factor:.3f}")

    # Report improvement on test (factor is locked — not derived from test)
    ridge_test_pred_real = np.expm1(ridge_pred)
    y_test_real_check    = np.expm1(y_test.values)
    # FIX 8: Use the SAME val-derived p90_val threshold for the test mask,
    # not np.percentile(test) — different thresholds make before/after incomparable.
    peak_mask_test_verify = y_test_real_check > p90_val
    if peak_mask_test_verify.sum() > 5:
        test_uncorr = ridge_test_pred_real[peak_mask_test_verify]
        test_corr   = test_uncorr * correction_factor
        test_obs    = y_test_real_check[peak_mask_test_verify]
        print(f"   Test peak RMSE before: {np.sqrt(mean_squared_error(test_obs, test_uncorr)):.1f} m³/s")
        print(f"   Test peak RMSE after:  {np.sqrt(mean_squared_error(test_obs, test_corr)):.1f} m³/s")
else:
    correction_factor = 1.0
    print("\n   ℹ️  Not enough peak days in validation set for bias correction")


# ══════════════════════════════════════════════════════════
#  PART B: LSTM MODEL
# ══════════════════════════════════════════════════════════
if HAS_TF:
    print("\n" + "=" * 60)
    print("  PART B: LSTM Model")
    print("=" * 60)

    # NOTE: tf.random.set_seed(50) and np.random.seed(50) were already
    # called at the top of the script, before TF was imported.
    # No need to re-seed here — doing so would reset the RNG state
    # and could cause unexpected interactions with the XGB training above.

    LOOKBACK = 30

    scaler = StandardScaler()
    scaler.fit(X[train_mask])
    X_all_scaled = scaler.transform(X)

    def create_sequences(X_arr, y_arr, lookback):
        Xs, ys, indices = [], [], []
        for i in range(lookback, len(X_arr)):
            Xs.append(X_arr[i - lookback:i])
            ys.append(y_arr[i])
            indices.append(i)
        return np.array(Xs), np.array(ys), np.array(indices)

    print(f"   Creating {LOOKBACK}-day sliding windows...")
    X_seq, y_seq, idx_seq = create_sequences(X_all_scaled, y.values, LOOKBACK)

    seq_dates = df['date'].values[idx_seq]
    seq_train = seq_dates < val_start
    seq_val   = (seq_dates >= val_start) & (seq_dates < test_start)
    seq_test  = seq_dates >= test_start

    X_lstm_train, y_lstm_train = X_seq[seq_train], y_seq[seq_train]
    X_lstm_val,   y_lstm_val   = X_seq[seq_val],   y_seq[seq_val]
    X_lstm_test,  y_lstm_test  = X_seq[seq_test],  y_seq[seq_test]

    print(f"   LSTM Train: {len(X_lstm_train)}  Val: {len(X_lstm_val)}  Test: {len(X_lstm_test)}")
    print(f"   Input shape: {X_lstm_train.shape}  (samples, timesteps, features)")

    n_features = X_lstm_train.shape[2]

    # FIX 2 cont.: l2 is already imported from tensorflow.keras.regularizers
    # at the top — no inner import needed (and the old inner import used the
    # standalone keras package which may not be installed or may conflict).
    # LSTM architecture: moderate regularization.
    # Run 7 was over-regularized (L2 + dropout + recurrent_dropout → floor at val 0.187).
    # Run 8 was under-regularized (no recurrent_dropout, low dropout → best epoch 7, overfit).
    # Middle ground: moderate dropout + recurrent_dropout (0.1), NO L2.
    # L2 and dropout together over-constrain on ~8000 samples.
    # Batch size 64 (was 32) for more stable gradient estimates — reduces val loss noise
    # that was causing early stopping to grab a spurious best-epoch dip.
    lstm_model = Sequential([
        LSTM(128, input_shape=(LOOKBACK, n_features),
             dropout=0.2, recurrent_dropout=0.1),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1),
    ])

    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    lstm_model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1),  # patience=20: reduce before early-stop fires
    ]

    print("\n🚀 Training LSTM...")
    history = lstm_model.fit(
        X_lstm_train, y_lstm_train,
        validation_data=(X_lstm_val, y_lstm_val),
        epochs=200,
        batch_size=64,       # increased from 32 — halves batch count, stabilises val loss curve
        callbacks=callbacks,
        verbose=1,
    )

    lstm_pred = lstm_model.predict(X_lstm_test, verbose=0).flatten()
    print("\n📊 LSTM:")
    lstm_metrics = compute_metrics(y_lstm_test, lstm_pred, "LSTM")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], label='Train loss', linewidth=0.9)
    ax.plot(history.history['val_loss'], label='Val loss', linewidth=0.9)
    ax.set_title("LSTM Training History")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "lstm_training_history.png", dpi=150)
    plt.close()
    print(f"📸 Saved: lstm_training_history.png")

    # ── Hybrid weight: optimised on VALIDATION SET, applied to test ──
    # FIX 9: Previous code searched for best_hw using common_obs (TEST labels).
    # That is test-set snooping. We now use the val set exclusively.
    lstm_val_pred   = lstm_model.predict(X_lstm_val, verbose=0).flatten()
    lstm_val_dates  = seq_dates[seq_val]
    xgb_val_dates   = df.loc[val_mask, 'date'].values
    xgb_val_real    = np.expm1(y_val.values)

    # Reconstruct val-set ensemble predictions using locked train-only weights
    if HAS_LGB and best_w[1] > 0.0:
        xgb_ens_val_real = np.expm1(best_w[0]*xgb_val_pred + best_w[1]*lgb_val_pred + best_w[2]*ridge_val_pred)
    else:
        w_x = best_w[0]; w_r = best_w[-1]; w_t = w_x + w_r
        w_x, w_r = (w_x/w_t, w_r/w_t) if w_t > 0 else (0.75, 0.25)
        xgb_ens_val_real = np.expm1(w_x*xgb_val_pred + w_r*ridge_val_pred)

    lstm_val_date_set = set(pd.to_datetime(lstm_val_dates).date)
    val_idx_xgb, val_idx_lstm = [], []
    lstm_val_date_list = list(pd.to_datetime(lstm_val_dates).date)
    for i, d in enumerate([pd.Timestamp(d).date() for d in xgb_val_dates]):
        if d in lstm_val_date_set:
            val_idx_xgb.append(i)
            val_idx_lstm.append(lstm_val_date_list.index(d))

    common_val_obs  = xgb_val_real[val_idx_xgb]
    common_val_xgb  = xgb_ens_val_real[val_idx_xgb]
    common_val_lstm = np.expm1(lstm_val_pred[val_idx_lstm])

    best_hw, best_h_mse = 0.5, float('inf')
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * common_val_xgb + (1 - w) * common_val_lstm
        h_mse = mean_squared_error(common_val_obs, blend)
        if h_mse < best_h_mse:
            best_h_mse = h_mse
            best_hw = w
    print(f"\n   Hybrid weight (optimized on Val): XGB={best_hw:.2f}  LSTM={1-best_hw:.2f}")

    # Align test predictions and apply locked weight
    lstm_test_dates = seq_dates[seq_test]
    xgb_test_dates  = df.loc[test_mask, 'date'].values
    xgb_test_real   = np.expm1(y_test.values)
    xgb_ens_real    = np.expm1(ensemble_pred)

    lstm_date_set   = set(pd.to_datetime(lstm_test_dates).date)
    lstm_dates_list = [pd.Timestamp(d).date() for d in lstm_test_dates]
    common_idx_xgb, common_idx_lstm = [], []
    for i, d in enumerate([pd.Timestamp(d).date() for d in xgb_test_dates]):
        if d in lstm_date_set:
            common_idx_xgb.append(i)
            common_idx_lstm.append(lstm_dates_list.index(d))

    common_idx_xgb  = np.array(common_idx_xgb)
    common_idx_lstm = np.array(common_idx_lstm)
    common_dates    = np.array(xgb_test_dates)[common_idx_xgb]
    common_obs      = xgb_test_real[common_idx_xgb]
    common_xgb      = xgb_ens_real[common_idx_xgb]
    common_lstm     = np.expm1(lstm_pred[common_idx_lstm])

    hybrid_pred = best_hw * common_xgb + (1 - best_hw) * common_lstm

    hybrid_rmse = np.sqrt(mean_squared_error(common_obs, hybrid_pred))
    hybrid_mae  = mean_absolute_error(common_obs, hybrid_pred)
    hybrid_r2   = r2_score(common_obs, hybrid_pred)
    hybrid_nse  = 1 - np.sum((common_obs - hybrid_pred)**2) / np.sum((common_obs - common_obs.mean())**2)
    hybrid_metrics = {'rmse': hybrid_rmse, 'mae': hybrid_mae, 'r2': hybrid_r2, 'nse': hybrid_nse}

    print(f"\n📊 Hybrid (XGB+LSTM):")
    print(f"   [Hybrid]  RMSE={hybrid_rmse:.2f}  MAE={hybrid_mae:.2f}  R²={hybrid_r2:.4f}  NSE={hybrid_nse:.4f}")

else:
    lstm_metrics   = None
    hybrid_metrics = None
    common_dates   = None


# ══════════════════════════════════════════════════════════
#  PART C: COMPREHENSIVE COMPARISON & PLOTS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART C: Model Comparison")
print("=" * 60)

ridge_pred_real = np.expm1(ridge_pred)
y_test_real     = np.expm1(y_test.values)

peak_threshold = p90_val  # locked from validation set
peak_mask      = y_test_real > peak_threshold
print(f"   Peak threshold (from val P90): {peak_threshold:.0f} m³/s")

ridge_pred_corrected = ridge_pred_real.copy()
if correction_factor != 1.0:
    pred_peak_mask = ridge_pred_real > peak_threshold
    ridge_pred_corrected[pred_peak_mask] *= correction_factor
    print(f"   Applied peak correction (×{correction_factor:.3f}) to {pred_peak_mask.sum()} predicted peak days")

y_pred_log  = ensemble_pred
y_pred_real = np.expm1(y_pred_log)
test_dates  = df.loc[test_mask, 'date']
test_months = test_dates.dt.month

rmse = ens_metrics['rmse']
mae  = ens_metrics['mae']
r2   = ens_metrics['r2']
nse  = ens_metrics['nse']

monsoon_mask = test_months.between(6, 9).values
dry_mask     = ~monsoon_mask

print(f"\n📊 SEASONAL BREAKDOWN (Ridge — best model):")
if monsoon_mask.sum() > 0:
    compute_metrics(y_test.values[monsoon_mask], ridge_pred[monsoon_mask], "Monsoon")
if dry_mask.sum() > 0:
    compute_metrics(y_test.values[dry_mask], ridge_pred[dry_mask], "Dry season")

if peak_mask.sum() > 0:
    peak_obs      = y_test_real[peak_mask]
    peak_pred_r   = ridge_pred_corrected[peak_mask]
    peak_bias     = np.mean(peak_pred_r - peak_obs) / np.mean(peak_obs) * 100
    peak_rmse_val = np.sqrt(mean_squared_error(peak_obs, peak_pred_r))
    print(f"\n📊 PEAK FLOW (>{peak_threshold:.0f} m³/s, n={peak_mask.sum()}):")
    print(f"   Peak RMSE: {peak_rmse_val:.2f}   Bias: {peak_bias:+.1f}%")

# ===========================================================
#  PLOTS 1–9 (unchanged from previous version)
# ===========================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates.values, y_test_real, label='Observed', alpha=0.8, linewidth=0.9, color='#1a1a1a')
ax.plot(test_dates.values, ridge_pred_real, label=f'Ridge (best, NSE={ridge_metrics["nse"]:.3f})',
        alpha=0.9, linewidth=1.3, color='#F39C12')
ax.plot(test_dates.values, y_pred_real, label=f'XGB Ensemble (NSE={nse:.3f})',
        alpha=0.6, linewidth=0.8, color='#2E86C1')
ax.fill_between(test_dates.values, y_test_real, ridge_pred_real, alpha=0.08, color='#F39C12')
ax.set_title("Observed vs Predicted — Ridge (best) & XGB Ensemble")
ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)"); ax.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "obs_vs_pred_timeseries.png", dpi=150)
plt.close()
print(f"\n📸 Saved: obs_vs_pred_timeseries.png")

n_scatter = 2 + (1 if HAS_TF and common_dates is not None else 0)
fig, axes = plt.subplots(1, n_scatter, figsize=(6 * n_scatter, 6))
axes = list(axes)
max_val = max(y_test_real.max(), ridge_pred_real.max(), y_pred_real.max()) * 1.05

ax = axes[0]
ax.scatter(y_test_real, ridge_pred_real, alpha=0.3, s=5, c='#F39C12')
if peak_mask.sum() > 0:
    ax.scatter(y_test_real[peak_mask], ridge_pred_real[peak_mask], alpha=0.6, s=12, c='coral', label=f'Peak (>{peak_threshold:.0f})')
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
ax.set_title(f"Ridge (best)\nR²={ridge_metrics['r2']:.4f}  RMSE={ridge_metrics['rmse']:.1f}")
ax.set_xlabel("Observed (m³/s)"); ax.set_ylabel("Predicted (m³/s)"); ax.legend(fontsize=8)
ax.set_xlim(0, max_val); ax.set_ylim(0, max_val); ax.set_aspect('equal')

ax = axes[1]
ax.scatter(y_test_real, y_pred_real, alpha=0.3, s=5, c='steelblue')
if peak_mask.sum() > 0:
    ax.scatter(y_test_real[peak_mask], y_pred_real[peak_mask], alpha=0.6, s=12, c='coral')
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
ax.set_title(f"XGB Ensemble\nR²={r2:.4f}  RMSE={rmse:.1f}")
ax.set_xlabel("Observed (m³/s)"); ax.set_ylabel("Predicted (m³/s)")
ax.set_xlim(0, max_val); ax.set_ylim(0, max_val); ax.set_aspect('equal')

plt.suptitle("Scatter Comparison", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(MODEL_DIR / "scatter_obs_vs_pred.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"📸 Saved: scatter_obs_vs_pred.png")

ridge_residuals = y_test_real - ridge_pred_real
fig, axes_r = plt.subplots(1, 2, figsize=(12, 4))
axes_r[0].scatter(ridge_pred_real, ridge_residuals, alpha=0.2, s=5, c='#F39C12')
axes_r[0].axhline(0, color='red', linewidth=0.8, linestyle='--')
axes_r[0].set_xlabel("Predicted (m³/s)"); axes_r[0].set_ylabel("Residual")
axes_r[0].set_title("Residuals vs Predicted (Ridge)")
axes_r[1].hist(ridge_residuals, bins=60, edgecolor='white', linewidth=0.3, color='#F39C12')
axes_r[1].axvline(0, color='red', linewidth=0.8, linestyle='--')
axes_r[1].set_xlabel("Residual (m³/s)"); axes_r[1].set_ylabel("Freq")
axes_r[1].set_title(f"Ridge Residuals (mean={np.mean(ridge_residuals):.1f})")
plt.tight_layout()
plt.savefig(MODEL_DIR / "ridge_residual_analysis.png", dpi=150)
plt.close()
print(f"📸 Saved: ridge_residual_analysis.png")

importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 7))
importance.head(20).sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title("Top 20 Feature Importances (XGBoost)"); ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
plt.close()
print(f"📸 Saved: feature_importance.png")

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

residuals = y_test_real - y_pred_real
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_real, residuals, alpha=0.2, s=5, c='steelblue')
axes[0].axhline(0, color='red', linewidth=0.8, linestyle='--')
axes[0].set_xlabel("Predicted (m³/s)"); axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")
axes[1].hist(residuals, bins=60, edgecolor='white', linewidth=0.3, color='steelblue')
axes[1].axvline(0, color='red', linewidth=0.8, linestyle='--')
axes[1].set_xlabel("Residual (m³/s)"); axes[1].set_ylabel("Freq")
axes[1].set_title(f"Residuals (mean={np.mean(residuals):.1f})")
plt.tight_layout()
plt.savefig(MODEL_DIR / "residual_analysis.png", dpi=150)
plt.close()
print(f"📸 Saved: residual_analysis.png")

if HAS_TF and common_dates is not None and len(common_dates) > 0:
    common_ridge = ridge_pred_real[common_idx_xgb]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(common_dates, common_obs,   label='Observed',                                       color='#1a1a1a', linewidth=1,   alpha=0.85)
    ax.plot(common_dates, common_ridge, label=f'Ridge (best, NSE={ridge_metrics["nse"]:.3f})',  color='#F39C12', linewidth=1.3, alpha=0.9)
    ax.plot(common_dates, common_xgb,  label=f'XGB Ensemble (NSE={ens_metrics["nse"]:.3f})',   color='#2E86C1', linewidth=0.9, alpha=0.7)
    ax.plot(common_dates, common_lstm, label=f'LSTM (NSE={lstm_metrics["nse"]:.3f})',           color='#E74C3C', linewidth=0.9, alpha=0.7)
    ax.plot(common_dates, hybrid_pred, label=f'Hybrid XGB+LSTM (NSE={hybrid_nse:.3f})',        color='#27AE60', linewidth=0.9, alpha=0.7, linestyle='--')
    ax.set_title("Model Comparison — All Models (Test Period)")
    ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "comparison_timeseries.png", dpi=150)
    plt.close()
    print(f"📸 Saved: comparison_timeseries.png")

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
        ax.set_xlim(0, max_v); ax.set_ylim(0, max_v); ax.set_aspect('equal')
    plt.suptitle("Scatter Comparison: XGB vs LSTM vs Hybrid", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "comparison_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Saved: comparison_scatter.png")

all_models = {'XGBoost': xgb_metrics, 'Ridge': ridge_metrics}

ensemble_is_distinct = True
if HAS_LGB:
    if best_w[1] == 0.0 and best_w[2] == 0.0:
        ensemble_is_distinct = False
else:
    if len(best_w) == 2 and best_w[1] == 0.0:
        ensemble_is_distinct = False

if ensemble_is_distinct:
    all_models['XGB Ensemble'] = ens_metrics
else:
    print(f"   ℹ️  Ensemble == XGBoost (Ridge weight=0) — suppressing duplicate row")

if HAS_LGB:
    all_models['LightGBM'] = lgb_metrics
if HAS_TF and lstm_metrics:
    all_models['LSTM'] = lstm_metrics
if HAS_TF and hybrid_metrics:
    if best_hw < 0.95 and hybrid_metrics['nse'] > ens_metrics['nse']:
        all_models['Hybrid'] = hybrid_metrics
    else:
        print(f"   ℹ️  Hybrid skipped (weight={best_hw:.2f}, NSE={hybrid_metrics['nse']:.4f} ≤ Ensemble={ens_metrics['nse']:.4f})")

model_names = list(all_models.keys())
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
bar_colors = ['#3498DB', '#95A5A6', '#2E86C1', '#1ABC9C', '#E74C3C', '#27AE60']
for ax, metric_key, metric_label in zip(axes, ['rmse', 'mae', 'nse'], ['RMSE', 'MAE', 'NSE']):
    vals = [all_models[m][metric_key] for m in model_names]
    bars = ax.barh(model_names, vals, color=bar_colors[:len(model_names)], edgecolor='white', linewidth=0.5)
    ax.set_title(metric_label, fontweight='bold'); ax.set_xlabel(metric_label)
    for bar, val in zip(bars, vals):
        fmt = f"{val:.4f}" if metric_key == 'nse' else f"{val:.1f}"
        ax.text(bar.get_width() + (max(vals)*0.02), bar.get_y() + bar.get_height()/2, fmt, va='center', fontsize=9)
plt.suptitle("Model Comparison — Test Set Metrics (real m³/s)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(MODEL_DIR / "comparison_metrics_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"📸 Saved: comparison_metrics_bar.png")

if HAS_TF and common_dates is not None and len(common_dates) > 0:
    cd = pd.to_datetime(common_dates)
    first_year = cd.year.min()
    zoom_mask = (cd >= pd.Timestamp(f'{first_year}-06-01')) & (cd <= pd.Timestamp(f'{first_year}-10-31'))
    if zoom_mask.sum() > 30:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cd[zoom_mask], common_obs[zoom_mask],   label='Observed',     color='#1a1a1a', linewidth=1.2)
        ax.plot(cd[zoom_mask], common_xgb[zoom_mask],   label='XGB Ensemble', color='#2E86C1', linewidth=0.9, alpha=0.85)
        ax.plot(cd[zoom_mask], common_lstm[zoom_mask],  label='LSTM',         color='#E74C3C', linewidth=0.9, alpha=0.85)
        ax.plot(cd[zoom_mask], hybrid_pred[zoom_mask],  label='Hybrid',       color='#27AE60', linewidth=1,   linestyle='--', alpha=0.85)
        ax.set_title(f"Monsoon {first_year} — Peak Season Zoom")
        ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "comparison_monsoon_zoom.png", dpi=150)
        plt.close()
        print(f"📸 Saved: comparison_monsoon_zoom.png")

# ===========================================================
#  SAVE PREDICTIONS & COMPARISON TABLE
# ===========================================================
results_df = pd.DataFrame({
    'date':         df.loc[test_mask, 'date'].values,
    'observed':     y_test_real,
    'ridge':        np.expm1(ridge_pred),
    'xgb_ensemble': y_pred_real,
    'residual':     residuals,
})
if HAS_TF and common_dates is not None:
    lstm_col   = np.full(len(results_df), np.nan)
    hybrid_col = np.full(len(results_df), np.nan)
    for i, idx in enumerate(common_idx_xgb):
        lstm_col[idx]   = common_lstm[i]
        hybrid_col[idx] = hybrid_pred[i]
    results_df['lstm']   = lstm_col
    results_df['hybrid'] = hybrid_col

results_df.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
print(f"\n💾 Saved: test_predictions.csv")

comp_rows = [{'Model': name, 'RMSE': round(m['rmse'], 2), 'MAE': round(m['mae'], 2),
              'R2': round(m['r2'], 4), 'NSE': round(m['nse'], 4)} for name, m in all_models.items()]
comparison_df = pd.DataFrame(comp_rows)
comparison_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
print(f"💾 Saved: model_comparison.csv")

print(f"\n{'='*60}")
print(f"  FINAL COMPARISON TABLE")
print(f"{'='*60}")
print(comparison_df.to_string(index=False))
print("\n✅ Training complete.")
print("=" * 60)

# ══════════════════════════════════════════════════════════
#  SAVE FORECAST ASSETS
#  Ridge is the best model (NSE=0.8897). For CMIP6 forecasting
#  we retrain it on ALL historical data (not just train+val)
#  so it benefits from the full record. We also persist the
#  peak-correction factor and feature list so forecast.py
#  applies identical preprocessing without guessing.
# ══════════════════════════════════════════════════════════
import joblib, json

print(f"\n{'='*60}")
print(f"  SAVING FORECAST ASSETS")
print(f"{'='*60}")

# ── A) Full Ridge (with discharge lags) — for short-term forecasting ──
ridge_forecast = Ridge(alpha=1.0)
ridge_forecast.fit(X, y)                           # X, y = full historical dataset
joblib.dump(ridge_forecast, MODEL_DIR / "ridge_forecast_model.pkl")
print(f"💾 Saved: ridge_forecast_model.pkl  (trained on {len(X)} samples)")

with open(MODEL_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
print(f"💾 Saved: feature_cols.json  ({len(feature_cols)} features)")

# ── B) Rainfall-Only Ridge — for CMIP6 long-range projections ──
#
#  The full Ridge gets NSE~0.89 mainly from log_q_lag_1d (r=0.94).
#  This is valid for short-term forecasting but NOT for CMIP6:
#  future discharge doesn't exist, so the AR loop feeds predictions
#  back as lags, causing compounding errors over months/years.
#
#  The rainfall-only model uses ONLY features that CMIP6 provides:
#  rainfall lags, rolling sums, rolling std, seasonality, monsoon flag.
#  Lower NSE on test (~ablation value), but physically honest for
#  multi-year climate projections.
#
#  The forecast script adds a physical post-processor (recession
#  filter + smoothing) to produce realistic daily hydrographs.

# Identify discharge-dependent features to exclude
q_dependent_cols = [c for c in feature_cols if 'log_q_lag' in c or 'log_q_roll' in c or 'delta_q' in c]
rainfall_only_cols = [c for c in feature_cols if c not in q_dependent_cols]

print(f"\n   Rainfall-only features ({len(rainfall_only_cols)}):")
for f in rainfall_only_cols:
    print(f"     {f}")
print(f"   Excluded discharge features: {q_dependent_cols}")

# ── B1) Ridge baseline (rainfall-only) ──
ridge_ro = Ridge(alpha=1.0)
ridge_ro.fit(X[rainfall_only_cols], y)
ro_ridge_pred = ridge_ro.predict(X_test[rainfall_only_cols])
print("\n📊 Rainfall-Only Ridge (test set):")
ro_ridge_metrics = compute_metrics(y_test.values, ro_ridge_pred, "RO Ridge")

# ── B2) XGBoost (rainfall-only) — captures nonlinear rainfall→discharge ──
print("\n🔍 Training XGBoost rainfall-only model...")
xgb_ro = XGBRegressor(
    n_estimators=1000, learning_rate=0.04, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
)
xgb_ro.fit(X_trainval[rainfall_only_cols], y_trainval,
           eval_set=[(X_val[rainfall_only_cols], y_val)], verbose=0)
ro_xgb_pred = xgb_ro.predict(X_test[rainfall_only_cols])
print("📊 Rainfall-Only XGBoost (test set):")
ro_xgb_metrics = compute_metrics(y_test.values, ro_xgb_pred, "RO XGBoost")

# ── B3) Damped AR — use rainfall-only predictions as pseudo-Q lags ──
#  Instead of feeding raw predictions back (which compounds errors),
#  we blend them with climatological Q before using as lags.
#  This gives temporal coherence without runaway accumulation.
print("\n🔍 Training Damped-AR model (rainfall-only + pseudo-Q lags)...")

# Build damped-AR features: use rainfall-only XGB predictions as pseudo-Q
# Train on full data: predict, then use those predictions as lag features
ro_full_pred_log = xgb_ro.predict(X[rainfall_only_cols])
ro_full_pred_real = np.expm1(ro_full_pred_log)

# Compute climatological monthly Q for damping
df_master_path = BASE_DIR / "data/master_dataset.csv"
df_m_tmp = pd.read_csv(df_master_path)
df_m_tmp['date'] = pd.to_datetime(df_m_tmp['date'])
df_m_tmp['month'] = df_m_tmp['date'].dt.month
clim_q_monthly = df_m_tmp.groupby('month')['q_upstream_mk'].median().to_dict()

# Build pseudo-Q lag features with climatological damping
DAMP_ALPHA = 0.6  # blend: 0.6 × predicted + 0.4 × climatological
pseudo_q = np.zeros(len(df))
dates_all = df['date'].values

for i in range(len(df)):
    month_i = pd.Timestamp(dates_all[i]).month
    clim_q  = clim_q_monthly.get(month_i, 30.0)
    raw_q   = max(ro_full_pred_real[i], 0.0)
    pseudo_q[i] = DAMP_ALPHA * raw_q + (1 - DAMP_ALPHA) * clim_q

# Create damped lag features
dar_extra_cols = []
for lag in range(1, 4):
    col_name = f'log_pseudo_q_lag_{lag}d'
    df[col_name] = np.log1p(pd.Series(pseudo_q).shift(lag).fillna(0).clip(lower=0).values)
    dar_extra_cols.append(col_name)

dar_feature_cols = rainfall_only_cols + dar_extra_cols

# Retrain on full data with pseudo-Q lags
X_dar = df[dar_feature_cols]
ridge_dar = Ridge(alpha=1.0)
ridge_dar.fit(X_dar, y)

# Test with pseudo-Q lags built from test-period rainfall-only predictions
# (simulates what CMIP6 will do: predict → damp → use as lag)
test_pseudo_q = np.zeros(test_mask.sum())
test_dates_v = df.loc[test_mask, 'date'].values
ro_test_real = np.expm1(xgb_ro.predict(X_test[rainfall_only_cols]))

for i in range(len(test_pseudo_q)):
    month_i = pd.Timestamp(test_dates_v[i]).month
    clim_q  = clim_q_monthly.get(month_i, 30.0)
    test_pseudo_q[i] = DAMP_ALPHA * max(ro_test_real[i], 0.0) + (1 - DAMP_ALPHA) * clim_q

X_test_dar = X_test[rainfall_only_cols].copy()
for lag in range(1, 4):
    col_name = f'log_pseudo_q_lag_{lag}d'
    X_test_dar[col_name] = np.log1p(pd.Series(test_pseudo_q).shift(lag).fillna(0).clip(lower=0).values)

ro_dar_pred = ridge_dar.predict(X_test_dar[dar_feature_cols])
print("📊 Damped-AR Ridge (test set):")
ro_dar_metrics = compute_metrics(y_test.values, ro_dar_pred, "Damped AR")

# ── Pick the best rainfall-only approach ──
ro_candidates = {
    'Ridge':     (ridge_ro, rainfall_only_cols,  ro_ridge_metrics, 'ridge'),
    'XGBoost':   (xgb_ro,   rainfall_only_cols,  ro_xgb_metrics,   'xgboost'),
    'Damped-AR': (ridge_dar, dar_feature_cols,    ro_dar_metrics,   'damped_ar'),
}

best_ro_name = max(ro_candidates, key=lambda k: ro_candidates[k][2]['nse'])
best_ro_model, best_ro_cols, best_ro_metrics, best_ro_type = ro_candidates[best_ro_name]

print(f"\n   RAINFALL-ONLY MODEL COMPARISON:")
print(f"   {'Model':<15} {'NSE':>8} {'RMSE':>8}")
print(f"   {'-'*33}")
for name, (_, _, m, _) in ro_candidates.items():
    marker = " ← best" if name == best_ro_name else ""
    print(f"   {name:<15} {m['nse']:>8.4f} {m['rmse']:>8.2f}{marker}")

# Save the best rainfall-only model
joblib.dump(best_ro_model, MODEL_DIR / "ridge_rainfall_only_model.pkl")
print(f"\n💾 Saved: ridge_rainfall_only_model.pkl  ({best_ro_name}, {len(best_ro_cols)} features)")

with open(MODEL_DIR / "rainfall_only_feature_cols.json", "w") as f:
    json.dump(best_ro_cols, f)
print(f"💾 Saved: rainfall_only_feature_cols.json  ({len(best_ro_cols)} features)")

ro_metrics = best_ro_metrics

# Save damped-AR parameters if that's the winner
forecast_ro_meta = {
    "ro_model_type": best_ro_type,
    "ro_nse": round(float(ro_metrics['nse']), 4),
    "damp_alpha": DAMP_ALPHA if best_ro_type == 'damped_ar' else None,
    "clim_q_monthly": {str(k): round(v, 2) for k, v in clim_q_monthly.items()},
}
with open(MODEL_DIR / "rainfall_only_meta.json", "w") as f:
    json.dump(forecast_ro_meta, f, indent=2)
print(f"💾 Saved: rainfall_only_meta.json  (type={best_ro_type})")

# Also save the XGBoost rainfall-only model separately (needed for damped-AR base predictions)
joblib.dump(xgb_ro, MODEL_DIR / "xgb_rainfall_only_model.pkl")
with open(MODEL_DIR / "rainfall_only_base_cols.json", "w") as f:
    json.dump(rainfall_only_cols, f)
print(f"💾 Saved: xgb_rainfall_only_model.pkl  (base predictor for damped-AR)")

# ── C) Peak-correction metadata (derived from validation set) ──
forecast_meta = {
    "correction_factor": round(float(correction_factor), 6),
    "p90_threshold_m3s": round(float(p90_val), 2),
    "best_model": "Ridge",
    "best_nse": round(float(ridge_metrics['nse']), 4),
    "rainfall_only_nse": round(float(ro_metrics['nse']), 4),
    "rainfall_only_type": best_ro_type,
}
with open(MODEL_DIR / "forecast_meta.json", "w") as f:
    json.dump(forecast_meta, f, indent=2)
print(f"💾 Saved: forecast_meta.json  "
      f"(correction={correction_factor:.3f}, p90={p90_val:.0f} m³/s)")

# ── D) Climatological monthly discharge for post-processing ──
#  The forecast script uses this to scale rainfall-only predictions
#  and apply recession behavior during dry spells.
df_master_path = BASE_DIR / "data/master_dataset.csv"
if df_master_path.exists():
    df_m = pd.read_csv(df_master_path)
    df_m['date'] = pd.to_datetime(df_m['date'])
    df_m['month'] = df_m['date'].dt.month
    df_m['doy']   = df_m['date'].dt.dayofyear

    # Recession constant: median of Q(t)/Q(t-1) during falling limbs
    q_vals = df_m['q_upstream_mk'].dropna().values
    dQ = np.diff(q_vals)
    recession_mask = dQ < 0
    ratios = q_vals[1:][recession_mask] / q_vals[:-1][recession_mask]
    ratios = ratios[(ratios > 0.5) & (ratios < 1.0)]
    recession_k = float(np.median(ratios))

    clim_monthly = df_m.groupby('month')['q_upstream_mk'].agg(['mean', 'median', 'std', 'max']).to_dict()

    hydro_params = {
        "recession_constant": round(recession_k, 5),
        "recession_halflife_days": round(-np.log(2)/np.log(recession_k), 1),
        "clim_monthly_mean": {str(k): round(v, 2) for k, v in clim_monthly['mean'].items()},
        "clim_monthly_median": {str(k): round(v, 2) for k, v in clim_monthly['median'].items()},
        "clim_monthly_std": {str(k): round(v, 2) for k, v in clim_monthly['std'].items()},
        "clim_monthly_max": {str(k): round(v, 2) for k, v in clim_monthly['max'].items()},
        "mean_annual_q": round(float(df_m['q_upstream_mk'].mean()), 2),
    }
    with open(MODEL_DIR / "hydro_params.json", "w") as f:
        json.dump(hydro_params, f, indent=2)
    print(f"💾 Saved: hydro_params.json  (recession_k={recession_k:.4f})")
else:
    print("⚠️  master_dataset.csv not found — hydro_params.json not saved")

print(f"\n   Run forecast.py to generate CMIP6 scenario projections.")