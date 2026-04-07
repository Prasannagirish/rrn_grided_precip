import streamlit as st
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Kabini River · Discharge Forecast",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
EDA_DIR     = BASE_DIR / "eda_outputs"
HYDRO_DIR   = BASE_DIR / "hydrology_outputs"
FEAT_DIR    = BASE_DIR / "feature_outputs"
MODEL_DIR   = BASE_DIR / "model_outputs"

# ============================================================
# CSS — dark-native
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .block-container { padding-top: 1.5rem; max-width: 1180px; }

  .hero {
    background: linear-gradient(135deg, #0c1a2e 0%, #0f2b4a 50%, #0a3550 100%);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 2.6rem 2.8rem 2.2rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
  }
  .hero::before {
    content: ''; position: absolute; inset: 0;
    background:
      radial-gradient(ellipse 50% 50% at 75% 15%, rgba(56,189,248,.09) 0%, transparent 70%),
      radial-gradient(ellipse 35% 55% at 10% 85%, rgba(14,165,233,.06) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero h1 { font-family:'DM Sans',sans-serif; font-weight:700; font-size:2.2rem; color:#e0f2fe; margin:0 0 .4rem; letter-spacing:-0.4px; }
  .hero p  { color:#8faabe; font-size:.98rem; margin:0; line-height:1.55; }
  .hero .hl { color:#38bdf8; font-weight:500; }

  .steps-row { display:flex; gap:.55rem; margin-bottom:1.4rem; }
  .step-card {
    flex:1; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
    border-radius:10px; padding:.85rem .95rem; transition:border-color .2s, background .2s;
  }
  .step-card.done  { border-color:rgba(34,197,94,.3); background:rgba(34,197,94,.04); }
  .step-card.active { border-color:#38bdf8; background:rgba(56,189,248,.06); }
  .step-num {
    font-family:'JetBrains Mono',monospace; font-weight:600; font-size:.63rem;
    letter-spacing:1.4px; text-transform:uppercase; margin-bottom:.25rem;
  }
  .step-card.active .step-num { color:#38bdf8; }
  .step-card.done .step-num   { color:#4ade80; }
  .step-card .step-num        { color:rgba(255,255,255,.3); }
  .step-title { font-weight:600; font-size:.88rem; color:rgba(255,255,255,.85); line-height:1.3; }
  .step-card.done .step-title::after { content:' ✓'; color:#4ade80; font-size:.85rem; }

  .sec-desc { color:rgba(255,255,255,.5); font-size:.9rem; line-height:1.5; margin-bottom:1rem; }

  .metric-row { display:flex; gap:.8rem; margin:1.2rem 0 .6rem; }
  .metric-tile {
    flex:1; background:rgba(56,189,248,.06); border:1px solid rgba(56,189,248,.18);
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
  }
  .metric-tile .label {
    font-family:'JetBrains Mono',monospace; font-size:.68rem; font-weight:600;
    color:#38bdf8; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:.25rem;
  }
  .metric-tile .value { font-weight:700; font-size:1.45rem; color:#e0f2fe; }

  .metric-tile-green {
    flex:1; background:rgba(34,197,94,.06); border:1px solid rgba(34,197,94,.18);
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
  }
  .metric-tile-green .label {
    font-family:'JetBrains Mono',monospace; font-size:.68rem; font-weight:600;
    color:#4ade80; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:.25rem;
  }
  .metric-tile-green .value { font-weight:700; font-size:1.45rem; color:#d1fae5; }

  .badge {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:.67rem; font-weight:600; letter-spacing:1px; text-transform:uppercase;
    padding:.22rem .65rem; border-radius:5px;
  }
  .badge-ready { background:rgba(34,197,94,.15); color:#4ade80; border:1px solid rgba(34,197,94,.25); }
  .badge-empty { background:rgba(250,204,21,.1); color:#fbbf24; border:1px solid rgba(250,204,21,.2); }

  .gallery-caption {
    font-family:'JetBrains Mono',monospace; font-size:.7rem;
    color:rgba(255,255,255,.35); text-align:center; margin-top:.25rem; letter-spacing:.4px;
  }

  .foot {
    text-align:center; color:rgba(255,255,255,.25); font-size:.8rem;
    padding:1.8rem 0 .8rem; margin-top:2rem; border-top:1px solid rgba(255,255,255,.06);
  }
  .foot code {
    font-family:'JetBrains Mono',monospace; background:rgba(255,255,255,.06);
    padding:.12rem .4rem; border-radius:4px; font-size:.75rem; color:rgba(255,255,255,.4);
  }

  div[data-testid="stExpander"] details summary span {
    font-family:'JetBrains Mono',monospace; font-size:.8rem;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
for key in ["data_uploaded", "eda_done", "hydro_done", "feat_done", "model_done", "forecast_done"]:
    if key not in st.session_state:
        st.session_state[key] = False

# ============================================================
# HELPERS
# ============================================================
def dir_has_outputs(directory: Path) -> bool:
    return directory.exists() and any(directory.glob("*.png"))

def run_script(script_name: str) -> bool:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        st.error(f"Script not found: `{script_name}`")
        return False
    with st.spinner(f"Running `{script_name}` …"):
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True,
        )
    if result.returncode == 0:
        st.success(f"Completed `{script_name}`")
        with st.expander("Terminal output"):
            st.code(result.stdout, language="text")
        return True
    else:
        st.error(f"Failed — `{script_name}`")
        with st.expander("Error log", expanded=True):
            st.code(result.stderr, language="text")
        return False

def show_images(directory: Path, cols: int = 2):
    if not directory.exists():
        st.info("No outputs yet — run this step first.")
        return
    images = sorted(directory.glob("*.png"))
    if not images:
        st.info("No plots generated yet.")
        return
    grid = st.columns(cols)
    for i, img_path in enumerate(images):
        with grid[i % cols]:
            img = Image.open(img_path)
            st.image(img, width="stretch")
            st.markdown(
                f'<p class="gallery-caption">{img_path.stem.replace("_", " ").title()}</p>',
                unsafe_allow_html=True,
            )

def get_pred_column(df_res):
    """Find the prediction column — handles both old ('predicted')
    and new ('xgb_ensemble') column naming."""
    for col in ['xgb_ensemble', 'predicted']:
        if col in df_res.columns:
            return col
    return None

# Auto-detect prior runs
if (DATA_DIR / "master_dataset.csv").exists() and (DATA_DIR / "chirps_kabini_daily.csv").exists():
    st.session_state.data_uploaded = True
if dir_has_outputs(EDA_DIR):
    st.session_state.eda_done = True
if dir_has_outputs(HYDRO_DIR):
    st.session_state.hydro_done = True
if dir_has_outputs(FEAT_DIR):
    st.session_state.feat_done = True
if dir_has_outputs(MODEL_DIR):
    st.session_state.model_done = True
FORECAST_DIR = BASE_DIR / "forecast_outputs"
if dir_has_outputs(FORECAST_DIR):
    st.session_state.forecast_done = True

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
  <h1>🌊 Kabini River Discharge Forecasting</h1>
  <p>
    End-to-end ML pipeline — from raw <span class="hl">CHIRPS rainfall</span>
    and <span class="hl">CWC discharge</span> data through
    <span class="hl">hydrological analysis</span> to
    <span class="hl">XGBoost + LSTM predictions</span>.
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STEP INDICATOR — 7 steps
# ============================================================
if st.session_state.forecast_done:
    active_step = 7
elif st.session_state.model_done:
    active_step = 6
elif st.session_state.feat_done:
    active_step = 5
elif st.session_state.hydro_done:
    active_step = 4
elif st.session_state.eda_done:
    active_step = 3
elif st.session_state.data_uploaded:
    active_step = 2
else:
    active_step = 1

steps_info = [
    ("01", "Upload"),
    ("02", "EDA"),
    ("03", "Hydrology"),
    ("04", "Features"),
    ("05", "Training"),
    ("06", "Results"),
    ("07", "Forecast"),
]

cards_html = '<div class="steps-row">'
for idx, (num, title) in enumerate(steps_info):
    step_i = idx + 1
    if step_i < active_step:
        cls = "step-card done"
    elif step_i == active_step:
        cls = "step-card active"
    else:
        cls = "step-card"
    cards_html += f"""
    <div class="{cls}">
      <div class="step-num">Step {num}</div>
      <div class="step-title">{title}</div>
    </div>"""
cards_html += '</div>'
st.markdown(cards_html, unsafe_allow_html=True)

# ============================================================
# TABS — 6 tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂  Upload",
    "📊  EDA",
    "💧  Hydrology",
    "⚙️  Features",
    "🚀  Training",
    "📈  Results",
    "🔮  Forecast",
])

# ------ TAB 1: DATA UPLOAD ------
with tab1:
    st.markdown('<p class="sec-desc">Upload your two source files to begin. The master dataset needs daily discharge and rainfall; the CHIRPS file provides spatial rainfall statistics.</p>', unsafe_allow_html=True)

    ucol1, ucol2 = st.columns(2)
    with ucol1:
        master_file = st.file_uploader(
            "Master Dataset (CSV)", type=["csv"], key="master_upload",
            help="Must contain columns: date, q_upstream_mk, rainfall_max_mm",
        )
    with ucol2:
        chirps_file = st.file_uploader(
            "CHIRPS Rainfall (CSV)", type=["csv"], key="chirps_upload",
            help="Daily CHIRPS aggregates including rainfall_std_mm",
        )

    if master_file and chirps_file:
        DATA_DIR.mkdir(exist_ok=True)
        (DATA_DIR / "master_dataset.csv").write_bytes(master_file.getvalue())
        (DATA_DIR / "chirps_kabini_daily.csv").write_bytes(chirps_file.getvalue())
        st.session_state.data_uploaded = True
        st.success("Both files saved — ready to proceed.")

        with st.expander("Preview: Master Dataset"):
            st.dataframe(pd.read_csv(DATA_DIR / "master_dataset.csv", nrows=200), width="stretch", height=240)
        with st.expander("Preview: CHIRPS Rainfall"):
            st.dataframe(pd.read_csv(DATA_DIR / "chirps_kabini_daily.csv", nrows=200), width="stretch", height=240)

    elif st.session_state.data_uploaded:
        st.markdown('<span class="badge badge-ready">✓ data loaded</span>', unsafe_allow_html=True)
        st.caption("Files already on disk. Re-upload to replace them.")
    else:
        st.markdown('<span class="badge badge-empty">awaiting upload</span>', unsafe_allow_html=True)


# ------ TAB 2: EDA ------
with tab2:
    st.markdown('<p class="sec-desc">Generate exploratory charts — distributions, time-series, and correlation matrices — to understand the raw data before feature engineering.</p>', unsafe_allow_html=True)

    if not st.session_state.data_uploaded:
        st.warning("Upload data in Step 1 first.")
    else:
        bcol, _ = st.columns([1, 3])
        with bcol:
            if st.button("▶  Run EDA", type="primary", key="btn_eda", use_container_width=True):
                if run_script("eda.py"):
                    st.session_state.eda_done = True
                    st.rerun()
        if st.session_state.eda_done:
            st.divider()
            show_images(EDA_DIR, cols=2)


# ------ TAB 3: HYDROLOGY ANALYSIS ------
with tab3:
    st.markdown('<p class="sec-desc">Compute classical hydrology parameters: baseflow separation, unit hydrograph, SCS curve number, recession analysis, flow duration curve, and flood frequency analysis.</p>', unsafe_allow_html=True)

    if not st.session_state.data_uploaded:
        st.warning("Upload data in Step 1 first.")
    else:
        bcol, _ = st.columns([1, 3])
        with bcol:
            if st.button("▶  Run Hydrology Analysis", type="primary", key="btn_hydro", use_container_width=True):
                if run_script("hydrology_analysis.py"):
                    st.session_state.hydro_done = True
                    st.rerun()

        if st.session_state.hydro_done:
            params_csv = HYDRO_DIR / "hydrology_parameters.csv"
            if params_csv.exists():
                df_params = pd.read_csv(params_csv)
                param_dict = dict(zip(df_params['Parameter'], df_params['Value']))

                def safe_fmt(key, fmt=".3f"):
                    val = param_dict.get(key, "N/A")
                    try:
                        return f"{float(val):{fmt}}"
                    except (ValueError, TypeError):
                        return str(val)

                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-tile-green">
                    <div class="label">BFI</div>
                    <div class="value">{safe_fmt('Baseflow Index (BFI)')}</div>
                  </div>
                  <div class="metric-tile-green">
                    <div class="label">Curve Number</div>
                    <div class="value">{safe_fmt('SCS Curve Number (CN)', '.1f')}</div>
                  </div>
                  <div class="metric-tile-green">
                    <div class="label">Recession k</div>
                    <div class="value">{safe_fmt('Recession constant (k)', '.4f')}</div>
                  </div>
                  <div class="metric-tile-green">
                    <div class="label">Runoff Coeff</div>
                    <div class="value">{safe_fmt('Runoff coefficient (C)')}</div>
                  </div>
                </div>
                <div class="metric-row">
                  <div class="metric-tile-green">
                    <div class="label">UH Tp (days)</div>
                    <div class="value">{safe_fmt('UH Time to peak Tp (days)', '.0f')}</div>
                  </div>
                  <div class="metric-tile-green">
                    <div class="label">Q50 (m³/s)</div>
                    <div class="value">{safe_fmt('Q50 median flow (m³/s)', '.1f')}</div>
                  </div>
                  <div class="metric-tile-green">
                    <div class="label">Q90 (m³/s)</div>
                    <div class="value">{safe_fmt('Q90 low flow (m³/s)', '.1f')}</div>
                  </div>
                  <div class="metric-tile-green">
                    <div class="label">Mean Q (m³/s)</div>
                    <div class="value">{safe_fmt('Mean discharge (m³/s)', '.1f')}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                with st.expander("Full parameter table"):
                    st.dataframe(df_params, width="stretch", height=460)

                st.download_button(
                    label="⬇  Download parameters CSV",
                    data=params_csv.read_bytes(),
                    file_name="hydrology_parameters.csv",
                    mime="text/csv",
                )

            st.divider()
            show_images(HYDRO_DIR, cols=2)


# ------ TAB 4: FEATURE ENGINEERING ------
with tab4:
    st.markdown('<p class="sec-desc">Build predictive features: rainfall lags (1–7 d), rolling sums &amp; std, discharge auto-regression, sin/cos seasonality, monsoon flag, and log transforms.</p>', unsafe_allow_html=True)

    if not st.session_state.data_uploaded:
        st.warning("Complete Step 1 first.")
    else:
        bcol, _ = st.columns([1, 3])
        with bcol:
            if st.button("▶  Run Feature Engineering", type="primary", key="btn_feat", use_container_width=True):
                if run_script("feature_engineering.py"):
                    st.session_state.feat_done = True
                    st.rerun()

        if st.session_state.feat_done:
            corr_csv = FEAT_DIR / "feature_target_correlation.csv"
            if corr_csv.exists():
                with st.expander("Feature → log_q correlations"):
                    st.dataframe(pd.read_csv(corr_csv, index_col=0), width="stretch", height=300)
            st.divider()
            show_images(FEAT_DIR, cols=2)


# ------ TAB 5: MODEL TRAINING ------
with tab5:
    st.markdown('<p class="sec-desc">Train XGBoost ensemble + LSTM with a strict temporal split. Compares all models on RMSE, MAE, R², and Nash-Sutcliffe Efficiency.</p>', unsafe_allow_html=True)

    if not st.session_state.feat_done:
        st.warning("Run Feature Engineering (Step 4) first.")
    else:
        bcol, _ = st.columns([1, 3])
        with bcol:
            if st.button("▶  Train Models", type="primary", key="btn_train", use_container_width=True):
                if run_script("train_model.py"):
                    st.session_state.model_done = True
                    st.rerun()

        if st.session_state.model_done:
            # Show model comparison table if available
            comp_csv = MODEL_DIR / "model_comparison.csv"
            if comp_csv.exists():
                st.subheader("Model Comparison")
                st.dataframe(pd.read_csv(comp_csv), width="stretch", height=240)

            st.divider()
            show_images(MODEL_DIR, cols=2)


# ------ TAB 6: RESULTS ------
with tab6:
    st.markdown('<p class="sec-desc">Final model performance and downloadable predictions.</p>', unsafe_allow_html=True)

    preds_csv = MODEL_DIR / "test_predictions.csv"

    if not st.session_state.model_done and not preds_csv.exists():
        st.warning("Train the model (Step 5) to see results.")
    elif preds_csv.exists():
        df_res = pd.read_csv(preds_csv)

        # Find the right prediction column (handles both old and new formats)
        pred_col = get_pred_column(df_res)
        if pred_col is None:
            st.error("Could not find prediction column in test_predictions.csv. "
                     f"Available columns: {list(df_res.columns)}")
        else:
            obs  = df_res["observed"].values
            pred = df_res[pred_col].values

            rmse = np.sqrt(np.mean((obs - pred) ** 2))
            mae  = np.mean(np.abs(obs - pred))
            ss_res = np.sum((obs - pred) ** 2)
            ss_tot = np.sum((obs - obs.mean()) ** 2)
            r2  = 1 - ss_res / ss_tot
            nse = r2

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-tile">
                <div class="label">RMSE (m³/s)</div>
                <div class="value">{rmse:,.2f}</div>
              </div>
              <div class="metric-tile">
                <div class="label">MAE (m³/s)</div>
                <div class="value">{mae:,.2f}</div>
              </div>
              <div class="metric-tile">
                <div class="label">R²</div>
                <div class="value">{r2:.4f}</div>
              </div>
              <div class="metric-tile">
                <div class="label">NSE</div>
                <div class="value">{nse:.4f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # If LSTM column exists, show its metrics too
            if 'lstm' in df_res.columns:
                lstm_valid = df_res.dropna(subset=['lstm'])
                if len(lstm_valid) > 0:
                    obs_l  = lstm_valid["observed"].values
                    pred_l = lstm_valid["lstm"].values
                    rmse_l = np.sqrt(np.mean((obs_l - pred_l) ** 2))
                    nse_l  = 1 - np.sum((obs_l - pred_l)**2) / np.sum((obs_l - obs_l.mean())**2)

                    hybrid_valid = df_res.dropna(subset=['hybrid'])
                    if len(hybrid_valid) > 0:
                        obs_h  = hybrid_valid["observed"].values
                        pred_h = hybrid_valid["hybrid"].values
                        rmse_h = np.sqrt(np.mean((obs_h - pred_h) ** 2))
                        nse_h  = 1 - np.sum((obs_h - pred_h)**2) / np.sum((obs_h - obs_h.mean())**2)
                    else:
                        rmse_h, nse_h = 0, 0

                    st.markdown(f"""
                    <div class="metric-row">
                      <div class="metric-tile" style="border-color:rgba(231,76,60,.25); background:rgba(231,76,60,.06);">
                        <div class="label" style="color:#e74c3c;">LSTM RMSE</div>
                        <div class="value">{rmse_l:,.2f}</div>
                      </div>
                      <div class="metric-tile" style="border-color:rgba(231,76,60,.25); background:rgba(231,76,60,.06);">
                        <div class="label" style="color:#e74c3c;">LSTM NSE</div>
                        <div class="value">{nse_l:.4f}</div>
                      </div>
                      <div class="metric-tile" style="border-color:rgba(39,174,96,.25); background:rgba(39,174,96,.06);">
                        <div class="label" style="color:#27ae60;">Hybrid RMSE</div>
                        <div class="value">{rmse_h:,.2f}</div>
                      </div>
                      <div class="metric-tile" style="border-color:rgba(39,174,96,.25); background:rgba(39,174,96,.06);">
                        <div class="label" style="color:#27ae60;">Hybrid NSE</div>
                        <div class="value">{nse_h:.4f}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.write("")
            show_images(MODEL_DIR, cols=2)

            st.divider()
            st.subheader("Test Set Predictions")

            # Build format dict dynamically based on available columns
            fmt = {"observed": "{:,.2f}"}
            if pred_col in df_res.columns:
                fmt[pred_col] = "{:,.2f}"
            if "residual" in df_res.columns:
                fmt["residual"] = "{:+,.2f}"
            if "lstm" in df_res.columns:
                fmt["lstm"] = "{:,.2f}"
            if "hybrid" in df_res.columns:
                fmt["hybrid"] = "{:,.2f}"

            st.dataframe(
                df_res.style.format(fmt, na_rep="—"),
                width="stretch", height=340,
            )
            st.download_button(
                label="⬇  Download predictions CSV",
                data=preds_csv.read_bytes(),
                file_name="test_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Predictions file not found.")


# ------ TAB 7: FORECAST ------
with tab7:
    st.markdown('<p class="sec-desc">Generate scenario-based discharge forecasts to 2030 using historical rainfall climatology. Six scenarios: Average, Wet (+20%), Dry (-20%), and Extreme (+40%) monsoon, SSP245, SSP585.</p>', unsafe_allow_html=True)

    if not st.session_state.model_done:
        st.warning("Train the model (Step 5) first.")
    else:
        bcol, _ = st.columns([1, 3])
        with bcol:
            if st.button("▶  Run Forecast", type="primary", key="btn_forecast", use_container_width=True):
                if run_script("forecast.py"):
                    st.session_state.forecast_done = True
                    st.rerun()

        if st.session_state.forecast_done:
            # Annual summary table
            summary_csv = FORECAST_DIR / "forecast_annual_summary.csv"
            if summary_csv.exists():
                st.subheader("Annual Forecast Summary")
                df_summary = pd.read_csv(summary_csv)
                st.dataframe(
                    df_summary.style.format({
                        "mean_discharge_m3s": "{:,.1f}",
                        "peak_discharge_m3s": "{:,.1f}",
                        "total_rainfall_mm": "{:,.0f}",
                    }),
                    width="stretch", height=320,
                )

                st.download_button(
                    label="⬇  Download forecast summary",
                    data=summary_csv.read_bytes(),
                    file_name="forecast_annual_summary.csv",
                    mime="text/csv",
                )

            st.divider()
            show_images(FORECAST_DIR, cols=2)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="foot">
  Kabini Basin Discharge Forecasting &nbsp;·&nbsp;
  <code>XGBoost</code> &nbsp;·&nbsp; <code>LSTM</code> &nbsp;·&nbsp; <code>CHIRPS</code> &nbsp;·&nbsp; <code>CWC</code>
  &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)