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

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
EDA_DIR    = BASE_DIR / "eda_outputs"
FEAT_DIR   = BASE_DIR / "feature_outputs"
MODEL_DIR  = BASE_DIR / "model_outputs"

# ============================================================
# CSS — fully dark-native, uses rgba white for all text/borders
# so nothing clashes with Streamlit's dark background
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  .block-container { padding-top: 1.5rem; max-width: 1180px; }

  /* ---------- hero ---------- */
  .hero {
    background: linear-gradient(135deg, #0c1a2e 0%, #0f2b4a 50%, #0a3550 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 2.6rem 2.8rem 2.2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
      radial-gradient(ellipse 50% 50% at 75% 15%, rgba(56,189,248,.09) 0%, transparent 70%),
      radial-gradient(ellipse 35% 55% at 10% 85%, rgba(14,165,233,.06) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero h1 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700; font-size: 2.2rem;
    color: #e0f2fe; margin: 0 0 .4rem;
    letter-spacing: -0.4px;
  }
  .hero p {
    color: #8faabe; font-size: .98rem;
    margin: 0; line-height: 1.55;
  }
  .hero .hl { color: #38bdf8; font-weight: 500; }

  /* ---------- step cards ---------- */
  .steps-row {
    display: flex; gap: .7rem;
    margin-bottom: 1.4rem;
  }
  .step-card {
    flex: 1;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    transition: border-color .2s, background .2s;
  }
  .step-card.done {
    border-color: rgba(34,197,94,.3);
    background: rgba(34,197,94,.04);
  }
  .step-card.active {
    border-color: #38bdf8;
    background: rgba(56,189,248,.06);
  }
  .step-num {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600; font-size: .68rem;
    letter-spacing: 1.4px; text-transform: uppercase;
    margin-bottom: .3rem;
  }
  .step-card.active .step-num { color: #38bdf8; }
  .step-card.done .step-num { color: #4ade80; }
  .step-card .step-num { color: rgba(255,255,255,.3); }
  .step-title {
    font-weight: 600; font-size: .95rem;
    color: rgba(255,255,255,.85);
    line-height: 1.3;
  }
  .step-card.done .step-title::after {
    content: ' ✓'; color: #4ade80; font-size: .85rem;
  }

  /* ---------- section desc ---------- */
  .sec-desc {
    color: rgba(255,255,255,.5);
    font-size: .9rem; line-height: 1.5;
    margin-bottom: 1rem;
  }

  /* ---------- metric tiles ---------- */
  .metric-row {
    display: flex; gap: .8rem;
    margin: 1.2rem 0 .6rem;
  }
  .metric-tile {
    flex: 1;
    background: rgba(56,189,248,.06);
    border: 1px solid rgba(56,189,248,.18);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-tile .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem; font-weight: 600;
    color: #38bdf8; text-transform: uppercase;
    letter-spacing: 1.2px; margin-bottom: .25rem;
  }
  .metric-tile .value {
    font-weight: 700; font-size: 1.45rem;
    color: #e0f2fe;
  }

  /* ---------- badges ---------- */
  .badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: .67rem; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase;
    padding: .22rem .65rem;
    border-radius: 5px;
  }
  .badge-ready { background: rgba(34,197,94,.15); color: #4ade80; border: 1px solid rgba(34,197,94,.25); }
  .badge-empty { background: rgba(250,204,21,.1); color: #fbbf24; border: 1px solid rgba(250,204,21,.2); }

  /* ---------- gallery caption ---------- */
  .gallery-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; color: rgba(255,255,255,.35);
    text-align: center; margin-top: .25rem;
    letter-spacing: .4px;
  }

  /* ---------- footer ---------- */
  .foot {
    text-align: center;
    color: rgba(255,255,255,.25);
    font-size: .8rem;
    padding: 1.8rem 0 .8rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(255,255,255,.06);
  }
  .foot code {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(255,255,255,.06);
    padding: .12rem .4rem;
    border-radius: 4px; font-size: .75rem;
    color: rgba(255,255,255,.4);
  }

  /* ---------- streamlit overrides ---------- */
  div[data-testid="stExpander"] details summary span {
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
for key in ["data_uploaded", "eda_done", "feat_done", "model_done"]:
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
            st.image(img, use_container_width=True)
            st.markdown(
                f'<p class="gallery-caption">{img_path.stem.replace("_", " ").title()}</p>',
                unsafe_allow_html=True,
            )


# Auto-detect prior runs on page load
if (DATA_DIR / "master_dataset.csv").exists() and (DATA_DIR / "chirps_kabini_daily.csv").exists():
    st.session_state.data_uploaded = True
if dir_has_outputs(EDA_DIR):
    st.session_state.eda_done = True
if dir_has_outputs(FEAT_DIR):
    st.session_state.feat_done = True
if dir_has_outputs(MODEL_DIR):
    st.session_state.model_done = True


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
  <h1>🌊 Kabini River Discharge Forecasting</h1>
  <p>
    End-to-end ML pipeline — from raw <span class="hl">CHIRPS rainfall</span>
    and <span class="hl">CWC discharge</span> data to
    <span class="hl">XGBoost predictions</span>.
    Walk through each phase below.
  </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STEP INDICATOR
# ============================================================
active_step = 1
if st.session_state.model_done:
    active_step = 5
elif st.session_state.feat_done:
    active_step = 4
elif st.session_state.eda_done:
    active_step = 3
elif st.session_state.data_uploaded:
    active_step = 2

steps_info = [
    ("01", "Data Upload"),
    ("02", "Exploratory Analysis"),
    ("03", "Feature Engineering"),
    ("04", "Model Training"),
    ("05", "Results"),
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
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂  Data Upload",
    "📊  EDA",
    "⚙️  Features",
    "🚀  Training",
    "📈  Results",
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
            st.dataframe(pd.read_csv(DATA_DIR / "master_dataset.csv", nrows=200), use_container_width=True, height=240)
        with st.expander("Preview: CHIRPS Rainfall"):
            st.dataframe(pd.read_csv(DATA_DIR / "chirps_kabini_daily.csv", nrows=200), use_container_width=True, height=240)

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


# ------ TAB 3: FEATURE ENGINEERING ------
with tab3:
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
                    st.dataframe(pd.read_csv(corr_csv, index_col=0), use_container_width=True, height=300)
            st.divider()
            show_images(FEAT_DIR, cols=2)


# ------ TAB 4: MODEL TRAINING ------
with tab4:
    st.markdown('<p class="sec-desc">Train an XGBoost regressor on log-transformed discharge with a strict temporal split (pre-2018 train / 2018+ test). Evaluates RMSE, MAE, R², and Nash-Sutcliffe Efficiency.</p>', unsafe_allow_html=True)

    if not st.session_state.feat_done:
        st.warning("Run Feature Engineering (Step 3) first.")
    else:
        bcol, _ = st.columns([1, 3])
        with bcol:
            if st.button("▶  Train Model", type="primary", key="btn_train", use_container_width=True):
                if run_script("train_model.py"):
                    st.session_state.model_done = True
                    st.rerun()

        if st.session_state.model_done:
            st.divider()
            show_images(MODEL_DIR, cols=2)


# ------ TAB 5: RESULTS ------
with tab5:
    st.markdown('<p class="sec-desc">Final model performance and downloadable predictions.</p>', unsafe_allow_html=True)

    preds_csv = MODEL_DIR / "test_predictions.csv"

    if not st.session_state.model_done and not preds_csv.exists():
        st.warning("Train the model (Step 4) to see results.")
    elif preds_csv.exists():
        df_res = pd.read_csv(preds_csv)
        obs  = df_res["observed"].values
        pred = df_res["predicted"].values

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

        st.write("")
        show_images(MODEL_DIR, cols=2)

        st.divider()
        st.subheader("Test Set Predictions")
        st.dataframe(
            df_res.style.format({
                "observed": "{:,.2f}",
                "predicted": "{:,.2f}",
                "residual": "{:+,.2f}",
            }),
            use_container_width=True, height=340,
        )
        st.download_button(
            label="⬇  Download predictions CSV",
            data=preds_csv.read_bytes(),
            file_name="test_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Predictions file not found.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="foot">
  Kabini Basin Discharge Forecasting &nbsp;·&nbsp;
  <code>XGBoost</code> &nbsp;·&nbsp; <code>CHIRPS</code> &nbsp;·&nbsp; <code>CWC</code>
  &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)