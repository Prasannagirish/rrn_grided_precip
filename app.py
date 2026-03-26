import streamlit as st
import subprocess
import os
from pathlib import Path
from PIL import Image

# --- Configuration ---
st.set_page_config(page_title="Hydrology ML Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
EDA_DIR = BASE_DIR / "eda_outputs"
FEAT_DIR = BASE_DIR / "feature_outputs"
MODEL_DIR = BASE_DIR / "model_outputs"

# --- Helper Functions ---
def run_script(script_name):
    """Executes a python script and displays the terminal output."""
    script_path = BASE_DIR / script_name
    
    if not script_path.exists():
        st.error(f"Could not find `{script_name}` in {BASE_DIR}")
        return False

    with st.spinner(f"Running {script_name}... This might take a moment."):
        # Run the script as a subprocess
        result = subprocess.run(
            ["python", str(script_path)], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            st.success(f"✅ Successfully executed `{script_name}`")
            with st.expander("View Terminal Logs", expanded=False):
                st.code(result.stdout)
            return True
        else:
            st.error(f"❌ Error executing `{script_name}`")
            with st.expander("View Error Logs", expanded=True):
                st.code(result.stderr)
            return False

def display_images_from_dir(directory):
    """Finds and displays all PNG images in a given directory."""
    if not directory.exists():
        st.info(f"Directory `{directory.name}` does not exist yet. Run the pipeline to generate outputs.")
        return

    image_files = list(directory.glob("*.png"))
    
    if not image_files:
        st.info(f"No images found in `{directory.name}`. Run the pipeline to generate them.")
        return

    # Display images in a grid (2 columns)
    cols = st.columns(2)
    for idx, img_path in enumerate(image_files):
        with cols[idx % 2]:
            try:
                img = Image.open(img_path)
                st.image(img, caption=img_path.name, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load {img_path.name}: {e}")

# --- Dashboard UI ---
st.title("🌊 River Discharge Prediction Pipeline")
st.write("Execute pipeline phases and view the generated hydrological analytics and model evaluations.")

# Create tabs for each phase of the pipeline
tab1, tab2, tab3 = st.tabs(["📊 Phase 1: EDA", "⚙️ Phase 2: Feature Engineering", "🚀 Phase 3: Model Training"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("Generate distributions, time-series, and correlation matrices for rainfall and discharge.")
        if st.button("Run EDA Phase", type="primary", key="btn_eda"):
            run_script("eda.py")
            
    with col2:
        st.subheader("EDA Outputs")
        display_images_from_dir(EDA_DIR)

# --- TAB 2: Feature Engineering ---
with tab2:
    st.header("Feature Engineering")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("Apply lag transformations, rolling windows, and log-transforms to prepare the dataset.")
        if st.button("Run Feature Engineering", type="primary", key="btn_feat"):
            run_script("feature_engineering.py")
            
    with col2:
        st.subheader("Feature Outputs")
        display_images_from_dir(FEAT_DIR)

# --- TAB 3: Model Training ---
with tab3:
    st.header("XGBoost Model Training")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("Train the XGBoost regressor, calculate NSE/RMSE metrics, and plot predictions.")
        if st.button("Run Model Training", type="primary", key="btn_train"):
            run_script("train_model.py")
            
    with col2:
        st.subheader("Model Evaluation Outputs")
        display_images_from_dir(MODEL_DIR)

st.divider()
st.caption("Dashboard connected to local pipeline execution. Outputs are refreshed directly from the local file system.")