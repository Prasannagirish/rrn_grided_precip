import pandas as pd
from pathlib import Path

print("=" * 55)
print("  PHASE 2: DATA CLEANING & ALIGNMENT")
print("=" * 55)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

chirps_path = BASE_DIR / "data/chirps_kabini_daily.csv"
mk_path     = BASE_DIR / "data/River Water Discharge_Muthankera.xlsx"

output_path = BASE_DIR / "data/master_dataset.csv"

# --------------------------------------------------
# LOAD CHIRPS
# --------------------------------------------------
print(f"Loading CHIRPS from {chirps_path}...")
df_chirps = pd.read_csv(chirps_path)

df_chirps.columns = df_chirps.columns.str.strip().str.lower()

if 'date' not in df_chirps.columns:
    raise ValueError("❌ CHIRPS file must contain a 'date' column")

df_chirps['date'] = pd.to_datetime(df_chirps['date'], errors='coerce')
df_chirps['date'] = df_chirps['date'].dt.floor('D')
df_chirps = df_chirps.dropna(subset=['date'])

# --------------------------------------------------
# Detect and standardize the rainfall column name.
# CHIRPS columns vary (e.g. 'precipitation', 'chirps_v2.0_...',
# 'rainfall', 'prcp'). We find whichever is present and rename
# it to 'rainfall_max_mm' so both scripts share one agreed name.
# --------------------------------------------------
_rain_candidates = [
    c for c in df_chirps.columns
    if any(kw in c for kw in ('precip', 'rain', 'chirps', 'prcp', 'rf'))
    and c != 'date'
]

if not _rain_candidates:
    raise ValueError(
        f"❌ Could not detect a rainfall column in CHIRPS. "
        f"Available columns: {list(df_chirps.columns)}"
    )

_original_rain_col = _rain_candidates[0]
print(f"✅ Rainfall column detected: '{_original_rain_col}' → renamed to 'rainfall_max_mm'")

df_chirps = df_chirps.rename(columns={_original_rain_col: 'rainfall_max_mm'})
df_chirps = df_chirps[['date', 'rainfall_max_mm']]

# --------------------------------------------------
# WRIS CLEANER
# --------------------------------------------------
def clean_wris_file(path, flow_col_name):
    print(f"Loading Discharge from {path}...")

    df_raw = pd.read_excel(path, header=None)

    header_row_idx = None
    for i in range(len(df_raw)):
        row = df_raw.iloc[i].astype(str).str.lower()
        if row.str.contains('data time').any():
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError(f"❌ Could not find 'Data Time' header in {path.name}")

    print(f"✅ Found header at row: {header_row_idx}")

    df = pd.read_excel(path, header=header_row_idx)
    df.columns = df.columns.str.strip().str.lower()

    if 'data time' not in df.columns or 'data value' not in df.columns:
        raise ValueError(f"❌ Required columns missing in {path.name}")

    df_final = df[['data time', 'data value']].copy()
    df_final.columns = ['date', flow_col_name]

    df_final['date'] = pd.to_datetime(df_final['date'], errors='coerce')
    df_final[flow_col_name] = (
        df_final[flow_col_name]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('--', '', regex=False)
        .str.strip()
    )
    df_final[flow_col_name] = pd.to_numeric(df_final[flow_col_name], errors='coerce')

    df_final['date'] = df_final['date'].dt.floor('D')
    df_final = df_final.dropna(subset=['date'])
    df_final = df_final.groupby('date').mean().reset_index()
    df_final = df_final.sort_values('date')

    print(f"✅ Cleaned {path.name} | Rows: {len(df_final)}")
    return df_final

# --------------------------------------------------
# LOAD DISCHARGE
# --------------------------------------------------
df_mk = clean_wris_file(mk_path, 'q_upstream_mk')

# --------------------------------------------------
# DEBUG DATE ALIGNMENT
# --------------------------------------------------
print("\n🔍 DATE CHECK")
print("CHIRPS:", df_chirps['date'].min(), "→", df_chirps['date'].max())
print("MK     :", df_mk['date'].min(),     "→", df_mk['date'].max())

# --------------------------------------------------
# MERGE
# --------------------------------------------------
print("\nMerging datasets...")
df_master = df_chirps.merge(df_mk, on='date', how='left')
df_master = df_master.sort_values('date')

# --------------------------------------------------
# Interpolate MK discharge
# --------------------------------------------------
valid_count = df_master['q_upstream_mk'].notna().sum()
if valid_count > 0:
    df_master['q_upstream_mk'] = df_master['q_upstream_mk'].interpolate(method='linear')
    print(f"✅ Interpolated 'q_upstream_mk' ({valid_count} valid values present)")
else:
    print("⚠️  'q_upstream_mk' has NO valid values — skipping interpolation")

# --------------------------------------------------
# SAVE
# --------------------------------------------------
df_master.to_csv(output_path, index=False)

print(f"\n✅ Master dataset saved to {output_path}")
print(f"Final shape: {df_master.shape}")
print("\nMissing values:")
print(df_master[['q_upstream_mk']].isna().sum())
print("=" * 55)