import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

print("=" * 60)
print("  HYDROLOGY ANALYSIS: Unit Hydrograph & Basin Parameters")
print("=" * 60)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
HYDRO_DIR  = BASE_DIR / "hydrology_outputs"
HYDRO_DIR.mkdir(exist_ok=True)

master_path = BASE_DIR / "data/master_dataset.csv"

# --------------------------------------------------
# LOAD
# --------------------------------------------------
df = pd.read_csv(master_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Rename for convenience
df.rename(columns={
    'rainfall_max_mm': 'P',        # Precipitation (mm)
    'q_upstream_mk':   'Q_total',  # Total discharge (m³/s)
}, inplace=True)

print(f"Dataset: {df.shape[0]} rows, {df['date'].min().date()} → {df['date'].max().date()}")


# ==========================================================
#  1. BASEFLOW SEPARATION (Eckhardt recursive digital filter)
#
#  The Eckhardt (2005) filter is one of the most widely
#  used automated methods in hydrology:
#
#    b(t) = [(1 - BFImax) * a * b(t-1) + (1 - a) * BFImax * Q(t)]
#           / (1 - a * BFImax)
#
#  where a = recession constant, BFImax = maximum baseflow
#  index. Constraint: b(t) <= Q(t).
# ==========================================================
print("\n" + "=" * 60)
print("  1. BASEFLOW SEPARATION (Eckhardt filter)")
print("=" * 60)

def eckhardt_filter(Q, a=0.975, BFImax=0.80):
    """Eckhardt (2005) recursive digital baseflow filter."""
    b = np.zeros_like(Q, dtype=float)
    b[0] = min(Q[0], Q[0] * BFImax)  # initialize
    for t in range(1, len(Q)):
        b[t] = ((1 - BFImax) * a * b[t-1] + (1 - a) * BFImax * Q[t]) / (1 - a * BFImax)
        b[t] = min(b[t], Q[t])  # baseflow cannot exceed total flow
    return b

Q_total = df['Q_total'].values

# Estimate recession constant from falling limbs
# (ratio of Q(t) / Q(t-1) during recession periods)
dQ = np.diff(Q_total)
recession_mask = dQ < 0  # falling limb
ratios = Q_total[1:][recession_mask] / Q_total[:-1][recession_mask]
ratios = ratios[(ratios > 0.5) & (ratios < 1.0)]  # filter outliers
recession_constant = np.median(ratios)
print(f"   Estimated recession constant (a): {recession_constant:.4f}")

# Apply filter
df['Q_base']   = eckhardt_filter(Q_total, a=recession_constant, BFImax=0.80)
df['Q_direct'] = df['Q_total'] - df['Q_base']
df['Q_direct'] = df['Q_direct'].clip(lower=0)

BFI = df['Q_base'].sum() / df['Q_total'].sum()
print(f"   Baseflow Index (BFI): {BFI:.3f}")
print(f"   Mean baseflow:  {df['Q_base'].mean():.2f} m³/s")
print(f"   Mean direct RO: {df['Q_direct'].mean():.2f} m³/s")

# --- Plot: Baseflow separation (1 year sample) ---
sample = df[(df['date'].dt.year == 2017)].copy()
fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(sample['date'], 0, sample['Q_base'],
                alpha=0.4, color='#3B8BD4', label='Baseflow')
ax.fill_between(sample['date'], sample['Q_base'], sample['Q_total'],
                alpha=0.4, color='#E8593C', label='Direct runoff')
ax.plot(sample['date'], sample['Q_total'],
        color='#1a1a1a', linewidth=0.8, label='Total discharge')
ax.set_title("Baseflow Separation — 2017 (Eckhardt Filter)")
ax.set_xlabel("Date")
ax.set_ylabel("Discharge (m³/s)")
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.tight_layout()
plt.savefig(HYDRO_DIR / "baseflow_separation.png", dpi=150)
plt.close()
print(f"   📸 Saved: baseflow_separation.png")


# ==========================================================
#  2. RECESSION ANALYSIS & RECESSION CONSTANT
#
#  During dry periods, discharge decays exponentially:
#    Q(t) = Q0 * k^t
#
#  k is the recession constant — characterizes how fast
#  the basin drains. Values close to 1 = slow drainage
#  (large groundwater storage); close to 0 = flashy.
# ==========================================================
print("\n" + "=" * 60)
print("  2. RECESSION ANALYSIS")
print("=" * 60)

# Extract recession segments (>= 5 consecutive falling days)
min_length = 5
segments = []
current = []

for i in range(1, len(Q_total)):
    if Q_total[i] < Q_total[i-1]:
        current.append(i)
    else:
        if len(current) >= min_length:
            segments.append(current.copy())
        current = [i]

# Fit exponential to each segment
k_values = []
for seg in segments:
    q_seg = Q_total[seg]
    if q_seg.min() <= 0:
        continue
    t_seg = np.arange(len(q_seg))
    log_q = np.log(q_seg)
    try:
        slope, intercept = np.polyfit(t_seg, log_q, 1)
        k = np.exp(slope)
        if 0.5 < k < 1.0:
            k_values.append(k)
    except:
        continue

k_median = np.median(k_values) if k_values else recession_constant
print(f"   Recession segments analysed: {len(k_values)}")
print(f"   Recession constant (k):  {k_median:.4f}")
print(f"   Half-life of recession:   {-np.log(2)/np.log(k_median):.1f} days")

# Master recession curve
fig, ax = plt.subplots(figsize=(8, 5))
for seg in segments[:30]:
    q_seg = Q_total[seg]
    if q_seg.min() > 0 and len(q_seg) >= min_length:
        t_norm = np.arange(len(q_seg))
        ax.semilogy(t_norm, q_seg, color='#3B8BD4', alpha=0.25, linewidth=0.8)

# Overlay theoretical curve
t_fit = np.arange(0, 60)
q_fit = np.median(Q_total) * k_median ** t_fit
ax.semilogy(t_fit, q_fit, 'r--', linewidth=2,
            label=f'Theoretical (k={k_median:.3f})')
ax.set_title("Master Recession Curve")
ax.set_xlabel("Days since peak")
ax.set_ylabel("Discharge (m³/s, log scale)")
ax.legend()
ax.set_xlim(0, 50)
plt.tight_layout()
plt.savefig(HYDRO_DIR / "recession_curve.png", dpi=150)
plt.close()
print(f"   📸 Saved: recession_curve.png")


# ==========================================================
#  3. UNIT HYDROGRAPH DERIVATION
#
#  The unit hydrograph (UH) is the direct runoff response
#  of the basin to 1 unit (1 mm) of excess rainfall in one
#  time step (1 day). Derived by:
#    1. Identify isolated storm events
#    2. Separate baseflow → get direct runoff hydrograph
#    3. Compute excess rainfall (P - losses)
#    4. Normalize: UH = direct_runoff / excess_rainfall_depth
#
#  We derive an "average" UH from multiple events and also
#  fit the SCS dimensionless UH for the Tp/Qp parameters.
# ==========================================================
print("\n" + "=" * 60)
print("  3. UNIT HYDROGRAPH DERIVATION")
print("=" * 60)

# Identify storm events: days where P > threshold followed
# by a clear hydrograph response (Q rises next 1-3 days)
P = df['P'].values
Q_dir = df['Q_direct'].values
rain_threshold = 10  # mm, minimum for a usable event

event_starts = []
for i in range(1, len(P) - 15):
    if P[i] > rain_threshold and P[i-1] < 2:
        # Check for a clear response within 3 days
        if Q_dir[i+1:i+4].max() > Q_dir[i] * 1.5:
            event_starts.append(i)

# Extract UH from each event
uh_collection = []
response_len = 15  # days to capture the full response

for start in event_starts:
    end = min(start + response_len, len(Q_dir))
    q_event = Q_dir[start:end].copy()

    # Excess rainfall (simplified: total event rainfall)
    rain_event = P[start:start+3].sum()  # 3-day storm window
    if rain_event < 5:
        continue

    # Normalize to unit hydrograph (response per mm of excess)
    uh = q_event / rain_event
    if len(uh) == response_len:
        uh_collection.append(uh)

print(f"   Usable storm events found: {len(uh_collection)}")

if len(uh_collection) >= 3:
    # Average unit hydrograph
    uh_avg = np.mean(uh_collection, axis=0)
    uh_std = np.std(uh_collection, axis=0)
    t_uh = np.arange(response_len)

    # UH parameters
    Qp_uh = uh_avg.max()
    Tp_uh = t_uh[uh_avg.argmax()]
    Tb_uh = response_len  # time base

    # Find time to centroid
    Tc_uh = np.sum(t_uh * uh_avg) / np.sum(uh_avg)

    print(f"   UH Peak (Qp):       {Qp_uh:.4f} (m³/s)/mm")
    print(f"   Time to peak (Tp):  {Tp_uh} days")
    print(f"   Time base (Tb):     {Tb_uh} days")
    print(f"   Lag time (Tc):      {Tc_uh:.1f} days")

    # --- Plot: Unit Hydrograph ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # Individual event UHs (faint)
    for uh_i in uh_collection:
        ax.plot(t_uh, uh_i, color='#3B8BD4', alpha=0.15, linewidth=0.7)

    # Average UH with uncertainty band
    ax.fill_between(t_uh, uh_avg - uh_std, uh_avg + uh_std,
                    alpha=0.2, color='#E8593C')
    ax.plot(t_uh, uh_avg, color='#E8593C', linewidth=2.5,
            label=f'Average UH (Tp={Tp_uh}d, Qp={Qp_uh:.3f})')

    # Annotate peak
    ax.annotate(f'Qp = {Qp_uh:.3f}\nTp = {Tp_uh} d',
                xy=(Tp_uh, Qp_uh), xytext=(Tp_uh + 2, Qp_uh * 0.9),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_title(f"Derived Unit Hydrograph (1-day, {len(uh_collection)} events)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Discharge per mm excess rainfall (m³/s)/mm")
    ax.legend(loc='upper right')
    ax.set_xlim(0, response_len - 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(HYDRO_DIR / "unit_hydrograph.png", dpi=150)
    plt.close()
    print(f"   📸 Saved: unit_hydrograph.png")

    # --- Plot: SCS Dimensionless UH comparison ---
    # Normalize: t/Tp on x-axis, Q/Qp on y-axis
    t_dimless = t_uh / max(Tp_uh, 1)
    q_dimless = uh_avg / max(Qp_uh, 1e-9)

    # SCS standard dimensionless UH (tabulated ratios)
    scs_t_tp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                         1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                         2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0])
    scs_q_qp = np.array([0.000, 0.030, 0.100, 0.190, 0.310, 0.470, 0.660,
                          0.820, 0.930, 0.990, 1.000, 0.990, 0.930, 0.860,
                          0.780, 0.680, 0.560, 0.460, 0.390, 0.330, 0.280,
                          0.207, 0.147, 0.107, 0.077, 0.055, 0.025, 0.011,
                          0.005, 0.000])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_dimless, q_dimless, 'o-', color='#E8593C', markersize=4,
            linewidth=1.5, label='Derived (Kabini)')
    ax.plot(scs_t_tp, scs_q_qp, 's--', color='#3B8BD4', markersize=3,
            linewidth=1.5, label='SCS standard UH')
    ax.set_title("Dimensionless Unit Hydrograph Comparison")
    ax.set_xlabel("t / Tp")
    ax.set_ylabel("Q / Qp")
    ax.legend()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(HYDRO_DIR / "dimensionless_uh_comparison.png", dpi=150)
    plt.close()
    print(f"   📸 Saved: dimensionless_uh_comparison.png")

else:
    print("   ⚠️  Not enough isolated events for UH derivation.")
    Tp_uh, Qp_uh, Tb_uh, Tc_uh = None, None, None, None


# ==========================================================
#  4. SCS CURVE NUMBER ESTIMATION
#
#  The SCS-CN method relates rainfall to runoff:
#    Q = (P - 0.2S)² / (P + 0.8S)  for P > 0.2S
#  where S = (25400/CN) - 254 (in mm)
#
#  We back-calculate CN from observed P-Q event pairs.
# ==========================================================
print("\n" + "=" * 60)
print("  4. SCS CURVE NUMBER ESTIMATION")
print("=" * 60)

# Aggregate events: daily P and direct runoff depth
# Convert Q_direct (m³/s) to mm/day — requires catchment area
# Kabini upstream catchment ~2100 km² (typical estimate)
# If you have the exact area, replace this value.
CATCHMENT_AREA_KM2 = 2100
CATCHMENT_AREA_M2  = CATCHMENT_AREA_KM2 * 1e6

# Convert m³/s to mm/day: Q(m³/s) * 86400(s/day) / Area(m²) * 1000(mm/m)
df['Q_direct_mm'] = df['Q_direct'] * 86400 / CATCHMENT_AREA_M2 * 1000
df['Q_total_mm']  = df['Q_total']  * 86400 / CATCHMENT_AREA_M2 * 1000

# Event-based: aggregate 3-day windows
event_P = []
event_Q = []
for i in range(0, len(df) - 3, 3):
    p_sum = df['P'].iloc[i:i+3].sum()
    q_sum = df['Q_direct_mm'].iloc[i:i+3].sum()
    if p_sum > 10 and q_sum > 0:
        event_P.append(p_sum)
        event_Q.append(q_sum)

event_P = np.array(event_P)
event_Q = np.array(event_Q)

# Back-calculate S for each event using SCS equation:
# Q = (P - 0.2S)² / (P + 0.8S)
# Solve for S given P and Q:
# This is a quadratic — use numerical approach
cn_values = []
for p, q in zip(event_P, event_Q):
    if q >= p or q <= 0:
        continue
    # Search for S
    for S_try in np.arange(1, 500, 0.5):
        Ia = 0.2 * S_try  # Initial abstraction (0.2S for US; 0.3S for India)
        if p <= Ia:
            continue
        Q_calc = (p - Ia) ** 2 / (p - Ia + S_try)
        if abs(Q_calc - q) < 0.5:
            cn = 25400 / (S_try + 254)
            if 30 < cn < 100:
                cn_values.append(cn)
            break

if cn_values:
    CN_median = np.median(cn_values)
    CN_mean   = np.mean(cn_values)
    S_avg     = (25400 / CN_median) - 254
    print(f"   Events used for CN estimation: {len(cn_values)}")
    print(f"   Median Curve Number (CN):  {CN_median:.1f}")
    print(f"   Mean Curve Number (CN):    {CN_mean:.1f}")
    print(f"   Potential max retention (S): {S_avg:.1f} mm")
    print(f"   Initial abstraction (Ia=0.2S): {0.2*S_avg:.1f} mm")

    # Plot: P-Q scatter with SCS curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(event_P, event_Q, alpha=0.3, s=15, c='#3B8BD4', label='Observed events')

    # Overlay SCS curves for different CN values
    P_range = np.linspace(0, event_P.max() * 1.1, 200)
    for cn_plot in [60, 70, 80, 90]:
        S_plot = (25400 / cn_plot) - 254
        Ia_plot = 0.2 * S_plot
        Q_plot = np.where(P_range > Ia_plot,
                          (P_range - Ia_plot) ** 2 / (P_range - Ia_plot + S_plot), 0)
        ax.plot(P_range, Q_plot, '--', linewidth=1,
                label=f'CN={cn_plot}', alpha=0.7)

    # Highlight the estimated CN
    S_est = (25400 / CN_median) - 254
    Ia_est = 0.2 * S_est
    Q_est = np.where(P_range > Ia_est,
                     (P_range - Ia_est) ** 2 / (P_range - Ia_est + S_est), 0)
    ax.plot(P_range, Q_est, '-', linewidth=2.5, color='#E8593C',
            label=f'Estimated CN={CN_median:.0f}')

    ax.set_title("SCS Curve Number — Observed vs Theoretical")
    ax.set_xlabel("3-day Rainfall (mm)")
    ax.set_ylabel("3-day Direct Runoff (mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(HYDRO_DIR / "scs_curve_number.png", dpi=150)
    plt.close()
    print(f"   📸 Saved: scs_curve_number.png")
else:
    CN_median = None
    print("   ⚠️  Could not estimate CN — check data.")


# ==========================================================
#  5. RUNOFF COEFFICIENT & WATER BALANCE
# ==========================================================
print("\n" + "=" * 60)
print("  5. RUNOFF COEFFICIENT & WATER BALANCE")
print("=" * 60)

# Annual summaries
df['year'] = df['date'].dt.year
annual = df.groupby('year').agg(
    P_total   = ('P', 'sum'),
    Q_total   = ('Q_total_mm', 'sum'),
    Q_base    = ('Q_base', lambda x: (x * 86400 / CATCHMENT_AREA_M2 * 1000).sum()),
    Q_direct  = ('Q_direct_mm', 'sum'),
).reset_index()
annual['C_runoff'] = annual['Q_total'] / annual['P_total']
annual['C_runoff'] = annual['C_runoff'].clip(0, 1)

print(f"\n   Annual Water Balance Summary:")
print(f"   {'Year':>6}  {'Rain(mm)':>10}  {'Runoff(mm)':>11}  {'Coeff':>7}")
print(f"   {'-'*40}")
for _, row in annual.iterrows():
    print(f"   {int(row['year']):>6}  {row['P_total']:>10.1f}  {row['Q_total']:>11.1f}  {row['C_runoff']:>7.3f}")

C_avg = annual['C_runoff'].mean()
print(f"\n   Average runoff coefficient (C): {C_avg:.3f}")

# Plot: Annual water balance
fig, ax1 = plt.subplots(figsize=(10, 5))
x = annual['year']
w = 0.35
ax1.bar(x - w/2, annual['P_total'], w, label='Rainfall', color='#3B8BD4', alpha=0.7)
ax1.bar(x + w/2, annual['Q_total'], w, label='Runoff', color='#E8593C', alpha=0.7)
ax1.set_xlabel("Year")
ax1.set_ylabel("Depth (mm)")
ax1.set_title(f"Annual Water Balance (Avg C = {C_avg:.3f})")
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(x, annual['C_runoff'], 'ko-', markersize=5, label='Runoff coefficient')
ax2.set_ylabel("Runoff Coefficient")
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(HYDRO_DIR / "annual_water_balance.png", dpi=150)
plt.close()
print(f"   📸 Saved: annual_water_balance.png")


# ==========================================================
#  6. TIME OF CONCENTRATION (Kirpich Formula)
#
#  Tc = 0.0195 * L^0.77 * S^(-0.385)
#  where L = longest flow path (m), S = slope (m/m)
#
#  For Kabini: approximate values (replace with actual
#  DEM-derived values for your presentation)
# ==========================================================
print("\n" + "=" * 60)
print("  6. TIME OF CONCENTRATION")
print("=" * 60)

# Approximate values for Kabini upstream — REPLACE WITH ACTUAL
L_km = 85       # longest flow path (km) — REPLACE
L_m  = L_km * 1000
H_m  = 600      # elevation difference (m) — REPLACE
S_slope = H_m / L_m  # average slope

# Kirpich formula
Tc_kirpich = 0.0195 * (L_m ** 0.77) * (S_slope ** (-0.385))
Tc_hours = Tc_kirpich / 60  # minutes to hours
print(f"   Longest flow path (L):     {L_km} km")
print(f"   Elevation difference (H):  {H_m} m")
print(f"   Average slope (S):         {S_slope:.5f} m/m")
print(f"   Tc (Kirpich):              {Tc_hours:.1f} hours ({Tc_hours/24:.1f} days)")

# Also estimate from observed lag
# Lag time from peak rainfall to peak discharge
lag_days = []
for i in range(2, len(df) - 5):
    if df['P'].iloc[i] > 30:  # significant rain
        q_window = df['Q_total'].iloc[i:i+5].values
        lag = np.argmax(q_window)
        if lag > 0:
            lag_days.append(lag)

if lag_days:
    lag_obs = np.median(lag_days)
    print(f"   Observed median lag time:   {lag_obs:.0f} days")
    # SCS relation: Tp ≈ 0.6 * Tc (for SCS UH)
    Tc_from_lag = lag_obs / 0.6
    print(f"   Tc estimated from lag:      {Tc_from_lag:.1f} days ({Tc_from_lag*24:.0f} hours)")


# ==========================================================
#  7. FLOW DURATION CURVE (FDC)
#
#  Shows the percentage of time a given flow is equalled
#  or exceeded. Essential for water availability assessment.
# ==========================================================
print("\n" + "=" * 60)
print("  7. FLOW DURATION CURVE")
print("=" * 60)

Q_sorted = np.sort(df['Q_total'].values)[::-1]
n = len(Q_sorted)
exceedance = np.arange(1, n + 1) / n * 100

# Key percentiles
Q50 = np.interp(50, exceedance, Q_sorted)
Q75 = np.interp(75, exceedance, Q_sorted)
Q90 = np.interp(90, exceedance, Q_sorted)
Q95 = np.interp(95, exceedance, Q_sorted)

print(f"   Q50  (median flow):        {Q50:.2f} m³/s")
print(f"   Q75  (dependable flow):    {Q75:.2f} m³/s")
print(f"   Q90  (drought indicator):  {Q90:.2f} m³/s")
print(f"   Q95  (low flow):           {Q95:.2f} m³/s")

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(exceedance, Q_sorted, color='#3B8BD4', linewidth=1.5)
ax.axhline(Q50, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'Q50 = {Q50:.1f}', xy=(50, Q50), fontsize=9,
            xytext=(55, Q50*1.5), arrowprops=dict(arrowstyle='->', color='gray'))
ax.axhline(Q90, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'Q90 = {Q90:.1f}', xy=(90, Q90), fontsize=9,
            xytext=(75, Q90*0.5), arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_title("Flow Duration Curve")
ax.set_xlabel("Exceedance Probability (%)")
ax.set_ylabel("Discharge (m³/s, log scale)")
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(HYDRO_DIR / "flow_duration_curve.png", dpi=150)
plt.close()
print(f"   📸 Saved: flow_duration_curve.png")


# ==========================================================
#  8. FLOOD FREQUENCY ANALYSIS (Annual maxima + Gumbel)
# ==========================================================
print("\n" + "=" * 60)
print("  8. FLOOD FREQUENCY ANALYSIS")
print("=" * 60)

annual_max = df.groupby('year')['Q_total'].max().values
n_years = len(annual_max)

# Gumbel distribution fitting (method of moments)
y_mean = np.mean(annual_max)
y_std  = np.std(annual_max, ddof=1)
alpha_g = y_std * np.sqrt(6) / np.pi       # scale
mu_g    = y_mean - 0.5772 * alpha_g         # location

# Return period flows
return_periods = [2, 5, 10, 25, 50, 100]
print(f"\n   {'T (years)':>10}  {'Q (m³/s)':>10}")
print(f"   {'-'*25}")
for T in return_periods:
    yT = -np.log(-np.log(1 - 1/T))
    QT = mu_g + alpha_g * yT
    print(f"   {T:>10}  {QT:>10.1f}")

# Plot: Gumbel probability
fig, ax = plt.subplots(figsize=(8, 5))
annual_max_sorted = np.sort(annual_max)
plotting_pos = np.arange(1, n_years + 1) / (n_years + 1)
return_period_obs = 1 / (1 - plotting_pos)
gumbel_y = -np.log(-np.log(plotting_pos))

ax.scatter(gumbel_y, annual_max_sorted, c='#E8593C', s=40, zorder=5, label='Observed')
y_line = np.linspace(-1, 5, 100)
q_line = mu_g + alpha_g * y_line
ax.plot(y_line, q_line, '--', color='#3B8BD4', linewidth=1.5, label='Gumbel fit')

# Add return period axis labels
ax2 = ax.twiny()
rp_ticks = [2, 5, 10, 25, 50, 100]
rp_y     = [-np.log(-np.log(1 - 1/T)) for T in rp_ticks]
ax2.set_xticks(rp_y)
ax2.set_xticklabels([str(t) for t in rp_ticks])
ax2.set_xlabel("Return Period (years)")
ax2.set_xlim(ax.get_xlim())

ax.set_title("Flood Frequency Analysis (Gumbel)")
ax.set_xlabel("Gumbel Reduced Variate")
ax.set_ylabel("Annual Maximum Discharge (m³/s)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(HYDRO_DIR / "flood_frequency.png", dpi=150)
plt.close()
print(f"   📸 Saved: flood_frequency.png")


# ==========================================================
#  9. SUMMARY TABLE — All Parameters
# ==========================================================
print("\n" + "=" * 60)
print("  9. PARAMETER SUMMARY")
print("=" * 60)

params = {
    'Catchment area (km²)':          CATCHMENT_AREA_KM2,
    'Record period':                  f"{df['date'].min().date()} to {df['date'].max().date()}",
    'Mean annual rainfall (mm)':     annual['P_total'].mean(),
    'Mean annual runoff (mm)':       annual['Q_total'].mean(),
    'Runoff coefficient (C)':        C_avg,
    'Baseflow Index (BFI)':          BFI,
    'Recession constant (k)':        k_median,
    'Recession half-life (days)':    -np.log(2)/np.log(k_median),
    'SCS Curve Number (CN)':         CN_median if CN_median else 'N/A',
    'Potential max retention S (mm)': S_avg if CN_median else 'N/A',
    'UH Time to peak Tp (days)':     Tp_uh if Tp_uh else 'N/A',
    'UH Peak Qp (m³/s/mm)':         f"{Qp_uh:.4f}" if Qp_uh else 'N/A',
    'UH Time base Tb (days)':        Tb_uh if Tb_uh else 'N/A',
    'UH Lag time (days)':            f"{Tc_uh:.1f}" if Tc_uh else 'N/A',
    'Tc Kirpich (hours)':            f"{Tc_hours:.1f}",
    'Q50 median flow (m³/s)':        Q50,
    'Q90 low flow (m³/s)':           Q90,
    'Q95 drought flow (m³/s)':       Q95,
    'Mean discharge (m³/s)':         df['Q_total'].mean(),
    'Max discharge (m³/s)':          df['Q_total'].max(),
    'Min discharge (m³/s)':          df['Q_total'].min(),
}

summary_df = pd.DataFrame([
    {'Parameter': k, 'Value': v} for k, v in params.items()
])
summary_df.to_csv(HYDRO_DIR / "hydrology_parameters.csv", index=False)

print(f"\n   {'Parameter':<40}  {'Value':>15}")
print(f"   {'='*58}")
for _, row in summary_df.iterrows():
    val = row['Value']
    if isinstance(val, float):
        val = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
    print(f"   {row['Parameter']:<40}  {str(val):>15}")

print(f"\n   💾 Saved: hydrology_parameters.csv")

print("\n✅ Hydrology analysis complete.")
print("=" * 60)