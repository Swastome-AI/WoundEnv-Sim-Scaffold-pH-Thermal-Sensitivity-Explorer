import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="WoundEnv Sim | pH & Thermal Sensitivity",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ WoundEnv Sim: Scaffold pH & Thermal Sensitivity Explorer")
st.markdown(
    "Interactive simulation comparing **Formulation A (Collagen + Chitosan Control)** vs. "
    "**Formulation B (Collagen + Hyaluronic Acid + Chitosan)** across wound bed pH and temperature regimes."
)

# --- SIDEBAR: PDB UPLOAD & CONCENTRATION CONTROLS ---
st.sidebar.header("1. Structure & Recipe Inputs")
uploaded_ha = st.sidebar.file_uploader(
    "Upload GLYCAM Hyaluronic Acid PDB (.pdb) [Optional]", type=["pdb"]
)

st.sidebar.subheader("Polymer Concentrations (% w/v)")
conc_col = st.sidebar.slider("Collagen (% w/v)", 0.1, 5.0, 1.0, 0.1)
conc_chi = st.sidebar.slider("Chitosan (% w/v)", 0.1, 5.0, 1.0, 0.1)
conc_ha = st.sidebar.slider("Hyaluronic Acid (% w/v) [Formulation B]", 0.1, 5.0, 1.0, 0.1)
film_thickness = st.sidebar.number_input("Film Thickness (mm)", 0.05, 5.0, 0.5, 0.05)

st.sidebar.divider()
st.sidebar.header("2. Wound Bed Presets")
wound_preset = st.sidebar.radio(
    "Select Clinical Scenario",
    [
        "Healthy / Healing Skin (pH 5.5, 36.5 °C)",
        "Physiological / Lab PBS (pH 7.4, 37.0 °C)",
        "Infected / Inflamed Wound (pH 8.5, 39.5 °C)",
        "Room Temp Storage / Prep (pH 7.0, 25.0 °C)",
        "Custom Slider Values"
    ],
    index=1
)

preset_map = {
    "Healthy / Healing Skin (pH 5.5, 36.5 °C)": (5.5, 36.5),
    "Physiological / Lab PBS (pH 7.4, 37.0 °C)": (7.4, 37.0),
    "Infected / Inflamed Wound (pH 8.5, 39.5 °C)": (8.5, 39.5),
    "Room Temp Storage / Prep (pH 7.0, 25.0 °C)": (7.0, 25.0),
}

if wound_preset in preset_map:
    default_ph, default_temp = preset_map[wound_preset]
else:
    default_ph, default_temp = 7.4, 37.0

# --- PARSE GLYCAM HA PDB ---
def parse_glycam_ha(raw_bytes):
    residues = {}
    atom_count = 0
    for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        if line.startswith("ATOM"):
            atom_count += 1
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_seq = int(line[22:26].strip())
            residues.setdefault(res_seq, {"res_name": res_name, "atoms": []})
            residues[res_seq]["atoms"].append(atom_name)

    carboxyls = sum(
        1 for r in residues.values() if "O6A" in r["atoms"] and "O6B" in r["atoms"]
    )
    n_glca = sum(1 for r in residues.values() if "ZB" in r["res_name"])
    n_glcnac = sum(1 for r in residues.values() if "YB" in r["res_name"])
    mw = (n_glca * 176.12) + (n_glcnac * 203.19) + 18.02
    return {
        "atoms": atom_count,
        "residues": len(residues),
        "carboxyls": max(carboxyls, 1),
        "mw": mw if mw > 50 else 1156.0,
        "source": "Uploaded GLYCAM PDB"
    }

if uploaded_ha is not None:
    ha_data = parse_glycam_ha(uploaded_ha.getvalue())
    st.sidebar.success(
        f"Loaded HA PDB: {ha_data['residues']} residues, {ha_data['carboxyls']} -COO⁻ groups ({ha_data['mw']:.1f} Da)"
    )
else:
    ha_data = {
        "atoms": 141,
        "residues": 7,
        "carboxyls": 3,
        "mw": 1155.9,
        "source": "Default 6-mer GLYCAM HA Profile"
    }

# --- INTERACTIVE ENVIRONMENT SLIDERS ---
col_s1, col_s2 = st.columns(2)
active_ph = col_s1.slider("Active Wound Bed pH", 2.0, 12.0, float(default_ph), 0.1)
active_temp = col_s2.slider("Active Temperature (°C)", 4.0, 60.0, float(default_temp), 0.5)

# --- BIOPHYSICAL SIMULATION ENGINE ---
def water_viscosity_mPas(T_C):
    """Vogel-Fulcher-Tammann approximation for dynamic viscosity of water (mPa·s)."""
    T_K = T_C + 273.15
    return 0.02414 * (10.0 ** (247.8 / (T_K - 140.0)))

def simulate_state(pH_arr, T_C_arr, include_ha=False):
    """
    Computes:
    1. Net Electrostatic Charge (e/kDa) via temperature-corrected Henderson-Hasselbalch
    2. Cationic Antibacterial Charge Density (e/kDa)
    3. Swelling Index (relative water uptake driven by HA hydrophilicity + electrostatic repulsion + thermal expansion)
    4. Relative Bacterial Penetration Risk (0 to 1 scale based on mesh loosening & thermal mobility)
    """
    T_K = T_C_arr + 273.15
    # Van't Hoff temperature shift on amine pKa (d(pKa)/dT ~ -0.028 / K for primary amines)
    pKa_chi = 6.50 - 0.028 * (T_C_arr - 25.0)
    pKa_col_base = 10.50 - 0.025 * (T_C_arr - 25.0)
    pKa_col_acid = 4.10 - 0.002 * (T_C_arr - 25.0)
    pKa_ha = 3.20 - 0.002 * (T_C_arr - 25.0)

    # Moles of chains per L
    m_col = (conc_col * 10.0) / 8200.0
    m_chi = (conc_chi * 10.0) / 1650.0
    m_ha = ((conc_ha * 10.0) / ha_data["mw"]) if include_ha else 0.0

    total_mass = (conc_col + conc_chi + (conc_ha if include_ha else 0.0)) * 10.0

    # Ionization fractions
    f_col_neg = 1.0 / (1.0 + 10.0 ** (pKa_col_acid - pH_arr))
    f_col_pos = 1.0 / (1.0 + 10.0 ** (pH_arr - pKa_col_base))
    f_chi_pos = 1.0 / (1.0 + 10.0 ** (pH_arr - pKa_chi))
    f_ha_neg = 1.0 / (1.0 + 10.0 ** (pKa_ha - pH_arr))

    pos_total = (m_col * 6.0 * f_col_pos) + (m_chi * 8.5 * f_chi_pos)
    neg_total = (m_col * 6.0 * f_col_neg) + (m_ha * ha_data["carboxyls"] * f_ha_neg)

    net_per_kda = (pos_total - neg_total) / total_mass
    cat_per_kda = pos_total / total_mass

    # Swelling & Mesh Loosening Model:
    # Base hydrophilicity increases with HA fraction; ionic repulsion (|net_charge|) expands network;
    # PEC ionic pairing (min(pos, neg)) tightens network; temperature increases chain mobility.
    pec_pairing = np.minimum(pos_total, neg_total) / total_mass
    electrostatic_repulsion = np.abs(net_per_kda)
    ha_hydration_boost = 1.45 if include_ha else 1.0
    thermal_factor = (T_K / 298.15) ** 1.5

    swelling_ratio = (
        120.0
        * ha_hydration_boost
        * (1.0 + 1.8 * electrostatic_repulsion - 1.2 * pec_pairing)
        * thermal_factor
    )

    # Stokes-Einstein thermal mobility factor (T / viscosity) normalized to 25 C
    mob_ratio = (T_K / water_viscosity_mPas(T_C_arr)) / (298.15 / water_viscosity_mPas(25.0))
    # Barrier permeability proxy (higher swelling + higher thermal mobility + lower cationic killing = higher passage risk)
    passage_index = np.clip(
        (swelling_ratio / 350.0) * (mob_ratio ** 0.4) * (1.0 / (1.0 + 0.8 * cat_per_kda)) / (film_thickness / 0.5),
        0.01,
        1.0
    )

    return net_per_kda, cat_per_kda, swelling_ratio, passage_index

# Point evaluation for active sliders
nA, cA, swA, pA = simulate_state(np.array([active_ph]), np.array([active_temp]), include_ha=False)
nB, cB, swB, pB = simulate_state(np.array([active_ph]), np.array([active_temp]), include_ha=True)

# --- TOP KPI METRICS ---
st.subheader(f"Snapshot at pH {active_ph:.1f} & {active_temp:.1f} °C")
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Formulation A Net Charge",
    f"{nA[0]:+.3f} e/kDa",
    f"Cationic: {cA[0]:.3f} e/kDa"
)
k2.metric(
    "Formulation B (+HA) Net Charge",
    f"{nB[0]:+.3f} e/kDa",
    f"{nB[0] - nA[0]:+.3f} vs Control"
)
k3.metric(
    "Predicted Swelling (B vs A)",
    f"{swB[0]:.0f}% vs {swA[0]:.0f}%",
    f"{swB[0] - swA[0]:+.0f}% water uptake"
)
k4.metric(
    "S. epidermidis Passage Risk (B vs A)",
    f"{pB[0]*100:.1f}% vs {pA[0]*100:.1f}%",
    f"{(pB[0] - pA[0])*100:+.1f}% change",
    delta_color="inverse"
)

# --- TABS FOR DETAILED SWEEPS ---
tab_ph, tab_temp, tab_2d, tab_data = st.tabs([
    "🧪 1. pH Sweep (Wound Stages)",
    "🔥 2. Temperature Sweep (Thermal Kinetics)",
    "🗺️ 3. Combined 2D pH × Temp Heatmap",
    "📥 4. Export Data & Summary Table"
])

# === TAB 1: pH SWEEP ===
with tab_ph:
    st.markdown(f"#### Effect of Wound pH at Constant Temperature ({active_temp:.1f} °C)")
    ph_grid = np.linspace(3.0, 11.0, 161)
    t_const = np.full_like(ph_grid, active_temp)

    netA_ph, catA_ph, swA_ph, passA_ph = simulate_state(ph_grid, t_const, include_ha=False)
    netB_ph, catB_ph, swB_ph, passB_ph = simulate_state(ph_grid, t_const, include_ha=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_charge = go.Figure()
        fig_charge.add_trace(go.Scatter(x=ph_grid, y=netA_ph, name="Formulation A (Control)", line=dict(color="#0072B2", width=3)))
        fig_charge.add_trace(go.Scatter(x=ph_grid, y=netB_ph, name="Formulation B (+ HA)", line=dict(color="#D55E00", width=3)))
        fig_charge.add_hline(y=0, line_dash="dot", line_color="gray")
        for label, ph_mark, col in [("Healthy (5.5)", 5.5, "#009E73"), ("PBS (7.4)", 7.4, "#56B4E9"), ("Infected (8.5)", 8.5, "#CC79A7")]:
            fig_charge.add_vline(x=ph_mark, line_dash="dash", line_color=col, annotation_text=label)
        fig_charge.update_layout(title="Net Electrostatic Charge vs. pH", xaxis_title="pH", yaxis_title="Net Charge (e / kDa)")
        st.plotly_chart(fig_charge, use_container_width=True)

    with c2:
        fig_swell = go.Figure()
        fig_swell.add_trace(go.Scatter(x=ph_grid, y=swA_ph, name="Formulation A (Control)", line=dict(color="#0072B2", width=3)))
        fig_swell.add_trace(go.Scatter(x=ph_grid, y=swB_ph, name="Formulation B (+ HA)", line=dict(color="#D55E00", width=3)))
        fig_swell.add_vline(x=active_ph, line_dash="solid", line_color="black", annotation_text="Selected pH")
        fig_swell.update_layout(title="Predicted Water Swelling (%) vs. pH", xaxis_title="pH", yaxis_title="Equilibrium Swelling (%)")
        st.plotly_chart(fig_swell, use_container_width=True)

# === TAB 2: TEMPERATURE SWEEP ===
with tab_temp:
    st.markdown(f"#### Effect of Temperature at Constant pH ({active_ph:.1f})")
    st.caption("Accounts for Van't Hoff pKa shift of Chitosan amines, water viscosity drop, and thermal polymer expansion.")
    temp_grid = np.linspace(4.0, 50.0, 120)
    ph_const = np.full_like(temp_grid, active_ph)

    netA_t, catA_t, swA_t, passA_t = simulate_state(ph_const, temp_grid, include_ha=False)
    netB_t, catB_t, swB_t, passB_t = simulate_state(ph_const, temp_grid, include_ha=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_t_cat = go.Figure()
        fig_t_cat.add_trace(go.Scatter(x=temp_grid, y=catA_t, name="Formulation A (Control)", line=dict(color="#0072B2", width=3)))
        fig_t_cat.add_trace(go.Scatter(x=temp_grid, y=catB_t, name="Formulation B (+ HA)", line=dict(color="#D55E00", width=3)))
        for label, t_mark, col in [("Room (25°C)", 25.0, "#7F8C8D"), ("Body (37°C)", 37.0, "#009E73"), ("Fever/Inflamed (39.5°C)", 39.5, "#E74C3C")]:
            fig_t_cat.add_vline(x=t_mark, line_dash="dash", line_color=col, annotation_text=label)
        fig_t_cat.update_layout(title="Antibacterial Cationic Charge vs. Temperature", xaxis_title="Temperature (°C)", yaxis_title="Positive Charge (e / kDa)")
        st.plotly_chart(fig_t_cat, use_container_width=True)

    with c2:
        fig_t_pass = go.Figure()
        fig_t_pass.add_trace(go.Scatter(x=temp_grid, y=passA_t * 100, name="Formulation A (Control)", line=dict(color="#0072B2", width=3)))
        fig_t_pass.add_trace(go.Scatter(x=temp_grid, y=passB_t * 100, name="Formulation B (+ HA)", line=dict(color="#D55E00", width=3)))
        fig_t_pass.add_vline(x=active_temp, line_dash="solid", line_color="black", annotation_text="Selected Temp")
        fig_t_pass.update_layout(title="S. epidermidis Penetration Risk (%) vs. Temperature", xaxis_title="Temperature (°C)", yaxis_title="Relative Passage Index (%)")
        st.plotly_chart(fig_t_pass, use_container_width=True)

# === TAB 3: 2D COMBINED HEATMAP ===
with tab_2d:
    st.markdown("#### Combined pH × Temperature Landscape for Formulation B (HA Treatment)")
    ph_2d = np.linspace(4.5, 9.5, 50)
    t_2d = np.linspace(20.0, 45.0, 50)
    PH_M, T_M = np.meshgrid(ph_2d, t_2d)

    _, _, SW_M, PASS_M = simulate_state(PH_M, T_M, include_ha=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_hm1 = px.imshow(
            SW_M, x=np.round(ph_2d, 2), y=np.round(t_2d, 1),
            origin="lower", aspect="auto", color_continuous_scale="YlGnBu",
            labels=dict(x="Wound pH", y="Temperature (°C)", color="Swelling %"),
            title="Formulation B Swelling Landscape (%)"
        )
        st.plotly_chart(fig_hm1, use_container_width=True)

    with c2:
        fig_hm2 = px.imshow(
            PASS_M * 100, x=np.round(ph_2d, 2), y=np.round(t_2d, 1),
            origin="lower", aspect="auto", color_continuous_scale="OrRd",
            labels=dict(x="Wound pH", y="Temperature (°C)", color="Passage %"),
            title="Formulation B Bacterial Penetration Risk (%)"
        )
        st.plotly_chart(fig_hm2, use_container_width=True)

# === TAB 4: EXPORT DATA TABLE ===
with tab_data:
    st.markdown("#### Clinical Benchmark Comparison Table")
    scenarios = [
        ("Healthy / Healing Skin", 5.5, 36.5),
        ("Physiological / PBS Buffer", 7.4, 37.0),
        ("Infected Chronic Wound", 8.5, 39.5),
        ("Room Temperature Prep", 7.0, 25.0),
        ("Custom Active Setting", active_ph, active_temp),
    ]
    rows = []
    for name, ph_v, t_v in scenarios:
        na, ca, swa, pa = simulate_state(np.array([ph_v]), np.array([t_v]), include_ha=False)
        nb, cb, swb, pb = simulate_state(np.array([ph_v]), np.array([t_v]), include_ha=True)
        rows.append({
            "Scenario": name,
            "pH": ph_v,
            "Temp (°C)": t_v,
            "Form A Net Charge (e/kDa)": round(float(na[0]), 4),
            "Form B (+HA) Net Charge (e/kDa)": round(float(nb[0]), 4),
            "Form A Swelling (%)": round(float(swa[0]), 1),
            "Form B (+HA) Swelling (%)": round(float(swb[0]), 1),
            "Form A Passage Risk (%)": round(float(pa[0] * 100), 1),
            "Form B (+HA) Passage Risk (%)": round(float(pb[0] * 100), 1),
        })

    df_summary = pd.DataFrame(rows)
    st.dataframe(df_summary, hide_index=True, use_container_width=True)
    st.download_button(
        "⬇️ Download pH & Temperature Simulation Table (CSV)",
        data=df_summary.to_csv(index=False).encode("utf-8"),
        file_name="WoundEnv_Sim_pH_Temp_Results.csv",
        mime="text/csv",
        type="primary"
    )
