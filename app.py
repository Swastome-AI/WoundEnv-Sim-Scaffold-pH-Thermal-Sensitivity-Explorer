import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

st.set_page_config(
    page_title="WoundEnv Sim | Dynamic pH & Thermal Simulator",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 WoundEnv Sim: Dynamic Scaffold pH & Thermal Simulator")
st.markdown("Build your own custom formulations, choose your concentrations, and test any pH and temperature points.")

# --- SESSION STATE INITIALIZATION ---
if "molecules" not in st.session_state:
    st.session_state.molecules = {
        "Collagen": {"mw": 8200.0, "acid_groups": 6.0, "pKa_acid": 4.1, "base_groups": 6.0, "pKa_base": 10.5, "hydro_factor": 1.0},
        "Chitosan": {"mw": 1650.0, "acid_groups": 0.0, "pKa_acid": 0.0, "base_groups": 8.5, "pKa_base": 6.5, "hydro_factor": 1.15},
        "Hyaluronic Acid": {"mw": 1155.9, "acid_groups": 3.0, "pKa_acid": 3.2, "base_groups": 0.0, "pKa_base": 0.0, "hydro_factor": 1.50},
        "PCL (Polycaprolactone)": {"mw": 588.0, "acid_groups": 0.0, "pKa_acid": 0.0, "base_groups": 0.0, "pKa_base": 0.0, "hydro_factor": 0.55},
    }

if "conditions" not in st.session_state:
    st.session_state.conditions = {}

# --- HELPER: PARSE OPTIONAL PDB UPLOAD ---
def parse_pdb_for_ha(raw_bytes):
    residues = {}
    for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_seq = int(line[22:26].strip())
            residues.setdefault(res_seq, {"res_name": res_name, "atoms": []})
            residues[res_seq]["atoms"].append(atom_name)
    carboxyls = sum(1 for r in residues.values() if "O6A" in r["atoms"] and "O6B" in r["atoms"])
    n_glca = sum(1 for r in residues.values() if "ZB" in r["res_name"])
    n_glcnac = sum(1 for r in residues.values() if "YB" in r["res_name"])
    mw = (n_glca * 176.12) + (n_glcnac * 203.19) + 18.02
    if mw > 50 and carboxyls > 0:
        st.session_state.molecules["Hyaluronic Acid"]["mw"] = mw
        st.session_state.molecules["Hyaluronic Acid"]["acid_groups"] = float(carboxyls)
        return True, mw, carboxyls
    return False, 0, 0

# --- SIDEBAR: OPTIONAL PDB & CUSTOM MOLECULES ---
with st.sidebar:
    st.header("1. Molecule Library")
    upl = st.file_uploader("Upload GLYCAM HA PDB (optional)", type=["pdb"])
    if upl is not None:
        ok, mw, carb = parse_pdb_for_ha(upl.getvalue())
        if ok:
            st.success(f"Updated HA from PDB: {mw:.1f} Da, {carb} carboxyl groups")

    with st.expander("➕ Add Custom Molecule"):
        m_name = st.text_input("Molecule Name", placeholder="e.g., Alginate")
        m_mw = st.number_input("Molecular Weight (Da)", 50.0, 500000.0, 2000.0)
        m_acid = st.number_input("Acidic Groups per Chain (-COOH)", 0.0, 100.0, 0.0)
        m_pka_a = st.number_input("Acidic pKa", 1.0, 14.0, 3.5)
        m_base = st.number_input("Basic Groups per Chain (-NH2)", 0.0, 100.0, 0.0)
        m_pka_b = st.number_input("Basic pKa", 1.0, 14.0, 6.5)
        m_hydro = st.slider("Hydration Multiplier (0.5=Hydrophobic, 1.5=Hydrophilic)", 0.4, 2.0, 1.0, 0.05)
        if st.button("Add to Library") and m_name.strip():
            st.session_state.molecules[m_name.strip()] = {
                "mw": m_mw, "acid_groups": m_acid, "pKa_acid": m_pka_a,
                "base_groups": m_base, "pKa_base": m_pka_b, "hydro_factor": m_hydro
            }
            st.rerun()

    st.divider()
    st.header("2. Test Environments (pH & Temp)")
    ph_input = st.text_input("Enter pH values to test (comma-separated)", value="5.5, 7.4, 8.5")
    temp_input = st.text_input("Enter Temperatures in °C (comma-separated)", value="25.0, 37.0, 39.5")
    film_thickness = st.number_input("Film Thickness (mm)", 0.05, 10.0, 0.5, 0.05)

# Parse pH and Temperature lists safely
try:
    ph_list = sorted({float(x.strip()) for x in ph_input.split(",") if x.strip()})
except ValueError:
    st.error("Please enter valid numbers for pH separated by commas (e.g., 5.5, 7.4, 8.5).")
    ph_list = [7.4]

try:
    temp_list = sorted({float(x.strip()) for x in temp_input.split(",") if x.strip()})
except ValueError:
    st.error("Please enter valid numbers for Temperature separated by commas (e.g., 25, 37, 40).")
    temp_list = [37.0]

# --- STEP 1: DYNAMIC CONDITION BUILDER ---
st.subheader("Step 1: Define Your Conditions (Formulations & Concentrations)")
col_btn1, col_btn2, _ = st.columns([1.2, 1, 4])
if col_btn1.button("➕ Add New Condition", type="primary"):
    cid = uuid.uuid4().hex[:6]
    idx = len(st.session_state.conditions)
    label = f"Condition {chr(65 + idx)}" if idx < 26 else f"Condition {idx + 1}"
    st.session_state.conditions[cid] = {
        "name": label,
        "components": {}
    }
    st.rerun()

if col_btn2.button("🗑️ Clear All Conditions", disabled=not st.session_state.conditions):
    st.session_state.conditions = {}
    st.rerun()

if not st.session_state.conditions:
    st.info("Click **➕ Add New Condition** above to create Condition A, Condition B, etc., and choose their ingredients and concentrations.")
else:
    available_mols = list(st.session_state.molecules.keys())
    to_delete = None

    for cid, cond in st.session_state.conditions.items():
        with st.container(border=True):
            c_top1, c_top2 = st.columns([5, 1])
            cond["name"] = c_top1.text_input("Condition Name", cond["name"], key=f"name_{cid}")
            c_top2.write("")
            if c_top2.button("🗑️ Remove", key=f"del_{cid}"):
                to_delete = cid

            selected_mols = st.multiselect(
                "Select Polymers / Components",
                options=available_mols,
                default=[m for m in cond["components"] if m in available_mols],
                key=f"mols_{cid}"
            )

            updated_comps = {}
            if selected_mols:
                conc_cols = st.columns(min(len(selected_mols), 4))
                for i, m in enumerate(selected_mols):
                    prev_val = cond["components"].get(m, 1.0)
                    val = conc_cols[i % len(conc_cols)].number_input(
                        f"{m} (% w/v)",
                        min_value=0.01,
                        max_value=50.0,
                        value=float(prev_val),
                        step=0.1,
                        key=f"conc_{cid}_{m}"
                    )
                    updated_comps[m] = val
            else:
                st.caption("No polymers selected (acts as a Water-Only Control).")
            cond["components"] = updated_comps

    if to_delete:
        del st.session_state.conditions[to_delete]
        st.rerun()

# --- BIOPHYSICAL SIMULATION FUNCTION ---
def water_viscosity_mPas(T_C):
    T_K = T_C + 273.15
    return 0.02414 * (10.0 ** (247.8 / (T_K - 140.0)))

def run_simulation(components_dict, pH_arr, T_C_arr):
    if not components_dict:
        # Water-only control
        T_K = T_C_arr + 273.15
        mob_ratio = (T_K / water_viscosity_mPas(T_C_arr)) / (298.15 / water_viscosity_mPas(25.0))
        return (
            np.zeros_like(pH_arr, dtype=float),
            np.zeros_like(pH_arr, dtype=float),
            np.zeros_like(pH_arr, dtype=float),
            np.clip(0.85 * (mob_ratio ** 0.2), 0.01, 1.0)
        )

    T_K = T_C_arr + 273.15
    pos_total = np.zeros_like(pH_arr, dtype=float)
    neg_total = np.zeros_like(pH_arr, dtype=float)
    total_mass_gL = 0.0
    weighted_hydro = 0.0

    for mol_name, conc_wv in components_dict.items():
        props = st.session_state.molecules[mol_name]
        mass_gL = conc_wv * 10.0
        moles_chains = mass_gL / props["mw"]
        total_mass_gL += mass_gL
        weighted_hydro += mass_gL * props["hydro_factor"]

        if props["acid_groups"] > 0:
            pka_a = props["pKa_acid"] - 0.002 * (T_C_arr - 25.0)
            f_neg = 1.0 / (1.0 + 10.0 ** (pka_a - pH_arr))
            neg_total += moles_chains * props["acid_groups"] * f_neg

        if props["base_groups"] > 0:
            pka_b = props["pKa_base"] - 0.028 * (T_C_arr - 25.0)
            f_pos = 1.0 / (1.0 + 10.0 ** (pH_arr - pka_b))
            pos_total += moles_chains * props["base_groups"] * f_pos

    net_per_kda = (pos_total - neg_total) / total_mass_gL
    cat_per_kda = pos_total / total_mass_gL
    pec_pairing = np.minimum(pos_total, neg_total) / total_mass_gL
    repulsion = np.abs(net_per_kda)
    avg_hydro = weighted_hydro / total_mass_gL

    # Higher total polymer concentration tightens network; higher hydrophilicity & repulsion increase swelling
    conc_tightening = 3.0 / max(total_mass_gL / 10.0, 0.2)
    thermal_factor = (T_K / 298.15) ** 1.5

    swelling_pct = (
        100.0
        * avg_hydro
        * (conc_tightening ** 0.35)
        * (1.0 + 1.8 * repulsion - 1.2 * pec_pairing)
        * thermal_factor
    )

    mob_ratio = (T_K / water_viscosity_mPas(T_C_arr)) / (298.15 / water_viscosity_mPas(25.0))
    passage_risk = np.clip(
        (swelling_pct / 350.0) * (mob_ratio ** 0.4) * (1.0 / (1.0 + 0.8 * cat_per_kda)) / (film_thickness / 0.5),
        0.01,
        1.0
    )

    return net_per_kda, cat_per_kda, swelling_pct, passage_risk

# --- STEP 2: RUN & VISUALIZE ---
st.divider()
st.subheader("Step 2: Run Simulation & Compare Conditions")

if st.session_state.conditions and ph_list and temp_list:
    if st.button("▶ Run Dynamic Simulation", type="primary"):
        st.session_state.has_run = True

    if st.session_state.get("has_run"):
        # Build full results table for every Condition x pH x Temperature combination
        records = []
        for cid, cond in st.session_state.conditions.items():
            comp_str = " + ".join(f"{m} ({c:g}%)" for m, c in cond["components"].items()) or "Water Only"
            for t_val in temp_list:
                for ph_val in ph_list:
                    net_q, cat_q, swell, pass_r = run_simulation(
                        cond["components"], np.array([ph_val]), np.array([t_val])
                    )
                    records.append({
                        "Condition": cond["name"],
                        "Composition": comp_str,
                        "pH": ph_val,
                        "Temperature (°C)": t_val,
                        "Net Charge (e/kDa)": round(float(net_q[0]), 4),
                        "Cationic Antibacterial Charge (e/kDa)": round(float(cat_q[0]), 4),
                        "Predicted Swelling (%)": round(float(swell[0]), 1),
                        "Bacterial Passage Risk (%)": round(float(pass_r[0] * 100), 1),
                    })

        df_res = pd.DataFrame(records)

        tab1, tab2, tab3 = st.tabs([
            "📊 Condition Comparison Charts",
            "📈 Continuous pH & Temp Curves",
            "📥 Full Results Table & Export"
        ])

        with tab1:
            sel_temp = st.selectbox("Select Temperature (°C) to compare across your pH points:", temp_list)
            df_sub = df_res[df_res["Temperature (°C)"] == sel_temp].copy()
            df_sub["pH Label"] = "pH " + df_sub["pH"].astype(str)

            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.bar(
                    df_sub, x="pH Label", y="Predicted Swelling (%)", color="Condition",
                    barmode="group", title=f"Swelling (%) by Condition at {sel_temp} °C"
                )
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.bar(
                    df_sub, x="pH Label", y="Bacterial Passage Risk (%)", color="Condition",
                    barmode="group", title=f"Bacterial Passage Risk (%) at {sel_temp} °C"
                )
                st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig3 = px.bar(
                    df_sub, x="pH Label", y="Net Charge (e/kDa)", color="Condition",
                    barmode="group", title=f"Net Electrostatic Charge (e/kDa) at {sel_temp} °C"
                )
                fig3.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig3, use_container_width=True)
            with c4:
                fig4 = px.bar(
                    df_sub, x="pH Label", y="Cationic Antibacterial Charge (e/kDa)", color="Condition",
                    barmode="group", title=f"Antibacterial Cationic Charge at {sel_temp} °C"
                )
                st.plotly_chart(fig4, use_container_width=True)

        with tab2:
            col_c1, col_c2 = st.columns(2)
            ref_temp = col_c1.selectbox("Fix Temperature for pH Curve (°C):", temp_list, key="curve_t")
            ref_ph = col_c2.selectbox("Fix pH for Temperature Curve:", ph_list, key="curve_ph")

            ph_sweep = np.linspace(min(ph_list) - 1.5, max(ph_list) + 1.5, 120)
            ph_sweep = np.clip(ph_sweep, 2.0, 12.0)
            t_sweep = np.linspace(max(4.0, min(temp_list) - 10.0), min(60.0, max(temp_list) + 10.0), 120)

            fig_ph_c = go.Figure()
            fig_t_c = go.Figure()

            for cid, cond in st.session_state.conditions.items():
                _, _, sw_p, _ = run_simulation(cond["components"], ph_sweep, np.full_like(ph_sweep, ref_temp))
                _, _, _, pa_t = run_simulation(cond["components"], np.full_like(t_sweep, ref_ph), t_sweep)
                fig_ph_c.add_trace(go.Scatter(x=ph_sweep, y=sw_p, name=cond["name"], mode="lines", line=dict(width=3)))
                fig_t_c.add_trace(go.Scatter(x=t_sweep, y=pa_t * 100, name=cond["name"], mode="lines", line=dict(width=3)))

            for p_pt in ph_list:
                fig_ph_c.add_vline(x=p_pt, line_dash="dash", line_color="gray")
            for t_pt in temp_list:
                fig_t_c.add_vline(x=t_pt, line_dash="dash", line_color="gray")

            fig_ph_c.update_layout(title=f"Swelling vs. pH (at {ref_temp} °C)", xaxis_title="pH", yaxis_title="Swelling (%)")
            fig_t_c.update_layout(title=f"Bacterial Passage Risk vs. Temp (at pH {ref_ph})", xaxis_title="Temperature (°C)", yaxis_title="Passage Risk (%)")

            g1, g2 = st.columns(2)
            g1.plotly_chart(fig_ph_c, use_container_width=True)
            g2.plotly_chart(fig_t_c, use_container_width=True)

        with tab3:
            st.dataframe(df_res, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇️ Download Complete Simulation Results (CSV)",
                data=df_res.to_csv(index=False).encode("utf-8"),
                file_name="WoundEnv_Dynamic_Results.csv",
                mime="text/csv",
                type="primary"
            )
else:
    st.warning("Add at least one Condition above and specify valid pH and Temperature lists in the sidebar to run.")
