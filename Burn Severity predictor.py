import streamlit as st
import pandas as pd
import numpy as np
import os, json, joblib, io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="🔥 Burn Severity Predictor",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# =====================================
# CSS STYLING
# =====================================
st.markdown("""
<style>
    .severe-badge { background-color: #ef4444; color: white; padding: 8px 12px; border-radius: 5px; font-weight: bold; }
    .moderate-badge { background-color: #f59e0b; color: white; padding: 8px 12px; border-radius: 5px; font-weight: bold; }
    .minor-badge { background-color: #10b981; color: white; padding: 8px 12px; border-radius: 5px; font-weight: bold; }
    
    /* Add padding to bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding-top: 10px;
        padding-bottom: 20px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# CONFIG & CONSTANTS
# =====================================
MODEL_DIR = "artifacts"
CSV_PATH = "burn_cases_history.csv"
OUTCOMES_CSV = "burn_outcomes.csv"

COST_STRUCTURE = {
    "ICU": {"daily": 15000},
    "Ward": {"daily": 5000},
    "Outpatient": {"daily": 1000},
    "Surgery": {"per_session": 50000},
    "Dressing": {"per_session": 2000},
    "Medications": {"daily": 3000},
    "Skin_Graft": {"per_sqcm": 100},
    "Consultation": {"per_session": 500}
}

MEDICATION_RECOMMENDATIONS = {
    "Minor": ["Topical antibiotics (Mupirocin)", "Topical silver sulfadiazine"],
    "Moderate": ["IV Flucloxacillin 500mg QID", "Topical antibiotics", "Paracetamol 500mg"],
    "Severe": ["IV broad-spectrum antibiotics", "IV opioid analgesia", "Prophylactic antibiotics", "Ranitidine 50mg IV"]
}

SEVERITY_BADGE_MAP = {
    "Minor": "minor-badge",
    "Moderate": "moderate-badge",
    "Severe": "severe-badge"
}

# =====================================
# SESSION STATE
# =====================================
if "cases" not in st.session_state:
    st.session_state.cases = []
if "outcomes" not in st.session_state:
    st.session_state.outcomes = []

# =====================================
# CORE FUNCTIONS
# =====================================
def load_models():
    """Load ML models if available"""
    sev_model, trt_model, meta = None, None, {}
    try:
        sev_path = os.path.join(MODEL_DIR, "pipeline_severity.joblib")
        if os.path.exists(sev_path):
            sev_model = joblib.load(sev_path)
    except Exception:
        pass
    try:
        trt_path = os.path.join(MODEL_DIR, "pipeline_treatment.joblib")
        if os.path.exists(trt_path):
            trt_model = joblib.load(trt_path)
    except Exception:
        pass
    return sev_model, trt_model, meta

@st.cache_resource
def get_models():
    return load_models()

def load_csv(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
        except Exception:
            pass
    return []

def save_csv(data, path):
    if data:
        try:
            pd.DataFrame(data).to_csv(path, index=False)
        except Exception:
            pass

def compute_score(form):
    """Rule-based severity scoring"""
    tbsa, age = form["tbsa"], form["age"]
    depth_weights = {"superficial": 0, "partial_thickness": 1.0, "full_thickness": 2.0}
    depth_score = depth_weights.get(form["burn_depth"], 1.0)
    inhalation = 2.0 if form["inhalation_injury"] else 0.0
    comorb = form["comorbidities"] * 0.3
    time_penalty = (form["time_to_treatment_hours"] / 24.0) * 0.5
    age_penalty = (age / 80.0) * 0.5
    score = (tbsa / 100.0) * 3.0 + depth_score * 1.5 + inhalation + comorb + time_penalty + age_penalty
    return round(score, 3)

def score_to_severity(score):
    """Convert score to severity classification"""
    if score < 1.2:
        return "Minor", "Low", "Outpatient Care"
    elif score < 2.5:
        return "Moderate", "Medium", "Ward Admission"
    else:
        return "Severe", "High", "ICU/HDU"

def parkland_formula(tbsa_pct, weight_kg):
    """Calculate fluid resuscitation requirements"""
    # CORRECTED: Removed division by 100. Formula is 4 * kg * TBSA(%)
    total_24h = 4 * weight_kg * tbsa_pct
    first_8h = total_24h / 2.0
    return int(round(total_24h)), int(round(first_8h))

def baux_score(age, tbsa):
    """Calculate Baux index for mortality prediction"""
    return age + tbsa

def curreri_formula(weight_kg, tbsa_pct):
    """Calculate nutritional requirements (kcal/day)"""
    # This formula is correct as it uses the decimal form of TBSA
    return 25 * weight_kg + 40 * weight_kg * (tbsa_pct / 100.0)

def calculate_costs(severity, tbsa, weight_kg, los=7):
    """Calculate comprehensive treatment costs in INR"""
    costs = {}
    
    if severity == "Minor":
        costs["Bed"] = COST_STRUCTURE["Outpatient"]["daily"] * los
        costs["Dressing"] = COST_STRUCTURE["Dressing"]["per_session"] * 3
        costs["Medications"] = COST_STRUCTURE["Medications"]["daily"] * los
        costs["Consultation"] = COST_STRUCTURE["Consultation"]["per_session"] * 2
    
    elif severity == "Moderate":
        costs["Bed"] = COST_STRUCTURE["Ward"]["daily"] * los
        costs["Dressing"] = COST_STRUCTURE["Dressing"]["per_session"] * los
        costs["Medications"] = COST_STRUCTURE["Medications"]["daily"] * los
        costs["Surgery"] = COST_STRUCTURE["Surgery"]["per_session"] * 0.5
        costs["Consultation"] = COST_STRUCTURE["Consultation"]["per_session"] * 3
        # CORRECTED: Changed graft_area logic. Assumes tbsa * weight_kg is a proxy for area in sqcm.
        graft_area = tbsa * weight_kg
        costs["Skin_Graft"] = COST_STRUCTURE["Skin_Graft"]["per_sqcm"] * graft_area * 0.3
    
    else:  # Severe
        costs["Bed"] = COST_STRUCTURE["ICU"]["daily"] * los
        costs["Dressing"] = COST_STRUCTURE["Dressing"]["per_session"] * los
        costs["Medications"] = COST_STRUCTURE["Medications"]["daily"] * los * 1.5
        costs["Surgery"] = COST_STRUCTURE["Surgery"]["per_session"] * 2
        costs["Consultation"] = COST_STRUCTURE["Consultation"]["per_session"] * 5
        # CORRECTED: Changed graft_area logic.
        graft_area = tbsa * weight_kg
        costs["Skin_Graft"] = COST_STRUCTURE["Skin_Graft"]["per_sqcm"] * graft_area * 0.7
    
    return costs

def make_record(form, ml_outputs=None):
    """Create case record"""
    score = compute_score(form)
    severity_rule, risk_rule, treatment_rule = score_to_severity(score)
    parkland_total, parkland_8h = parkland_formula(form["tbsa"], form["weight_kg"])
    baux = baux_score(form["age"], form["tbsa"])
    curreri = curreri_formula(form["weight_kg"], form["tbsa"])
    ts = datetime.now()
    
    record = {
        "id": int(ts.timestamp()*1000),
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "first_name": form.get("first_name", ""),
        "last_name": form.get("last_name", ""),
        "age": form["age"],
        "gender": form["gender"],
        "tbsa": form["tbsa"],
        "weight_kg": form["weight_kg"],
        "burn_depth": form["burn_depth"],
        "cause": form["cause"],
        "time_to_treatment_hours": form["time_to_treatment_hours"],
        "inhalation_injury": int(form["inhalation_injury"]),
        "comorbidities": form["comorbidities"],
        "score_rule": score,
        "severity_rule": severity_rule,
        "risk_level_rule": risk_rule,
        "treatment_rule": treatment_rule,
        "parkland_total_ml": parkland_total,
        "parkland_first8h_ml": parkland_8h,
        "baux": baux,
        "curreri_kcal": int(curreri),
    }
    
    if ml_outputs:
        record.update(ml_outputs)
    
    return record

# =====================================
# LOAD DATA AT START
# =====================================
st.session_state.cases = load_csv(CSV_PATH)
st.session_state.outcomes = load_csv(OUTCOMES_CSV)
sev_model, trt_model, meta = get_models()
ML_AVAILABLE = sev_model is not None and trt_model is not None

# =====================================
# PREDICTION VIEW
# =====================================
def predict_view():
    st.header("🧬 Predict Burn Severity")
    
    if ML_AVAILABLE:
        st.success("✅ ML models loaded successfully.")
    else:
        st.info("⚠️ Machine learning models not found. Running in rule-based mode.")
    
    uploaded_file = st.file_uploader("Upload burn image (optional)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, width=250)
    
    with st.form("burn_form"):
        # VISUAL: Grouped form fields into bordered containers
        with st.container(border=True):
            st.subheader("👤 Patient Info")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                fname = st.text_input("First Name", "John")
            with c2:
                lname = st.text_input("Last Name", "Doe")
            with c3:
                gender = st.selectbox("Gender", ["M", "F", "O"])
            with c4:
                age = st.slider("Age", 0, 120, 35)
            
            c5, c6 = st.columns(2)
            with c5:
                weight = st.number_input("Weight (kg)", 30.0, 200.0, 65.0)
            with c6:
                comorbid = st.slider("Comorbidities (Count)", 0, 10, 0)
        
        with st.container(border=True):
            st.subheader("🔥 Burn Details")
            c7, c8, c9, c10 = st.columns(4)
            with c7:
                tbsa = st.slider("TBSA (%)", 0, 100, 15)
            with c8:
                depth = st.selectbox("Depth", ["superficial", "partial_thickness", "full_thickness"])
            with c9:
                cause = st.selectbox("Cause", ["flame", "scald", "chemical", "electric", "contact"])
            with c10:
                time_treat = st.number_input("Time to Treatment (hrs)", 0.0, 168.0, 2.0)
        
        with st.container(border=True):
            st.subheader("⚕️ Clinical")
            c11, c12 = st.columns(2)
            with c11:
                inhal = st.checkbox("Inhalation Injury")
            with c12:
                face = st.checkbox("Face Burn")
        
        # VISUAL: Made submit button primary
        submit = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")
    
    if submit:
        form_data = {
            "first_name": fname, "last_name": lname,
            "age": int(age), "gender": gender, "weight_kg": float(weight),
            "tbsa": float(tbsa), "burn_depth": depth, "cause": cause,
            "time_to_treatment_hours": float(time_treat),
            "inhalation_injury": inhal, "comorbidities": int(comorbid)
        }
        
        ml_out = {}
        if ML_AVAILABLE:
            try:
                baux_val = form_data["age"] + form_data["tbsa"]
                # CORRECTED: Removed division by 100
                parkland_24 = 4.0 * form_data["weight_kg"] * form_data["tbsa"]
                
                ml_input_data = {
                    "age": form_data["age"], "weight_kg": form_data["weight_kg"], "tbsa": form_data["tbsa"],
                    "time_to_treatment_hours": form_data["time_to_treatment_hours"], "comorbidities": form_data["comorbidities"],
                    "baux": baux_val, "parkland_24h_ml": parkland_24, "shock_index": None,
                    "hr": 90, "sbp": 120, "temp": 37, "spo2": 98, "rr": 16,
                    "gender": form_data["gender"], "burn_depth": form_data["burn_depth"], "cause": form_data["cause"],
                    "face": int(face), "hands": 0, "circumferential": 0, "first_aid": 0, "fluid_given": 0,
                    "inhalation_injury": int(inhal)
                }
                
                # Ensure all columns expected by the model are present, even if as None
                # This is a safer way to handle models with many features
                expected_cols = sev_model.steps[0][1].feature_names_in_ # Get feature names from preprocessor
                ml_input_dict = {col: ml_input_data.get(col) for col in expected_cols}
                ml_input = pd.DataFrame([ml_input_dict])

                sev_pred = sev_model.predict(ml_input)[0]
                sev_probs = dict(zip(sev_model.classes_, sev_model.predict_proba(ml_input)[0]))
                trt_pred = trt_model.predict(ml_input)[0]
                ml_out = {
                    "severity_ml": sev_pred,
                    "treatment_ml": trt_pred,
                    "confidence_ml": round(max(sev_probs.values())*100, 1),
                    "severity_ml_probs": sev_probs
                }
            except Exception as e:
                st.error(f"Error during ML prediction: {str(e)}")
                st.warning("ML prediction failed. Showing rule-based results only.")
        
        rec = make_record(form_data, ml_out)
        st.session_state.cases.insert(0, rec)
        save_csv(st.session_state.cases, CSV_PATH)
        
        st.success(f"✅ Case saved for {fname} {lname} (ID: {rec['id']})")
        
        tab1, tab2, tab3 = st.tabs(["📊 Severity", "💊 Treatment Plan", "💰 Cost Projection"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Rule-Based Assessment")
                # VISUAL: Using the custom CSS badge
                badge_class = SEVERITY_BADGE_MAP.get(rec['severity_rule'], 'minor-badge')
                st.markdown(f"**Severity**: <span class='{badge_class}'>{rec['severity_rule']}</span>", unsafe_allow_html=True)
                st.metric("Rule Score", f"{rec['score_rule']:.2f}")
                st.metric("Risk Level", rec['risk_level_rule'])
                st.metric("Baux Index (Mortality Risk)", rec['baux'])
                if rec['baux'] > 60:
                    st.warning("⚠️ High mortality risk (Baux > 60)")
            
            with col_b:
                st.subheader("ML Model Assessment")
                if ml_out:
                    # VISUAL: Using the custom CSS badge
                    badge_class_ml = SEVERITY_BADGE_MAP.get(ml_out['severity_ml'], 'minor-badge')
                    st.markdown(f"**Severity**: <span class='{badge_class_ml}'>{ml_out['severity_ml']}</span>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{ml_out['confidence_ml']}%")
                    st.metric("Predicted Treatment", ml_out['treatment_ml'])
                    
                    df_probs = pd.DataFrame(list(ml_out['severity_ml_probs'].items()), columns=["Class", "Probability"])
                    fig = px.bar(df_probs, x="Class", y="Probability", color="Class",
                                 color_discrete_map={"Minor":"#10b981","Moderate":"#f59e0b","Severe":"#ef4444"},
                                 title="Model Confidence per Class")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ML Model not available or prediction failed.")
        
        with tab2:
            st.subheader("🚰 Fluid Resuscitation (Parkland)")
            st.info(f"• **Total in 24h**: **{rec['parkland_total_ml']} mL** (Lactated Ringer's)\n"
                    f"• **First 8h**: **{rec['parkland_first8h_ml']} mL**")
            
            st.subheader("🥗 Nutritional Needs (Curreri)")
            st.info(f"• **Calories**: **{rec['curreri_kcal']} kcal/day**\n"
                    f"• **Protein (Est.)**: {int(rec['curreri_kcal']*0.18/4)} g/day")

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.subheader("📊 Monitoring Plan")
                if rec['severity_rule'] == "Minor":
                    st.write("• Outpatient follow-up in 24h\n• Daily wound checks")
                elif rec['severity_rule'] == "Moderate":
                    st.write("• Ward admission\n• Hourly vitals\n• Daily dressings\n• Surgical consultation")
                else:
                    st.write("• ICU/HDU admission\n• Continuous monitoring\n• Hourly dressings\n• Immediate surgical consult")
            
            with col_t2:
                st.subheader("💊 Medication")
                for med in MEDICATION_RECOMMENDATIONS[rec['severity_rule']]:
                    st.write(f"• {med}")
        
        with tab3:
            costs = calculate_costs(rec['severity_rule'], rec['tbsa'], rec['weight_kg'], 7)
            total = sum(costs.values())
            
            st.subheader("💵 Estimated Cost Breakdown (7 Days)")
            
            col_c1, col_c2 = st.columns([1,2])
            with col_c1:
                st.metric("Total Est. Cost (7 days)", f"₹{total:,.0f}")
                df_cost = pd.DataFrame(costs.items(), columns=["Item", "Cost"])
                df_cost["Cost"] = df_cost["Cost"].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(df_cost, use_container_width=True, hide_index=True)
            
            with col_c2:
                df_cost_pie = pd.DataFrame(costs.items(), columns=["Item", "Cost"])
                fig_cost = px.pie(df_cost_pie, values='Cost', names='Item', title='Cost Distribution')
                st.plotly_chart(fig_cost, use_container_width=True)

            st.divider()
            st.subheader("💸 Financial Impact Analysis")
            delay_h = form_data['time_to_treatment_hours']
            # Assuming 15% cost increase for every 24h delay
            delay_cost = total * (1 + (delay_h / 24) * 0.15)
            st.warning(f"**Impact of {delay_h}h Delay**: Estimated cost could rise to **₹{delay_cost:,.0f}** (+{(delay_cost-total)/total*100:.1f}%) due to complications.")
            st.success(f"**Potential Savings**: Early intervention and active management can reduce costs by up to 30% (Save **₹{total*0.3:,.0f}**).")

# =====================================
# CASES VIEW
# =====================================
def cases_view():
    st.header("🧾 Case History")
    
    if not st.session_state.cases:
        st.info("No cases have been recorded yet.")
        return
    
    df = pd.DataFrame(st.session_state.cases)
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sev_filt = st.multiselect("Filter by Severity", ["Minor", "Moderate", "Severe"], 
                                     default=["Minor", "Moderate", "Severe"], key="sev_filt")
        with col2:
            cause_opts = df['cause'].unique().tolist()
            cause_filt = st.multiselect("Filter by Cause", cause_opts, default=cause_opts, key="cause_filt")
        with col3:
            fmt = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    
    df_filt = df[(df['severity_rule'].isin(sev_filt)) & (df['cause'].isin(cause_filt))]
    
    st.dataframe(df_filt, use_container_width=True)
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        if not df_filt.empty:
            if fmt == "CSV":
                csv_str = df_filt.to_csv(index=False)
                st.download_button("📥 Download CSV", csv_str, "cases.csv", "text/csv", use_container_width=True)
            elif fmt == "Excel":
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    df_filt.to_excel(w, index=False)
                buf.seek(0)
                st.download_button("📥 Download Excel", buf, "cases.xlsx", "application/vnd.ms-excel", use_container_width=True)
            else:
                json_str = df_filt.to_json(orient='records')
                st.download_button("📥 Download JSON", json_str, "cases.json", "application/json", use_container_width=True)
    
    with col_e2:
        if st.button("🔄 Reload Data", use_container_width=True):
            st.session_state.cases = load_csv(CSV_PATH)
            st.rerun()
    
    with col_e3:
        if st.checkbox("Enable Deletion"):
            if st.button("🗑️ Delete All Cases", use_container_width=True, type="secondary"):
                st.session_state.cases = []
                save_csv([], CSV_PATH)
                st.success("All cases deleted.")
                st.rerun()

# =====================================
# DASHBOARD VIEW
# =====================================
def dashboard_view():
    st.header("📊 Dashboard")
    
    if not st.session_state.cases:
        st.info("No data available to display. Please add cases from the 'Predict' tab.")
        return
    
    df = pd.DataFrame(st.session_state.cases)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Cases", len(df))
    col2.metric("Avg. TBSA", f"{df['tbsa'].mean():.1f}%")
    col3.metric("Avg. Age", f"{df['age'].mean():.1f}")
    col4.metric("Severe Cases", (df['severity_rule']=="Severe").sum())
    col5.metric("Avg. Baux Score", f"{df['baux'].mean():.1f}")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.pie(df, names='severity_rule', title='Case Severity Distribution',
                     color_discrete_map={"Minor":"#10b981","Moderate":"#f59e0b","Severe":"#ef4444"})
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        fig2 = px.pie(df, names='cause', title='Burn Cause Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    
    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(df, x='tbsa', nbins=15, title='TBSA % Distribution')
        st.plotly_chart(fig3, use_container_width=True)
    
    with c4:
        fig4 = px.histogram(df, x='age', nbins=15, title='Patient Age Distribution')
        st.plotly_chart(fig4, use_container_width=True)
    
    c5, c6 = st.columns(2)
    with c5:
        depth_counts = df.groupby('burn_depth').size().reset_index(name='count')
        fig5 = px.bar(depth_counts, x='burn_depth', y='count', title='Burn Depth Frequency')
        st.plotly_chart(fig5, use_container_width=True)
    
    with c6:
        fig6 = px.scatter(df, x='time_to_treatment_hours', y='score_rule', color='severity_rule', 
                         title='Time to Treatment vs. Severity Score',
                         color_discrete_map={"Minor":"#10b981","Moderate":"#f59e0b","Severe":"#ef4444"})
        st.plotly_chart(fig6, use_container_width=True)
    
    st.subheader("Summary Statistics")
    summary = df[['age', 'weight_kg', 'tbsa', 'score_rule', 'baux', 'time_to_treatment_hours']].describe()
    st.dataframe(summary)

# =====================================
# OUTCOMES VIEW
# =====================================
def outcomes_view():
    st.header("📅 Patient Outcomes")
    
    if not st.session_state.cases:
        st.info("No cases found. Please add a case before logging an outcome.")
        return
    
    case_opts = [f"{c['first_name']} {c['last_name']} (ID: {c['id']})" for c in st.session_state.cases]
    
    with st.form("outcome_form"):
        st.subheader("Log New Outcome")
        sel_idx = st.selectbox("Select Case", range(len(case_opts)), format_func=lambda i: case_opts[i])
        sel_case = st.session_state.cases[sel_idx]
        
        st.write(f"**Patient**: {sel_case['first_name']} {sel_case['last_name']}")
        st.write(f"**Initial Severity**: {sel_case['severity_rule']}")
        
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            los = st.number_input("Length of Stay (days)", 0, 365, 7)
            surg = st.number_input("Number of Surgeries", 0, 20, 1)
        with col_o2:
            disc = st.selectbox("Discharge Status", ["Active", "Discharged Home", "Transferred", "Deceased", "Left AMA"])
            cost_act = st.number_input("Actual Total Cost (INR)", 0.0, 1e7, 1e5, step=1000.0)
        
        comp = st.multiselect("Complications", ["Infection", "Sepsis", "Respiratory Failure", "AKI (Kidney Injury)", "None"], default=["None"])
        notes = st.text_area("Discharge Notes / Summary")
        sub_out = st.form_submit_button("💾 Save Outcome", use_container_width=True, type="primary")
    
    if sub_out:
        # Check for duplicate
        existing_outcome = next((o for o in st.session_state.outcomes if o['case_id'] == sel_case['id']), None)
        
        outcome = {
            "case_id": sel_case['id'],
            "patient": f"{sel_case['first_name']} {sel_case['last_name']}",
            "severity_initial": sel_case['severity_rule'],
            "los_days": los,
            "surgeries": surg,
            "discharge_status": disc,
            "complications": ";".join(comp),
            "cost_inr_actual": cost_act,
            "notes": notes,
            "date_logged": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if existing_outcome:
            # Update existing
            st.session_state.outcomes = [o if o['case_id'] != sel_case['id'] else outcome for o in st.session_state.outcomes]
            st.success(f"✅ Outcome updated for {sel_case['first_name']}")
        else:
            # Add new
            st.session_state.outcomes.append(outcome)
            st.success(f"✅ Outcome saved for {sel_case['first_name']}")

        save_csv(st.session_state.outcomes, OUTCOMES_CSV)
    
    if st.session_state.outcomes:
        st.divider()
        st.subheader("Outcomes Summary")
        out_df = pd.DataFrame(st.session_state.outcomes)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Outcomes Logged", len(out_df))
        col_s2.metric("Avg. Length of Stay", f"{out_df['los_days'].mean():.1f} days")
        col_s3.metric("Avg. Actual Cost", f"₹{out_df['cost_inr_actual'].mean():,.0f}")
        
        st.dataframe(out_df, use_container_width=True)

# =====================================
# VALIDATION VIEW
# =====================================
def validation_view():
    st.header("🔍 Model Validation")
    
    if not ML_AVAILABLE:
        st.error("ML Models are not loaded. Validation view is unavailable.")
        return
        
    if not st.session_state.cases:
        st.info("No cases have been recorded. Run predictions to validate the model.")
        return
    
    df = pd.DataFrame(st.session_state.cases)
    if 'severity_ml' not in df.columns or df['severity_ml'].isna().all():
        st.warning("No ML predictions are available in the case history. Please run new predictions.")
        return
    
    # Filter out rows where ML prediction wasn't made
    df_val = df.dropna(subset=['severity_ml', 'severity_rule'])
    
    if df_val.empty:
        st.warning("No completed ML predictions found to validate.")
        return

    st.subheader("Rule-Based vs. ML Model Agreement")
    
    y_rule = df_val['severity_rule'].tolist()
    y_ml = df_val['severity_ml'].tolist()
    
    match_count = sum(1 for r, m in zip(y_rule, y_ml) if r == m)
    accuracy = (match_count / len(y_rule)) * 100
    
    st.metric("Model-Rule Agreement Rate", f"{accuracy:.1f}%", 
              help="Percentage of cases where the Rule-Based severity matches the ML-Predicted severity.")
    
    st.subheader("Agreement Confusion Matrix")
    
    labels = ['Minor', 'Moderate', 'Severe']
    conf_data = []
    for rule_sev in labels:
        for ml_sev in labels:
            count = len(df_val[(df_val['severity_rule'] == rule_sev) & (df_val['severity_ml'] == ml_sev)])
            conf_data.append({"Rule-Based": rule_sev, "ML-Predicted": ml_sev, "Count": count})
    
    conf_df = pd.DataFrame(conf_data)
    conf_pivot = conf_df.pivot(index="Rule-Based", columns="ML-Predicted", values="Count").reindex(index=labels, columns=labels).fillna(0)
    
    fig_conf = go.Figure(data=go.Heatmap(
        z=conf_pivot.values,
        x=conf_pivot.columns,
        y=conf_pivot.index,
        text=conf_pivot.values.astype(int),
        texttemplate='%{text}',
        colorscale='Blues'
    ))
    fig_conf.update_layout(title='Rule-Based vs. ML Model Agreement', 
                           xaxis_title='ML Model Prediction', 
                           yaxis_title='Rule-Based Result')
    st.plotly_chart(fig_conf, use_container_width=True)
    
    st.subheader("ML Model Confidence Distribution")
    if 'confidence_ml' in df_val.columns:
        fig_conf_dist = px.histogram(df_val, x='confidence_ml', nbins=20, 
                                     title='Distribution of ML Prediction Confidence')
        fig_conf_dist.update_layout(xaxis_title="Confidence (%)", yaxis_title="Count")
        st.plotly_chart(fig_conf_dist, use_container_width=True)

# =====================================
# MAIN ROUTER
# =====================================
st.sidebar.title("🔥 Burn Severity Predictor")
st.sidebar.markdown("---")

# VISUAL: Added icons to navigation
view = st.sidebar.radio("Navigation", 
                        ["📊 Dashboard", "🧬 Predict", "🧾 Cases", "📅 Outcomes", "🔍 Validation"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Upload images for potential future AI analysis in the 'Predict' tab.")
st.sidebar.info("📊 **Dashboard** shows all statistics and trends from recorded cases.")
st.sidebar.info("📅 **Outcomes** tracks the actual hospital course against the initial prediction.")

if view == "📊 Dashboard":
    dashboard_view()
elif view == "🧬 Predict":
    predict_view()
elif view == "🧾 Cases":
    cases_view()
elif view == "📅 Outcomes":
    outcomes_view()
else:
    validation_view()