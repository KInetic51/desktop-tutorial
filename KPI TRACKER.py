import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Kinetic Health OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Brand Colors (Kinetic Theme)
COLORS = {
    "primary": "#0ea5e9",    # Kinetic Teal/Blue
    "secondary": "#3b82f6",  # Deep Blue
    "background": "#f8fafc",
    "surface": "#ffffff",
    "success": "#10b981",
    "warning": "#f59e0b",
    "critical": "#ef4444",
    "text_main": "#0f172a",
    "text_muted": "#64748b"
}

# Advanced Custom CSS for Modern UI
st.markdown(f"""
    <style>
        /* Global Background */
        .stApp {{
            background-color: {COLORS['background']};
        }}
        
        /* Typography */
        h1, h2, h3 {{
            color: {COLORS['text_main']};
            font-family: 'Inter', 'Segoe UI', sans-serif;
            font-weight: 700;
        }}
        
        /* Metric Card Styling */
        div[data-testid="metric-container"] {{
            background-color: {COLORS['surface']};
            border: 1px solid #e2e8f0;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border-left: 5px solid {COLORS['primary']};
            transition: transform 0.2s ease;
        }}
        div[data-testid="metric-container"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}
        
        /* Metric Labels & Values */
        div[data-testid="stMetricLabel"] {{
            color: {COLORS['text_muted']} !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {COLORS['text_main']} !important;
            font-weight: 800 !important;
            font-size: 2rem !important;
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }}
        
        /* Expander/Cards */
        .streamlit-expanderHeader {{
            background-color: white;
            border-radius: 8px;
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DYNAMIC CONFIGURATION (SIDEBAR)
# ==========================================
st.sidebar.title("⚡ Kinetic Health")
st.sidebar.caption("Intelligent Hospital Management OS")
st.sidebar.markdown("---")

# Navigation
module = st.sidebar.radio(
    "Modules",
    ["📊 Operations Dashboard", "💰 Financial Intelligence", "☢️ Diagnostics Hub", "🔪 Surgical & Critical", "🔮 AI Forecasting", "🧪 Scenario Simulator", "🛡️ Quality & Risk"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Hospital Profile Settings")
st.sidebar.caption("Adjust to fit your facility's scale")

# Dynamic Parameters
TOTAL_BEDS = st.sidebar.number_input("Total Ward Beds", min_value=50, max_value=2000, value=150, step=10)
ICU_BEDS = st.sidebar.number_input("Total ICU Beds", min_value=10, max_value=200, value=20, step=2)
BASE_OPD = st.sidebar.number_input("Avg Daily OPD Volume", min_value=100, max_value=5000, value=500, step=50)

# ==========================================
# 3. DYNAMIC DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600) # Cache clears if inputs change or after 1 hour
def load_dynamic_data(total_beds, icu_beds, base_opd):
    """Generates synthetic data scaled to the user's hospital size."""
    dates = pd.date_range(end=datetime.today(), periods=30)
    
    # Scale factors
    adm_rate = 0.10 # 10% of OPD gets admitted
    
    ops_data = pd.DataFrame({
        "Date": dates,
        "OPD_Footfall": np.random.normal(base_opd, base_opd*0.1, 30).astype(int),
        "Admissions": np.random.normal(base_opd*adm_rate, (base_opd*adm_rate)*0.1, 30).astype(int),
        "Discharges": np.random.normal(base_opd*adm_rate*0.95, (base_opd*adm_rate)*0.15, 30).astype(int),
        "Ward_Census": np.clip(np.random.normal(total_beds*0.85, total_beds*0.1, 30), 0, total_beds*1.1).astype(int),
        "ICU_Census": np.clip(np.random.normal(icu_beds*0.80, icu_beds*0.15, 30), 0, icu_beds).astype(int)
    })
    
    # Financials (Scaled roughly to volume)
    avg_ticket = 25000 # Generic currency unit
    daily_rev = ops_data["Admissions"] * avg_ticket
    
    fin_data = pd.DataFrame({
        "Date": dates,
        "Gross_Billed": daily_rev * np.random.uniform(0.9, 1.1, 30),
        "Cash_Collected": daily_rev * np.random.uniform(0.6, 0.85, 30),
        "Denial_Rate": np.random.uniform(0.02, 0.15, size=30)
    })
    
    dept_data = pd.DataFrame({
        "Department": ["Internal Medicine", "Orthopedics", "General Surgery", "Gynecology", "Cardiology"],
        "ALOS": [3.5, 4.2, 3.8, 2.9, 4.5],
        "ARPOB": [9000, 15000, 13000, 8500, 22000],
        "Revenue_Share": [0.25, 0.20, 0.25, 0.15, 0.15]
    })

    dx_data = pd.DataFrame({
        "Modality": ["X-Ray", "USG", "CT", "MRI", "Pathology"],
        "Capacity_Per_Day": [base_opd*0.4, base_opd*0.2, base_opd*0.08, base_opd*0.04, base_opd*1.5],
        "Current_Usage": [base_opd*0.35, base_opd*0.18, base_opd*0.07, base_opd*0.04, base_opd*1.2]
    })
    
    return ops_data, fin_data, dept_data, dx_data

# Load Data dynamically based on sidebar inputs
ops_df, fin_df, dept_df, dx_df = load_dynamic_data(TOTAL_BEDS, ICU_BEDS, BASE_OPD)

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def calculate_bor(census, capacity):
    if capacity == 0: return 0
    return round((census / capacity) * 100, 1)

def format_currency(value):
    if value >= 10000000: return f"${value/1000000:.1f}M"
    if value >= 1000: return f"${value/1000:.1f}K"
    return f"${value:.0f}"

# ==========================================
# 5. GLOBAL ALERT ENGINE
# ==========================================
# Evaluates background data to push global alerts to the top of the app
curr_ward_bor = calculate_bor(ops_df['Ward_Census'].iloc[-1], TOTAL_BEDS)
curr_icu_bor = calculate_bor(ops_df['ICU_Census'].iloc[-1], ICU_BEDS)
avg_denial = fin_df['Denial_Rate'].mean() * 100

st.container()
alert_cols = st.columns(1)
with alert_cols[0]:
    if curr_ward_bor > 95:
        st.error(f"⚠️ **CAPACITY CRITICAL:** Ward Occupancy is currently at {curr_ward_bor}% (Total Beds: {TOTAL_BEDS}). Implement early discharge protocols.")
    if curr_icu_bor > 90:
        st.warning(f"🚨 **ICU ALERT:** ICU Capacity is at {curr_icu_bor}%. Only {ICU_BEDS - ops_df['ICU_Census'].iloc[-1]} beds remaining.")

# ==========================================
# MODULE 1: OPERATIONS
# ==========================================
if module == "📊 Operations Dashboard":
    st.header("Operations Command Center")
    st.markdown("Real-time facility throughput and occupancy monitoring.")
    
    # KPIs
    curr_opd = int(ops_df['OPD_Footfall'].iloc[-1])
    curr_admits = int(ops_df['Admissions'].iloc[-1])
    curr_disch = int(ops_df['Discharges'].iloc[-1])
    conversion = round((curr_admits / curr_opd) * 100, 1) if curr_opd > 0 else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Today's OPD Volume", f"{curr_opd:,}", f"{int(curr_opd - ops_df['OPD_Footfall'].iloc[-2])} vs yesterday")
    k2.metric("Admissions", curr_admits, f"{curr_admits - int(ops_df['Admissions'].iloc[-2])}")
    k3.metric("Discharges", curr_disch, f"{curr_disch - int(ops_df['Discharges'].iloc[-2])}", delta_color="inverse")
    k4.metric("OPD to IP Conversion", f"{conversion}%", "Target: 10%")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Bed Occupancy Trend (30 Days)")
        chart_bor = alt.Chart(ops_df).mark_area(
            line={'color': COLORS['primary']},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color=COLORS['primary'], offset=0),
                       alt.GradientStop(color='white', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Date', title=''),
            y=alt.Y('Ward_Census', scale=alt.Scale(domain=[0, TOTAL_BEDS*1.2]), title='Patients'),
            tooltip=['Date', 'Ward_Census']
        ).interactive()
        
        # Add capacity line
        capacity_line = alt.Chart(pd.DataFrame({'y': [TOTAL_BEDS]})).mark_rule(strokeDash=[5, 5], color=COLORS['critical']).encode(y='y')
        st.altair_chart(chart_bor + capacity_line, use_container_width=True)

    with c2:
        st.subheader("Patient Flow (Admissions vs Discharges)")
        chart_flow = alt.Chart(ops_df.tail(14)).transform_fold(
            ['Admissions', 'Discharges'],
            as_=['Flow Type', 'Count']
        ).mark_bar(opacity=0.8).encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('Count:Q', stack=False),
            color=alt.Color('Flow Type:N', scale=alt.Scale(domain=['Admissions', 'Discharges'], range=[COLORS['primary'], COLORS['warning']])),
            tooltip=['Date', 'Flow Type', 'Count']
        ).interactive()
        st.altair_chart(chart_flow, use_container_width=True)

# ==========================================
# MODULE 2: FINANCIALS
# ==========================================
elif module == "💰 Financial Intelligence":
    st.header("Financial Performance")
    
    total_rev = fin_df['Gross_Billed'].sum()
    avg_arpob = dept_df['ARPOB'].mean()
    denial_impact = total_rev * (avg_denial/100)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MTD Gross Revenue", format_currency(total_rev))
    col2.metric("Avg ARPOB", format_currency(avg_arpob))
    col3.metric("Claim Denial Rate", f"{avg_denial:.1f}%", "-0.5% vs Last Month", delta_color="inverse")
    col4.metric("Revenue Leakage", format_currency(denial_impact), "Requires Action", delta_color="inverse")
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Billing vs Realized Cash Trend")
        fin_melt = fin_df.melt(id_vars=['Date'], value_vars=['Gross_Billed', 'Cash_Collected'], var_name='Metric', value_name='Amount')
        line_fin = alt.Chart(fin_melt).mark_line(point=True).encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('Amount:Q', title='Amount ($)'),
            color=alt.Color('Metric:N', scale=alt.Scale(range=[COLORS['primary'], COLORS['success']]))
        ).interactive()
        st.altair_chart(line_fin, use_container_width=True)
        
    with c2:
        st.subheader("ARPOB by Specialty")
        dept_df_sorted = dept_df.sort_values('ARPOB', ascending=False)
        st.dataframe(
            dept_df_sorted[['Department', 'ARPOB']],
            column_config={
                "ARPOB": st.column_config.ProgressColumn(
                    "Avg Rev per Occupied Bed",
                    format="$%f",
                    min_value=0,
                    max_value=max(dept_df['ARPOB'])
                ),
            },
            hide_index=True,
            use_container_width=True
        )

# ==========================================
# MODULE 3: DIAGNOSTICS
# ==========================================
elif module == "☢️ Diagnostics Hub":
    st.header("Diagnostics & Lab Utilization")
    
    dx_df['Utilization'] = round((dx_df['Current_Usage'] / dx_df['Capacity_Per_Day']) * 100, 1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Asset Utilization Grid")
        bars = alt.Chart(dx_df).mark_bar().encode(
            x=alt.X('Utilization:Q', title='Utilization (%)', scale=alt.Scale(domain=[0, 120])),
            y=alt.Y('Modality:N', sort='-x', title=''),
            color=alt.condition(
                alt.datum.Utilization > 90,
                alt.value(COLORS['critical']),
                alt.condition(alt.datum.Utilization > 75, alt.value(COLORS['warning']), alt.value(COLORS['primary']))
            ),
            tooltip=['Modality', 'Current_Usage', 'Capacity_Per_Day', 'Utilization']
        ).properties(height=350)
        
        rule = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(strokeDash=[5,5], color='red').encode(x='y')
        st.altair_chart(bars + rule, use_container_width=True)

    with col2:
        st.subheader("Optimization Alerts")
        for index, row in dx_df.sort_values('Utilization', ascending=False).iterrows():
            if row['Utilization'] > 95:
                st.error(f"🚨 **{row['Modality']}**: Over capacity ({row['Utilization']}%). Consider diverting non-urgent cases.")
            elif row['Utilization'] > 80:
                st.warning(f"⚠️ **{row['Modality']}**: High load ({row['Utilization']}%). Monitor turnaround times.")
            else:
                st.success(f"✅ **{row['Modality']}**: Optimal flow ({row['Utilization']}%).")

# ==========================================
# MODULE 4: SURGICAL & CRITICAL
# ==========================================
elif module == "🔪 Surgical & Critical":
    st.header("Surgical & Critical Care Core")
    
    tab1, tab2 = st.tabs(["Operating Theatres", "Intensive Care Units"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Scheduled Surgeries (Today)", "24", "On Track")
        c2.metric("OT Turnaround Time", "28 mins", "-4 mins", delta_color="inverse")
        c3.metric("First Case On-Time Starts", "82%", "+5%")
        
        st.markdown("---")
        st.subheader("Active Theatre Status")
        ot_data = pd.DataFrame({
            'Theatre': ['OT-1 (Cardio)', 'OT-2 (Neuro)', 'OT-3 (Ortho)', 'OT-4 (Gen Surg)', 'OT-5 (Emergency)'],
            'Status': ['In Procedure', 'Cleaning', 'In Procedure', 'Idle', 'Reserved'],
            'Current Case Duration': ['1h 45m', '-', '45m', '-', '-'],
            'Next Scheduled': ['14:00', '13:15', '14:30', '15:00', 'Standby']
        })
        st.dataframe(ot_data, use_container_width=True, hide_index=True)

    with tab2:
        ic1, ic2 = st.columns(2)
        current_icu = int(ops_df['ICU_Census'].iloc[-1])
        
        with ic1:
            st.metric(f"Total ICU Occupancy", f"{current_icu} / {ICU_BEDS}")
            st.progress(min(current_icu / ICU_BEDS, 1.0))
            if current_icu >= ICU_BEDS:
                st.error("ICU is currently at maximum capacity. Activate step-down protocols immediately.")
                
        with ic2:
            st.metric("Ventilator Utilization", f"{int(current_icu * 0.45)} Active")
            st.metric("Patients Ready for Step-Down (Ward)", "4")

# ==========================================
# MODULE 5: FORECASTING
# ==========================================
elif module == "🔮 AI Forecasting":
    st.header("Predictive Volume Modeling")
    st.markdown("Uses linear regression on your historical 30-day facility data to project upcoming demand.")
    
    forecast_days = st.slider("Select Forecast Horizon (Days)", 7, 30, 14)
    
    # Simple linear regression forecasting
    df = ops_df.copy()
    df['Day_Index'] = range(len(df))
    X = df[['Day_Index']]
    y = df['OPD_Footfall']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.array(range(len(df), len(df) + forecast_days)).reshape(-1, 1)
    future_pred = model.predict(future_X)
    
    future_dates = pd.date_range(start=datetime.today() + timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "OPD_Footfall": future_pred, "Type": "Forecast"})
    
    hist_df = df[['Date', 'OPD_Footfall']].copy()
    hist_df['Type'] = "Historical"
    
    combined_df = pd.concat([hist_df, forecast_df])
    
    chart = alt.Chart(combined_df).mark_line(point=True).encode(
        x=alt.X('Date:T', title=''),
        y=alt.Y('OPD_Footfall:Q', title='Patient Volume', scale=alt.Scale(zero=False)),
        color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], range=[COLORS['primary'], COLORS['warning']])),
        strokeDash=alt.condition(
            alt.datum.Type == 'Forecast',
            alt.value([5, 5]),
            alt.value([0])
        )
    ).properties(height=400).interactive()
    
    st.altair_chart(chart, use_container_width=True)

# ==========================================
# MODULE 6: SIMULATOR
# ==========================================
elif module == "🧪 Scenario Simulator":
    st.header("Financial & Operational Simulator")
    st.markdown("Test strategic interventions to forecast bottom-line impact.")
    
    with st.form("sim_form"):
        col1, col2, col3 = st.columns(3)
        sim_alos = col1.slider("Target ALOS Reduction (Days)", 0.0, 1.5, 0.5, 0.1)
        sim_denial = col2.slider("Target Denial Rate Reduction (%)", 0.0, 5.0, 2.0, 0.5)
        sim_vol = col3.slider("Expected Volume Growth (%)", 0, 20, 5, 1)
        
        submitted = st.form_submit_button("Run Simulation ⚡")
    
    if submitted:
        # Base logic adapted dynamically based on inputs
        current_rev = fin_df['Gross_Billed'].iloc[-1] * 30 # Monthly roughly
        
        # Simulated savings
        alos_savings = (sim_alos * TOTAL_BEDS * 500) # Proxy calculation
        denial_savings = current_rev * (sim_denial / 100)
        vol_revenue = current_rev * (sim_vol / 100)
        
        total_impact = alos_savings + denial_savings + vol_revenue
        
        st.success(f"Simulation Complete: Projected Monthly Impact **+{format_currency(total_impact)}**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("ALOS Efficiency Gains", format_currency(alos_savings))
        c2.metric("Revenue Recovered (Denials)", format_currency(denial_savings))
        c3.metric("New Volume Revenue", format_currency(vol_revenue))

# ==========================================
# MODULE 7: QUALITY & RISK
# ==========================================
elif module == "🛡️ Quality & Risk":
    st.header("Clinical Quality & Safety Index")
    
    q1, q2, q3 = st.columns(3)
    q1.metric("Reported Incidents (MTD)", "4", "Requires Review", delta_color="inverse")
    q2.metric("Hospital Acquired Infections", "0.8%", "-0.2%", delta_color="normal")
    q3.metric("Medication Errors", "0", "0", delta_color="off")
    
    st.markdown("---")
    st.subheader("Open Compliance & Risk Tickets")
    
    # Dynamic styling for risk tickets
    def highlight_risk(val):
        if val == 'High': return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
        if val == 'Medium': return 'background-color: #fef3c7; color: #92400e; font-weight: bold'
        return ''

    risk_df = pd.DataFrame({
        "Incident ID": ["INC-104", "INC-108", "CMP-092", "AUD-014"],
        "Category": ["Equipment", "Protocol", "Patient Feedback", "Pharmacy"],
        "Description": ["Backup generator fail-test delay", "Surgical checklist missing signature", "Excessive discharge wait time (>4hrs)", "Narcotics cabinet audit discrepancy"],
        "Risk Level": ["High", "Medium", "Medium", "High"],
        "Status": ["Investigating", "Open", "Resolved", "Urgent Review"]
    })
    
    st.dataframe(risk_df.style.applymap(highlight_risk, subset=['Risk Level']), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #94a3b8; font-size: 13px; font-weight: 500;'>
        Powered by Kinetic Health OS ⚡ | Unified Hospital Intelligence
    </div>
    """, 
    unsafe_allow_html=True
)