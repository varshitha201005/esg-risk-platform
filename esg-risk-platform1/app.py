import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
import warnings
import streamlit_authenticator as stauth
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TripleLens - ESG Risk Analysis Platform",
    page_icon="esg_logo.png",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 2rem; }
    .stButton>button {
        background: linear-gradient(135deg, #1a3c5e, #2ecc71);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1rem;
    }
    .login-box {
        max-width: 400px;
        margin: auto;
        padding: 30px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ─── Authentication Setup ──────────────────────────────────────────────────────
credentials = {
    "usernames": {
        "admin": {
            "email": "admin@esg.com",
            "name": "Admin User",
            "password": "$2b$12$oHqfhncj2mjFPvERKWUm8e9LOkf021H8zVlA01F2EYpd1xhz26v.6"
        },
        "analyst": {
            "email": "analyst@esg.com",
            "name": "ESG Analyst",
            "password": "$2b$12$oHqfhncj2mjFPvERKWUm8e9LOkf021H8zVlA01F2EYpd1xhz26v.6"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "esg_auth_cookie",
    "esg_platform_secret_key",
    cookie_expiry_days=1
)

# ─── Login Page ────────────────────────────────────────────────────────────────
# Show login header
if not st.session_state.get("authentication_status"):
    st.markdown("""
    <div style='text-align:center; padding: 20px;'>
        <h1>🌿 TripleLens - ESG Risk Analysis Platform</h1>
        <p style='color:#666'>Please login to access the platform</p>
    </div>
    """, unsafe_allow_html=True)

authenticator.login(location="main")
name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

if authentication_status == False:
    st.error("❌ Incorrect username or password.")
    st.stop()

elif authentication_status == None:
    st.stop()

elif authentication_status:

    # ─── Sample Dataset ──────────────────────────────────────────────────────
    def get_sample_data():
        np.random.seed(42)
        companies = [
            "Tesla", "Apple", "Microsoft", "ExxonMobil", "Coal India",
            "Infosys", "Reliance", "Google", "Amazon", "BP",
            "Wipro", "TCS", "Shell", "HDFC", "Nestle",
            "Unilever", "Toyota", "Samsung", "HSBC", "Goldman Sachs"
        ]
        data = {
            "company": companies,
            "carbon_emissions": np.random.randint(10, 100, 20),
            "energy_usage": np.random.randint(20, 100, 20),
            "environmental_score": np.random.randint(20, 100, 20),
            "employee_turnover": np.random.randint(5, 50, 20),
            "diversity_ratio": np.random.randint(20, 80, 20),
            "social_score": np.random.randint(20, 100, 20),
            "board_structure": np.random.randint(30, 100, 20),
            "compliance_score": np.random.randint(30, 100, 20),
            "governance_score": np.random.randint(20, 100, 20),
        }
        return pd.DataFrame(data)

    # ─── Preprocessing ───────────────────────────────────────────────────────
    def preprocess_data(df):
        df = df.copy()
        df = df.rename(columns={'name': 'company', 'environment_score': 'environmental_score'})
        if 'company' not in df.columns and 'ticker' in df.columns:
            df['company'] = df['ticker']
        elif 'company' not in df.columns:
            df['company'] = [f"Company {i+1}" for i in range(len(df))]
        for col in ['environmental_score', 'social_score', 'governance_score']:
            if col not in df.columns:
                df[col] = 50
        if 'carbon_emissions' not in df.columns:
            df['carbon_emissions'] = np.random.randint(10, 100, len(df))
        if 'energy_usage' not in df.columns:
            df['energy_usage'] = np.random.randint(20, 100, len(df))
        if 'employee_turnover' not in df.columns:
            df['employee_turnover'] = np.random.randint(5, 50, len(df))
        if 'diversity_ratio' not in df.columns:
            df['diversity_ratio'] = np.random.randint(20, 80, len(df))
        if 'board_structure' not in df.columns:
            df['board_structure'] = np.random.randint(30, 100, len(df))
        if 'compliance_score' not in df.columns:
            df['compliance_score'] = np.random.randint(30, 100, len(df))
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        for col in ['environmental_score', 'social_score', 'governance_score']:
            if df[col].max() > 100:
                df[col] = (df[col] / df[col].max()) * 100
        keep_cols = [
            'company', 'environmental_score', 'social_score', 'governance_score',
            'carbon_emissions', 'energy_usage', 'employee_turnover',
            'diversity_ratio', 'board_structure', 'compliance_score'
        ]
        df = df[[c for c in keep_cols if c in df.columns]]
        return df

    # ─── ESG Scoring ─────────────────────────────────────────────────────────
    def calculate_esg_score(row):
        env = row['environmental_score']
        soc = row['social_score']
        gov = row['governance_score']
        max_val = max(env, soc, gov)
        if max_val > 100:
            env = (env / max_val) * 100
            soc = (soc / max_val) * 100
            gov = (gov / max_val) * 100
        weighted = round((env * 0.4) + (soc * 0.3) + (gov * 0.3), 1)
        return min(weighted, 100)

    def get_risk_label(score):
        if score >= 70:
            return "Low Risk"
        elif score >= 45:
            return "Medium Risk"
        else:
            return "High Risk"

    def get_risk_color(risk):
        return {
            "Low Risk": "#2ecc71",
            "Medium Risk": "#f39c12",
            "High Risk": "#e74c3c"
        }.get(risk, "#ccc")

    # ─── PDF Generation ───────────────────────────────────────────────────────
    def generate_pdf(df, filtered_df):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()
        pdf.set_fill_color(26, 60, 94)
        pdf.rect(0, 0, 210, 60, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_y(15)
        pdf.cell(0, 10, "ESG Risk & Insight Platform", ln=True, align='C')
        pdf.set_font("Helvetica", "", 14)
        pdf.cell(0, 10, "Investment Risk Analysis Report", ln=True, align='C')
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 10, f"Generated on: {pd.Timestamp.now().strftime('%B %d, %Y')}", ln=True, align='C')

        pdf.set_text_color(0, 0, 0)
        pdf.set_y(75)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(26, 60, 94)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.set_draw_color(46, 204, 113)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        total = len(filtered_df)
        low = len(filtered_df[filtered_df['risk_label'] == 'Low Risk'])
        med = len(filtered_df[filtered_df['risk_label'] == 'Medium Risk'])
        high = len(filtered_df[filtered_df['risk_label'] == 'High Risk'])
        avg_score = round(filtered_df['esg_score'].mean(), 1)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        for label, value in [
            ("Total Companies Analyzed", str(total)),
            ("Average ESG Score", str(avg_score)),
            ("Safe Investments (Low Risk)", str(low)),
            ("Moderate Risk Companies", str(med)),
            ("High Risk Companies", str(high)),
        ]:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(100, 8, label + ":", ln=False)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, value, ln=True)

        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(26, 60, 94)
        pdf.cell(0, 10, "Risk Distribution", ln=True)
        pdf.set_draw_color(46, 204, 113)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_fill_color(26, 60, 94)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(60, 9, "Risk Level", border=1, fill=True, align='C')
        pdf.cell(40, 9, "Companies", border=1, fill=True, align='C')
        pdf.cell(40, 9, "Percentage", border=1, fill=True, align='C')
        pdf.cell(50, 9, "Avg ESG Score", border=1, fill=True, align='C')
        pdf.ln()

        risk_colors = {
            "Low Risk": (46, 204, 113),
            "Medium Risk": (243, 156, 18),
            "High Risk": (231, 76, 60)
        }

        pdf.set_font("Helvetica", "", 11)
        for risk, color in risk_colors.items():
            count = len(filtered_df[filtered_df['risk_label'] == risk])
            pct = round((count / total * 100), 1) if total > 0 else 0
            avg = round(filtered_df[filtered_df['risk_label'] == risk]['esg_score'].mean(), 1) if count > 0 else 0
            pdf.set_fill_color(*color)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(60, 8, risk, border=1, fill=True, align='C')
            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(40, 8, str(count), border=1, fill=True, align='C')
            pdf.cell(40, 8, f"{pct}%", border=1, fill=True, align='C')
            pdf.cell(50, 8, str(avg), border=1, fill=True, align='C')
            pdf.ln()

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(26, 60, 94)
        pdf.cell(0, 10, "Company-wise ESG Analysis", ln=True)
        pdf.set_draw_color(46, 204, 113)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_fill_color(26, 60, 94)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 9, "Company", border=1, fill=True, align='C')
        pdf.cell(25, 9, "Env Score", border=1, fill=True, align='C')
        pdf.cell(25, 9, "Soc Score", border=1, fill=True, align='C')
        pdf.cell(25, 9, "Gov Score", border=1, fill=True, align='C')
        pdf.cell(25, 9, "ESG Score", border=1, fill=True, align='C')
        pdf.cell(40, 9, "Risk Level", border=1, fill=True, align='C')
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for i, (_, row) in enumerate(filtered_df.sort_values('esg_score', ascending=False).iterrows()):
            if i % 2 == 0:
                pdf.set_fill_color(245, 245, 245)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(50, 8, str(row['company'])[:20], border=1, fill=True)
            pdf.cell(25, 8, str(round(row['environmental_score'], 1)), border=1, fill=True, align='C')
            pdf.cell(25, 8, str(round(row['social_score'], 1)), border=1, fill=True, align='C')
            pdf.cell(25, 8, str(round(row['governance_score'], 1)), border=1, fill=True, align='C')
            pdf.cell(25, 8, str(row['esg_score']), border=1, fill=True, align='C')
            pdf.set_fill_color(*risk_colors.get(row['risk_label'], (128, 128, 128)))
            pdf.set_text_color(255, 255, 255)
            pdf.cell(40, 8, row['risk_label'], border=1, fill=True, align='C')
            pdf.set_text_color(0, 0, 0)
            pdf.ln()

        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, "Generated by ESG Risk & Insight Platform | Confidential Report", align='C')

        return bytes(pdf.output())

    # ─── ML Training ─────────────────────────────────────────────────────────
    def train_models(df):
        features = ['environmental_score', 'social_score', 'governance_score']
        X = df[features]
        y = df['risk_label'].map({"Low Risk": 0, "Medium Risk": 1, "High Risk": 2})
        if y.nunique() < 2:
            st.error("⚠️ Dataset has only one risk class. Please upload a more diverse dataset.")
            st.stop()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "model": model,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "y_test": y_test,
                "y_pred": y_pred
            }
        return results, scaler

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    from PIL import Image
import os

logo_path = os.path.join(os.path.dirname(__file__), "esg_logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
    st.sidebar.title("TripleLens")
    st.sidebar.markdown(f"👤 Welcome, **{name}**!")
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader("📁 Upload ESG Dataset (CSV)", type=["csv"])
    selected_model = st.sidebar.selectbox(
        "🤖 Select ML Model",
        ["Random Forest", "Logistic Regression", "Gradient Boosting"]
    )
    st.sidebar.markdown("---")
    risk_filter = st.sidebar.multiselect(
        "🔍 Filter by Risk Level",
        ["Low Risk", "Medium Risk", "High Risk"],
        default=["Low Risk", "Medium Risk", "High Risk"]
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Filter by ESG Score Range**")
    score_min, score_max = st.sidebar.slider("ESG Score Range", 0, 100, (0, 100))
    st.sidebar.markdown("---")
    authenticator.logout(location="sidebar")
    st.sidebar.info("Upload your Kaggle ESG dataset or use sample data to get started.")

    # ─── Load & Process Data ─────────────────────────────────────────────────
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        df = preprocess_data(raw_df)
        st.sidebar.success(f"✅ Loaded {len(df)} companies")
    else:
        df = get_sample_data()
        df = preprocess_data(df)

    df['esg_score'] = df.apply(calculate_esg_score, axis=1)
    df['risk_label'] = df['esg_score'].apply(get_risk_label)
    filtered_df = df[
        (df['risk_label'].isin(risk_filter)) &
        (df['esg_score'] >= score_min) &
        (df['esg_score'] <= score_max)
    ]

    # ─── Header ──────────────────────────────────────────────────────────────
    st.markdown("## TripleLens - ESG Risk Analysis Platform")
    st.markdown("Analyze companies based on Environmental, Social, and Governance data using Machine Learning.")
    if len(risk_filter) < 3:
        st.info(f"🔍 Filtering by: **{', '.join(risk_filter)}** | ESG Score: **{score_min} – {score_max}**")
    st.markdown("---")

    # ─── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "🔬 Data Preview",
        "🤖 ML Predictions",
        "📈 Model Evaluation",
        "🏢 Company Insights",
        "📄 Export Report"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("📊 ESG Risk Overview")
        if filtered_df.empty:
            st.warning("⚠️ No companies match the current filters.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Filtered Companies", len(filtered_df))
            with col2:
                st.metric("✅ Low Risk", len(filtered_df[filtered_df['risk_label'] == 'Low Risk']))
            with col3:
                st.metric("⚠️ Medium Risk", len(filtered_df[filtered_df['risk_label'] == 'Medium Risk']))
            with col4:
                st.metric("🚨 High Risk", len(filtered_df[filtered_df['risk_label'] == 'High Risk']))

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    filtered_df, names='risk_label', title='Risk Distribution (Filtered)',
                    color='risk_label',
                    color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                fig_bar = px.bar(
                    filtered_df.sort_values('esg_score', ascending=False).head(20),
                    x='company', y='esg_score', color='risk_label',
                    title='Top 20 Companies ESG Scores (Filtered)',
                    color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_scatter = px.scatter(
                    filtered_df, x='environmental_score', y='social_score',
                    size='esg_score', color='risk_label', hover_name='company',
                    title='Environmental vs Social Score (Filtered)',
                    color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            with col2:
                fig_box = px.box(
                    filtered_df, x='risk_label', y='esg_score', color='risk_label',
                    title='ESG Score Distribution by Risk (Filtered)',
                    color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
                )
                st.plotly_chart(fig_box, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DATA PREVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🔬 Dataset Preview & Preprocessing")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📋 Total Rows: {df.shape[0]}")
        with col2:
            st.info(f"🔍 Filtered Rows: {filtered_df.shape[0]}")
        with col3:
            st.success(f"✅ Missing Values: {df.isnull().sum().sum()}")

        st.markdown("### 📋 Filtered Dataset")
        search = st.text_input("🔍 Search company", "")
        display_df = (
            filtered_df[filtered_df['company'].str.contains(search, case=False, na=False)]
            if search else filtered_df
        )
        st.dataframe(
            display_df.style.background_gradient(subset=['esg_score'], cmap='RdYlGn'),
            use_container_width=True
        )
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(filtered_df.describe(), use_container_width=True)

        st.markdown("### 🔥 Correlation Heatmap")
        numeric_df = filtered_df.select_dtypes(include=np.number)
        fig_heat, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig_heat)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — ML PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("🤖 Machine Learning Risk Prediction")
        if len(df) < 5:
            st.warning("⚠️ Need at least 5 companies to train models.")
        else:
            with st.spinner("Training ML models..."):
                results, scaler = train_models(df)
            st.success(f"✅ Models trained on {len(df)} companies!")

            st.markdown("### 📊 Model Performance Comparison")
            perf_data = {
                "Model": list(results.keys()),
                "Accuracy": [round(r['accuracy'], 3) for r in results.values()],
                "Precision": [round(r['precision'], 3) for r in results.values()],
                "Recall": [round(r['recall'], 3) for r in results.values()],
                "F1 Score": [round(r['f1'], 3) for r in results.values()]
            }
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
            fig_perf = px.bar(
                perf_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric', barmode='group',
                title='Model Performance Comparison'
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            st.markdown("### 🔮 Predict Risk for New Company")
            col1, col2, col3 = st.columns(3)
            with col1:
                env = st.slider("🌱 Environmental Score", 0, 100, 50)
            with col2:
                soc = st.slider("🤝 Social Score", 0, 100, 50)
            with col3:
                gov = st.slider("🏛️ Governance Score", 0, 100, 50)

            if st.button("🔮 Predict Risk Level"):
                model = results[selected_model]['model']
                input_data = scaler.transform([[env, soc, gov]])
                prediction = model.predict(input_data)[0]
                risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
                predicted_risk = risk_map[prediction]
                esg = round((env * 0.4) + (soc * 0.3) + (gov * 0.3), 1)
                color = get_risk_color(predicted_risk)
                st.markdown(f"""
                <div style='background:white; border-radius:12px; padding:25px;
                            text-align:center; border-top: 5px solid {color};
                            box-shadow: 0 2px 10px rgba(0,0,0,0.08);'>
                    <h2 style='color:{color}'>{predicted_risk}</h2>
                    <p style='font-size:1.1rem'>Overall ESG Score: <strong>{esg} / 100</strong></p>
                    <p>Model used: <strong>{selected_model}</strong></p>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — MODEL EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("📈 Model Evaluation")
        if len(df) < 5:
            st.warning("⚠️ Need at least 5 companies to evaluate models.")
        else:
            with st.spinner("Evaluating models..."):
                results, scaler = train_models(df)

            selected_eval = st.selectbox("Select Model to Evaluate", list(results.keys()))
            r = results[selected_eval]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{r['accuracy']:.2%}")
            col2.metric("Precision", f"{r['precision']:.2%}")
            col3.metric("Recall", f"{r['recall']:.2%}")
            col4.metric("F1 Score", f"{r['f1']:.2%}")

            st.markdown("### 🔲 Confusion Matrix")
            fig_cm, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                r['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"]
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig_cm)

            if selected_eval == "Random Forest":
                st.markdown("### 🌟 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': ['environmental_score', 'social_score', 'governance_score'],
                    'Importance': r['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                fig_imp = px.bar(
                    importance_df, x='Feature', y='Importance',
                    title='Feature Importance', color='Importance',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_imp, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — COMPANY INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("🏢 Company Insights")
        if filtered_df.empty:
            st.warning("⚠️ No companies match the current filters.")
        else:
            selected_company = st.selectbox("Select a Company", filtered_df['company'].tolist())
            company = filtered_df[filtered_df['company'] == selected_company].iloc[0]
            risk_color = get_risk_color(company['risk_label'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style='background:white; border-radius:12px; padding:25px;
                            border-top: 5px solid {risk_color};
                            box-shadow: 0 2px 10px rgba(0,0,0,0.08);'>
                    <h2>{company['company']}</h2>
                    <h3 style='color:{risk_color}'>{company['risk_label']}</h3>
                    <p style='font-size:1.1rem'>Overall ESG Score: <strong>{company['esg_score']} / 100</strong></p>
                    <hr/>
                    <p>🌱 Environmental Score: <strong>{round(company['environmental_score'], 1)}</strong></p>
                    <p>🤝 Social Score: <strong>{round(company['social_score'], 1)}</strong></p>
                    <p>🏛️ Governance Score: <strong>{round(company['governance_score'], 1)}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=[company['environmental_score'], company['social_score'], company['governance_score']],
                    theta=['Environmental', 'Social', 'Governance'],
                    fill='toself', line_color=risk_color
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title=f"{company['company']} ESG Radar Chart"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("### 💡 Investment Recommendation")
            if company['risk_label'] == 'Low Risk':
                st.success(f"✅ **Safe Investment** — {company['company']} has strong ESG practices and is a safe investment.")
            elif company['risk_label'] == 'Medium Risk':
                st.warning(f"⚠️ **Moderate Risk** — {company['company']} shows average ESG performance. Invest with caution.")
            else:
                st.error(f"🚨 **High Risk – Caution** — {company['company']} has poor ESG scores. Investment is not recommended.")

            st.markdown("### 📊 Score Breakdown")
            score_df = pd.DataFrame({
                'Category': ['Environmental (40%)', 'Social (30%)', 'Governance (30%)'],
                'Score': [round(company['environmental_score'], 1), round(company['social_score'], 1), round(company['governance_score'], 1)],
                'Weighted Score': [round(company['environmental_score'] * 0.4, 1), round(company['social_score'] * 0.3, 1), round(company['governance_score'] * 0.3, 1)]
            })
            fig_scores = px.bar(
                score_df, x='Category', y='Score', color='Category',
                title='Category Score Breakdown', text='Weighted Score'
            )
            st.plotly_chart(fig_scores, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — EXPORT REPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab6:
        st.subheader("📄 Export ESG Analysis Report")
        if filtered_df.empty:
            st.warning("⚠️ No data to export. Please adjust your filters.")
        else:
            st.markdown("### 📋 Report Preview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Companies", len(filtered_df))
            with col2:
                st.metric("Avg ESG Score", round(filtered_df['esg_score'].mean(), 1))
            with col3:
                st.metric("✅ Low Risk", len(filtered_df[filtered_df['risk_label'] == 'Low Risk']))
            with col4:
                st.metric("🚨 High Risk", len(filtered_df[filtered_df['risk_label'] == 'High Risk']))

            st.markdown("### 🔍 Data to be Exported")
            st.dataframe(
                filtered_df[['company', 'environmental_score', 'social_score',
                             'governance_score', 'esg_score', 'risk_label']]
                .sort_values('esg_score', ascending=False),
                use_container_width=True
            )

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📄 Download PDF Report")
                st.markdown("Professional report with cover page, summary and company table.")
                if st.button("📄 Generate & Download PDF"):
                    with st.spinner("Generating PDF report..."):
                        pdf_bytes = generate_pdf(df, filtered_df)
                    st.download_button(
                        label="⬇️ Click to Download PDF",
                        data=pdf_bytes,
                        file_name=f"ESG_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF ready! Click the button above to download.")
            with col2:
                st.markdown("#### 📊 Download CSV Data")
                st.markdown("Raw data export with all ESG scores and risk labels.")
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"ESG_Data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )