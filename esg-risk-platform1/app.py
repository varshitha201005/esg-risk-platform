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
import bcrypt
import json
import os
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .stButton>button {
        background: linear-gradient(135deg, #1a3c5e, #2ecc71);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 204, 113, 0.4);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #1a3c5e !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #666 !important;
    }
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #2ecc71;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        color: #666;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a3c5e, #2ecc71) !important;
        color: white !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3c5e 0%, #0d2137 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stFileUploader {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    [data-testid="stSidebar"] .stFileUploader button {
        background: #2ecc71 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }
    [data-testid="stSidebar"] .stFileUploader label {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        border-radius: 8px;
    }
    .main-header {
        background: linear-gradient(135deg, #1a3c5e 0%, #2ecc71 100%);
        padding: 30px 40px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 8px 30px rgba(26, 60, 94, 0.3);
    }
    .auth-container {
        max-width: 450px;
        margin: auto;
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    .auth-header {
        text-align: center;
        margin-bottom: 30px;
    }
    .auth-header h1 {
        color: #1a3c5e;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .auth-header p {
        color: #666;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── User Database ─────────────────────────────────────────────────────────────
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, email, password, name):
    users = load_users()
    if username in users:
        return False, "Username already exists!"
    if any(u['email'] == email for u in users.values()):
        return False, "Email already registered!"
    users[username] = {
        "name": name,
        "email": email,
        "password": hash_password(password)
    }
    save_users(users)
    return True, "Account created successfully!"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "Username not found!"
    if verify_password(password, users[username]['password']):
        return True, users[username]['name']
    return False, "Incorrect password!"

# ─── Session State ─────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'name' not in st.session_state:
    st.session_state.name = ""
if 'auth_page' not in st.session_state:
    st.session_state.auth_page = "login"

# ─── Auth Pages ────────────────────────────────────────────────────────────────
def show_login():
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px 0;'>
        <h1 style='color:#1a3c5e; font-size:2.5rem; font-weight:700;'>🌿 TripleLens</h1>
        <p style='color:#666; font-size:1.1rem;'>ESG Risk Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='background:white; border-radius:20px; padding:40px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);'>
            <h2 style='color:#1a3c5e; text-align:center; margin-bottom:5px;'>Welcome Back!</h2>
            <p style='color:#666; text-align:center; margin-bottom:25px;'>Login to your account</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔐 Login", use_container_width=True):
                if username and password:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.name = result
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.warning("⚠️ Please fill in all fields!")

        with col_b:
            if st.button("📝 Register", use_container_width=True):
                st.session_state.auth_page = "register"
                st.rerun()

def show_register():
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px 0;'>
        <h1 style='color:#1a3c5e; font-size:2.5rem; font-weight:700;'>🌿 TripleLens</h1>
        <p style='color:#666; font-size:1.1rem;'>ESG Risk Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='background:white; border-radius:20px; padding:40px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);'>
            <h2 style='color:#1a3c5e; text-align:center; margin-bottom:5px;'>Create Account</h2>
            <p style='color:#666; text-align:center; margin-bottom:25px;'>Join TripleLens today</p>
        </div>
        """, unsafe_allow_html=True)

        name = st.text_input("👤 Full Name", placeholder="Enter your full name")
        email = st.text_input("📧 Email", placeholder="Enter your email")
        username = st.text_input("🆔 Username", placeholder="Choose a username")
        password = st.text_input("🔒 Password", type="password", placeholder="Choose a password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Create Account", use_container_width=True):
                if name and email and username and password and confirm_password:
                    if password != confirm_password:
                        st.error("❌ Passwords do not match!")
                    elif len(password) < 6:
                        st.error("❌ Password must be at least 6 characters!")
                    else:
                        success, message = register_user(username, email, password, name)
                        if success:
                            st.success(f"✅ {message} Please login.")
                            st.session_state.auth_page = "login"
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please fill in all fields!")
        with col_b:
            if st.button("🔙 Back to Login", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()

# ─── Show Auth or Main App ─────────────────────────────────────────────────────
if not st.session_state.logged_in:
    if st.session_state.auth_page == "login":
        show_login()
    else:
        show_register()
    st.stop()

# ─── Sample Dataset ────────────────────────────────────────────────────────────
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

# ─── Preprocessing ─────────────────────────────────────────────────────────────
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

# ─── ESG Scoring ───────────────────────────────────────────────────────────────
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

# ─── PDF Generation ────────────────────────────────────────────────────────────
def generate_pdf(df, filtered_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(26, 60, 94)
    pdf.rect(0, 0, 210, 60, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_y(15)
    pdf.cell(0, 10, "TripleLens - ESG Risk Analysis Platform", ln=True, align='C')
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
    pdf.cell(0, 10, "Generated by TripleLens - ESG Risk Analysis Platform | Confidential Report", align='C')
    return bytes(pdf.output())

# ─── ML Training ───────────────────────────────────────────────────────────────
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

# ─── Sidebar ───────────────────────────────────────────────────────────────────
import os
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esg_logo.png")
if os.path.exists(logo_path):
    try:
        st.sidebar.image(logo_path, use_container_width=True)
    except:
        st.sidebar.markdown("### 🌿 TripleLens")
else:
    st.sidebar.markdown("### 🌿 TripleLens")

st.sidebar.title("TripleLens")
st.sidebar.markdown(f"👤 Welcome, **{st.session_state.name}**!")
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

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.name = ""
    st.rerun()

st.sidebar.info("Upload your Kaggle ESG dataset or use sample data to get started.")

# ─── Load & Process Data ───────────────────────────────────────────────────────
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

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1 style='margin:0; font-size:2rem; font-weight:700;'>🌿 TripleLens</h1>
    <p style='margin:5px 0 0 0; opacity:0.85; font-size:1.1rem;'>ESG Risk Analysis Platform — Insight. Impact. Integrity.</p>
</div>
""", unsafe_allow_html=True)

if len(risk_filter) < 3:
    st.info(f"🔍 Filtering by: **{', '.join(risk_filter)}** | ESG Score: **{score_min} – {score_max}**")

st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "🔬 Data Preview",
    "🤖 ML Predictions",
    "📈 Model Evaluation",
    "🏢 Company Insights",
    "📄 Export Report"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if filtered_df.empty:
        st.warning("⚠️ No companies match the current filters.")
    else:
        RISK_COLORS = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 1 — KPI SUMMARY CARDS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #2ecc71;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>📊 Key Performance Indicators</span>
        </div>""", unsafe_allow_html=True)

        avg_esg = round(filtered_df['esg_score'].mean(), 1)
        low_count  = len(filtered_df[filtered_df['risk_label'] == 'Low Risk'])
        med_count  = len(filtered_df[filtered_df['risk_label'] == 'Medium Risk'])
        high_count = len(filtered_df[filtered_df['risk_label'] == 'High Risk'])

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a3c5e,#2980b9);
                        border-radius:14px;padding:20px;text-align:center;
                        box-shadow:0 4px 15px rgba(0,0,0,0.1);'>
                <p style='color:rgba(255,255,255,0.8);margin:0;font-size:0.85rem;'>Total Companies</p>
                <h2 style='color:white;margin:5px 0;font-size:2rem;'>{len(filtered_df)}</h2>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#2ecc71,#27ae60);
                        border-radius:14px;padding:20px;text-align:center;
                        box-shadow:0 4px 15px rgba(0,0,0,0.1);'>
                <p style='color:rgba(255,255,255,0.8);margin:0;font-size:0.85rem;'>✅ Low Risk</p>
                <h2 style='color:white;margin:5px 0;font-size:2rem;'>{low_count}</h2>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#f39c12,#e67e22);
                        border-radius:14px;padding:20px;text-align:center;
                        box-shadow:0 4px 15px rgba(0,0,0,0.1);'>
                <p style='color:rgba(255,255,255,0.8);margin:0;font-size:0.85rem;'>⚠️ Medium Risk</p>
                <h2 style='color:white;margin:5px 0;font-size:2rem;'>{med_count}</h2>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#e74c3c,#c0392b);
                        border-radius:14px;padding:20px;text-align:center;
                        box-shadow:0 4px 15px rgba(0,0,0,0.1);'>
                <p style='color:rgba(255,255,255,0.8);margin:0;font-size:0.85rem;'>🚨 High Risk</p>
                <h2 style='color:white;margin:5px 0;font-size:2rem;'>{high_count}</h2>
            </div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#9b59b6,#8e44ad);
                        border-radius:14px;padding:20px;text-align:center;
                        box-shadow:0 4px 15px rgba(0,0,0,0.1);'>
                <p style='color:rgba(255,255,255,0.8);margin:0;font-size:0.85rem;'>📊 Avg ESG Score</p>
                <h2 style='color:white;margin:5px 0;font-size:2rem;'>{avg_esg}</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 2 — RISK OVERVIEW
        # Left : Donut — what % of companies fall in each risk tier
        # Right: Grouped bar — avg E / S / G score per risk tier
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='border-left:4px solid #2980b9;padding:4px 0 4px 14px;margin-bottom:12px;'>
                <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>🍩 Risk Tier Share</span>
            </div>""", unsafe_allow_html=True)
            fig_pie = px.pie(
                filtered_df, names='risk_label',
                color='risk_label', hole=0.5,
                color_discrete_map=RISK_COLORS
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                margin=dict(t=20, b=60)
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("""
            <div style='border-left:4px solid #2980b9;padding:4px 0 4px 14px;margin-bottom:12px;'>
                <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>📊 Avg E / S / G by Risk Tier</span>
            </div>""", unsafe_allow_html=True)
            avg_by_risk = filtered_df.groupby('risk_label')[
                ['environmental_score', 'social_score', 'governance_score']
            ].mean().reset_index()
            avg_by_risk_melted = avg_by_risk.melt(
                id_vars='risk_label', var_name='Category', value_name='Avg Score'
            )
            avg_by_risk_melted['Category'] = avg_by_risk_melted['Category'].map({
                'environmental_score': '🌱 Environmental',
                'social_score': '🤝 Social',
                'governance_score': '🏛️ Governance'
            })
            fig_grouped = px.bar(
                avg_by_risk_melted,
                x='risk_label', y='Avg Score', color='Category',
                barmode='group',
                color_discrete_sequence=['#27ae60', '#2980b9', '#8e44ad'],
                text_auto='.1f'
            )
            fig_grouped.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Risk Level', yaxis_title='Average Score (0–100)',
                legend_title='ESG Category', title_font_size=15,
                margin=dict(t=20),
                yaxis=dict(range=[0, 110], gridcolor='rgba(0,0,0,0.05)')
            )
            fig_grouped.update_traces(textposition='outside')
            st.plotly_chart(fig_grouped, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>", unsafe_allow_html=True)


        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 3 — COMPANY RANKINGS
        # Left : Horizontal bar — top 15 companies ranked by ESG score
        # Right: Box plot — ESG score spread within each risk tier
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #f39c12;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>🏆 Company Rankings & Score Spread</span>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            top15 = filtered_df.nlargest(15, 'esg_score').sort_values('esg_score')
            fig_hbar = px.bar(
                top15, y='company', x='esg_score',
                orientation='h', color='risk_label',
                title='Top 15 Companies by ESG Score',
                color_discrete_map=RISK_COLORS,
                text='esg_score'
            )
            fig_hbar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='ESG Score (0–100)', yaxis_title='',
                legend_title='Risk Level', title_font_size=15,
                xaxis=dict(range=[0, 110], gridcolor='rgba(0,0,0,0.05)'),
                margin=dict(l=10)
            )
            fig_hbar.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_hbar, use_container_width=True)

        with col2:
            fig_box = px.box(
                filtered_df, x='risk_label', y='esg_score',
                color='risk_label',
                title='ESG Score Spread per Risk Tier',
                color_discrete_map=RISK_COLORS,
                points='all',
                hover_name='company'
            )
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Risk Level', yaxis_title='ESG Score (0–100)',
                showlegend=False, title_font_size=15,
                yaxis=dict(range=[0, 110], gridcolor='rgba(0,0,0,0.05)')
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>", unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 4 — ESG PILLAR DEEP-DIVE
        # Left : Scatter — Environmental vs Social (size = ESG score)
        # Right: Grouped bar — avg E / S / G per company (top 10)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #9b59b6;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>🔬 ESG Pillar Deep-Dive</span>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = px.scatter(
                filtered_df,
                x='environmental_score', y='social_score',
                size='esg_score', color='risk_label',
                hover_name='company',
                hover_data={'governance_score': True, 'esg_score': True,
                            'environmental_score': False, 'social_score': False},
                title='Environmental vs Social Score<br><sup>Bubble size = overall ESG score</sup>',
                color_discrete_map=RISK_COLORS,
                labels={'environmental_score': 'Environmental Score',
                        'social_score': 'Social Score'}
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=15, legend_title='Risk Level',
                xaxis=dict(range=[0, 110], gridcolor='rgba(0,0,0,0.05)',
                           title='Environmental Score (0–100)'),
                yaxis=dict(range=[0, 110], gridcolor='rgba(0,0,0,0.05)',
                           title='Social Score (0–100)')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            top10_companies = filtered_df.nlargest(10, 'esg_score')
            pillar_df = top10_companies[
                ['company', 'environmental_score', 'social_score', 'governance_score']
            ].melt(id_vars='company', var_name='Pillar', value_name='Score')
            pillar_df['Pillar'] = pillar_df['Pillar'].map({
                'environmental_score': '🌱 Environmental',
                'social_score': '🤝 Social',
                'governance_score': '🏛️ Governance'
            })
            fig_pillar = px.bar(
                pillar_df, x='company', y='Score', color='Pillar',
                barmode='group',
                title='E / S / G Pillar Scores — Top 10 Companies',
                color_discrete_sequence=['#27ae60', '#2980b9', '#8e44ad']
            )
            fig_pillar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-35, xaxis_title='', yaxis_title='Score (0–100)',
                legend_title='ESG Pillar', title_font_size=15,
                yaxis=dict(range=[0, 115], gridcolor='rgba(0,0,0,0.05)')
            )
            st.plotly_chart(fig_pillar, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>", unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 5 — PILLAR SCORE DISTRIBUTIONS
        # Three histograms with distinct axis labels and descriptions
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #27ae60;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>📈 Pillar Score Distributions</span>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        shared_layout = dict(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Number of Companies',
            legend_title='Risk Level', title_font_size=14,
            yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
        )

        with col1:
            fig_env = px.histogram(
                filtered_df, x='environmental_score', color='risk_label',
                title='🌱 Environmental Score Distribution',
                labels={'environmental_score': 'Environmental Score (0–100)', 'count': 'Companies'},
                color_discrete_map=RISK_COLORS, nbins=20, barmode='stack'
            )
            fig_env.update_layout(**shared_layout,
                xaxis_title='Environmental Score (0–100)')
            st.plotly_chart(fig_env, use_container_width=True)

        with col2:
            fig_soc = px.histogram(
                filtered_df, x='social_score', color='risk_label',
                title='🤝 Social Score Distribution',
                labels={'social_score': 'Social Score (0–100)', 'count': 'Companies'},
                color_discrete_map=RISK_COLORS, nbins=20, barmode='stack'
            )
            fig_soc.update_layout(**shared_layout,
                xaxis_title='Social Score (0–100)')
            st.plotly_chart(fig_soc, use_container_width=True)

        with col3:
            fig_gov = px.histogram(
                filtered_df, x='governance_score', color='risk_label',
                title='🏛️ Governance Score Distribution',
                labels={'governance_score': 'Governance Score (0–100)', 'count': 'Companies'},
                color_discrete_map=RISK_COLORS, nbins=20, barmode='stack'
            )
            fig_gov.update_layout(**shared_layout,
                xaxis_title='Governance Score (0–100)')
            st.plotly_chart(fig_gov, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>", unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 6 — TOP 10 SAFEST INVESTMENTS TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #e74c3c;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>🏅 Top 10 Safest Companies to Invest</span>
        </div>""", unsafe_allow_html=True)

        top10 = filtered_df.nlargest(10, 'esg_score')[
            ['company', 'environmental_score', 'social_score', 'governance_score', 'esg_score', 'risk_label']
        ].reset_index(drop=True)
        top10.index += 1
        st.dataframe(
            top10.style.background_gradient(subset=['esg_score'], cmap='Greens'),
            use_container_width=True
        )
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
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

    st.markdown("### Model Performance Overview")

    cm = np.array(r['confusion_matrix'])

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total != 0 else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Correct Predictions", int(correct))
    col2.metric("Total Samples", int(total))
    col3.metric("Accuracy", f"{accuracy:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPANY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPORT REPORT
# ══════════════════════════════════════════════════════════════════════════════
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