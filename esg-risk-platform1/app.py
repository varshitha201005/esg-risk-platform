import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.inspection import permutation_importance
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import warnings
import bcrypt
import json
import os
import io
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import os
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esg_logo.png")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TripleLens - ESG Risk Analysis Platform",
    page_icon="🌿",
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
        "password": hash_password(password),
        "created_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "last_login": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "analyses_run": 0
    }
    save_users(users)
    return True, "Account created successfully!"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "Username not found!"
    if verify_password(password, users[username]['password']):
        users[username]['last_login'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        users[username].setdefault('analyses_run', 0)
        save_users(users)
        return True, users[username]['name']
    return False, "Incorrect password!"

# ─── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ('logged_in', False), ('username', ''), ('name', ''),
    ('auth_page', 'login'), ('pred_env', 50),
    ('pred_soc', 50), ('pred_gov', 50),
    ('risk_low', 45), ('risk_high', 70)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Auth Pages ────────────────────────────────────────────────────────────────
def password_strength(password):
    score = 0
    if len(password) >= 6:  score += 1
    if len(password) >= 10: score += 1
    if any(c.isupper() for c in password): score += 1
    if any(c.isdigit() for c in password): score += 1
    if any(c in '!@#$%^&*' for c in password): score += 1
    labels = ['', 'Very Weak', 'Weak', 'Fair', 'Strong', 'Very Strong']
    colors = ['', '#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    return score, labels[score], colors[score]

def show_login():
    st.markdown("""
    <div style='text-align:center;padding:40px 0 20px 0;'>
        <h1 style='color:#1a3c5e;font-size:2.5rem;font-weight:700;'>🌿 TripleLens</h1>
        <p style='color:#666;font-size:1.1rem;'>ESG Risk Analysis Platform</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='background:white;border-radius:20px;padding:40px;
                    box-shadow:0 10px 40px rgba(0,0,0,0.1);'>
            <h2 style='color:#1a3c5e;text-align:center;margin-bottom:5px;'>Welcome Back!</h2>
            <p style='color:#666;text-align:center;margin-bottom:25px;'>Login to your account</p>
        </div>""", unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔐 Login", use_container_width=True):
                if username and password:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username  = username
                        st.session_state.name      = result
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
    <div style='text-align:center;padding:40px 0 20px 0;'>
        <h1 style='color:#1a3c5e;font-size:2.5rem;font-weight:700;'>🌿 TripleLens</h1>
        <p style='color:#666;font-size:1.1rem;'>ESG Risk Analysis Platform</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='background:white;border-radius:20px;padding:40px;
                    box-shadow:0 10px 40px rgba(0,0,0,0.1);'>
            <h2 style='color:#1a3c5e;text-align:center;margin-bottom:5px;'>Create Account</h2>
            <p style='color:#666;text-align:center;margin-bottom:25px;'>Join TripleLens today</p>
        </div>""", unsafe_allow_html=True)

        name     = st.text_input("👤 Full Name",         placeholder="Enter your full name")
        email    = st.text_input("📧 Email",              placeholder="Enter your email")
        username = st.text_input("🆔 Username",           placeholder="Choose a username")
        password = st.text_input("🔒 Password",  type="password", placeholder="Choose a password")
        confirm  = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")

        # password strength indicator
        if password:
            score, label, color = password_strength(password)
            bar_width = score * 20
            st.markdown(f"""
            <div style='margin:-8px 0 8px 0;'>
                <div style='background:#f0f0f0;border-radius:6px;height:6px;'>
                    <div style='background:{color};width:{bar_width}%;height:6px;
                                border-radius:6px;transition:width 0.3s;'></div>
                </div>
                <span style='font-size:0.75rem;color:{color};font-weight:600;'>
                    {label}
                </span>
            </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Create Account", use_container_width=True):
                if name and email and username and password and confirm:
                    if password != confirm:
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

# ─── Auth Gate ─────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    if st.session_state.auth_page == "login":
        show_login()
    else:
        show_register()
    st.stop()

# ─── Constants ─────────────────────────────────────────────────────────────────
RISK_COLORS = {
    "Low Risk":    "#2ecc71",
    "Medium Risk": "#f39c12",
    "High Risk":   "#e74c3c"
}
RISK_RGBA = {
    "Low Risk":    "rgba(46,204,113,0.13)",
    "Medium Risk": "rgba(243,156,18,0.13)",
    "High Risk":   "rgba(231,76,60,0.13)"
}
SECTOR_MAP = {
    "Tesla": "Automotive", "Apple": "Technology", "Microsoft": "Technology",
    "ExxonMobil": "Energy", "Coal India": "Energy", "Infosys": "Technology",
    "Reliance": "Energy", "Google": "Technology", "Amazon": "Retail",
    "BP": "Energy", "Wipro": "Technology", "TCS": "Technology",
    "Shell": "Energy", "HDFC": "Finance", "Nestle": "Consumer Goods",
    "Unilever": "Consumer Goods", "Toyota": "Automotive",
    "Samsung": "Technology", "HSBC": "Finance", "Goldman Sachs": "Finance"
}

# ─── Sample Dataset ────────────────────────────────────────────────────────────
def get_sample_data():
    np.random.seed(42)
    companies = list(SECTOR_MAP.keys())
    data = {
        "company":             companies,
        "sector":              list(SECTOR_MAP.values()),
        "carbon_emissions":    np.random.randint(10, 100, 20),
        "energy_usage":        np.random.randint(20, 100, 20),
        "environmental_score": np.random.randint(20, 100, 20),
        "employee_turnover":   np.random.randint(5,  50,  20),
        "diversity_ratio":     np.random.randint(20, 80,  20),
        "social_score":        np.random.randint(20, 100, 20),
        "board_structure":     np.random.randint(30, 100, 20),
        "compliance_score":    np.random.randint(30, 100, 20),
        "governance_score":    np.random.randint(20, 100, 20),
    }
    # simulate 6-month score history
    for i in range(1, 7):
        base_e = np.array(data['environmental_score'], dtype=float)
        base_s = np.array(data['social_score'],        dtype=float)
        base_g = np.array(data['governance_score'],    dtype=float)
        noise  = np.random.randint(-8, 8, 20)
        data[f'env_m{i}']  = np.clip(base_e + noise,              0, 100).astype(int)
        data[f'soc_m{i}']  = np.clip(base_s + np.random.randint(-8, 8, 20), 0, 100).astype(int)
        data[f'gov_m{i}']  = np.clip(base_g + np.random.randint(-8, 8, 20), 0, 100).astype(int)
    return pd.DataFrame(data)

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_data(df):
    df = df.copy()
    df = df.rename(columns={'name': 'company', 'environment_score': 'environmental_score'})
    if 'company' not in df.columns and 'ticker' in df.columns:
        df['company'] = df['ticker']
    elif 'company' not in df.columns:
        df['company'] = [f"Company {i+1}" for i in range(len(df))]
    if 'sector' not in df.columns:
        df['sector'] = df['company'].map(SECTOR_MAP).fillna('Other')
    for col in ['environmental_score', 'social_score', 'governance_score']:
        if col not in df.columns:
            df[col] = 50
    for col, lo, hi in [
        ('carbon_emissions', 10, 100), ('energy_usage', 20, 100),
        ('employee_turnover', 5, 50),  ('diversity_ratio', 20, 80),
        ('board_structure', 30, 100),  ('compliance_score', 30, 100)
    ]:
        if col not in df.columns:
            df[col] = np.random.randint(lo, hi, len(df))
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    for col in ['environmental_score', 'social_score', 'governance_score']:
        if df[col].max() > 100:
            df[col] = (df[col] / df[col].max()) * 100
    # add history columns if missing
    for i in range(1, 7):
        for prefix, base in [('env', 'environmental_score'),
                              ('soc', 'social_score'),
                              ('gov', 'governance_score')]:
            col = f'{prefix}_m{i}'
            if col not in df.columns:
                noise = np.random.randint(-8, 8, len(df))
                df[col] = np.clip(df[base] + noise, 0, 100).astype(int)
    keep_cols = [
        'company', 'sector',
        'environmental_score', 'social_score', 'governance_score',
        'carbon_emissions', 'energy_usage', 'employee_turnover',
        'diversity_ratio', 'board_structure', 'compliance_score',
    ] + [f'{p}_m{i}' for p in ['env','soc','gov'] for i in range(1,7)]
    df = df[[c for c in keep_cols if c in df.columns]]
    return df

# ─── ESG Scoring ───────────────────────────────────────────────────────────────
def calculate_esg_score(row, w_env=0.4, w_soc=0.3, w_gov=0.3):
    env = row['environmental_score']
    soc = row['social_score']
    gov = row['governance_score']
    mx  = max(env, soc, gov)
    if mx > 100:
        env, soc, gov = (env/mx)*100, (soc/mx)*100, (gov/mx)*100
    return min(round((env*w_env)+(soc*w_soc)+(gov*w_gov), 1), 100)

def get_risk_label(score, low_thresh=45, high_thresh=70):
    if score >= high_thresh: return "Low Risk"
    elif score >= low_thresh: return "Medium Risk"
    else: return "High Risk"

def get_risk_color(risk):
    return RISK_COLORS.get(risk, "#ccc")

# ─── ML Training (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_models(df_json):
    df = pd.read_json(io.StringIO(df_json))
    features = ['environmental_score', 'social_score', 'governance_score']
    X = df[features]
    y = df['risk_label'].map({"Low Risk": 0, "Medium Risk": 1, "High Risk": 2})
    if y.nunique() < 2:
        return None, None
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    models_def = {
        "Logistic Regression":  LogisticRegression(max_iter=1000),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    for name, model in models_def.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # cross-val
        cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(df)//2),
                                    scoring='f1_weighted')

        # per-class metrics
        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[0,1,2], zero_division=0
        )

        # feature importance
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
        else:
            fi = np.array([1/3, 1/3, 1/3])

        results[name] = {
            "model":            model,
            "accuracy":         accuracy_score(y_test, y_pred),
            "precision":        precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall":           recall_score(y_test, y_pred,    average='weighted', zero_division=0),
            "f1":               f1_score(y_test, y_pred,        average='weighted', zero_division=0),
            "cv_mean":          cv_scores.mean(),
            "cv_std":           cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "per_class_p":      p, "per_class_r": r, "per_class_f": f,
            "feature_importance": fi,
            "y_test":           y_test, "y_pred": y_pred
        }
    return results, scaler

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
    pdf.cell(0, 10, f"Generated: {pd.Timestamp.now().strftime('%B %d, %Y')}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(75)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(26, 60, 94)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_draw_color(46, 204, 113)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    total    = len(filtered_df)
    low      = len(filtered_df[filtered_df['risk_label'] == 'Low Risk'])
    med      = len(filtered_df[filtered_df['risk_label'] == 'Medium Risk'])
    high     = len(filtered_df[filtered_df['risk_label'] == 'High Risk'])
    avg_score = round(filtered_df['esg_score'].mean(), 1)
    best_co  = filtered_df.loc[filtered_df['esg_score'].idxmax(), 'company']
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    for label, value in [
        ("Total Companies Analyzed", str(total)),
        ("Average ESG Score",        str(avg_score)),
        ("Best ESG Company",         str(best_co)),
        ("Safe Investments (Low Risk)", str(low)),
        ("Moderate Risk Companies",  str(med)),
        ("High Risk Companies",      str(high)),
    ]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(100, 8, label + ":", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, value, ln=True)
    pdf.ln(5)
    # risk distribution table
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(26, 60, 94)
    pdf.cell(0, 10, "Risk Distribution", ln=True)
    pdf.set_draw_color(46, 204, 113)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    risk_colors_pdf = {
        "Low Risk":    (46,  204, 113),
        "Medium Risk": (243, 156, 18),
        "High Risk":   (231, 76,  60)
    }
    pdf.set_fill_color(26, 60, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    for hdr, w in [("Risk Level",60),("Companies",40),("Percentage",40),("Avg ESG",50)]:
        pdf.cell(w, 9, hdr, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font("Helvetica", "", 11)
    for risk, color in risk_colors_pdf.items():
        count = len(filtered_df[filtered_df['risk_label'] == risk])
        pct   = round(count / total * 100, 1) if total > 0 else 0
        avg   = round(filtered_df[filtered_df['risk_label'] == risk]['esg_score'].mean(), 1) if count > 0 else 0
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(60, 8, risk, border=1, fill=True, align='C')
        pdf.set_fill_color(245, 245, 245)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(40, 8, str(count),    border=1, fill=True, align='C')
        pdf.cell(40, 8, f"{pct}%",     border=1, fill=True, align='C')
        pdf.cell(50, 8, str(avg),      border=1, fill=True, align='C')
        pdf.ln()
    # company table
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
    for hdr, w in [("Company",50),("Env",25),("Soc",25),("Gov",25),("ESG",25),("Risk",40)]:
        pdf.cell(w, 9, hdr, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i, (_, row) in enumerate(
        filtered_df.sort_values('esg_score', ascending=False).iterrows()
    ):
        pdf.set_fill_color(245,245,245) if i%2==0 else pdf.set_fill_color(255,255,255)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(50, 8, str(row['company'])[:20], border=1, fill=True)
        pdf.cell(25, 8, str(round(row['environmental_score'],1)), border=1, fill=True, align='C')
        pdf.cell(25, 8, str(round(row['social_score'],1)),        border=1, fill=True, align='C')
        pdf.cell(25, 8, str(round(row['governance_score'],1)),    border=1, fill=True, align='C')
        pdf.cell(25, 8, str(row['esg_score']),                    border=1, fill=True, align='C')
        pdf.set_fill_color(*risk_colors_pdf.get(row['risk_label'], (128,128,128)))
        pdf.set_text_color(255, 255, 255)
        pdf.cell(40, 8, row['risk_label'], border=1, fill=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln()
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, "Generated by TripleLens — ESG Risk Analysis Platform | Confidential", align='C')
    return bytes(pdf.output())

# ─── Sidebar ───────────────────────────────────────────────────────────────────
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esg_logo.png")
if os.path.exists(logo_path):
    import os
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esg_logo.png")
try:
    st.sidebar.image(logo_path, use_container_width=True)
except:
    st.sidebar.markdown("### 🌿 TripleLens")
else:
    st.sidebar.markdown("### 🌿 TripleLens")

st.sidebar.markdown(f"👤 Welcome, **{st.session_state.name}**!")
st.sidebar.markdown("---")

# ── Data Upload ──
st.sidebar.markdown("**📁 Data**")
uploaded_file = st.sidebar.file_uploader("Upload ESG Dataset (CSV)", type=["csv"])
st.sidebar.markdown("---")

# ── Model ──
st.sidebar.markdown("**🤖 Model**")
selected_model = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Logistic Regression", "Gradient Boosting"]
)
st.sidebar.markdown("---")

# ── Filters ──
st.sidebar.markdown("**🔍 Filters**")
risk_filter = st.sidebar.multiselect(
    "Filter by Risk Level",
    ["Low Risk", "Medium Risk", "High Risk"],
    default=["Low Risk", "Medium Risk", "High Risk"]
)
score_min, score_max = st.sidebar.slider("ESG Score Range", 0, 100, (0, 100))
st.sidebar.markdown("---")

# ── Risk Thresholds ──
st.sidebar.markdown("**⚙️ Risk Thresholds**")
high_thresh = st.sidebar.slider(
    "Low Risk threshold (≥)", 50, 90,
    st.session_state.risk_high, step=5
)
low_thresh = st.sidebar.slider(
    "High Risk threshold (<)", 20, 60,
    st.session_state.risk_low, step=5
)
st.session_state.risk_high = high_thresh
st.session_state.risk_low  = low_thresh
st.sidebar.markdown("---")

# ── Logout ──
if st.sidebar.button("🚪 Logout"):
    for k in ['logged_in','username','name']:
        st.session_state[k] = '' if k != 'logged_in' else False
    st.rerun()

st.sidebar.info("Upload a Kaggle ESG dataset or use sample data to get started.")

# ─── Load & Process Data ───────────────────────────────────────────────────────
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    df     = preprocess_data(raw_df)
    st.sidebar.success(f"✅ Loaded {len(df)} companies")
else:
    df = preprocess_data(get_sample_data())

df['esg_score']  = df.apply(
    lambda r: calculate_esg_score(r), axis=1
)
df['risk_label'] = df['esg_score'].apply(
    lambda s: get_risk_label(s, low_thresh, high_thresh)
)

filtered_df = df[
    (df['risk_label'].isin(risk_filter)) &
    (df['esg_score'] >= score_min) &
    (df['esg_score'] <= score_max)
].copy()

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1 style='margin:0;font-size:2rem;font-weight:700;'>🌿 TripleLens</h1>
    <p style='margin:5px 0 0 0;opacity:0.85;font-size:1.1rem;'>
        ESG Risk Analysis Platform — Insight. Impact. Integrity.
    </p>
</div>""", unsafe_allow_html=True)

if len(risk_filter) < 3 or score_min > 0 or score_max < 100:
    st.info(f"🔍 Filters active — Risk: **{', '.join(risk_filter) if risk_filter else 'None'}** | "
            f"ESG Score: **{score_min}–{score_max}** | "
            f"Thresholds: Low≥{high_thresh} / High<{low_thresh}")

st.markdown("---")

# ─── Empty state guard ─────────────────────────────────────────────────────────
def empty_state():
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;background:white;
                border-radius:16px;box-shadow:0 2px 10px rgba(0,0,0,0.06);'>
        <h2 style='color:#1a3c5e;'>🔍 No companies match your filters</h2>
        <p style='color:#888;'>Try widening your ESG score range or selecting more risk levels
        in the sidebar.</p>
    </div>""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "🔬 Data Preview", "🤖 ML Predictions",
    "📈 Model Evaluation", "🏢 Company Insights", "📄 Export Report"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if filtered_df.empty:
        empty_state()
    else:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 1 — KPI CARDS + BEST COMPANY SPOTLIGHT
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #2ecc71;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
                📊 Key Performance Indicators
            </span>
        </div>""", unsafe_allow_html=True)

        avg_esg    = round(filtered_df['esg_score'].mean(), 1)
        low_count  = len(filtered_df[filtered_df['risk_label'] == 'Low Risk'])
        med_count  = len(filtered_df[filtered_df['risk_label'] == 'Medium Risk'])
        high_count = len(filtered_df[filtered_df['risk_label'] == 'High Risk'])
        best_row   = filtered_df.loc[filtered_df['esg_score'].idxmax()]

        col1, col2, col3, col4, col5 = st.columns(5)
        kpi_cards = [
            (col1, "#1a3c5e,#2980b9", "Total Companies", len(filtered_df)),
            (col2, "#2ecc71,#27ae60", "✅ Low Risk",      low_count),
            (col3, "#f39c12,#e67e22", "⚠️ Medium Risk",  med_count),
            (col4, "#e74c3c,#c0392b", "🚨 High Risk",     high_count),
            (col5, "#9b59b6,#8e44ad", "📊 Avg ESG Score", avg_esg),
        ]
        for col, grad, label, val in kpi_cards:
            with col:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#{grad.split(",")[0] if "#" not in grad else grad});
                            background:linear-gradient(135deg,{grad.replace(",",",")} );
                            border-radius:14px;padding:20px;text-align:center;
                            box-shadow:0 4px 15px rgba(0,0,0,0.1);'>
                    <p style='color:rgba(255,255,255,0.8);margin:0;font-size:0.85rem;'>{label}</p>
                    <h2 style='color:white;margin:5px 0;font-size:2rem;'>{val}</h2>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Best company spotlight
        best_color = get_risk_color(best_row['risk_label'])
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#eafaf1,#d5f5e3);
                    border-radius:14px;padding:18px 24px;
                    border-left:5px solid #2ecc71;margin-bottom:20px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.06);
                    display:flex;align-items:center;gap:20px;flex-wrap:wrap;'>
            <span style='font-size:1.8rem;'>🏆</span>
            <div>
                <p style='color:#888;font-size:0.8rem;margin:0;'>Top ESG Company</p>
                <h2 style='color:#1a3c5e;margin:2px 0;'>{best_row['company']}</h2>
                <span style='background:{best_color};color:white;border-radius:20px;
                             padding:3px 12px;font-size:0.8rem;font-weight:600;'>
                    {best_row['risk_label']}
                </span>
            </div>
            <div style='margin-left:auto;text-align:right;'>
                <p style='color:#888;font-size:0.8rem;margin:0;'>ESG Score</p>
                <h1 style='color:#2ecc71;margin:0;font-size:2.8rem;font-weight:700;'>
                    {best_row['esg_score']}
                </h1>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                    unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 2 — COMPANY RANKINGS & SCORE SPREAD
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #f39c12;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
                🏆 Company Rankings & Score Spread
            </span>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            top15 = filtered_df.nlargest(15, 'esg_score').sort_values('esg_score')
            fig_hbar = px.bar(
                top15, y='company', x='esg_score', orientation='h',
                color='risk_label', title='Top 15 Companies by ESG Score',
                color_discrete_map=RISK_COLORS, text='esg_score'
            )
            fig_hbar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='ESG Score (0–100)', yaxis_title='',
                legend_title='Risk Level', title_font_size=15,
                xaxis=dict(range=[0,110], gridcolor='rgba(0,0,0,0.05)'),
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
                points='all', hover_name='company'
            )
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Risk Level', yaxis_title='ESG Score (0–100)',
                showlegend=False, title_font_size=15,
                yaxis=dict(range=[0,110], gridcolor='rgba(0,0,0,0.05)')
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                    unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 3 — ESG PILLAR DEEP-DIVE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #9b59b6;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
                🔬 ESG Pillar Deep-Dive
            </span>
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
                xaxis=dict(range=[0,110], gridcolor='rgba(0,0,0,0.05)',
                           title='Environmental Score (0–100)'),
                yaxis=dict(range=[0,110], gridcolor='rgba(0,0,0,0.05)',
                           title='Social Score (0–100)')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            top10_companies = filtered_df.nlargest(10, 'esg_score')
            pillar_df = top10_companies[
                ['company','environmental_score','social_score','governance_score']
            ].melt(id_vars='company', var_name='Pillar', value_name='Score')
            pillar_df['Pillar'] = pillar_df['Pillar'].map({
                'environmental_score': '🌱 Environmental',
                'social_score':        '🤝 Social',
                'governance_score':    '🏛️ Governance'
            })
            fig_pillar = px.bar(
                pillar_df, x='company', y='Score', color='Pillar',
                barmode='group',
                title='E / S / G Pillar Scores — Top 10 Companies',
                color_discrete_sequence=['#27ae60','#2980b9','#8e44ad']
            )
            fig_pillar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-35, xaxis_title='', yaxis_title='Score (0–100)',
                legend_title='ESG Pillar', title_font_size=15,
                yaxis=dict(range=[0,115], gridcolor='rgba(0,0,0,0.05)')
            )
            st.plotly_chart(fig_pillar, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                    unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 4 — RISK OVERVIEW
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #e67e22;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>⚠️ Risk Overview</span>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            risk_counts = (
                filtered_df['risk_label']
                .value_counts()
                .rename_axis('risk_label')
                .reset_index(name='count')
            )
            fig_donut = px.pie(
                risk_counts, names='risk_label', values='count', hole=0.55,
                title='Company Distribution by Risk Tier<br><sup>Share of companies in each risk category</sup>',
                color='risk_label', color_discrete_map=RISK_COLORS
            )
            fig_donut.update_traces(
                textposition='outside', textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Companies: %{value}<br>Share: %{percent}<extra></extra>'
            )
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=15, showlegend=True, legend_title='Risk Level',
                annotations=[dict(
                    text=f"<b>{len(filtered_df)}</b><br>Total",
                    x=0.5, y=0.5, font_size=16, showarrow=False
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            risk_pillar_df = (
                filtered_df
                .groupby('risk_label')[['environmental_score','social_score','governance_score']]
                .mean().round(1).reset_index()
                .melt(id_vars='risk_label', var_name='Pillar', value_name='Avg Score')
            )
            risk_pillar_df['Pillar'] = risk_pillar_df['Pillar'].map({
                'environmental_score': '🌱 Environmental',
                'social_score':        '🤝 Social',
                'governance_score':    '🏛️ Governance'
            })
            fig_risk_bar = px.bar(
                risk_pillar_df, x='risk_label', y='Avg Score', color='Pillar',
                barmode='group', title='Avg E / S / G Scores by Risk Tier',
                color_discrete_sequence=['#27ae60','#2980b9','#8e44ad'],
                labels={'risk_label':'Risk Tier','Avg Score':'Avg Score (0–100)'},
                text_auto='.1f'
            )
            fig_risk_bar.update_traces(textposition='outside', textfont_size=11)
            fig_risk_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Risk Tier', yaxis_title='Avg Score (0–100)',
                legend_title='ESG Pillar', title_font_size=15,
                yaxis=dict(range=[0,115], gridcolor='rgba(0,0,0,0.05)'),
                uniformtext_minsize=8, uniformtext_mode='hide'
            )
            st.plotly_chart(fig_risk_bar, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                    unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 5 — SECTOR BREAKDOWN (NEW)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if 'sector' in filtered_df.columns:
            st.markdown("""
            <div style='border-left:4px solid #16a085;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
                    🏭 Sector Analysis
                </span>
            </div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                sector_avg = (
                    filtered_df.groupby('sector')['esg_score']
                    .mean().round(1).reset_index()
                    .sort_values('esg_score', ascending=True)
                )
                fig_sector = px.bar(
                    sector_avg, y='sector', x='esg_score', orientation='h',
                    title='Average ESG Score by Sector',
                    text='esg_score',
                    color='esg_score',
                    color_continuous_scale='Greens'
                )
                fig_sector.update_traces(texttemplate='%{text}', textposition='outside')
                fig_sector.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(range=[0,115], gridcolor='rgba(0,0,0,0.05)',
                               title='Avg ESG Score'),
                    yaxis_title='', title_font_size=15,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_sector, use_container_width=True)

            with col2:
                sector_risk = (
                    filtered_df.groupby(['sector','risk_label'])
                    .size().reset_index(name='count')
                )
                fig_sector_risk = px.bar(
                    sector_risk, x='sector', y='count', color='risk_label',
                    barmode='stack',
                    title='Risk Distribution by Sector',
                    color_discrete_map=RISK_COLORS
                )
                fig_sector_risk.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_tickangle=-30, xaxis_title='',
                    yaxis_title='Number of Companies',
                    legend_title='Risk Level', title_font_size=15,
                    yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
                )
                st.plotly_chart(fig_sector_risk, use_container_width=True)

            st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                        unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 6 — PILLAR SCORE DISTRIBUTIONS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #27ae60;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
                📈 Pillar Score Distributions
            </span>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        shared_layout = dict(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Number of Companies', legend_title='Risk Level',
            title_font_size=14,
            yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
        )
        with col1:
            fig_env = px.histogram(
                filtered_df, x='environmental_score', color='risk_label',
                title='🌱 Environmental Score Distribution',
                color_discrete_map=RISK_COLORS, nbins=20, barmode='stack'
            )
            fig_env.update_layout(**shared_layout, xaxis_title='Environmental Score (0–100)')
            st.plotly_chart(fig_env, use_container_width=True)
        with col2:
            fig_soc = px.histogram(
                filtered_df, x='social_score', color='risk_label',
                title='🤝 Social Score Distribution',
                color_discrete_map=RISK_COLORS, nbins=20, barmode='stack'
            )
            fig_soc.update_layout(**shared_layout, xaxis_title='Social Score (0–100)')
            st.plotly_chart(fig_soc, use_container_width=True)
        with col3:
            fig_gov = px.histogram(
                filtered_df, x='governance_score', color='risk_label',
                title='🏛️ Governance Score Distribution',
                color_discrete_map=RISK_COLORS, nbins=20, barmode='stack'
            )
            fig_gov.update_layout(**shared_layout, xaxis_title='Governance Score (0–100)')
            st.plotly_chart(fig_gov, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                    unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION 7 — TOP 10 SAFEST INVESTMENTS TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #e74c3c;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
                🏅 Top 10 Safest Companies to Invest
            </span>
        </div>""", unsafe_allow_html=True)

        top10 = filtered_df.nlargest(10, 'esg_score')[
            ['company','sector','environmental_score','social_score',
             'governance_score','esg_score','risk_label']
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
    st.markdown("""
    <div style='border-left:4px solid #2980b9;padding:4px 0 4px 14px;margin-bottom:16px;'>
        <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
            🔬 Dataset Preview & Preprocessing
        </span>
    </div>""", unsafe_allow_html=True)

    if filtered_df.empty:
        empty_state()
    else:
        col1, col2, col3 = st.columns(3)
        with col1: st.info(f"📋 Total Rows: {df.shape[0]}")
        with col2: st.info(f"🔍 Filtered Rows: {filtered_df.shape[0]}")
        with col3: st.success(f"✅ Missing Values: {df.isnull().sum().sum()}")

        st.markdown("### 📋 Filtered Dataset")
        search = st.text_input("🔍 Search company", "")
        display_df = (
            filtered_df[filtered_df['company'].str.contains(search, case=False, na=False)]
            if search else filtered_df
        )
        base_cols = ['company','sector','environmental_score','social_score',
                     'governance_score','esg_score','risk_label']
        st.dataframe(
            display_df[base_cols].style.background_gradient(subset=['esg_score'], cmap='RdYlGn'),
            use_container_width=True
        )

        st.markdown("### 📊 Statistical Summary")
        stat_cols = ['environmental_score','social_score','governance_score','esg_score']
        st.dataframe(filtered_df[stat_cols].describe().round(2), use_container_width=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Column profiler (NEW)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("### 🧮 Column Profiler")
        profile_cols = ['environmental_score','social_score','governance_score',
                        'carbon_emissions','energy_usage','diversity_ratio']
        profile_cols = [c for c in profile_cols if c in filtered_df.columns]
        prof_data = []
        for c in profile_cols:
            s = filtered_df[c]
            prof_data.append({
                "Column": c, "Min": round(s.min(),1), "Max": round(s.max(),1),
                "Mean": round(s.mean(),1), "Std": round(s.std(),1),
                "Nulls": int(s.isnull().sum())
            })
        st.dataframe(pd.DataFrame(prof_data).set_index("Column"), use_container_width=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Plotly correlation heatmap (replaces seaborn)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("### 🔥 Correlation Heatmap")
        numeric_df = filtered_df[profile_cols + ['esg_score']].select_dtypes(include=np.number)
        corr = numeric_df.corr().round(2)
        fig_corr = px.imshow(
            corr, text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Feature Correlation Heatmap'
        )
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=15,
            height=450
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Outlier detection (NEW)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("### ⚠️ Outlier Detection")
        outlier_col = st.selectbox("Select column to check for outliers",
                                   ['environmental_score',
                                    'social_score','governance_score'])
        mean_v = filtered_df[outlier_col].mean()
        std_v  = filtered_df[outlier_col].std()
        outliers = filtered_df[
            (filtered_df[outlier_col] > mean_v + 2*std_v) |
            (filtered_df[outlier_col] < mean_v - 2*std_v)
        ][['company','sector', outlier_col,'esg_score','risk_label']]

        if outliers.empty:
            st.success(f"✅ No outliers detected in **{outlier_col}** (±2 std dev)")
        else:
            st.warning(f"⚠️ {len(outliers)} outlier(s) detected in **{outlier_col}**")
            st.dataframe(
                outliers.style.background_gradient(subset=[outlier_col], cmap='RdYlGn'),
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='border-left:4px solid #2ecc71;padding:4px 0 4px 14px;margin-bottom:16px;'>
        <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
            🤖 Machine Learning Risk Prediction
        </span>
    </div>""", unsafe_allow_html=True)

    if len(df) < 5:
        st.warning("⚠️ Need at least 5 companies to train models.")
    else:
        with st.spinner("Training ML models..."):
            results, scaler = train_models(df.to_json())

        if results is None:
            st.error("⚠️ Dataset has only one risk class. Please upload a more diverse dataset.")
        else:
            st.success(f"✅ Models trained on {len(df)} companies!")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SECTION A — MODEL PERFORMANCE CARDS + BAR
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("""
            <div style='border-left:4px solid #2980b9;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                    📊 Model Performance Comparison
                </span>
            </div>""", unsafe_allow_html=True)

            model_colors = ['#2ecc71','#e74c3c','#f39c12']
            cols = st.columns(len(results))
            for col, (name, res), color in zip(cols, results.items(), model_colors):
                with col:
                    st.markdown(f"""
                    <div style='background:white;border-radius:14px;padding:18px;
                                text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.07);
                                border-top:4px solid {color};'>
                        <p style='color:#1a3c5e;font-size:0.9rem;font-weight:700;
                                  margin:0 0 10px 0;'>{name}</p>
                        <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
                            <div style='background:#f8f9fa;border-radius:8px;padding:8px;'>
                                <p style='color:#888;font-size:0.7rem;margin:0;'>Accuracy</p>
                                <p style='color:{color};font-size:1.1rem;
                                          font-weight:700;margin:0;'>{res['accuracy']:.1%}</p>
                            </div>
                            <div style='background:#f8f9fa;border-radius:8px;padding:8px;'>
                                <p style='color:#888;font-size:0.7rem;margin:0;'>Precision</p>
                                <p style='color:{color};font-size:1.1rem;
                                          font-weight:700;margin:0;'>{res['precision']:.1%}</p>
                            </div>
                            <div style='background:#f8f9fa;border-radius:8px;padding:8px;'>
                                <p style='color:#888;font-size:0.7rem;margin:0;'>Recall</p>
                                <p style='color:{color};font-size:1.1rem;
                                          font-weight:700;margin:0;'>{res['recall']:.1%}</p>
                            </div>
                            <div style='background:#f8f9fa;border-radius:8px;padding:8px;'>
                                <p style='color:#888;font-size:0.7rem;margin:0;'>CV F1</p>
                                <p style='color:{color};font-size:1.1rem;
                                          font-weight:700;margin:0;'>{res['cv_mean']:.1%}
                                    <span style='font-size:0.7rem;'>±{res['cv_std']:.2f}</span>
                                </p>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            perf_df = pd.DataFrame([{
                "Model": name, "Accuracy": round(res['accuracy'],3),
                "Precision": round(res['precision'],3),
                "Recall": round(res['recall'],3), "F1 Score": round(res['f1'],3)
            } for name, res in results.items()])

            fig_perf = px.bar(
                perf_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric', barmode='group',
                text_auto='.1%',
                title='Model Performance — All Metrics',
                color_discrete_sequence=['#2ecc71','#2980b9','#9b59b6','#e67e22']
            )
            fig_perf.update_traces(textposition='outside', textfont_size=10)
            fig_perf.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0,1.15], tickformat='.0%',
                           gridcolor='rgba(0,0,0,0.05)', title='Score'),
                xaxis_title='', legend_title='Metric', title_font_size=15,
                uniformtext_minsize=8, uniformtext_mode='hide'
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                        unsafe_allow_html=True)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SECTION B — FEATURE IMPORTANCE (NEW)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("""
            <div style='border-left:4px solid #9b59b6;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                    🔍 Feature Importance — Which Pillar Drives Risk?
                </span>
            </div>""", unsafe_allow_html=True)

            fi_rows = []
            for name, res in results.items():
                for feat, imp in zip(
                    ['Environmental','Social','Governance'],
                    res['feature_importance']
                ):
                    fi_rows.append({"Model": name, "Pillar": feat,
                                    "Importance": round(float(imp), 4)})
            fi_df = pd.DataFrame(fi_rows)

            fig_fi = px.bar(
                fi_df, x='Pillar', y='Importance', color='Model',
                barmode='group', text_auto='.3f',
                title='Feature Importance by Pillar — All Models',
                color_discrete_sequence=['#2ecc71','#e74c3c','#f39c12']
            )
            fig_fi.update_traces(textposition='outside', textfont_size=11)
            fig_fi.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0,1.0], gridcolor='rgba(0,0,0,0.05)',
                           title='Importance Score'),
                xaxis_title='ESG Pillar', legend_title='Model',
                title_font_size=15
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                        unsafe_allow_html=True)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SECTION C — SINGLE PREDICTION
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("""
            <div style='border-left:4px solid #e67e22;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                    🔮 Predict Risk for New Company
                </span>
            </div>""", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                env = st.slider("🌱 Environmental Score", 0, 100,
                                st.session_state.pred_env, key='pred_env')
            with col2:
                soc = st.slider("🤝 Social Score", 0, 100,
                                st.session_state.pred_soc, key='pred_soc')
            with col3:
                gov = st.slider("🏛️ Governance Score", 0, 100,
                                st.session_state.pred_gov, key='pred_gov')


            if st.button("🔮 Predict Risk Level"):
                model      = results[selected_model]['model']
                input_data = scaler.transform([[env, soc, gov]])
                prediction = model.predict(input_data)[0]
                has_proba  = hasattr(model, "predict_proba")
                proba      = model.predict_proba(input_data)[0] if has_proba else None
                risk_map   = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
                predicted_risk = risk_map[prediction]
                esg   = round((env*0.4)+(soc*0.3)+(gov*0.3), 1)
                color = get_risk_color(predicted_risk)
                env_c = round(env*0.4, 1)
                soc_c = round(soc*0.3, 1)
                gov_c = round(gov*0.3, 1)

                col1, col2 = st.columns([1,1])
                with col1:
                    st.markdown(f"""
                    <div style='background:white;border-radius:14px;padding:28px;
                                text-align:center;border-top:5px solid {color};
                                box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
                        <p style='color:#888;font-size:0.85rem;margin:0;'>Predicted Risk Level</p>
                        <h1 style='color:{color};margin:8px 0;font-size:2.2rem;'>
                            {predicted_risk}
                        </h1>
                        <div style='background:{color}20;border-radius:20px;
                                    padding:8px 20px;display:inline-block;margin:8px 0;'>
                            <span style='color:{color};font-weight:700;font-size:1.3rem;'>
                                ESG Score: {esg} / 100
                            </span>
                        </div>
                        <p style='color:#666;font-size:0.85rem;margin:8px 0 0 0;'>
                            Model: <strong>{selected_model}</strong>
                        </p>
                    </div>""", unsafe_allow_html=True)

                    if proba is not None:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style='background:white;border-radius:12px;padding:18px;
                                    box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                            <p style='color:#1a3c5e;font-weight:600;margin:0 0 12px 0;'>
                                🎲 Confidence Probabilities
                            </p>""", unsafe_allow_html=True)
                        for label, prob, pc in zip(
                            ["Low Risk","Medium Risk","High Risk"],
                            proba,
                            ['#2ecc71','#f39c12','#e74c3c']
                        ):
                            st.markdown(f"""
                            <div style='margin-bottom:10px;'>
                                <div style='display:flex;justify-content:space-between;
                                            margin-bottom:4px;'>
                                    <span style='font-size:0.85rem;color:#444;'>{label}</span>
                                    <span style='font-size:0.85rem;font-weight:700;
                                                 color:{pc};'>{prob:.1%}</span>
                                </div>
                                <div style='background:#f0f0f0;border-radius:10px;height:8px;'>
                                    <div style='background:{pc};width:{prob*100:.1f}%;
                                                height:8px;border-radius:10px;'></div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    fig_contrib = go.Figure(data=[go.Pie(
                        labels=['🌱 Environmental (40%)',
                                '🤝 Social (30%)',
                                '🏛️ Governance (30%)'],
                        values=[env_c, soc_c, gov_c],
                        hole=0.55,
                        marker_colors=['#27ae60','#2980b9','#8e44ad'],
                        textinfo='label+percent',
                        hovertemplate='<b>%{label}</b><br>Contribution: %{value}<extra></extra>'
                    )])
                    fig_contrib.update_layout(
                        title=dict(text="Pillar Contribution to ESG Score", font_size=14),
                        paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                        annotations=[dict(text=f"<b>{esg}</b>", x=0.5, y=0.5,
                                          font_size=22, showarrow=False)],
                        height=300, margin=dict(l=10,r=10,t=40,b=10)
                    )
                    st.plotly_chart(fig_contrib, use_container_width=True)

                    avg_env = round(df['environmental_score'].mean(), 1)
                    avg_soc = round(df['social_score'].mean(), 1)
                    avg_gov = round(df['governance_score'].mean(), 1)
                    fig_vs = go.Figure()
                    fig_vs.add_trace(go.Bar(
                        name='Your Company',
                        x=['🌱 Environmental','🤝 Social','🏛️ Governance'],
                        y=[env, soc, gov], marker_color=color,
                        text=[env, soc, gov], textposition='outside'
                    ))
                    fig_vs.add_trace(go.Bar(
                        name='Dataset Avg',
                        x=['🌱 Environmental','🤝 Social','🏛️ Governance'],
                        y=[avg_env, avg_soc, avg_gov],
                        marker_color='rgba(0,0,0,0.15)',
                        text=[avg_env, avg_soc, avg_gov], textposition='outside'
                    ))
                    fig_vs.update_layout(
                        barmode='group',
                        title=dict(text="Your Scores vs Dataset Average", font_size=14),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(range=[0,120], gridcolor='rgba(0,0,0,0.05)'),
                        xaxis_title='', yaxis_title='Score (0–100)',
                        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center'),
                        height=280, margin=dict(l=10,r=10,t=40,b=10)
                    )
                    st.plotly_chart(fig_vs, use_container_width=True)

                verdict_map = {
                    "Low Risk":    ("✅ Safe Investment",     "#eafaf1","#27ae60",
                        "Strong ESG practices across all pillars. Recommended for ESG-conscious portfolios."),
                    "Medium Risk": ("⚠️ Moderate Risk",       "#fef9e7","#e67e22",
                        "Average ESG performance. Monitor pillar scores before committing capital."),
                    "High Risk":   ("🚨 High Risk — Caution", "#fdedec","#e74c3c",
                        "Poor ESG scores indicate sustainability concerns. Investment not recommended."),
                }
                vl, bg, vc, msg = verdict_map[predicted_risk]
                st.markdown(f"""
                <div style='background:{bg};border-radius:12px;padding:16px 22px;
                            margin-top:16px;border-left:5px solid {vc};'>
                    <strong style='color:{vc};font-size:1rem;'>{vl}</strong>
                    <p style='color:#444;margin:6px 0 0 0;font-size:0.9rem;'>{msg}</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                        unsafe_allow_html=True)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SECTION D — BATCH PREDICTION (NEW)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("""
            <div style='border-left:4px solid #16a085;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                    📦 Batch Prediction — Upload New Companies
                </span>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            Upload a CSV with columns:
            `company`, `environmental_score`, `social_score`, `governance_score`
            """)
            batch_file = st.file_uploader("📁 Upload batch CSV", type=["csv"],
                                          key="batch_upload")
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                req_cols = ['environmental_score','social_score','governance_score']
                if all(c in batch_df.columns for c in req_cols):
                    batch_scaled = scaler.transform(batch_df[req_cols])
                    model        = results[selected_model]['model']
                    preds        = model.predict(batch_scaled)
                    risk_map_b   = {0:"Low Risk",1:"Medium Risk",2:"High Risk"}
                    batch_df['predicted_risk'] = [risk_map_b[p] for p in preds]
                    batch_df['esg_score']      = batch_df.apply(
                        lambda r: calculate_esg_score(r), axis=1
                    )
                    if 'company' not in batch_df.columns:
                        batch_df['company'] = [f"Company {i+1}"
                                               for i in range(len(batch_df))]
                    st.success(f"✅ Predicted risk for {len(batch_df)} companies!")
                    st.dataframe(
                        batch_df[['company','environmental_score','social_score',
                                  'governance_score','esg_score','predicted_risk']]
                        .style.background_gradient(subset=['esg_score'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                    csv_out = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Predictions CSV", csv_out,
                        "batch_predictions.csv", "text/csv"
                    )
                else:
                    st.error(f"❌ CSV must contain: {', '.join(req_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style='border-left:4px solid #2980b9;padding:4px 0 4px 14px;margin-bottom:16px;'>
        <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
            📈 Model Evaluation & Performance
        </span>
    </div>""", unsafe_allow_html=True)

    if len(df) < 5:
        st.warning("⚠️ Need at least 5 companies to evaluate models.")
    else:
        with st.spinner("Evaluating models..."):
            results, scaler = train_models(df.to_json())

        if results is None:
            st.error("⚠️ Dataset has only one risk class.")
        else:
            selected_eval = st.selectbox(
                "🤖 Select Model to Evaluate", list(results.keys()),
                key="eval_model_select"
            )
            r  = results[selected_eval]
            cm = np.array(r['confusion_matrix'])
            total   = int(cm.sum())
            correct = int(np.trace(cm))

            st.markdown("<br>", unsafe_allow_html=True)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SECTION A — KPI METRIC CARDS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("""
            <div style='border-left:4px solid #2ecc71;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                    🎯 Performance Metrics
                </span>
            </div>""", unsafe_allow_html=True)

            metrics = [
                ("🎯 Accuracy",  r['accuracy'],  "#2ecc71", "Overall correct predictions"),
                ("🔍 Precision", r['precision'], "#2980b9", "Positive prediction accuracy"),
                ("📡 Recall",    r['recall'],    "#9b59b6", "True positive detection rate"),
                ("⚖️ F1 Score",  r['f1'],        "#e67e22", "Harmonic mean of P & R"),

            ]
            cols = st.columns(5)
            for col, (label, value, color, subtitle) in zip(cols, metrics):
                with col:
                    st.markdown(f"""
                    <div style='background:white;border-radius:14px;padding:18px;
                                text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.07);
                                border-top:4px solid {color};'>
                        <p style='color:#888;font-size:0.75rem;margin:0 0 4px 0;'>
                            {subtitle}
                        </p>
                        <h2 style='color:{color};margin:0;font-size:1.8rem;font-weight:700;'>
                            {value:.1%}
                        </h2>
                        <p style='color:#1a3c5e;font-size:0.85rem;font-weight:600;
                                  margin:6px 0 0 0;'>{label}</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SECTION B — CONFUSION MATRIX + RADAR
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("""
            <div style='border-left:4px solid #9b59b6;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                    🔢 Confusion Matrix & Model Radar
                </span>
            </div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                risk_labels_cm = ["Low Risk","Medium Risk","High Risk"]
                n = cm.shape[0]
                labels_used = risk_labels_cm[:n]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f"Pred:<br>{l}" for l in labels_used],
                    y=[f"True:<br>{l}" for l in labels_used],
                    colorscale=[[0.0,"#eaf4fb"],[0.5,"#2980b9"],[1.0,"#1a3c5e"]],
                    text=cm, texttemplate="<b>%{text}</b>",
                    textfont={"size":18}, hoverongaps=False, showscale=True
                ))
                fig_cm.update_layout(
                    title=dict(text=f"Confusion Matrix — {selected_eval}", font_size=15),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title="Predicted Label", side="bottom"),
                    yaxis=dict(title="True Label", autorange="reversed"),
                    height=360, margin=dict(l=20,r=20,t=50,b=20)
                )
                for i in range(n):
                    fig_cm.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5, y0=i-0.5, y1=i+0.5,
                        line=dict(color="#2ecc71", width=3)
                    )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.markdown(f"""
                <div style='display:flex;gap:12px;justify-content:center;margin-top:-10px;'>
                    <div style='background:#eafaf1;border-radius:20px;padding:6px 18px;
                                font-size:0.85rem;color:#27ae60;font-weight:600;'>
                        ✅ Correct: {correct}
                    </div>
                    <div style='background:#fdedec;border-radius:20px;padding:6px 18px;
                                font-size:0.85rem;color:#e74c3c;font-weight:600;'>
                        ❌ Wrong: {total - correct}
                    </div>
                    <div style='background:#eaf4fb;border-radius:20px;padding:6px 18px;
                                font-size:0.85rem;color:#2980b9;font-weight:600;'>
                        📊 Total: {total}
                    </div>
                </div>""", unsafe_allow_html=True)

            with col2:
                radar_metrics = ['Accuracy','Precision','Recall','F1 Score']
                radar_colors  = [
                    ('#2ecc71','rgba(46,204,113,0.12)'),
                    ('#e74c3c','rgba(231,76,60,0.12)'),
                    ('#f39c12','rgba(243,156,18,0.12)'),
                ]
                fig_radar = go.Figure()
                for (name, res), (lc, fc) in zip(results.items(), radar_colors):
                    vals = [res['accuracy'],res['precision'],res['recall'],res['f1']]
                    vals += [vals[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=radar_metrics + [radar_metrics[0]],
                        fill='toself', name=name,
                        line_color=lc, fillcolor=fc, opacity=0.85
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0,1],
                                        tickformat='.0%',
                                        gridcolor='rgba(0,0,0,0.08)'),
                        angularaxis=dict(gridcolor='rgba(0,0,0,0.08)')
                    ),
                    title=dict(text="All Models — Metric Radar", font_size=15),
                    paper_bgcolor='rgba(0,0,0,0)', showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom',
                                y=-0.25, xanchor='center', x=0.5),
                    height=360, margin=dict(l=20,r=20,t=50,b=60)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                        unsafe_allow_html=True)



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SECTION C — ALL-MODELS COMPARISON + LEADERBOARD
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("""
            <div style='border-left:4px solid #1a3c5e;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                    📊 All Models — Side-by-Side Comparison & Leaderboard
                </span>
            </div>""", unsafe_allow_html=True)

            fig_bar = px.bar(
                perf_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Metric', y='Score', color='Model', barmode='group',
                text_auto='.1%',
                color_discrete_sequence=['#2ecc71','#e74c3c','#f39c12'],
                title='Accuracy · Precision · Recall · F1 — All Models'
            )
            fig_bar.update_traces(textposition='outside', textfont_size=11)
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0,1.15], tickformat='.0%',
                           gridcolor='rgba(0,0,0,0.05)', title='Score'),
                xaxis_title='', legend_title='Model', title_font_size=15,
                uniformtext_minsize=8, uniformtext_mode='hide'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            best_model = perf_df.loc[perf_df['F1 Score'].idxmax(), 'Model']
            best_f1    = perf_df['F1 Score'].max()
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#eafaf1,#d5f5e3);
                        border-radius:12px;padding:14px 20px;margin-bottom:16px;
                        border-left:4px solid #2ecc71;'>
                🏆 <strong>Best Model:</strong> {best_model} —
                highest F1 Score of <strong>{best_f1:.1%}</strong>
            </div>""", unsafe_allow_html=True)

            st.dataframe(
                perf_df.style
                    .format({"Accuracy":"{:.1%}","Precision":"{:.1%}",
                             "Recall":"{:.1%}","F1 Score":"{:.1%}"})
                    .background_gradient(
                        subset=["Accuracy","Precision","Recall","F1 Score"],
                        cmap="Greens")
                    .highlight_max(
                        subset=["Accuracy","Precision","Recall","F1 Score"],
                        color="#d5f5e3"),
                use_container_width=True, hide_index=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPANY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div style='border-left:4px solid #e67e22;padding:4px 0 4px 14px;margin-bottom:16px;'>
        <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>🏢 Company Insights</span>
    </div>""", unsafe_allow_html=True)

    if filtered_df.empty:
        empty_state()
    else:
        selected_company = st.selectbox(
            "Select a Company", filtered_df['company'].tolist()
        )
        company    = filtered_df[filtered_df['company'] == selected_company].iloc[0]
        risk_color = get_risk_color(company['risk_label'])
        risk_rgba  = RISK_RGBA[company['risk_label']]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION A — PROFILE CARD + RADAR
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #2ecc71;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>🪪 Company Profile</span>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            rank = int(filtered_df['esg_score'].rank(ascending=False)
                       .loc[filtered_df['company'] == selected_company].values[0])
            pct  = round((1 - (rank / len(filtered_df))) * 100, 1)
            sector_val = company.get('sector', 'N/A')

            st.markdown(f"""
            <div style='background:white;border-radius:14px;padding:28px;
                        border-top:5px solid {risk_color};
                        box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <h2 style='color:#1a3c5e;margin:0;'>{company['company']}</h2>
                        <p style='color:#888;font-size:0.85rem;margin:4px 0;'>
                            🏭 {sector_val}
                        </p>
                        <span style='background:{risk_color};color:white;
                                     border-radius:20px;padding:4px 14px;
                                     font-size:0.85rem;font-weight:600;'>
                            {company['risk_label']}
                        </span>
                    </div>
                    <div style='text-align:right;'>
                        <p style='color:#888;font-size:0.8rem;margin:0;'>Dataset Rank</p>
                        <h2 style='color:{risk_color};margin:0;'>#{rank}</h2>
                        <p style='color:#888;font-size:0.75rem;margin:0;'>
                            Top {100-pct:.1f}%
                        </p>
                    </div>
                </div>
                <hr style='border:1px solid #f0f0f0;margin:16px 0;'/>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;'>
                    <div style='background:#f8f9fa;border-radius:10px;
                                padding:14px;text-align:center;'>
                        <p style='color:#27ae60;font-size:0.75rem;margin:0;'>🌱 Environmental</p>
                        <h3 style='color:#1a3c5e;margin:4px 0;'>
                            {round(company['environmental_score'],1)}
                        </h3>
                    </div>
                    <div style='background:#f8f9fa;border-radius:10px;
                                padding:14px;text-align:center;'>
                        <p style='color:#2980b9;font-size:0.75rem;margin:0;'>🤝 Social</p>
                        <h3 style='color:#1a3c5e;margin:4px 0;'>
                            {round(company['social_score'],1)}
                        </h3>
                    </div>
                    <div style='background:#f8f9fa;border-radius:10px;
                                padding:14px;text-align:center;'>
                        <p style='color:#8e44ad;font-size:0.75rem;margin:0;'>🏛️ Governance</p>
                        <h3 style='color:#1a3c5e;margin:4px 0;'>
                            {round(company['governance_score'],1)}
                        </h3>
                    </div>
                </div>
                <div style='background:linear-gradient(135deg,#f8f9fa,#eaf4fb);
                            border-radius:10px;padding:14px;text-align:center;
                            margin-top:12px;'>
                    <p style='color:#888;font-size:0.8rem;margin:0;'>Overall ESG Score</p>
                    <h1 style='color:{risk_color};margin:4px 0;font-size:2.5rem;'>
                        {company['esg_score']}
                        <span style='font-size:1rem;color:#888;'> / 100</span>
                    </h1>
                </div>
            </div>""", unsafe_allow_html=True)

        with col2:
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[company['environmental_score'], company['social_score'],
                   company['governance_score']],
                theta=['Environmental','Social','Governance'],
                fill='toself', line_color=risk_color,
                fillcolor=risk_rgba, name=selected_company
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[round(filtered_df['environmental_score'].mean(),1),
                   round(filtered_df['social_score'].mean(),1),
                   round(filtered_df['governance_score'].mean(),1)],
                theta=['Environmental','Social','Governance'],
                fill='toself',
                line_color='rgba(0,0,0,0.2)',
                fillcolor='rgba(0,0,0,0.04)',
                name='Dataset Avg', line_dash='dash'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                title=dict(text=f"{company['company']} vs Dataset Average",
                           font_size=15),
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
                height=380
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                    unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION B — SCORE HISTORY SPARKLINES (NEW)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        hist_cols = [f'env_m{i}' for i in range(1,7)]
        if all(c in company.index for c in hist_cols):
            st.markdown("""
            <div style='border-left:4px solid #16a085;padding:4px 0 4px 14px;margin-bottom:16px;'>
                <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                    📉 6-Month Score History
                </span>
            </div>""", unsafe_allow_html=True)

            months = [f"Month {i}" for i in range(1,7)]
            env_hist = [company[f'env_m{i}'] for i in range(1,7)]
            soc_hist = [company[f'soc_m{i}'] for i in range(1,7)]
            gov_hist = [company[f'gov_m{i}'] for i in range(1,7)]

            fig_hist = go.Figure()
            for label, vals, color in [
                ('🌱 Environmental', env_hist, '#27ae60'),
                ('🤝 Social',        soc_hist, '#2980b9'),
                ('🏛️ Governance',   gov_hist, '#8e44ad'),
            ]:
                fig_hist.add_trace(go.Scatter(
                    x=months, y=vals, name=label,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=7)
                ))
            fig_hist.update_layout(
                title=dict(text=f"{selected_company} — Pillar Score Trend",
                           font_size=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0,110], gridcolor='rgba(0,0,0,0.05)',
                           title='Score (0–100)'),
                xaxis_title='', legend_title='Pillar',
                legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center')
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                        unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION C — INVESTMENT RECOMMENDATION + SCORE BREAKDOWN
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #2980b9;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                💡 Investment Recommendation & Score Breakdown
            </span>
        </div>""", unsafe_allow_html=True)

        verdict_map = {
            "Low Risk":    ("✅ Safe Investment",     "#eafaf1","#27ae60",
                "Strong ESG practices. Recommended for ESG-conscious portfolios."),
            "Medium Risk": ("⚠️ Moderate Risk",       "#fef9e7","#e67e22",
                "Average ESG performance. Monitor pillar trends before committing."),
            "High Risk":   ("🚨 High Risk — Caution", "#fdedec","#e74c3c",
                "Poor ESG scores. Significant sustainability concerns. Not recommended."),
        }
        vl, bg, vc, msg = verdict_map[company['risk_label']]
        st.markdown(f"""
        <div style='background:{bg};border-radius:12px;padding:16px 22px;
                    margin-bottom:16px;border-left:5px solid {vc};'>
            <strong style='color:{vc};font-size:1rem;'>{vl}</strong>
            <p style='color:#444;margin:6px 0 0 0;font-size:0.9rem;'>{msg}</p>
        </div>""", unsafe_allow_html=True)

        score_df = pd.DataFrame({
            'Category': ['Environmental (40%)','Social (30%)','Governance (30%)'],
            'Raw Score': [round(company['environmental_score'],1),
                          round(company['social_score'],1),
                          round(company['governance_score'],1)],
            'Weighted Score': [round(company['environmental_score']*0.4,1),
                               round(company['social_score']*0.3,1),
                               round(company['governance_score']*0.3,1)]
        })
        fig_scores = px.bar(
            score_df, x='Category', y='Raw Score', color='Category',
            title='Pillar Score Breakdown', text='Weighted Score',
            color_discrete_sequence=['#27ae60','#2980b9','#8e44ad']
        )
        fig_scores.update_traces(
            texttemplate='Weighted: %{text}', textposition='outside'
        )
        fig_scores.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0,120], gridcolor='rgba(0,0,0,0.05)',
                       title='Score (0–100)'),
            xaxis_title='', showlegend=False, title_font_size=15
        )
        st.plotly_chart(fig_scores, use_container_width=True)

        st.markdown("<hr style='border:1px solid #dce3ea;margin:8px 0 20px;'>",
                    unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION D — PEER COMPARISON
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #9b59b6;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                🔎 Peer Comparison — Same Risk Tier
            </span>
        </div>""", unsafe_allow_html=True)

        peers = filtered_df[filtered_df['risk_label'] == company['risk_label']].copy()

        if len(peers) <= 1:
            st.info("ℹ️ No peers in the same risk tier to compare against.")
        else:
            tier_avg_env = round(peers['environmental_score'].mean(), 1)
            tier_avg_soc = round(peers['social_score'].mean(), 1)
            tier_avg_gov = round(peers['governance_score'].mean(), 1)
            tier_avg_esg = round(peers['esg_score'].mean(), 1)

            deltas = [
                ("🌱 Environmental", company['environmental_score'], tier_avg_env, "#27ae60"),
                ("🤝 Social",        company['social_score'],        tier_avg_soc, "#2980b9"),
                ("🏛️ Governance",   company['governance_score'],    tier_avg_gov, "#8e44ad"),
                ("📊 ESG Score",     company['esg_score'],           tier_avg_esg, risk_color),
            ]
            dcols = st.columns(4)
            for dcol, (label, val, avg, color) in zip(dcols, deltas):
                diff   = round(float(val) - avg, 1)
                arrow  = "▲" if diff >= 0 else "▼"
                dcolor = "#27ae60" if diff >= 0 else "#e74c3c"
                with dcol:
                    st.markdown(f"""
                    <div style='background:white;border-radius:12px;padding:16px;
                                text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);
                                border-top:3px solid {color};'>
                        <p style='color:#888;font-size:0.75rem;margin:0;'>{label}</p>
                        <h3 style='color:#1a3c5e;margin:6px 0;'>{round(float(val),1)}</h3>
                        <span style='color:{dcolor};font-size:0.85rem;font-weight:600;'>
                            {arrow} {abs(diff)} vs tier avg
                        </span>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            peers_sorted = peers.sort_values('esg_score', ascending=False).head(15).copy()
            peers_sorted['highlight'] = peers_sorted['company'].apply(
                lambda x: selected_company if x == selected_company else 'Peers'
            )
            fig_peer = px.bar(
                peers_sorted, y='company', x='esg_score', orientation='h',
                color='highlight',
                color_discrete_map={selected_company: risk_color,
                                    'Peers': 'rgba(180,180,180,0.5)'},
                text='esg_score',
                title=f"ESG Score vs Peers — {company['risk_label']} Tier (Top 15)"
            )
            fig_peer.update_traces(texttemplate='%{text}', textposition='outside')
            fig_peer.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[0,115], gridcolor='rgba(0,0,0,0.05)',
                           title='ESG Score (0–100)'),
                yaxis_title='', legend_title='', title_font_size=15,
                margin=dict(l=10)
            )
            fig_peer.add_vline(
                x=tier_avg_esg, line_dash="dash", line_color="#888",
                annotation_text=f"Tier Avg: {tier_avg_esg}",
                annotation_position="top right",
                annotation_font_color="#888"
            )
           
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECTION E — SIDE-BY-SIDE COMPARATOR (NEW)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("""
        <div style='border-left:4px solid #e74c3c;padding:4px 0 4px 14px;margin-bottom:16px;'>
            <span style='font-size:1rem;font-weight:600;color:#1a3c5e;'>
                ⚖️ Side-by-Side Company Comparator
            </span>
        </div>""", unsafe_allow_html=True)

        other_companies = [c for c in filtered_df['company'].tolist()
                           if c != selected_company]
        if len(other_companies) == 0:
            st.info("ℹ️ Need at least 2 companies for comparison.")
        else:
            compare_with = st.multiselect(
                "Select up to 2 companies to compare with",
                other_companies, max_selections=2,
                default=other_companies[:min(2, len(other_companies))]
            )
            if compare_with:
                compare_list = [selected_company] + compare_with
                compare_df   = filtered_df[
                    filtered_df['company'].isin(compare_list)
                ].copy()

                comp_melt = compare_df[
                    ['company','environmental_score','social_score',
                     'governance_score','esg_score']
                ].melt(id_vars='company', var_name='Metric', value_name='Score')
                comp_melt['Metric'] = comp_melt['Metric'].map({
                    'environmental_score': '🌱 Environmental',
                    'social_score':        '🤝 Social',
                    'governance_score':    '🏛️ Governance',
                    'esg_score':           '📊 ESG Score'
                })
                fig_comp = px.bar(
                    comp_melt, x='Metric', y='Score', color='company',
                    barmode='group', text_auto='.1f',
                    title='Company Comparison — All Pillars + ESG Score',
                    color_discrete_sequence=['#2ecc71','#e74c3c','#f39c12']
                )
                fig_comp.update_traces(textposition='outside', textfont_size=11)
                fig_comp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0,120], gridcolor='rgba(0,0,0,0.05)',
                               title='Score'),
                    xaxis_title='', legend_title='Company', title_font_size=15
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # summary table
                summary_cols = ['company','sector','environmental_score',
                                'social_score','governance_score',
                                'esg_score','risk_label']
                summary_cols = [c for c in summary_cols if c in compare_df.columns]
                st.dataframe(
                    compare_df[summary_cols]
                    .set_index('company')
                    .style.background_gradient(subset=['esg_score'], cmap='Greens'),
                    use_container_width=True
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPORT REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    <div style='border-left:4px solid #1a3c5e;padding:4px 0 4px 14px;margin-bottom:16px;'>
        <span style='font-size:1.05rem;font-weight:600;color:#1a3c5e;'>
            📄 Export ESG Analysis Report
        </span>
    </div>""", unsafe_allow_html=True)

    if filtered_df.empty:
        st.markdown("""
        <div style='background:#fef9e7;border-radius:12px;padding:16px 22px;
                    border-left:5px solid #f39c12;'>
            <strong style='color:#e67e22;'>⚠️ No data to export</strong>
            <p style='color:#444;margin:6px 0 0 0;font-size:0.9rem;'>
                Your current filters return no companies. Try widening the
                ESG score range or selecting more risk levels in the sidebar.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Companies", len(filtered_df))
        with col2: st.metric("Avg ESG Score",   round(filtered_df['esg_score'].mean(),1))
        with col3: st.metric("✅ Low Risk",
                             len(filtered_df[filtered_df['risk_label']=='Low Risk']))
        with col4: st.metric("🚨 High Risk",
                             len(filtered_df[filtered_df['risk_label']=='High Risk']))

        if len(filtered_df) < 5:
            st.warning(f"⚠️ Only {len(filtered_df)} companies in current filter — "
                       "the exported report will be sparse. Consider widening filters.")

        st.markdown("### 🔍 Data to be Exported")
        export_cols = ['company','sector','environmental_score','social_score',
                       'governance_score','esg_score','risk_label']
        export_cols = [c for c in export_cols if c in filtered_df.columns]
        st.dataframe(
            filtered_df[export_cols]
            .sort_values('esg_score', ascending=False)
            .style.background_gradient(subset=['esg_score'], cmap='RdYlGn'),
            use_container_width=True
        )

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📄 PDF Report")
            st.markdown("Professional report with cover, summary & company table.")
            if st.button("📄 Generate & Download PDF"):
                with st.spinner("Generating PDF report..."):
                    pdf_bytes = generate_pdf(df, filtered_df)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"ESG_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                st.success("✅ PDF ready!")

        with col2:
            st.markdown("#### 📊 CSV Export")
            st.markdown("Raw data with all ESG scores and risk labels.")
            csv = filtered_df[export_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"ESG_Data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col3:
            st.markdown("#### 📑 Excel Export ")
            st.markdown("Separate sheets per risk tier + summary.")
            if st.button("📑 Generate Excel"):
                with st.spinner("Building Excel workbook..."):
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                        # summary sheet
                        summary_data = pd.DataFrame([{
                            "Metric": m, "Value": v
                        } for m, v in [
                            ("Total Companies",   len(filtered_df)),
                            ("Avg ESG Score",      round(filtered_df['esg_score'].mean(),1)),
                            ("Low Risk Count",     len(filtered_df[filtered_df['risk_label']=='Low Risk'])),
                            ("Medium Risk Count",  len(filtered_df[filtered_df['risk_label']=='Medium Risk'])),
                            ("High Risk Count",    len(filtered_df[filtered_df['risk_label']=='High Risk'])),
                            ("Best Company",       filtered_df.loc[filtered_df['esg_score'].idxmax(),'company']),
                            ("Best ESG Score",     filtered_df['esg_score'].max()),
                        ]])
                        summary_data.to_excel(writer, sheet_name='Summary', index=False)
                        # all data
                        filtered_df[export_cols].sort_values(
                            'esg_score', ascending=False
                        ).to_excel(writer, sheet_name='All Companies', index=False)
                        # per-tier sheets
                        for tier in ['Low Risk','Medium Risk','High Risk']:
                            tier_df = filtered_df[
                                filtered_df['risk_label'] == tier
                            ][export_cols]
                            if not tier_df.empty:
                                sheet_name = tier.replace(' ','_')
                                tier_df.sort_values(
                                    'esg_score', ascending=False
                                ).to_excel(writer, sheet_name=sheet_name, index=False)
                    excel_buf.seek(0)
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_buf.getvalue(),
                    file_name=f"ESG_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✅ Excel ready!")