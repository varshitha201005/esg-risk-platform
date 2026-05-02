# 🌿 TripleLens — ESG Risk Analysis Platform

> **Insight. Impact. Integrity.**

A full-stack ESG (Environmental, Social, and Governance) Risk Analysis Platform that helps investors analyze companies and predict investment risk levels using Machine Learning.

![TripleLens Banner](https://raw.githubusercontent.com/varshitha201005/esg-risk-platform/main/esg-risk-platform1/esg_logo.png)

---

## 🚀 Live Demo

🌐 **Web App:** [Click here to open TripleLens](https://esg-risk-platform-q9uc8ktfzwtu3ny6mpeepe.streamlit.app)

---

## 📌 Features

- 🔐 **User Authentication** — Register & Login system with encrypted passwords
- 📁 **CSV Upload** — Upload any Kaggle ESG dataset
- 📊 **Interactive Dashboard** — Real-time charts and KPI cards
- 🤖 **Machine Learning** — 3 ML models (Random Forest, Logistic Regression, Gradient Boosting)
- 📈 **Model Evaluation** — Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- 🏢 **Company Insights** — Radar charts, score breakdown, investment recommendations
- ⚖️ **Company Comparator** — Side-by-side comparison of companies
- 🔍 **Smart Filters** — Filter by risk level and ESG score range
- 📄 **Export Reports** — Download as PDF, CSV, or Excel
- 📱 **Android App** — Available as APK (WebView wrapper)

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Seaborn, Matplotlib |
| Authentication | Bcrypt |
| Export | FPDF2, OpenPyXL |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |

---

## 📊 ML Models Used

| Model | Purpose |
|---|---|
| Random Forest | Primary risk prediction |
| Logistic Regression | Baseline classification |
| Gradient Boosting | Advanced risk prediction |

**Risk Categories:**
- ✅ **Low Risk** — ESG Score ≥ 70
- ⚠️ **Medium Risk** — ESG Score 45–70
- 🚨 **High Risk** — ESG Score < 45

---

## 📁 Dataset

The platform accepts any CSV dataset with the following columns:
- `name` / `company` — Company name
- `environment_score` — Environmental score
- `social_score` — Social score
- `governance_score` — Governance score

**Recommended Dataset:**
[Public Company ESG Ratings — Kaggle](https://www.kaggle.com/datasets/alistairking/public-company-esg-ratings-dataset)

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.8+
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/varshitha201005/esg-risk-platform.git

# Navigate to project folder
cd esg-risk-platform/esg-risk-platform1

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📱 Android App

An Android APK is available that wraps the web app in a native Android shell.

- Built with Android Studio
- Uses WebView to load the Streamlit app
- Works on Android 7.0+ (API 24+)

---

## 📂 Project Structure
esg-risk-platform/
├── esg-risk-platform1/
│   ├── app.py              # Main Streamlit application
│   ├── requirements.txt    # Python dependencies
│   ├── esg_logo.png        # TripleLens logo
│   ├── users.json          # User database
│   └── README.md           # Project documentation
└── esg-risk-platform/
└── backend/
├── app.py          # Flask backend (initial version)
└── esg_data.csv    # Sample dataset
---

## 🎯 ESG Scoring Formula
ESG Score = (Environmental × 40%) + (Social × 30%) + (Governance × 30%)

---

## 👩‍💻 Developer

**Varshitha Sharigudam** — B.Tech Data Science Student, Mahatma Gandhi Institute Of Technology  
📧 varshithasharigudam@gmail.com  
💼 [LinkedIn](https://www.linkedin.com/in/varshitha-sharigudam-8b36722b8)  
🐙 [GitHub](https://github.com/varshitha201005)  
🌐 [Live Project](https://esg-risk-platform-q9uc8ktfzwtu3ny6mpeepe.streamlit.app)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io) — Web framework
- [Scikit-learn](https://scikit-learn.org) — Machine learning
- [Kaggle](https://kaggle.com) — Dataset source
- [Plotly](https://plotly.com) — Interactive charts