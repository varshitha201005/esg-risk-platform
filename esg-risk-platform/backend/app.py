from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import io

app = Flask(__name__)
CORS(app)

def calculate_risk(env, soc, gov):
    total = (env + soc + gov) / 3
    if total >= 70:
        return "Safe", "green"
    elif total >= 45:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def process_df(df):
    results = []
    for _, row in df.iterrows():
        risk, color = calculate_risk(
            row['environmental_score'],
            row['social_score'],
            row['governance_score']
        )
        results.append({
            "company": row['company'],
            "environmental": row['environmental_score'],
            "social": row['social_score'],
            "governance": row['governance_score'],
            "overall": round((row['environmental_score'] + row['social_score'] + row['governance_score']) / 3, 1),
            "risk": risk,
            "color": color
        })
    return results

@app.route('/api/companies', methods=['GET'])
def get_companies():
    df = pd.read_csv('esg_data.csv')
    return jsonify(process_df(df))

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        return jsonify(process_df(df))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)