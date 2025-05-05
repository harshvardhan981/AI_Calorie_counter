from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import logging
import os
import threading
import webbrowser
import time
import pandas as pd
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Setup
logging.basicConfig(level=logging.INFO)
base_path = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask
app = Flask(__name__, static_folder=os.path.join(base_path, 'static'), static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Load models
model = joblib.load(os.path.join(base_path, 'calorie_predictor.pkl'))
label_encoder = joblib.load(os.path.join(base_path, 'food_encoder.pkl'))
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))

# Load nutrition data
try:
    df = pd.read_csv(os.path.join(base_path, 'usda_food_nutrition.csv'))
    df = df[['Description', 'Calories', 'Protein', 'TotalFat', 'Carbohydrate']]
    df.fillna(df.median(numeric_only=True), inplace=True)
    logging.info("Nutrition dataset loaded.")
except Exception as e:
    logging.error("Error loading CSV: " + str(e))
    df = pd.DataFrame()

# SQLite user database
DB_PATH = os.path.join(base_path, 'users.db')
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
init_db()

# Routes
@app.route('/')
def index():
    return app.send_static_file('web.html')

@app.route('/dashboard.html')
def dashboard():
    return app.send_static_file('dashboard.html')

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(os.path.join(base_path, 'data'), filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    food_name = data.get("food_name", "").strip().lower()
    classes_lower = [c.lower() for c in label_encoder.classes_]
    if food_name not in classes_lower:
        return jsonify({"error": "Food not found in dataset."})
    real_name = label_encoder.classes_[classes_lower.index(food_name)]
    food_id = label_encoder.transform([real_name])[0]
    scaled = scaler.transform([[food_id]])
    pred = model.predict(scaled)[0]
    return jsonify({
        "Food": real_name,
        "Calories": round(pred[0], 2),
        "Protein": round(pred[1], 2),
        "Fat": round(pred[2], 2),
        "Carbohydrates": round(pred[3], 2),
        "Suggestion": get_suggestion(pred[0])
    })

def get_suggestion(calories):
    if calories < 200:
        return "Low-calorie food. Consider pairing with protein."
    elif calories < 500:
        return "Moderate-calorie. Good for a balanced meal."
    else:
        return "High-calorie. Consider portion control."

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query', '').lower()
    if 'breakfast' in query:
        return jsonify(suggest_breakfast())
    return jsonify({"error": "Only breakfast suggestion supported."}), 400

def suggest_breakfast():
    if df.empty:
        return {"error": "No food data available"}
    keywords = ['egg', 'bread', 'milk', 'toast', 'cereal', 'banana', 'pancake', 'yogurt', 'oatmeal']
    filtered = df[df['Description'].str.lower().str.contains('|'.join(keywords))]
    if filtered.empty:
        return {"error": "No breakfast options found."}
    filtered['Score'] = filtered['Protein'] / (filtered['Calories'] + 1)
    top = filtered.sort_values(by='Score', ascending=False).head(5)
    return {"suggestions": [
        {
            "Food": row['Description'],
            "Calories": round(row['Calories'], 2),
            "Protein": round(row['Protein'], 2),
            "Fat": round(row['TotalFat'], 2),
            "Carbohydrates": round(row['Carbohydrate'], 2)
        } for _, row in top.iterrows()
    ]}

# User sign-up
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"message": "Username and password required."}), 400
    hashed = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                  (username, hashed, datetime.utcnow()))
        conn.commit()
        return jsonify({"message": "Signup successful!"}), 200
    except sqlite3.IntegrityError:
        return jsonify({"message": "Username already exists."}), 400
    finally:
        conn.close()

# User login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"message": "Username and password required."}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row and check_password_hash(row[0], password):
        return jsonify({"message": "Login successful!"}), 200
    return jsonify({"message": "Invalid credentials."}), 401

# Auto-open browser
def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
