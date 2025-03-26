from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS  # Optional: To allow frontend communication

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load saved models
try:
    model = joblib.load("calorie_predictor.pkl")
    label_encoder = joblib.load("food_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    model, label_encoder, scaler = None, None, None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask is running!"})

def predict_nutrients(food_name):
    try:
        if label_encoder is None or model is None or scaler is None:
            return {"error": "Model files not loaded properly."}

        if food_name not in label_encoder.classes_:
            return {"error": f"Food '{food_name}' not found in dataset."}
        
        # Encode and scale input
        food_id = label_encoder.transform([food_name])[0]
        food_scaled = scaler.transform(pd.DataFrame([[food_id]], columns=["food_id"]))
        
        # Predict nutrients
        prediction = model.predict(food_scaled)[0]  # Ensure it is a 1D array
        
        return {
            "Food": food_name,
            "Calories": round(prediction[0], 2),
            "Protein": round(prediction[1], 2),
            "Fat": round(prediction[2], 2),
            "Carbohydrates": round(prediction[3], 2),
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        food_name = data.get("food_name", "").strip()
        
        if not food_name:
            return jsonify({"error": "Invalid input. Provide a food name."}), 400

        result = predict_nutrients(food_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)