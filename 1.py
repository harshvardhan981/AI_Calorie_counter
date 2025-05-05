import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
file_path = "usda_food_nutrition.csv"
try:
    df = pd.read_csv(file_path)
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    exit()

# Select relevant columns
df = df[['Description', 'Calories', 'Protein', 'TotalFat', 'Carbohydrate']]

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode food descriptions
label_encoder = LabelEncoder()
df['food_id'] = label_encoder.fit_transform(df['Description'])

# Define features and target
X = df[['food_id']]
y = df[['Calories', 'Protein', 'TotalFat', 'Carbohydrate']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
base_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
logging.info(f"Model trained successfully. MAE: {mae:.2f}, R² Score: {r2:.2f}")

# Save models
joblib.dump(model, "calorie_predictor.pkl")
joblib.dump(label_encoder, "food_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
logging.info("Models saved successfully.")

def predict_nutrients(food_name):
    if food_name not in label_encoder.classes_:
        return {"error": "Food not found in dataset."}
    
    food_id = label_encoder.transform([food_name])[0]
    food_scaled = scaler.transform(pd.DataFrame([[food_id]], columns=["food_id"]))
    prediction = model.predict(food_scaled)[0]
    
    return {
        "Food": food_name,
        "Calories": round(prediction[0], 2),
        "Protein": round(prediction[1], 2),
        "Fat": round(prediction[2], 2),
        "Carbohydrates": round(prediction[3], 2),
        "Suggestion": get_suggestion(prediction[0])
    }

def get_suggestion(calories):
    if calories < 200:
        return "This is a low-calorie food. Consider pairing it with a protein source."
    elif calories < 500:
        return "Moderate-calorie food. It can be part of a balanced meal."
    else:
        return "High-calorie food. Consider portion control."

if __name__ == "__main__":
    food = input("Enter a food item: ")
    print(predict_nutrients(food))
