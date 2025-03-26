import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# 1️⃣ Load the Dataset
file_path = "usda_food_nutrition.csv"  # Update with actual file path
df = pd.read_csv(file_path)

# 2️⃣ Select Relevant Columns
df = df[['Description', 'Calories', 'Protein', 'TotalFat', 'Carbohydrate']]

# 3️⃣ Handle Missing Values (Fill with Median)
df.fillna(df.median(numeric_only=True), inplace=True)

# 4️⃣ Encode Food Descriptions (Convert Text to Numbers)
label_encoder = LabelEncoder()
df['food_id'] = label_encoder.fit_transform(df['Description'])

# 5️⃣ Define Features (X) and Target (y)
X = df[['food_id']]
y = df[['Calories', 'Protein', 'TotalFat', 'Carbohydrate']]

# 6️⃣ Train-Test Split (90% Train, 10% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 7️⃣ Scale Features (Using MinMaxScaler)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8️⃣ Train MultiOutput Gradient Boosting Model
base_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
model = MultiOutputRegressor(base_model)  # 🔹 Fix for multiple outputs
model.fit(X_train_scaled, y_train)

# 9️⃣ Model Evaluation
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
print(f"MAE: {mae:.2f}, R² Score: {r2:.2f}")

# 🔟 Save Model & Encoders for Future Use
joblib.dump(model, "calorie_predictor.pkl")
joblib.dump(label_encoder, "food_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# 🆕 Prediction Function
def predict_nutrients(food_name):
    if food_name not in label_encoder.classes_:
        return "Food not found in dataset."
    
    food_id = label_encoder.transform([food_name])[0]
    food_scaled = scaler.transform(pd.DataFrame([[food_id]], columns=["food_id"]))  # Fix warning
    prediction = model.predict(food_scaled)[0]
    
    return {
        "Calories": round(prediction[0], 2),
        "Protein": round(prediction[1], 2),
        "Fat": round(prediction[2], 2),
        "Carbohydrates": round(prediction[3], 2),
    }

# 🔹 Example Prediction
#print("Example I/O:")
#food = "CHEESE,BLUE"

s=input("Enter a food item for your diet : ")
food=s

print(f"Nutrient Prediction for {food}: {predict_nutrients(food)}")

