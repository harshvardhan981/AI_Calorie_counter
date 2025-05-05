import pickle

file_path = "calorie_predictor.pkl"

try:
    with open(file_path, "rb") as f:
        data = f.read()
    print(f"File size: {len(data)} bytes")
except Exception as e:
    print(f"Error reading file: {e}")
