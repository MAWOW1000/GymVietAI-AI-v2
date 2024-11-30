from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)

MODEL_DIR = "./models/exercise"
WORKOUT_PLAN_DIR = './models/exercise/workout_plans'

# Load the model
models = {
    "asia": joblib.load(os.path.join(MODEL_DIR, "random_forest_classifier_asian.pkl")),
    "europe": joblib.load(os.path.join(MODEL_DIR, "random_forest_classifier_european.pkl")),
}
default_model = models["asian"]

# Load workout plans from JSON files
workout_plans = {}
for filename in os.listdir(WORKOUT_PLAN_DIR):
    if filename.endswith(".json"):
        plan_number = int(filename.split("_")[1].split(".")[0])
        with open(os.path.join(WORKOUT_PLAN_DIR, filename)) as f:
            workout_plans[plan_number] = json.load(f)


@app.route('/', methods=['GET'])
def index():
    return """Welcome to the Workout Plan Prediction API! Use the /api/workout-plan endpoint with a POST request.
            Send a JSON object with the user's data to get a workout plan prediction.
            e.g. {'Gender': Male, 'Weight': 70, 'Height': 1.70, 'Age': 25}"""

@app.route('/api/workout-plan', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        features = {
            "Gender": str,
            "Weight": float,
            "Height": float,
            "Age": int
        }
        
        parsed_data = {}
        errors = []
        
        # Check and validate the request data
        if data is None:
            return jsonify({"error": "Invalid data format. Please send a JSON object."}), 400
        
        for feature, dtype in features.items():
            value = data.get(feature)
            if value is None:
                return jsonify({"error": f"Missing required field: {feature}"}), 400
            elif value in [None, "", "null"]:
                return jsonify({"error": f"Missing value found in field: {feature}"}), 400
            else:
                try:
                    parsed_data[feature] = dtype(value)
                    if feature in ["Weight", "Height", "Age"] and parsed_data[feature] <= 0:
                        errors.append(f"{feature} must be a positive number.")
                except ValueError:
                    errors.append(f"{feature} must be of type {dtype.__name__}")
        
        if errors:
            return jsonify({"error": f"Invalid data: {', '.join(errors)}"}), 400
        
        
        continent = data.get("continent", "asian").lower()  # default continent is asian
        model = models.get(continent, default_model)
        
        # separete the user profile data from the request
        user_profile = {feature: data[feature] for feature in features if feature in data}
        
        
        input_df = pd.DataFrame([user_profile])
        prediction = model.predict(input_df)[0]
        plan = workout_plans.get(prediction)

        if plan is None:
            return jsonify({"error": "No workout plan found for this prediction."}), 404
        return jsonify(plan)
    
    
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred."}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)