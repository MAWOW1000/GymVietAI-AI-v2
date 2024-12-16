from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import numpy as np
from recommendation import NutritionPlanner, NutritionPlanFormatter, DietaryRestriction, Allergen

app = Flask(__name__)

# --- Configuration ---
MODELS_DIR = "./models"
# ERROR_CODES = {
#     1001: "Invalid input data",
#     1002: "Model not found",
#     500: "An unexpected error occurred",
# }

# --- Load models ---
def load_workout_models():
    try:
        models = {
            "asia": joblib.load(os.path.join(MODELS_DIR, "./exercise/random_forest_classifier_asian.pkl")),
            "europe": joblib.load(os.path.join(MODELS_DIR, "./exercise/random_forest_classifier_european.pkl")),
        }
        default_model = models["asia"]
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        exit(1)
    return models, default_model

def load_nutrition_model():
    try:
        return joblib.load(os.path.join(MODELS_DIR, "./nutrition/random_forest_regressor.pkl"))
    except FileNotFoundError as e:
        print(f"Error loading nutrition model: {e}")
        exit(1)

wo_models, wo_default_model = load_workout_models()
nutrition_model = load_nutrition_model()

# --- Helper functions ---
def validate_input(data, required_fields, data_types):
    errors = []
    for field in required_fields:
        value = data.get(field)
        if value in [None, "", "null"] and field not in ["DietaryRestrictions", "Allergens"]:
            errors.append(f"Missing required field: {field}")
        elif field == "Gender":
            try:
                data[field] = str(value).strip().capitalize()
                if data[field] not in ["Male", "Female"]:
                    errors.append(f"Invalid value for {field}: Gender must be 'Male' or 'Female'")
            except (ValueError, TypeError):
                errors.append(f"Invalid data type for {field}: Expected {data_types[field].__name__}")
        elif field == "Goal":
            try:
                data[field] = str(value)
                if data[field] not in ["Loss Weight", "Stay Fit", "Muscle Gain"]:
                    errors.append(f"Invalid value for {field}: Goal must be 'Loss Weight', 'Stay Fit', or 'Muscle Gain'")
            except (ValueError, TypeError):
                errors.append(f"Invalid data type for {field}: Expected {data_types[field].__name__}")
        elif field == "ActivityLevel":
            try:
                data[field] = str(value).strip().title()
                valid_activity_levels = ["Sedentary", "Lightly Active", "Moderately Active", "Active", "Very Active"]
                if data[field] not in valid_activity_levels:
                    errors.append(f"Invalid value for {field}: ActivityLevel must be one of {', '.join(valid_activity_levels)}")
            except (ValueError, TypeError):
                errors.append(f"Invalid data type for {field}: Expected a string")
        elif field in ["DietaryRestrictions", "Allergens"]:
            if value is not None and not isinstance(value, list):
                errors.append(f"Invalid data type for {field}: Expected a list")
        else:
            try:
                if field in ["Weight", "Height", "Age"]:
                    data[field] = float(value) if field != "Age" else int(value)
                else:
                    data[field] = str(value).strip()
            except (ValueError, TypeError):
                errors.append(f"Invalid data type for {field}: Expected {data_types[field].__name__}")
                
    # Validate numeric fields          
    for field in ["Weight", "Height", "Age"]:
        value = data.get(field)
        if isinstance(value, (int, float)) and value <= 0:
            errors.append(f"{field} must be a positive number")
        elif field in "Weight" and value > 300:
            errors.append(f"{field} must be less than 300")
        elif field == "Height" and value > 2.5:
            errors.append(f"{field} must be less than or equal to 2.5 meters")
        elif field == "Age" and value > 120:
            errors.append(f"{field} must be less than 120")
    return errors

# --- API routes ---
@app.route('/', methods=['GET'])
def index():
    return """Welcome to the GymVietAI API!
    Use the /api/workout-plan endpoint with a POST request.
    Send a JSON object with the user's data to get a workout plan prediction.
    e.g. {'Gender': 'Male/Female', 'Weight': 70, 'Height': 1.70, 'Age': 25}
    
    Use the /api/nutrition-plan endpoint with a POST request.
    Send a JSON object with the user's data to get a nutrition plan prediction.
    e.g. {'Weight': 70, 'Height': 1.70, 'Age': 25, 'Gender': 'Male/Female', 'Goal': 'Loss Weight/Stay Fit/Muscle Gain', 'ActivityLevel': 'Sedentary/Lightly Active/Moderately Active/Active/Very Active'}
    Output will be a list of macronutrient targets [calories, protein, carbs, fat].
    """

@app.route('/api/workout-plan', methods=['POST'])
def predict():
    data = request.get_json()
    required_fields = ["Gender", "Weight", "Height", "Age", "continent"]
    data_types = {
        "Gender": str,
        "Weight": float,
        "Height": float,
        "Age": int,
        "continent": str,
    }

    # Validate input
    errors = validate_input(data, required_fields, data_types)
    if errors:
        return jsonify({"EC": 1001, "EM": ", ".join(errors), "DT": []}), 400

    # Predict
    try:
        continent = data["continent"].lower()
        model = wo_models.get(continent, wo_default_model)
        input_df = pd.DataFrame([data])
        prediction = int(model.predict(input_df)[0])
        return jsonify({"EC": 0, "EM": "", "DT": [prediction]}), 200
    except KeyError:
        return jsonify({"EC": 1002, "EM": f"Model not found for continent: {data.get('continent')}", "DT": []}), 404
    except Exception as e:
        return jsonify({"EC": 500, "EM": f"An unexpected error occurred: {str(e)}", "DT": []}), 500

@app.route('/api/nutrition-plan', methods=['POST'])
def predict_nutrition():
    data = request.get_json()
    required_fields = ["Weight", "Height", "Age", "Gender", "Goal", "ActivityLevel", "DietaryRestrictions", "Allergens"]
    data_types = {
        "Weight": float,
        "Height": float,
        "Age": int,
        "Gender": str,
        "Goal": str,
        "ActivityLevel": str,
        "DietaryRestrictions": list,
        "Allergens": list
    }
    
    # Validate input
    errors = validate_input(data, required_fields, data_types)
    if errors:
        return jsonify({"EC": 1001, "EM": ", ".join(errors), "DT": []}), 400
    
    # Validate and convert dietary restrictions
    try:
        dietary_restrictions = {DietaryRestriction[r.upper()] for r in data.get("DietaryRestrictions", [])}
        if not dietary_restrictions:
            dietary_restrictions = {DietaryRestriction.NONE}
    except KeyError as e:
        return jsonify({
            "EC": 1001, 
            "EM": f"Invalid dietary restriction. Valid values are: {[r.name for r in DietaryRestriction]}", 
            "DT": []
        }), 400
        
    # Validate and convert allergens
    try:
        allergens = {Allergen[a.upper()] for a in data.get("Allergens", [])}
        if not allergens:
            allergens = {Allergen.NONE}
    except KeyError as e:
        return jsonify({
            "EC": 1001, 
            "EM": f"Invalid allergen. Valid values are: {[a.name for a in Allergen]}", 
            "DT": []
        }), 400
    
    # Predict
    try:
        input_df = pd.DataFrame([data])
        macro_targets = nutrition_model.predict(input_df)[0]
        macro_targets = {
            'calories': float(macro_targets[0]),
            'protein': float(macro_targets[1]),
            'carbs': float(macro_targets[2]),
            'fat': float(macro_targets[3])
        }
        
        # Load food data
        df = pd.read_csv('./data/food_dataset_new.csv')
        
        # Create planner instance with dietary restrictions and allergens
        planner = NutritionPlanner(
            df,
            macro_targets,
            dietary_restrictions=dietary_restrictions,
            allergens=allergens
        )
        
        # Generate weekly plan
        weekly_plan = planner.generate_weekly_plan()
        weekly_nutrition = planner.calculate_weekly_nutrition(weekly_plan)
        
        # Format the plan
        formatter = NutritionPlanFormatter(planner)
        formatted_plan = formatter.format_weekly_plan(weekly_plan, weekly_nutrition)
        
        # Return the formatted plan directly as DT
        return jsonify({"EC": 0, "EM": "", "DT": formatted_plan}), 200
    except Exception as e:
        return jsonify({"EC": 500, "EM": f"An unexpected error occurred: {str(e)}", "DT": []}), 500
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)
