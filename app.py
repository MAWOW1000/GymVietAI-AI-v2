import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify
from recommendation import NutritionPlanner, DietaryRestriction, Allergen, NutritionPlanFormatter
import cv2
import numpy as np
import torch
import time
try:
    from posture_analyzer import load_trained_model, get_validation_transform, predict_squat_posture
except ImportError:
    print("Error: Could not import from posture_analyzer. Check file location and imports.")
    load_trained_model = lambda path, device: None
    get_validation_transform = lambda: None
    predict_squat_posture = lambda frame, model, transform, device: {'error': 'Module not loaded'}

app = Flask(__name__)

# --- Configuration ---
MODELS_DIR = "./models"
UPLOAD_FOLDER = 'uploads' # temporary folder to save uploaded files
POSTURE_MODEL_PATH = os.path.join(MODELS_DIR, 'posture', 'squat_baseline_cnn_model.pth')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ensure the models directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

def load_posture_model():
    """Load the posture assessment model."""
    try:
        model = load_trained_model(POSTURE_MODEL_PATH, DEVICE)
        transform = get_validation_transform()
        print(f"Posture model loaded successfully on device: {DEVICE}")
        return model, transform
    except Exception as e:
        print(f"FATAL ERROR: Could not load posture assessment model. API endpoint will not function correctly. Error: {e}")
        return None, None

wo_models, wo_default_model = load_workout_models()
nutrition_model = load_nutrition_model()
squat_model, squat_validation_transform = load_posture_model()

# --- Helper functions ---
def validate_input(data, required_fields, data_types):
    errors = []
    for field in required_fields:
        value = data.get(field)
        if value in [None, "", "null"]:
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
    required_fields = ["Weight", "Height", "Age", "Gender", "Goal", "ActivityLevel", "restrictions", "allergens"]
    data_types = {
        "Weight": float,
        "Height": float,
        "Age": int,
        "Gender": str,
        "Goal": str,
        "ActivityLevel": str,
        "restrictions": list,
        "allergens": list
    }
    
    # Get optional dietary preferences from request
    dietary_restrictions = set(map(DietaryRestriction, data.get('restrictions', ['none'])))
    allergens = set(map(Allergen, data.get('allergens', ['none'])))
    
    # Validate input
    errors = validate_input(data, required_fields, data_types)
    if errors:
        return jsonify({"EC": 1001, "EM": ", ".join(errors), "DT": []}), 400
    
    # Predict
    try:
        input_df = pd.DataFrame([data])
        macro_predictions = nutrition_model.predict(input_df)[0]
        
        # Format macro targets for the planner
        macro_targets = {
            'calories': float(macro_predictions[0]),
            'protein': float(macro_predictions[1]),
            'carbs': float(macro_predictions[2]),
            'fat': float(macro_predictions[3])
        }
        
        food_df = pd.read_csv('./data/food_dataset_new.csv')
        
        # Create nutrition planner instance
        planner = NutritionPlanner(
            food_data=food_df,
            macro_targets=macro_targets,
            dietary_restrictions=dietary_restrictions,
            allergens=allergens
        )
        
        # Generate weekly plan
        weekly_plan = planner.generate_weekly_plan()
        weekly_nutrition = planner.calculate_weekly_nutrition(weekly_plan)
        
        # Format the plan
        formatter = NutritionPlanFormatter(planner)
        formatted_plan = formatter.format_weekly_plan(weekly_plan, weekly_nutrition)
        
        response_data = {
            "macro_targets": macro_targets,
            "weekly_plan": formatted_plan
        }
        
        return jsonify({
            "EC": 0,
            "EM": "",
            "DT": response_data
        }), 200
        
    except Exception as e:
        return jsonify({
            "EC": 500, 
            "EM": f"An unexpected error occurred: {str(e)}", 
            "DT": []
        }), 500
    
@app.route('/api/squat', methods=['POST'])
def predict_squat_posture_api():
    """
    API endpoint to predict squat posture ("Correct" or "Incorrect") from an uploaded video.
    Includes an overall confidence score in the response.
    Expects a POST request with a 'video' file part.
    Samples 3 frames (25%, 50%, 75%) and aggregates predictions.
    Returns a JSON response with standard EC, EM, DT format.
    """
    # Standard Error Codes
    EC_SUCCESS = 0
    EC_NO_FILE_PART = 4001
    EC_NO_FILE_SELECTED = 4002
    EC_INVALID_FILE_TYPE = 4003
    EC_MODEL_UNAVAILABLE = 5031
    EC_VIDEO_PROCESSING_ERROR = 5001
    EC_PREDICTION_ERROR = 5002
    EC_UNKNOWN = 5000

    # Check if the posture assessment model was loaded successfully
    if squat_model is None or squat_validation_transform is None:
         return jsonify({'EC': EC_MODEL_UNAVAILABLE, 'EM': 'Posture assessment model is not available.', 'DT': None}), 503

    # 1. Validate video file presence in the request
    if 'video' not in request.files:
        return jsonify({'EC': EC_NO_FILE_PART, 'EM': 'No video file part in the request', 'DT': None}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'EC': EC_NO_FILE_SELECTED, 'EM': 'No selected video file', 'DT': None}), 400

    # 2. Save the uploaded video file temporarily
    video_path = ""
    try:
        video_filename = file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(video_path)
        print(f"Video saved temporarily to: {video_path}")

        frame_predictions = [] # Stores "Correct"/"Incorrect" strings
        frame_confidences = [] # Stores confidence_incorrect floats
        final_prediction = "Correct"
        overall_confidence = None # Initialize overall confidence
        error_message = ""
        frame_analysis_details = []
        processing_error_occurred = False

        # 3. Process the video and perform prediction on sampled frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3:
             raise ValueError("Video too short to sample 3 frames")

        frame_indices = [
            max(0, int(total_frames * 0.25) - 1),
            max(0, int(total_frames * 0.50) - 1),
            max(0, int(total_frames * 0.75) - 1)
        ]
        frame_indices = sorted(list(set(frame_indices)))
        print(f"Total frames: {total_frames}, Sampling indices: {frame_indices}")

        for i, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                print(f"Processing frame {i+1} at index {frame_index}...")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = predict_squat_posture(frame_rgb, squat_model, squat_validation_transform, DEVICE)

                frame_detail = {'frame_index': frame_index}
                if result['error']:
                    print(f"Error predicting frame {i+1}: {result['error']}")
                    frame_detail['prediction'] = None
                    frame_detail['error'] = result['error']
                    processing_error_occurred = True
                else:
                    current_prediction = result['prediction']
                    current_confidence = result.get('confidence_incorrect', None)
                    frame_predictions.append(current_prediction)
                    if current_confidence is not None:
                         frame_confidences.append(current_confidence) # Store confidence for overall calculation

                    frame_detail['prediction'] = current_prediction
                    frame_detail['confidence_incorrect'] = current_confidence
                    print(f"Frame {i+1} prediction: {current_prediction} (Confidence Incorrect: {current_confidence:.4f})")
                frame_analysis_details.append(frame_detail)

            else:
                 print(f"Warning: Could not read frame at index {frame_index}")
                 frame_analysis_details.append({'frame_index': frame_index, 'prediction': None, 'error': 'Could not read frame'})
                 processing_error_occurred = True

        cap.release()

        # 4. Aggregate frame predictions AND Calculate Overall Confidence
        if not frame_predictions and processing_error_occurred:
            final_prediction = "Error"
            error_message = "Could not successfully process any frames from the video."
            return jsonify({'EC': EC_VIDEO_PROCESSING_ERROR, 'EM': error_message, 'DT': None}), 500
        elif not frame_predictions and not processing_error_occurred:
            final_prediction = "Undetermined"
            error_message = "No valid frames could be processed or read."
            return jsonify({'EC': EC_SUCCESS, 'EM': error_message, 'DT': {'overall_prediction': final_prediction, 'confidence': None}}), 200
        elif "Incorrect" in frame_predictions:
            final_prediction = "Incorrect"
            # Confidence for "Incorrect" is the max confidence_incorrect found
            if frame_confidences: # Check if list is not empty
                 overall_confidence = max(frame_confidences)
        else: # All processed frames are "Correct"
            final_prediction = "Correct"
            # Confidence for "Correct" is 1.0 - min(confidence_incorrect)
            if frame_confidences: # Check if list is not empty
                 overall_confidence = 1.0 - min(frame_confidences)

        # Add a general message if some frames had errors but we still got an overall prediction
        if processing_error_occurred and final_prediction != "Error":
            error_message = "Processed successfully, but some frames encountered errors during analysis."


    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        error_message = f'An unexpected error occurred during video processing: {str(e)}'
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if os.path.exists(video_path):
             try: os.remove(video_path); print(f"Removed temporary video file after error: {video_path}")
             except Exception as remove_err: print(f"Error removing temporary file {video_path} after error: {remove_err}")
        return jsonify({'EC': EC_VIDEO_PROCESSING_ERROR, 'EM': error_message, 'DT': None}), 500
    finally:
        # 5. Always attempt to remove the temporary video file
         if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Removed temporary video file: {video_path}")
            except Exception as remove_err:
                print(f"Error removing temporary file {video_path}: {remove_err}")


    # 6. Prepare and return the final JSON response with confidence
    response_data = {
        'overall_prediction': final_prediction,
        'confidence': round(overall_confidence, 4) if overall_confidence is not None else None, # Add formatted confidence score
        # 'frame_analysis': frame_analysis_details # per-frame details
    }

    return jsonify({
        'EC': EC_SUCCESS,
        'EM': error_message, # Will be empty if no errors/warnings occurred
        'DT': response_data
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
