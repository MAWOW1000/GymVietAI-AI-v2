import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify, url_for
import json
import numpy as np
from recommendation import NutritionPlanner, DietaryRestriction, Allergen, NutritionPlanFormatter
import cv2
import torch
import time
import uuid
try:
    from posture_analyzer import load_trained_model, get_validation_transform, predict_squat_posture, analyze_squat_errors
except ImportError:
    print("FATAL ERROR: Could not import from posture_analyzer. Check file location and imports.")
    load_trained_model = lambda path, device: None
    get_validation_transform = lambda: None
    predict_squat_posture = lambda frame, model, transform, device: {'error': 'Module not loaded'}
    analyze_squat_errors = lambda frame: ['MODULE_LOAD_ERROR']

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

# --- Configuration ---
MODELS_DIR = "./models"
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ERROR_FRAMES_SUBDIR = 'error_frames'
ERROR_FRAMES_FOLDER = os.path.join(STATIC_FOLDER, ERROR_FRAMES_SUBDIR)
POSTURE_MODEL_PATH = os.path.join(MODELS_DIR, 'posture', 'squat_baseline_cnn_model.pth')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# Ensure the models directory exists
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(ERROR_FRAMES_FOLDER): os.makedirs(ERROR_FRAMES_FOLDER)
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    API endpoint to predict squat posture and identify specific errors from video.
    Returns overall prediction, confidence, and detailed errors including URLs to error frames.
    """
    # Standard Error Codes
    EC_SUCCESS = 0
    EC_NO_FILE_PART = 4001
    EC_NO_FILE_SELECTED = 4002
    EC_INVALID_FILE_TYPE = 4003
    EC_MODEL_UNAVAILABLE = 5031
    EC_VIDEO_PROCESSING_ERROR = 5001
    EC_UNKNOWN = 5000

    # Constants
    NUM_FRAMES_TO_SAMPLE = 20

    # Check if model is loaded
    if squat_model is None or squat_validation_transform is None:
         return jsonify({'EC': EC_MODEL_UNAVAILABLE, 'EM': 'Posture assessment model is not available.', 'DT': None}), 503

    # 1. Validate request
    if 'video' not in request.files:
        return jsonify({'EC': EC_NO_FILE_PART, 'EM': 'No video file part in the request', 'DT': None}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'EC': EC_NO_FILE_SELECTED, 'EM': 'No selected video file', 'DT': None}), 400
    if not allowed_file(file.filename):
        return jsonify({'EC': EC_INVALID_FILE_TYPE, 'EM': f"Invalid video file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}", 'DT': None}), 400

    # 2. Save temporary video file with unique name
    video_path = ""
    unique_id = str(uuid.uuid4()) # Generate unique ID for this request
    original_filename, original_ext = os.path.splitext(file.filename)
    video_filename = f"{unique_id}{original_ext}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

    try:
        file.save(video_path)
        print(f"Video saved temporarily to: {video_path}")

        # --- Variables for results ---
        cnn_frame_predictions = {}
        all_confidences_incorrect = []
        incorrect_frame_indices = []
        detected_errors_details = [] # List of dicts: {'frame_index': idx, 'errors': ['KIE'], 'error_frame_url': url}
        processing_error_message = ""
        final_overall_prediction = "Correct"
        overall_confidence = None

        # 3. Process video: Sample frames and run CNN
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3: # Need at least a few frames
             print(f"Warning: Video has only {total_frames} frames. Sampling available frames.")
             frame_indices = list(range(total_frames)) if total_frames > 0 else []
        elif total_frames < NUM_FRAMES_TO_SAMPLE:
            print(f"Warning: Video shorter than {NUM_FRAMES_TO_SAMPLE} frames. Sampling all available frames.")
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES_TO_SAMPLE, dtype=int)

        if not frame_indices.size > 0: # Check if frame_indices is not empty
             raise ValueError("Could not determine frames to sample.")

        print(f"Total frames: {total_frames}, Sampling {len(frame_indices)} indices: {frame_indices.tolist()}")

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                print(f"Processing frame at index {frame_index} (CNN)...")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cnn_result = predict_squat_posture(frame_rgb, squat_model, squat_validation_transform, DEVICE)

                cnn_prediction = None
                if cnn_result['error']:
                    print(f"  Error predicting frame {frame_index} (CNN): {cnn_result['error']}")
                    cnn_frame_predictions[frame_index] = None
                else:
                    cnn_prediction = cnn_result['prediction']
                    cnn_confidence = cnn_result.get('confidence_incorrect', None)
                    cnn_frame_predictions[frame_index] = cnn_prediction
                    if cnn_confidence is not None: all_confidences_incorrect.append(cnn_confidence)
                    print(f"  CNN Prediction: {cnn_prediction} (Confidence Incorrect: {cnn_confidence:.4f})")
                    if cnn_prediction == "Incorrect":
                        incorrect_frame_indices.append((frame_index, frame)) # Store index AND frame BGR for saving later
            else:
                 print(f"Warning: Could not read frame at index {frame_index}")
                 cnn_frame_predictions[frame_index] = None

        # 4. Determine Overall Prediction based on CNN results
        if not any(pred is not None for pred in cnn_frame_predictions.values()):
             processing_error_message = "Could not process any frames from the video."
             final_overall_prediction = "Error" # Mark as Error if no frames were processed
             # Fall through to finally block for cleanup, return error later
        elif "Incorrect" in cnn_frame_predictions.values():
            final_overall_prediction = "Incorrect"
        else:
            final_overall_prediction = "Correct"

        # Calculate overall confidence
        if all_confidences_incorrect:
            if final_overall_prediction == "Incorrect": overall_confidence = float(max(all_confidences_incorrect))
            elif final_overall_prediction == "Correct": overall_confidence = float(1.0 - min(all_confidences_incorrect))


        # 5. Perform Detailed Error Analysis AND Save Error Frames if overall prediction is Incorrect
        if final_overall_prediction == "Incorrect":
            print(f"\nAnalyzing {len(incorrect_frame_indices)} incorrect frames for specific errors...")
            for frame_index, frame_bgr in incorrect_frame_indices: # Iterate through stored incorrect frames
                 frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Convert the stored frame
                 print(f"  Analyzing frame {frame_index} (Rule-based)...")
                 specific_errors = analyze_squat_errors(frame_rgb)
                 print(f"    Detected specific errors: {specific_errors}")

                 if specific_errors: # Only save/report if rule-based analysis finds errors
                      # Save error frame
                      error_frame_filename = f"{unique_id}_frame_{frame_index}_error.jpg"
                      error_frame_path = os.path.join(ERROR_FRAMES_FOLDER, error_frame_filename)
                      save_success = cv2.imwrite(error_frame_path, frame_bgr) # Save the original BGR frame

                      error_frame_url = None
                      if save_success:
                           print(f"    Saved error frame to: {error_frame_path}")
                           try:
                               # Generate URL for the saved static file
                               error_frame_url = url_for('static', filename=f'{ERROR_FRAMES_SUBDIR}/{error_frame_filename}', _external=True)
                           except RuntimeError:
                                print("    Warning: Could not generate external URL for error frame (Flask context missing?). Returning relative path.")
                                # Provide a relative path if url_for fails outside request context (e.g., during testing)
                                error_frame_url = f"/{STATIC_FOLDER}/{ERROR_FRAMES_SUBDIR}/{error_frame_filename}"

                      else:
                           print(f"    Error: Failed to save error frame {error_frame_path}")

                      # Add error details to the list
                      detected_errors_details.append({
                           'frame_index': int(frame_index),  # Explicitly convert to Python int
                           'errors': [str(error) for error in specific_errors],  # Convert each error to string
                           'error_frame_url': error_frame_url
                      })

        # Release video capture outside the loop
        if 'cap' in locals() and cap.isOpened(): cap.release()

        # Check if any frame processing errors occurred during CNN prediction phase
        if not all(pred is not None for pred in cnn_frame_predictions.values()):
             processing_error_message = "Processed video, but some frames encountered errors during initial prediction."


    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        error_message = f'An unexpected error occurred during video processing: {str(e)}'
        # Cleanup logic in finally block handles cap release and file removal

        return jsonify({'EC': EC_VIDEO_PROCESSING_ERROR, 'EM': error_message, 'DT': None}), 500
    finally:
        # Always attempt to remove the temporary video file
         if os.path.exists(video_path):
            try: os.remove(video_path); print(f"Removed temporary video file: {video_path}")
            except Exception as remove_err: print(f"Error removing temporary file {video_path}: {remove_err}")

    # 6. Prepare Final Response Data
    response_data = {
        'overall_prediction': final_overall_prediction,
        'confidence': round(overall_confidence, 4) if overall_confidence is not None else None,
        'detected_errors': detected_errors_details # Return detailed errors list (empty if prediction is Correct)
    }
    if processing_error_message:
        response_data['processing_message'] = processing_error_message

    return jsonify({
        'EC': EC_SUCCESS,
        'EM': "",
        'DT': response_data
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
