import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import os
import math
import mediapipe as mp

# --- 1. Model Architecture Definition ---
def get_squat_model_architecture() -> nn.Module:
    """Defines the modified ResNet18 architecture."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Output 1 neuron
    return model

# --- 2. Validation/Inference Transform Pipeline Definition ---
def get_validation_transform() -> A.Compose:
    """Defines the Albumentations validation/inference transform pipeline."""
    val_transform = A.Compose([
        A.Resize(height=400, width=400),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return val_transform

# --- 3. Function to Load Trained Model Weights ---
def load_trained_model(model_weights_path: str, device: torch.device) -> nn.Module:
    """Loads the trained model weights onto the specified device."""
    print(f"Loading model weights from: {model_weights_path}")
    model = get_squat_model_architecture()
    try:
        checkpoint = torch.load(model_weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Loaded checkpoint dictionary. Extracting 'model_state_dict'.")
        elif isinstance(checkpoint, dict) and not ('model_state_dict' in checkpoint) and all(k in model.state_dict() for k in checkpoint.keys()):
             state_dict = checkpoint
             print("Loaded state_dict directly (legacy saving format).")
        else:
            state_dict = checkpoint
            print("Loaded state_dict directly.")

        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_weights_path}")
        raise
    except KeyError:
        print(f"Error: Key 'model_state_dict' not found in the loaded file at {model_weights_path}.")
        raise
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    model.to(device)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    return model

# --- 4. Function to Preprocess a Single Image Frame (for CNN) ---
def preprocess_frame(frame_rgb: np.ndarray, transform: A.Compose, device: torch.device) -> torch.Tensor | None:
    """Preprocesses a single RGB image frame using the validation transform."""
    try:
        augmented = transform(image=frame_rgb)
        image_tensor = augmented['image']
        image_tensor = image_tensor.unsqueeze(0) # Add batch dimension
        image_tensor = image_tensor.to(device)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

# --- 5. Main CNN Prediction Function ---
def predict_squat_posture(frame_rgb: np.ndarray, model: nn.Module, transform: A.Compose, device: torch.device) -> dict:
    """Predicts posture ('Correct'/'Incorrect') for a single frame using the CNN."""
    processed_frame_tensor = preprocess_frame(frame_rgb, transform, device)
    if processed_frame_tensor is None:
        return {'prediction': None, 'confidence_incorrect': None, 'error': 'Frame preprocessing failed'}
    try:
        with torch.no_grad():
            outputs = model(processed_frame_tensor)
        probs = torch.sigmoid(outputs)
        probability_incorrect = probs.item()
        prediction = "Incorrect" if probability_incorrect > 0.6 else "Correct"
        return {'prediction': prediction, 'confidence_incorrect': probability_incorrect, 'error': None}
    except Exception as e:
        print(f"Error during CNN prediction: {e}")
        return {'prediction': None, 'confidence_incorrect': None, 'error': f'CNN Prediction failed: {str(e)}'}

# --- 6. Function to Extract Keypoints using MediaPipe Pose ---
def extract_keypoints_mediapipe(frame_rgb: np.ndarray) -> list | None:
    """Extracts 33 2D pose landmarks from an RGB image frame using MediaPipe Pose."""
    mp_pose = mp.solutions.pose
    keypoints_list = None
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            height, width, _ = frame_rgb.shape
            keypoints_list = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                keypoints_list.append((x, y)) # Store as tuple (x, y)
    return keypoints_list

# --- 7. Helper Function to Calculate Angle ---
def calculate_angle(p1, p2):
    """Calculates the angle (degrees) of the vector p1 -> p2 relative to the horizontal axis."""
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1] # y increases downwards in image coordinates
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# --- 8. Rule-Based Error Evaluation Functions ---
# --- KIE (Knee Inward Excursion) ---
def evaluate_kie_error(keypoints: list, kie_ratio_threshold=0.85) -> bool:
    """
    Evaluates KIE error using the ratio of knee distance to ankle distance.
    More robust to varying scales and potentially different views than absolute distance.

    Args:
        keypoints (list): List of MediaPipe Pose keypoints.
        kie_ratio_threshold (float): The threshold for the ratio (knee_distance / ankle_distance).
                                     If the ratio is BELOW this threshold, KIE is detected.
                                     (e.g., 0.85 means knees are significantly closer than ankles)

    Returns:
        bool: True if KIE error is detected, False otherwise.
    """
    if not keypoints or len(keypoints) < 29:  # Need knees and ankles
        print("Warning: Insufficient keypoints for KIE ratio evaluation.")
        return False
    try:
        left_knee = keypoints[25]
        right_knee = keypoints[26]
        left_ankle = keypoints[27]
        right_ankle = keypoints[28]

        # Calculate horizontal distance between knees
        knee_distance_x = abs(left_knee[0] - right_knee[0])

        # Calculate horizontal distance between ankles
        ankle_distance_x = abs(left_ankle[0] - right_ankle[0])

        # Avoid division by zero if ankles are very close or occluded
        if ankle_distance_x < 10:  # Use a small epsilon threshold instead of zero
            print("Warning: Ankle distance is very small, cannot reliably calculate KIE ratio.")
            return False  # Cannot determine KIE if ankle distance is unreliable

        # Calculate the ratio
        knee_ankle_ratio = knee_distance_x / ankle_distance_x
        print(f"  KIE Check: KneeDist={knee_distance_x:.1f}, AnkleDist={ankle_distance_x:.1f}, Ratio={knee_ankle_ratio:.2f}")  # Debug print

        # Check if the ratio is below the threshold
        is_kie_error = (knee_ankle_ratio < kie_ratio_threshold)

        return is_kie_error
    except IndexError:
        print("Warning: Required keypoints (knees/ankles) for KIE evaluation not found.")
        return False
    except Exception as e:
        print(f"Error during KIE evaluation: {e}")
        return False

# --- KFE (Knee Forward Excursion) ---
def evaluate_kfe_error(keypoints: list, kfe_threshold_distance_knee_nose=50) -> bool:
    """Evaluates Knee Forward Excursion (KFE) error based on keypoints."""
    if not keypoints or len(keypoints) < 27: # Need knees, nose
        return False
    try:
        left_knee = keypoints[25]
        right_knee = keypoints[26]
        nose = keypoints[0] # Using nose as proxy for toes

        # Check if left knee is significantly forward of nose horizontally
        left_knee_nose_diff_x = left_knee[0] - nose[0]
        condition_1 = (left_knee_nose_diff_x > kfe_threshold_distance_knee_nose)

        # Check if right knee is significantly forward of nose horizontally
        right_knee_nose_diff_x = right_knee[0] - nose[0] # Logic assumes front/side view where higher x is 'forward'
        condition_2 = (right_knee_nose_diff_x > kfe_threshold_distance_knee_nose)
        # Note: This logic is simplistic and depends heavily on camera angle.
        # A better approach might involve comparing knee X to ankle X if available.

        is_kfe_error = (condition_1 or condition_2)
        return is_kfe_error
    except IndexError:
        print("Warning: Required keypoints for KFE evaluation not found.")
        return False


# --- SS (Shallow Squat) ---
def evaluate_ss_error(keypoints: list, ss_threshold_hip_knee_y=20, ss_threshold_angle_from_vertical=30) -> bool:
    """
    Evaluates Shallow Squat (SS) error based on keypoints.
    Combines vertical hip position and thigh angle relative to vertical.

    Args:
        keypoints (list): List of MediaPipe Pose keypoints.
        ss_threshold_hip_knee_y (int): Vertical distance threshold (pixels). If hip is less than
                                       this amount below the knee, it might be shallow.
        ss_threshold_angle_from_vertical (int): Angle threshold (degrees). If the absolute thigh angle
                                                is within this threshold from vertical (90 deg),
                                                it's considered too steep (shallow).

    Returns:
        bool: True if Shallow Squat error is detected, False otherwise.
    """
    if not keypoints or len(keypoints) < 27: # Need hips, knees
        print("Warning: Insufficient keypoints for SS evaluation.")
        return False
    try:
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_knee = keypoints[25]
        right_knee = keypoints[26]

        # --- Condition 1: Hips not sufficiently below knees vertically (y increases downwards) ---
        left_hip_knee_diff_y = left_hip[1] - left_knee[1]
        right_hip_knee_diff_y = right_hip[1] - right_knee[1]
        # If diff_y is less than threshold, hip is NOT low enough
        condition_1 = (left_hip_knee_diff_y < ss_threshold_hip_knee_y or right_hip_knee_diff_y < ss_threshold_hip_knee_y)
        # print(f"  SS Check Cond1: L_diffY={left_hip_knee_diff_y:.1f}, R_diffY={right_hip_knee_diff_y:.1f}, Threshold={ss_threshold_hip_knee_y}") # Debug

        # --- Condition 2: Thigh angle is too close to vertical (too steep) ---
        left_thigh_angle = calculate_angle(left_hip, left_knee)
        right_thigh_angle = calculate_angle(right_hip, right_knee)

        # Check how close the absolute angle is to 90 degrees (vertical)
        condition_2_left = abs(abs(left_thigh_angle) - 90) < ss_threshold_angle_from_vertical
        condition_2_right = abs(abs(right_thigh_angle) - 90) < ss_threshold_angle_from_vertical
        condition_2 = (condition_2_left or condition_2_right)
        # print(f"  SS Check Cond2: L_Angle={left_thigh_angle:.1f}, R_Angle={right_thigh_angle:.1f}, Threshold_Vert={ss_threshold_angle_from_vertical}") # Debug

        # --- Final Decision ---
        is_ss_error = (condition_1 or condition_2)
        # print(f"  SS Check Result: Cond1={condition_1}, Cond2={condition_2}, Final={is_ss_error}") # Debug

        # --- Acknowledge Limitations ---
        # print("  Note: SS evaluation from rear view has limitations regarding precise hip crease and thigh parallelism.")

        return is_ss_error
    except IndexError:
        print("Warning: Required keypoints (hips/knees) for SS evaluation not found.")
        return False
    except Exception as e:
        print(f"Error during SS evaluation: {e}")
        return False


# --- 9. Main Function for Detailed Error Analysis ---
def analyze_squat_errors(frame_rgb: np.ndarray) -> list:
    """
    Analyzes a single frame for specific Squat errors (KIE, KFE, SS) using rule-based logic.

    Args:
        frame_rgb (np.ndarray): Input image frame (HWC format, RGB channels).

    Returns:
        list: A list of detected error codes (e.g., ['KFE', 'SS']). Returns empty list
              if no errors detected or pose/keypoints not found.
    """
    detected_errors = []
    keypoints = extract_keypoints_mediapipe(frame_rgb)

    if not keypoints:
        print("Warning: Could not extract keypoints for detailed error analysis.")
        return [] # Return empty list if no keypoints

    # Call individual rule-based error evaluation functions
    if evaluate_kie_error(keypoints):
        detected_errors.append("Knee Inward Excursion (KIE)")
    # if evaluate_kfe_error(keypoints):
    #     detected_errors.append("Knee Forward Excursion (KFE)")
    if evaluate_ss_error(keypoints):
        detected_errors.append("Shallow Squat")

    return detected_errors


# --- 10. Standalone Testing Block ---
# if __name__ == '__main__':
    print("Running posture_analyzer.py as main script for testing...")

    # --- Configuration for Testing ---
    DEFAULT_MODEL_PATH = './models/posture/squat_baseline_cnn_model.pth'
    DEFAULT_TEST_IMAGE_PATH = './data/image_test/f3c6e17c-a8c5-5ec0-8f81-7257ce7fb229.jpg'
    TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device for testing: {TEST_DEVICE}")

    # --- Load Model and Transform ---
    try:
        model_path = os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)
        test_image_path = os.getenv('TEST_IMAGE_PATH', DEFAULT_TEST_IMAGE_PATH)

        test_cnn_model = load_trained_model(model_path, TEST_DEVICE)
        test_transform = get_validation_transform()

        # --- Load Test Image ---
        if not os.path.exists(test_image_path):
             print(f"Error: Test image not found at {test_image_path}.")
        else:
            test_frame_bgr = cv2.imread(test_image_path)
            if test_frame_bgr is None:
                print(f"Error: Could not read test image from {test_image_path}")
            else:
                test_frame_rgb = cv2.cvtColor(test_frame_bgr, cv2.COLOR_BGR2RGB)

                # --- Perform CNN Prediction First ---
                cnn_result = predict_squat_posture(test_frame_rgb, test_cnn_model, test_transform, TEST_DEVICE)
                print("\n--- Test CNN Prediction Result ---")
                if cnn_result['error']:
                    print(f"Error: {cnn_result['error']}")
                else:
                    print(f"CNN Predicted Posture: {cnn_result['prediction']}")
                    print(f"CNN Confidence (Incorrect): {cnn_result['confidence_incorrect']:.4f}")

                    # --- Perform Detailed Error Analysis if CNN predicts Incorrect ---
                    if cnn_result['prediction'] == "Incorrect":
                        print("\n--- Test Detailed Error Analysis Result ---")
                        detailed_errors = analyze_squat_errors(test_frame_rgb)
                        if not detailed_errors:
                             print("No specific errors detected by rule-based analysis (or keypoints not found).")
                        else:
                             print(f"Detected Specific Errors: {detailed_errors}")
                    else:
                        print("\nSkipping detailed error analysis as CNN prediction is 'Correct'.")


    except Exception as e:
        print(f"An error occurred during standalone testing: {e}")
        import traceback
        traceback.print_exc()