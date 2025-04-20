import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. Model Architecture Definition (Must match training) ---
def get_squat_model_architecture() -> nn.Module:
    """
    Defines the modified ResNet18 architecture used for Squat classification.

    Returns:
        torch.nn.Module: The ResNet18 model with the final classification layer modified.
    """
    model = models.resnet18(weights=None) # Load architecture without pre-trained weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Output 1 neuron for binary classification (logits)
    return model

# --- 2. Validation/Inference Transform Pipeline Definition ---
def get_validation_transform() -> A.Compose:
    """
    Defines the Albumentations transformation pipeline used for validation and inference.
    Ensures consistency with the validation transforms used during training.

    Returns:
        albumentations.Compose: The validation transformation pipeline.
    """
    val_transform = A.Compose([
        A.Resize(height=224, width=224), # Ensure initial consistent size
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
        ToTensorV2(),
    ])
    return val_transform

# --- 3. Function to Load Trained Model Weights ---
def load_trained_model(model_weights_path: str, device: torch.device) -> nn.Module:
    """
    Loads the trained Squat classification model weights onto the specified device.
    Handles cases where the .pth file contains a dictionary with the state_dict.

    Args:
        model_weights_path (str): Path to the saved model weights (.pth file).
        device (torch.device): The device (CPU or GPU) to load the model onto.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.

    Raises:
        FileNotFoundError: If the model weights file is not found.
        KeyError: If 'model_state_dict' key is missing in the loaded checkpoint.
        Exception: For other potential errors during model loading.
    """
    print(f"Loading model weights from: {model_weights_path}")
    model = get_squat_model_architecture() # Initialize the base architecture
    try:
        # Load the entire checkpoint dictionary first
        checkpoint = torch.load(model_weights_path, map_location=device)

        # --- CORRECTED LOGIC: Extract state_dict ---
        # Check if the loaded object is a dictionary and contains 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Loaded checkpoint dictionary. Extracting 'model_state_dict'.")
        # Check if the loaded object is directly the state_dict (legacy saving method)
        elif isinstance(checkpoint, dict) and not ('model_state_dict' in checkpoint) and all(k in model.state_dict() for k in checkpoint.keys()):
             state_dict = checkpoint
             print("Loaded state_dict directly (legacy saving format).")
        else:
            # Assume the loaded object IS the state_dict itself (if not a dict with the key)
            # This covers the standard torch.save(model.state_dict(), PATH) case
            state_dict = checkpoint
            print("Loaded state_dict directly.")
            # Optional: Add more robust checks here if needed

        # Load the extracted state dictionary into the model
        model.load_state_dict(state_dict)

    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_weights_path}")
        raise
    except KeyError as e:
        print(f"Error: Key 'model_state_dict' not found in the loaded file at {model_weights_path}. Saved checkpoint might be in an unexpected format. Error: {e}")
        raise
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise

    model.to(device) # Move the model to the specified device
    model.eval()     # !!! CRITICAL: Set the model to evaluation mode !!!
    print("Model loaded successfully and set to evaluation mode.")
    return model

# --- 4. Function to Preprocess a Single Image Frame ---
def preprocess_frame(frame_rgb: np.ndarray, transform: A.Compose, device: torch.device) -> torch.Tensor | None:
    """
    Preprocesses a single image frame (NumPy array in RGB format) using the provided
    Albumentations transformation pipeline.

    Args:
        frame_rgb (np.ndarray): Input image frame (HWC format, RGB channels).
        transform (albumentations.Compose): The Albumentations transformation pipeline
                                           (typically the validation transform).
        device (torch.device): The device (CPU or GPU) to move the resulting tensor to.

    Returns:
        torch.Tensor | None: The preprocessed image tensor ready for the model
                             (CHW format, batch size 1), or None if preprocessing fails.
    """
    try:
        # Apply the Albumentations transformations (expects NumPy array)
        augmented = transform(image=frame_rgb)
        image_tensor = augmented['image'] # Get the transformed tensor

        # Add the batch dimension (from [C, H, W] to [1, C, H, W])
        image_tensor = image_tensor.unsqueeze(0)

        # Move the tensor to the target device
        image_tensor = image_tensor.to(device)

        return image_tensor
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

# --- 5. Main Posture Prediction Function ---
def predict_squat_posture(frame_rgb: np.ndarray, model: nn.Module, transform: A.Compose, device: torch.device) -> dict:
    """
    Predicts the Squat posture ("Correct" or "Incorrect") for a single image frame.

    Args:
        frame_rgb (np.ndarray): Input image frame (HWC format, RGB channels).
        model (torch.nn.Module): The loaded and trained Squat classification model.
        transform (albumentations.Compose): The validation transformation pipeline.
        device (torch.device): The device (CPU or GPU) for inference.

    Returns:
        dict: A dictionary containing the prediction result and confidence.
              Example: {'prediction': 'Incorrect', 'confidence_incorrect': 0.85, 'error': None}
              or: {'prediction': None, 'confidence_incorrect': None, 'error': 'Error message'} if an error occurs.
    """
    # Preprocess the input frame
    processed_frame_tensor = preprocess_frame(frame_rgb, transform, device)

    if processed_frame_tensor is None:
        return {'prediction': None, 'confidence_incorrect': None, 'error': 'Frame preprocessing failed'}

    try:
        # Perform inference (prediction)
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model(processed_frame_tensor) # Forward pass

        # Apply sigmoid to the logits output (since using BCEWithLogitsLoss)
        probs = torch.sigmoid(outputs)
        probability_incorrect = probs.item() # The single output represents P(Incorrect)

        # Determine prediction based on 0.5 threshold
        prediction = "Incorrect" if probability_incorrect > 0.5 else "Correct"

        return {'prediction': prediction, 'confidence_incorrect': probability_incorrect, 'error': None}

    except Exception as e:
        print(f"Error during model prediction: {e}")
        return {'prediction': None, 'confidence_incorrect': None, 'error': f'Prediction failed: {str(e)}'}

# --- 6. Standalone Testing Block (Optional) ---
# if __name__ == '__main__':
#     print("Running posture_analyzer.py as main script for testing...")

#     # --- Configuration for Testing ---
#     # Adjust these paths according to your project structure
#     DEFAULT_MODEL_PATH = '../cap1_ai_feature/models/posture/squat_baseline_cnn_model.pth' 
#     DEFAULT_TEST_IMAGE_PATH = '../cap1_ai_feature/data/image_test/image_test_3.jpg' # Example test image path
#     TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     print(f"Using device for testing: {TEST_DEVICE}")
    
#     # --- Load Model ---
#     try:
#         # Use environment variables or command-line arguments for paths in production,
#         # but use defaults for simple testing here.
#         model_path = os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)
#         test_image_path = os.getenv('TEST_IMAGE_PATH', DEFAULT_TEST_IMAGE_PATH)

#         test_model = load_trained_model(model_path, TEST_DEVICE)
#         test_transform = get_validation_transform() # Get the standard validation transform

#         # --- Load Test Image ---
#         if not os.path.exists(test_image_path):
#              print(f"Error: Test image not found at {test_image_path}. Please provide a valid image path.")
#         else:
#             test_frame_bgr = cv2.imread(test_image_path) # OpenCV reads in BGR format
#             if test_frame_bgr is None:
#                 print(f"Error: Could not read test image from {test_image_path}")
#             else:
#                 test_frame_rgb = cv2.cvtColor(test_frame_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB

#                 # --- Perform Prediction ---
#                 result = predict_squat_posture(test_frame_rgb, test_model, test_transform, TEST_DEVICE)

#                 # --- Print Result ---
#                 print("\n--- Test Prediction Result ---")
#                 if result['error']:
#                     print(f"Error: {result['error']}")
#                 else:
#                     print(f"Predicted Posture: {result['prediction']}")
#                     print(f"Confidence (Incorrect): {result['confidence_incorrect']:.4f}")

#     except Exception as e:
#         print(f"An error occurred during standalone testing: {e}")
#         import traceback
#         traceback.print_exc() # Print traceback for debugging testing errors