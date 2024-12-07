# GymVietAI - AI-Driven Assistance Fitness Platform (Exercise Plan Prediction Feature)

This repository contains the code for the AI microservice responsible for predicting personalized workout plans, nutrition recommendation within the GymVietAI application.  It uses a machine learning model to recommend workout plans based on user inputs (gender, weight, height, and age) and macro nutrients target based on user inputs.

## Features

* **Machine Learning Model:**  Uses a trained Random Forest model to predict the most suitable workout plan (Classification Model) and macro nutrients prediction (Regression Model).
* **API Endpoint:**  Provides a `/api/workout-plan` endpoint that accepts user data and returns the recommended workout plan (or in JSON format) and `/api/nutrition-plan` that returns the macro nutrients predict in a day for user. 
* **Workout Plans:** The ID for the workout plan ( or includes a set of predefined workout plans in JSON format, tailored for different fitness levels and goals).
* **Macro Nutrients Prediction:** Includes a list of macronutrient targets [calories, protein, carbs, fat].
* **Microservice Architecture:** Designed to be integrated as a standalone microservice within a larger application ecosystem.

## API Usage

**Endpoint:** `/api/workout-plan`

**Method:** `POST`

**Request Body (JSON):**

```json
{
  "Gender": "Male",  // Or "Female"
  "Weight": 70,       // In kilograms
  "Height": 1.75,      // In meters
  "Age": 30
}
```
**Response (JSON):**
```
{
  "DT": [
      4     // The classification of plan
  ],
  "EC": 0,  // Error code
  "EM": ""  // Error message
}
```

**Endpoint:** `/api/nutrition-plan`

**Method:** `POST`

**Request Body (JSON):**

```json
{
  "Weight": "55",                 // in kg
  "Height": "1.72",               // in meters
  "Age": "23",
  "Gender": "Male",               // Male or Female
  "Goal": "Loss Weight",          // 'Loss Weight' or 'Stay Fit' or 'Muscle Gain'
  "ActivityLevel": "Sedentary"    // 'Sedentary' or 'Lightly Active' or 'Moderately Active' or 'Active' or 'Very Active'
}
```
**Response (JSON):**
```
{
  "DT": [
      1485.279079365079,      // calories
      111.32302539682549,     // protein
      148.52865238095245,     // carbs
      49.53463809523812       // fat
  ],
  "EC": 0,
  "EM": ""
}
```

**Endpoint:**
- 400 Bad Request: Invalid input data or format. The response will include an "error" message.
- 404 Not Found: No workout plan found for the predicted BMI category.

## Installation and Setup
1. Clone the Repository:
```
git clone https://github.com/mintwann/GymVietAI-AI.git
```

2. Create and Activate a Virtual Environment:
```
python -m venv ~\path\to\where\you\want\it
.\Scripts\activate (on Windows)
```

3. Install Dependencies:
```
pip install -r requirements.txt
```

## Running the API
1. Navigate to the Project Directory:
```
cd GymVietAI-AI
```
2. Run the Flask App:
```
python app.py
```
The API will then be running at port `5001`
