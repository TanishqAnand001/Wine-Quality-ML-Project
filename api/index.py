from flask import Flask, request, jsonify
import joblib
import numpy as np
import sys
import os

app = Flask(__name__)

try:
    model_path = os.path.join(os.path.dirname(__file__), 'wine_model.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'wine_scaler.joblib')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Define the feature order explicitly
    expected_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]

except Exception as e:
    print(f"Error loading model or scaler: {e}", file=sys.stderr)
    model = None
    scaler = None
    expected_cols = None


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not expected_cols:
        return jsonify({"error": "Model or scaler not loaded properly"}), 500

    try:
        data = request.get_json()

        # --- Start: Pandas Replacement ---

        # Create a list of values from the JSON data, in the correct order
        input_features = []
        for col in expected_cols:
            if col not in data:
                return jsonify({"error": f"Missing feature in JSON payload: {col}"}), 400
            input_features.append(data[col])

        # Convert the list to a 2D numpy array
        features_np = np.array([input_features])

        # --- End: Pandas Replacement ---

        # Scale the numpy array
        features_scaled = scaler.transform(features_np)

        # Make prediction
        prediction = model.predict(features_scaled)

        prediction_label = 'Good Quality' if prediction[0] == 1 else 'Bad Quality'

        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400