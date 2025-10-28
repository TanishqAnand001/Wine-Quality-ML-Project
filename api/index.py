from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys
import os


app = Flask(__name__)

try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'wine_model.joblib'))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'wine_scaler.joblib'))
except Exception as e:
    print(f"Error loading model or scaler: {e}", file=sys.stderr)
    model = None
    scaler = None


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model or scaler not loaded properly"}), 500

    try:
        data = request.get_json()

        features_df = pd.DataFrame(data, index=[0])

        expected_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                         'pH', 'sulphates', 'alcohol']

        features_df = features_df[expected_cols]

        features_scaled = scaler.transform(features_df)

        prediction = model.predict(features_scaled)

        prediction_label = 'Good Quality' if prediction[0] == 1 else 'Bad Quality'

        return jsonify({'prediction': prediction_label})

    except KeyError as e:
        return jsonify({"error": f"Missing feature in JSON payload: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)