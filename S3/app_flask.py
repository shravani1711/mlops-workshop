# app_flask.py
# Day 2, Session 3 - Deploying a model as a REST API (Flask)

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Iris Model API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True, port=5001)

# Run with:  python app_flask.py
# Test with: curl -X POST http://localhost:5001/predict \
#              -H "Content-Type: application/json" \
#              -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
