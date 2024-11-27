from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load pre-trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract data from form
        data = request.form
        features = np.array([[float(data['age']), float(data['bmi']), float(data['blood_pressure'])]])

        # Predict using model
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Convert the numpy int64 to a standard Python int for JSON serialization
        prediction = int(prediction)
        probability = probability.tolist()  # Convert numpy array to a regular list

        return jsonify({
            "Prediction": prediction,
            "Probability": probability
        })
    except Exception as e:
        return jsonify({"Error": str(e)})
