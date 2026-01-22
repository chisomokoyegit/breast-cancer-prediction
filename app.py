# app.py
from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load saved model
model_path = os.path.join("model", "breast_cancer_model.pkl")
with open(model_path, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    features = data["features"]

@app.route("/")
def home():
    return render_template("index.html", prediction="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect features from form
        feature_values = [float(request.form[feat]) for feat in features]
        feature_array = np.array(feature_values).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)

        # Predict
        prediction = model.predict(feature_scaled)[0]
        result = "Malignant" if prediction == 0 else "Benign"

        return render_template("index.html", prediction=f"Predicted Tumor: {result}")
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)







