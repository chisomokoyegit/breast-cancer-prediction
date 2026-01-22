from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load trained model
with open("breast_cancer_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

@app.route("/")
def home():
    return render_template("index.html", prediction="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["radius_mean"]),
            float(request.form["texture_mean"]),
            float(request.form["perimeter_mean"]),
            float(request.form["area_mean"]),
            float(request.form["smoothness_mean"])
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        result = "Malignant" if prediction == 0 else "Benign"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)





