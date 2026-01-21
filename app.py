from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
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
        # get input from form
        features = [
            float(request.form["radius_mean"]),
            float(request.form["texture_mean"]),
            float(request.form["perimeter_mean"]),
            float(request.form["area_mean"]),
            float(request.form["smoothness_mean"]),
        ]

        # scale and predict
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        result = "Malignant" if pred==1 else "Benign"

        return render_template("index.html", prediction=f"Tumor is {result}")
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)




