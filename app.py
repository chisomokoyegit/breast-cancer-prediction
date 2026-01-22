from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from pathlib import Path

app = Flask(__name__)

# Load or generate model
model_path = "breast_cancer_model.pkl"
if not Path(model_path).exists():
    # Generate model on first startup
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = data.target
    
    features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
    X = df[features]
    y = df["diagnosis"]
    X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

with open(model_path, "rb") as f:
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





