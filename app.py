from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from pathlib import Path

app = Flask(__name__)

# ======= Paths =======
MODEL_PATH = Path(__file__).parent / "breast_cancer_model.pkl"

# ======= Load or create model =======
if not MODEL_PATH.exists():
    # Create model if Pickle does not exist
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # Load dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = data.target  # 0 = malignant, 1 = benign

    # Select features
    features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
    feature_names = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
    X = df[feature_names]
    y = df["diagnosis"]

    # Fill missing values (just in case)
    X = X.fillna(X.mean())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model + scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "features": features
        }, f)

# Load model
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    features = data["features"]

# ======= Routes =======
@app.route("/")
def home():
    return render_template("index.html", prediction="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read input values dynamically
        user_input = []
        for feat in ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]:
            val = request.form.get(feat)
            if val is None or val.strip() == "":
                return render_template("index.html", prediction=f"Error: Missing value for {feat}")
            user_input.append(float(val))

        # Predict
        user_input = np.array(user_input).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)
        pred = model.predict(user_input_scaled)[0]

        result = "Benign" if pred == 1 else "Malignant"
        return render_template("index.html", prediction=f"Tumor is likely: {result}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

# ======= Run =======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)








