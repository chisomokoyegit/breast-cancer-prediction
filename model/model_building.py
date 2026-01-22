import pickle
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = data.target

# Rename columns to match assignment
df.rename(columns={
    "mean radius": "radius_mean",
    "mean texture": "texture_mean",
    "mean perimeter": "perimeter_mean",
    "mean area": "area_mean",
    "mean smoothness": "smoothness_mean"
}, inplace=True)

# Select 5 features
features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean"
]

X = df[features]
y = df["diagnosis"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save model
with open("breast_cancer_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "features": features
    }, f)

print("✅ Breast cancer model saved successfully")




