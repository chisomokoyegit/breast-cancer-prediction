import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
df = pd.read_csv("breast_cancer.csv")  # your dataset here

# 2. Select features (choose 5)
features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
X = df[features]

# 3. Encode target
y = LabelEncoder().fit_transform(df["diagnosis"])  # Benign=0, Malignant=1

# 4. Handle missing values
X = X.fillna(X.median())

# 5. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# 9. Save model and scaler
with open("breast_cancer_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print("✅ Model saved successfully")


