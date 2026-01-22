import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib
import pickle

# 1. Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target


# 2. Feature selection (exactly 5 features)
selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean concavity'
]

X = df[selected_features]
y = df['diagnosis']

# 3. Handle missing values
print(X.isnull().sum())


# 4. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


# 7. Evaluate model with all metrics
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# 8. Save model + scaler
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
loaded_model = joblib.load('breast_cancer_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

sample = X.iloc[0].values.reshape(1, -1)
sample_scaled = loaded_scaler.transform(sample)

prediction = loaded_model.predict(sample_scaled)
print("Prediction:", "Benign" if prediction[0] == 1 else "Malignant")
print("Model and scaler saved successfully.")




