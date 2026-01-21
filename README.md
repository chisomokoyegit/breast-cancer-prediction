# Breast Cancer Prediction System

A Flask web application for predicting breast cancer diagnosis using a KNN machine learning model.

## Features
- Predicts whether a tumor is benign or malignant
- Uses 5 key features: radius mean, texture mean, perimeter mean, area mean, smoothness mean
- Pre-trained KNN model with high accuracy

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (if needed):
```bash
python model/model_building.py
```

3. Run the Flask app:
```bash
python app.py
```

4. Open browser and go to: `http://localhost:5000`

## Deployment on Render

1. Push your code to a GitHub repository
2. Connect your GitHub repo to Render
3. Render will automatically:
   - Install dependencies from `requirements.txt`
   - Run the Flask app using gunicorn
4. Your app will be live on a Render URL

## Files Structure
- `app.py` - Flask application
- `model/model_building.py` - Model training script
- `templates/index.html` - Frontend interface
- `breast_cancer_model.pkl` - Pre-trained model and scaler
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration

## Important Notes
- Ensure `breast_cancer_model.pkl` is committed to your repository
- The model is pre-trained and ready to use
- No dataset is needed for deployment
