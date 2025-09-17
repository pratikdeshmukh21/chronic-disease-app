# Flask Application for Chronic Disease Prediction
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ---------------- Paths ---------------- #
MODEL_PATH = "app/models/best_disease_prediction_model.pkl"
PREPROCESSOR_PATH = "app/models/preprocessor_basic.pkl"
TFIDF_PATH = "app/models/tfidf_vectorizer.pkl"
ENCODER_PATH = "app/models/label_encoder.pkl"

# ---------------- Load Components ---------------- #
print("🔄 Loading trained model and preprocessing components...")

model = None
preprocessor_basic = None
tfidf_vectorizer = None
label_encoder = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded!")
    else:
        print("❌ Model file not found:", MODEL_PATH)

    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor_basic = joblib.load(PREPROCESSOR_PATH)
        print("✅ Preprocessor loaded!")

    if os.path.exists(TFIDF_PATH):
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        print("✅ TF-IDF loaded!")

    if os.path.exists(ENCODER_PATH):
        label_encoder = joblib.load(ENCODER_PATH)
        print("✅ Label encoder loaded!")

except Exception as e:
    print("❌ Error while loading components:", e)


# ---------------- Prediction Logic ---------------- #
def predict_disease(age, gender, bmi, systolic_bp, diastolic_bp, cholesterol,
                   blood_sugar, smoking, alcohol, activity, family_history,
                   symptoms, lifestyle):
    """Predict chronic disease based on patient data"""
    if model is None:
        return {"status": "error", "error": "Model is not loaded."}

    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'BMI': [bmi],
            'SystolicBP': [systolic_bp],
            'DiastolicBP': [diastolic_bp],
            'Cholesterol': [cholesterol],
            'BloodSugar': [blood_sugar],
            'Smoking': [smoking],
            'AlcoholConsumption': [alcohol],
            'PhysicalActivity': [activity],
            'FamilyHistory': [family_history],
            'Symptoms': [symptoms],
            'Lifestyle': [lifestyle]
        })

        # Preprocess numeric/categorical features
        X_basic_new = preprocessor_basic.transform(
            input_data.drop(['Symptoms', 'Lifestyle'], axis=1)
        )

        # Process text features
        combined_text_new = input_data['Symptoms'] + ' ' + input_data['Lifestyle']
        X_tfidf_new = tfidf_vectorizer.transform(combined_text_new).toarray()

        # Combine features
        X_combined_new = np.hstack([X_basic_new, X_tfidf_new])

        # Prediction
        prediction = model.predict(X_combined_new)[0]
        prediction_proba = model.predict_proba(X_combined_new)[0]

        # Get probabilities
        all_probabilities = {
            disease: float(prediction_proba[i])
            for i, disease in enumerate(label_encoder.classes_)
        }

        disease_name = label_encoder.inverse_transform([prediction])[0]

        return {
            "predicted_disease": disease_name,
            "confidence": float(prediction_proba.max()),
            "all_probabilities": all_probabilities,
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------- Routes ---------------- #
@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict_form():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        systolic_bp = int(request.form['systolic_bp'])
        diastolic_bp = int(request.form['diastolic_bp'])
        cholesterol = int(request.form['cholesterol'])
        blood_sugar = int(request.form['blood_sugar'])
        smoking = request.form['smoking']
        alcohol = request.form['alcohol']
        activity = request.form['activity']
        family_history = request.form['family_history']
        symptoms = request.form['symptoms']
        lifestyle = request.form['lifestyle']

        result = predict_disease(age, gender, bmi, systolic_bp, diastolic_bp,
                                 cholesterol, blood_sugar, smoking, alcohol,
                                 activity, family_history, symptoms, lifestyle)

        if result['status'] == 'success':
            return render_template('result.html',
                                   predicted_disease=result['predicted_disease'],
                                   confidence=round(result['confidence'] * 100, 2),
                                   all_probabilities=result['all_probabilities'])
        else:
            return render_template('error.html', error=result['error'])

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        result = predict_disease(
            data['age'], data['gender'], data['bmi'],
            data['systolic_bp'], data['diastolic_bp'], data['cholesterol'],
            data['blood_sugar'], data['smoking'], data['alcohol'],
            data['activity'], data['family_history'],
            data.get('symptoms', ''), data.get('lifestyle', '')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})


# ---------------- Run ---------------- #
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
