import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from django.conf import settings
from pathlib import Path
import joblib

# Paths to pre-trained model and preprocessors
MODEL_PATH = Path(settings.BASE_DIR) / "svm_model.pkl"
TFIDF_PATH = Path(settings.BASE_DIR) / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = Path(settings.BASE_DIR) / "label_encoder.pkl"

# Load pre-trained model and preprocessors
svm_model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_job(job_data):
    """Predict if a job posting is fake or real using pre-trained SVM."""
    combined_text = " ".join([
        clean_text(job_data.get("title", "")),
        clean_text(job_data.get("location", "")),
        clean_text(job_data.get("department", "")),
        clean_text(job_data.get("company_profile", "")),
        clean_text(job_data.get("description", "")),
        clean_text(job_data.get("requirements", "")),
        clean_text(job_data.get("benefits", "")),
        clean_text(job_data.get("required_experience", "")),
        clean_text(job_data.get("required_education", ""))
    ])

    text_features = tfidf.transform([combined_text]).toarray()

    employment_type = job_data.get("employment_type", "Unknown")
    industry = job_data.get("industry", "Unknown")
    function = job_data.get("function", "Unknown")

    employment_type_encoded = (label_encoder.transform([employment_type])[0] 
                               if employment_type in label_encoder.classes_ 
                               else label_encoder.transform(["Unknown"])[0])
    industry_encoded = (label_encoder.transform([industry])[0] 
                        if industry in label_encoder.classes_ 
                        else label_encoder.transform(["Unknown"])[0])
    function_encoded = (label_encoder.transform([function])[0] 
                        if function in label_encoder.classes_ 
                        else label_encoder.transform(["Unknown"])[0])

    numerical_features = np.array([[employment_type_encoded, industry_encoded, function_encoded]])
    features = np.hstack((text_features, numerical_features))

    prediction = svm_model.predict(features)[0]
    return "Fake" if prediction == 1 else "Real"