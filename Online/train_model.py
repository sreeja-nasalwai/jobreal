import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from django.conf import settings
from pathlib import Path
import joblib
import os

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "job.settings")  # Replace "Online.settings" if your project name differs

# Initialize Django
import django
django.setup()

# Now settings.BASE_DIR is accessible
DATA_PATH = Path(settings.BASE_DIR) / "fake_job_postings.csv"
data = pd.read_csv(DATA_PATH)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocess dataset
text_fields = ["title", "location", "department", "company_profile", "description",
               "requirements", "benefits", "required_experience", "required_education"]
data["combined_text"] = data[text_fields].apply(lambda x: " ".join(x.fillna("")), axis=1)
data["combined_text"] = data["combined_text"].apply(clean_text)

label_encoder = LabelEncoder()
data["employment_type"] = label_encoder.fit_transform(data["employment_type"].fillna("Unknown"))
data["industry"] = label_encoder.fit_transform(data["industry"].fillna("Unknown"))
data["function"] = label_encoder.fit_transform(data["function"].fillna("Unknown"))

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_text = tfidf.fit_transform(data["combined_text"])
X_numerical = data[["employment_type", "industry", "function"]]
X = np.hstack((X_text.toarray(), X_numerical))
y = data["fraudulent"]

svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X, y)

# Save everything to disk
joblib.dump(svm_model, Path(settings.BASE_DIR) / "svm_model.pkl")
joblib.dump(tfidf, Path(settings.BASE_DIR) / "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, Path(settings.BASE_DIR) / "label_encoder.pkl")
print("Model training complete and saved!")