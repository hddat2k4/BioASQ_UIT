# import joblib
# from sklearn.linear_model import LogisticRegression

# vec = joblib.load("./data/tfidf_vectorizer.pkl")
# model = joblib.load("./data/question_type_classifier.pkl")


# def question_type(question: str) -> str:
#     X = vec.transform([question])
#     return model.predict(X)[0]

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np

# Load model và tokenizer đã huấn luyện
model_dir = "final_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Load label encoder để decode kết quả
le = joblib.load("final_model/label_encoder.pkl")

def question_type(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
    return le.inverse_transform([pred])[0], confidence