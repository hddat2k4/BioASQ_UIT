# train_best_model.py

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    EarlyStoppingCallback
)
import joblib

# ========================
# Load and preprocess data
# ========================
if os.path.exists("bioasq_question_type_augmented.csv"):
    print("✅ Dùng dữ liệu đã augment")
    df = pd.read_csv("bioasq_question_type_augmented.csv")
else:
    print("⚠️  Chưa có augmented file. Dùng dữ liệu gốc.")
    df = pd.read_csv("data/train.csv")

# Shuffle trước khi split để tránh bias thứ tự
from sklearn.utils import shuffle

df = df[df["question"].str.len() >= 10].reset_index(drop=True)
df = shuffle(df, random_state=83).reset_index(drop=True)

le = LabelEncoder()
df["label"] = le.fit_transform(df["type"])
joblib.dump(le, "label_encoder.pkl")

# Fixed split để tái lập được kết quả
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=83
)
train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)

train_ds = Dataset.from_pandas(train_df[["question", "label"]])
val_ds = Dataset.from_pandas(val_df[["question", "label"]])

# ========================
# Tokenization
# ========================
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["question"], truncation=True)

train_ds = train_ds.map(tokenize)
val_ds = val_ds.map(tokenize)

# ========================
# Weighted Loss
# ========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label"]),
    y=df["label"]
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_tensor = weights_tensor.to(device)

# ========================
# Custom Trainer with weighted loss
# ========================
import torch.nn as nn
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ========================
# Train with best hyperparameters + cải tiến
# ========================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(le.classes_)
)

training_args = TrainingArguments(
    output_dir="./final_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    seed=83,
    logging_steps=20,
    save_total_limit=2,
    report_to="none"
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, classification_report
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, preds)
    print("\nClassification report:\n")
    print(classification_report(labels, preds, target_names=le.classes_))
    return {"accuracy": acc}

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    class_weights=weights_tensor
)

trainer.train()

# Save model
trainer.save_model("final_model")
tokenizer.save_pretrained("final_model")