"""
# Bộ công cụ đánh giá cho bài toán trả lời câu hỏi (QA Evaluation Toolkit)

## Độ đo cho từng loại câu hỏi:

1. Câu hỏi dạng Yes/No, Factoid:
   - `accuracy_exact`, `precision_exact`, `recall_exact`, `f1_exact`: dùng để so sánh câu trả lời ngắn, chính xác.
   - Ví dụ:
       - gold: "yes", pred: "yes" → chính xác
       - gold: "diabetes", pred: "diabetes" → chính xác

2. Câu hỏi dạng Summary (tự do, sinh câu):
   - `bleu_ideal`, `rougeL_ideal`, `fuzzy_ideal`: đánh giá câu trả lời tự động sinh ra (so với câu lý tưởng).
   - Ví dụ:
       - gold: "The patient has a fever and cough"
       - pred: "Patient suffers from fever and cough"
       → BLEU, ROUGE-L, Fuzzy match sẽ đo mức độ tương đồng.

3. Câu hỏi dạng List / Document retrieval:
   - `jaccard_documents`, `recall@k_documents`, `precision@k_documents`, `MAP_documents`: đánh giá mức độ tìm đúng tài liệu.
   - `jaccard_snippets`: đo mức độ trùng lặp đoạn trích giữa gold và pred.
   - Ví dụ:
       - gold_docs = [doc1, doc2, doc3], pred_docs = [doc1, doc3, doc4]
       → precision = 2/3, recall = 2/3, Jaccard = 2 / 4

## Hướng dẫn sử dụng:
- Chuẩn bị 2 file JSON: `predictions.json`, `gold_data.json`, trong thư mục `testcase/`
- Chạy script để in ra tất cả độ đo đánh giá.
"""

import difflib
import nltk
import json
import os
from sklearn.metrics import precision_recall_fscore_support
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import model

nltk.download("punkt")

def normalize_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    elif isinstance(text, list):
        return " ".join([str(t).strip().lower() for t in text])
    else:
        return str(text).strip().lower()

def jaccard_similarity(set1, set2):
    set1, set2 = set(set1), set(set2)
    inter = set1 & set2
    union = set1 | set2
    return len(inter) / len(union) if union else 0.0

def fuzzy_match(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def compute_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        [nltk.word_tokenize(reference)],
        nltk.word_tokenize(hypothesis),
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25)
    )

def compute_rouge_l(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

def average_precision(predicted, gold):
    if not gold:
        return 0.0
    hits = 0
    sum_precisions = 0
    for i, doc in enumerate(predicted):
        if doc in gold:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(gold)

def evaluate_output(predictions, gold_data, k=5):
    total = len(predictions)
    correct_exact = 0
    exact_preds, exact_labels = [], []
    bleu_scores, rouge_scores, fuzzy_scores = [], [], []
    jaccard_doc_scores, jaccard_snip_scores = [], []
    recall_at_k, precision_at_k, map_scores = [], [], []

    for pred in predictions:
        gold = next((g for g in gold_data if g["id"] == pred["id"]), None)
        if not gold:
            continue

        gold_filtered, pred_filtered = extract_common_fields(gold, pred)
        if not gold_filtered or not pred_filtered:
            continue

        pred_ans = normalize_text(pred_filtered.get("exact_answer", ""))
        gold_ans = normalize_text(gold_filtered.get("exact_answer", ""))
        exact_preds.append(pred_ans)
        exact_labels.append(gold_ans)
        if pred_ans == gold_ans:
            correct_exact += 1

        pred_ideal = normalize_text(pred_filtered.get("ideal_answer", ""))
        gold_ideal = gold_filtered.get("ideal_answer", "")
        
        if isinstance(gold_ideal, list):
            best_bleu = max(compute_bleu(ans, pred_ideal) for ans in gold_ideal)
            best_rouge = max(compute_rouge_l(ans, pred_ideal) for ans in gold_ideal)
            best_fuzzy = max(fuzzy_match(ans, pred_ideal) for ans in gold_ideal)
        else:
            best_bleu = compute_bleu(gold_ideal, pred_ideal)
            best_rouge = compute_rouge_l(gold_ideal, pred_ideal)
            best_fuzzy = fuzzy_match(gold_ideal, pred_ideal)

        bleu_scores.append(best_bleu)
        rouge_scores.append(best_rouge)
        fuzzy_scores.append(best_fuzzy)

        pred_docs = pred_filtered.get("documents", [])[:k]
        gold_docs = [doc.split("/")[-1] for doc in gold_filtered.get("documents", [])]
        jaccard_doc_scores.append(jaccard_similarity(pred_docs, gold_docs))
        recall_at_k.append(len(set(pred_docs) & set(gold_docs)) / len(gold_docs) if gold_docs else 0.0)
        precision_at_k.append(len(set(pred_docs) & set(gold_docs)) / len(pred_docs) if pred_docs else 0.0)
        map_scores.append(average_precision(pred_docs, gold_docs))

        pred_snip = [normalize_text(s["text"]) for s in pred_filtered.get("snippets", [])]
        gold_snip = [normalize_text(s["text"]) for s in gold_filtered.get("snippets", [])]
        jaccard_snip_scores.append(jaccard_similarity(pred_snip, gold_snip))

    precision, recall, f1, _ = precision_recall_fscore_support(
        exact_labels, exact_preds, average="macro", zero_division=0
    )

    return {
        "accuracy_exact (yesno/factoid)": correct_exact / total,
        "precision_exact (yesno/factoid)": precision,
        "recall_exact (yesno/factoid)": recall,
        "f1_exact (yesno/factoid)": f1,

        "bleu_ideal (summary)": sum(bleu_scores) / total,
        "rougeL_ideal (summary)": sum(rouge_scores) / total,
        "fuzzy_ideal (summary)": sum(fuzzy_scores) / total,

        "jaccard_documents (list)": sum(jaccard_doc_scores) / total,
        "jaccard_snippets (list)": sum(jaccard_snip_scores) / total,

        "recall@k_documents (list)": sum(recall_at_k) / total,
        "precision@k_documents (list)": sum(precision_at_k) / total,
        "MAP_documents (list)": sum(map_scores) / total,
    }

def extract_common_fields(gold_item, pred_item):
    common_keys = set(gold_item.keys()) & set(pred_item.keys())
    gold_filtered = {k: gold_item[k] for k in common_keys}
    pred_filtered = {k: pred_item[k] for k in common_keys}
    return gold_filtered, pred_filtered

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
predictions_path = os.path.join(current_dir, 'testcase', 'predictions.json')
gold_data_path = os.path.join(current_dir, 'testcase', 'gold_data.json')

# Load predictions and gold data
with open(predictions_path, "r", encoding="utf-8") as f:
    predictions = json.load(f)
with open(gold_data_path, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

# Evaluate output
metrics = evaluate_output(predictions, gold_data, k=5)
# In đẹp:
print(f"{'Metric':<35} {'Score':>10}")
print("-" * 47)
for k, v in metrics.items():
    print(f"{model.embed_model_name} - {model.llm_model_name}: {k:<35} {v:>10.4f}")