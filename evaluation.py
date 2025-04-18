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
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
from rouge_score import rouge_scorer
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import model

nltk.download("punkt")
current_dir = os.path.dirname(os.path.abspath(__file__))

def normalize_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    elif isinstance(text, list):
        return [str(t).strip().lower() for t in text]
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

def extract_common_fields(gold, pred):
    return gold.get("answers", {}), pred.get("answers", {})

def evaluate_output(predictions, gold_data, k=5):
    metrics = {
        "yesno": {"labels": [], "preds": [], "correct": 0, "count": 0},
        "factoid": {"labels": [], "preds": [], "correct": 0, "count": 0},
        "summary": {"bleu": [], "rouge": [], "fuzzy": []},
        "list": {
            "jaccard_doc": [], "jaccard_snip": [],
            "recall_at_k": [], "precision_at_k": [], "map": [],
            "labels": [], "preds": [], "correct": 0, "count": 0
        }
    }
    # print(predictions)
    for pred in predictions:
        # print(pred['type'])
        gold = next((g for g in gold_data if g["id"] == pred["id"]), None)
        if not gold:
            continue
        gold_filtered, pred_filtered = extract_common_fields(gold, pred)
        if not gold_filtered or not pred_filtered:
            continue

        q_type = gold.get("type", "").lower()
        pred_ans = normalize_text(pred_filtered.get("exact_answer", ""))
        gold_ans = normalize_text(gold_filtered.get("exact_answer", ""))

        if q_type in ["yesno", "factoid"]:
            metrics[q_type]["preds"].append(pred_ans)
            metrics[q_type]["labels"].append(gold_ans)
            metrics[q_type]["count"] += 1
            if q_type == 'factoid':
                if [str(pred_ans)] == gold_ans:
                    metrics[q_type]["correct"] += 1    
            if pred_ans == gold_ans:
                metrics[q_type]["correct"] += 1

        if q_type == "summary":
            pred_ideal = normalize_text(pred_filtered.get("ideal_answer", ""))
            gold_ideal = gold_filtered.get("ideal_answer", "")
            if isinstance(gold_ideal, list):
                gold_ideal = [normalize_text(ans) for ans in gold_ideal]
                metrics["summary"]["bleu"].append(max(compute_bleu(ans, pred_ideal) for ans in gold_ideal))
                metrics["summary"]["rouge"].append(max(compute_rouge_l(ans, pred_ideal) for ans in gold_ideal))
                metrics["summary"]["fuzzy"].append(max(fuzzy_match(ans, pred_ideal) for ans in gold_ideal))
            else:
                gold_ideal = normalize_text(gold_ideal)
                metrics["summary"]["bleu"].append(compute_bleu(gold_ideal, pred_ideal))
                metrics["summary"]["rouge"].append(compute_rouge_l(gold_ideal, pred_ideal))
                metrics["summary"]["fuzzy"].append(fuzzy_match(gold_ideal, pred_ideal))

        if q_type == "list":
            # Get top-k predictions and gold answers
            pred_docs = [normalize_text(doc[0]).lower() for doc in pred_filtered.get("exact_answer", [])][:k]
            gold_docs = [normalize_text(doc[0]).lower() for doc in gold_filtered.get("exact_answer", [])]

            # Print to debug
            # print(f"pred_docs: {pred_docs}")
            # print(f"gold_docs: {gold_docs}")

            # Jaccard similarity
            metrics["list"]["jaccard_doc"].append(jaccard_similarity(pred_docs, gold_docs))
            metrics["list"]["recall_at_k"].append(len(set(pred_docs) & set(gold_docs)) / len(gold_docs) if gold_docs else 0.0)
            metrics["list"]["precision_at_k"].append(len(set(pred_docs) & set(gold_docs)) / len(pred_docs) if pred_docs else 0.0)
            metrics["list"]["map"].append(average_precision(pred_docs, gold_docs))

            # Snippets (not changing for now)
            pred_snip = [normalize_text(s["text"]).lower() for s in pred_filtered.get("snippets", [])]
            gold_snip = [normalize_text(s["text"]).lower() for s in gold_filtered.get("snippets", [])]
            metrics["list"]["jaccard_snip"].append(jaccard_similarity(pred_snip, gold_snip))

            # Calculate precision, recall, F1 for the list of entities
            true_positives = len(set(pred_docs) & set(gold_docs))
            false_positives = len(set(pred_docs) - set(gold_docs))
            false_negatives = len(set(gold_docs) - set(pred_docs))

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics["list"]["labels"].append(",".join(sorted(gold_docs)))
            metrics["list"]["preds"].append(",".join(sorted(pred_docs)))
            metrics["list"]["count"] += 1
            if precision > 0 and recall > 0:  # Checking if there is any positive match
                metrics["list"]["correct"] += 1

    # Precision, Recall, F1 for yesno and factoid
    result = {}
    for qtype in ["yesno", "factoid"]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            metrics[qtype]["labels"], metrics[qtype]["preds"], average="macro", zero_division=0
        )
        acc = metrics[qtype]["correct"] / metrics[qtype]["count"] if metrics[qtype]["count"] else 0.0
        result[f"accuracy_exact ({qtype})"] = acc
        result[f"precision_exact ({qtype})"] = precision
        result[f"recall_exact ({qtype})"] = recall
        result[f"f1_exact ({qtype})"] = f1

    # Summary Metrics
    result.update({
        "bleu_ideal (summary)": sum(metrics["summary"]["bleu"]) / len(metrics["summary"]["bleu"]) if metrics["summary"]["bleu"] else 0.0,
        "rougeL_ideal (summary)": sum(metrics["summary"]["rouge"]) / len(metrics["summary"]["rouge"]) if metrics["summary"]["rouge"] else 0.0,
        "fuzzy_ideal (summary)": sum(metrics["summary"]["fuzzy"]) / len(metrics["summary"]["fuzzy"]) if metrics["summary"]["fuzzy"] else 0.0,
    })

    # List Metrics
    result.update({
        "jaccard_documents (list)": sum(metrics["list"]["jaccard_doc"]) / len(metrics["list"]["jaccard_doc"]) if metrics["list"]["jaccard_doc"] else 0.0,
        "jaccard_snippets (list)": sum(metrics["list"]["jaccard_snip"]) / len(metrics["list"]["jaccard_snip"]) if metrics["list"]["jaccard_snip"] else 0.0,
        "recall@k_documents (list)": sum(metrics["list"]["recall_at_k"]) / len(metrics["list"]["recall_at_k"]) if metrics["list"]["recall_at_k"] else 0.0,
        "precision@k_documents (list)": sum(metrics["list"]["precision_at_k"]) / len(metrics["list"]["precision_at_k"]) if metrics["list"]["precision_at_k"] else 0.0,
        "MAP_documents (list)": sum(metrics["list"]["map"]) / len(metrics["list"]["map"]) if metrics["list"]["map"] else 0.0,

    })

    # Accuracy/precision/recall/f1 for list
    precision_list, recall_list, f1_list, _ = precision_recall_fscore_support(
        metrics["list"]["labels"], metrics["list"]["preds"], average="macro", zero_division=0
    )
    acc_list = metrics["list"]["correct"] / metrics["list"]["count"] if metrics["list"]["count"] else 0.0
    result["accuracy (list)"] = acc_list
    result["precision (list)"] = precision_list
    result["recall (list)"] = recall_list
    result["f1 (list)"] = f1_list

    return result

import pandas as pd

def save_results_to_excel(result_dict, embed_model, llm_model, retrieval_mode, output_path="results.xlsx"):
    df = pd.DataFrame([{
        "embed_model": embed_model,
        "llm_model": llm_model,
        "retrieval_mode": retrieval_mode,
        **result_dict
    }])

    if os.path.exists(output_path):
        # Nếu đã tồn tại, thì append vào
        existing_df = pd.read_excel(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(output_path, index=False)
    print(f"[✔] Kết quả đã được lưu vào {output_path}")

def extract_common_fields(gold_item, pred_item):
    common_keys = set(gold_item.keys()) & set(pred_item.keys())
    gold_filtered = {k: gold_item[k] for k in common_keys}
    pred_filtered = {k: pred_item[k] for k in common_keys}
    return gold_filtered, pred_filtered


# --- Đánh giá ---

retrieval_modes = ["bm25"]
def all_evaluation(retrieval_modes="bm25", log=True, excel=True):
    for retrieval_mode in retrieval_modes:
        
        # --- Đường dẫn file ---
        predictions_path = os.path.join(current_dir, 'testcase', f'predictions_{retrieval_mode}.json')
        gold_data_path = os.path.join(current_dir, 'testcase', 'gold_data.json')

        # --- Load dữ liệu ---
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        with open(gold_data_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
            
        metrics = evaluate_output(predictions, gold_data, k=5)

        if log: 
            # --- In kết quả ---
            print(f"{'Metric':<35} {'Score':>10}")
            print("-" * 47)

            for k, v in metrics.items():
                print(f"{model.embed_model_name} - {model.llm_model_name} - {retrieval_mode}: {k:<35} {v:>10.4f}")

        if excel:
            save_results_to_excel(metrics, model.embed_model_name, model.llm_model_name, retrieval_mode)


####################################################################################################################
retrieval_modes = ["dense", "hybrid", "bm25"]
all_evaluation(retrieval_modes=retrieval_modes, log=True, excel=True)