import json
import os
import nltk
import difflib
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from utils import model

load_dotenv()

nltk.download("punkt")

def normalize_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    elif isinstance(text, list):
        return [str(t).strip().lower() for t in text]
    else:
        return str(text).strip().lower()

def extract_common_fields(gold_item, pred_item):
    common_keys = set(gold_item.keys()) & set(pred_item.keys())
    gold_filtered = {k: gold_item[k] for k in common_keys}
    pred_filtered = {k: pred_item[k] for k in common_keys}
    return gold_filtered, pred_filtered

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


def compute_f1_yes_no(labels, preds, positive_label):
    binarize = lambda x: [1 if i == positive_label else 0 for i in x]
    return f1_score(binarize(labels), binarize(preds), zero_division=0)

def compute_macro_f1(labels, preds):
    return f1_score(labels, preds, average="macro", zero_division=0)

def compute_strict_accuracy(preds, labels):
    correct = sum([pred[0] == label[0] for pred, label in zip(preds, labels)])
    return correct / len(labels) if labels else 0.0

def compute_lenient_accuracy(preds, labels):
    correct = 0
    for pred, label in zip(preds, labels):
        if any(p in label for p in pred):
            correct += 1
    return correct / len(labels) if labels else 0.0



def compute_mrr(preds, labels):
    mrr_total = 0.0
    for pred, label in zip(preds, labels):
        rank = 0
        for i, p in enumerate(pred):
            if p in label:
                rank = i + 1
                break
        if rank > 0:
            mrr_total += 1 / rank
    return mrr_total / len(labels) if labels else 0.0

def compute_extended_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rouge1'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge2_recall": scores["rouge2"].recall,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeSu4_recall": scores["rouge1"].recall,   
        "rougeSu4_f1": scores["rouge1"].fmeasure,
    }

def evaluate_output(predictions, gold_data, k=10):
    metrics = {
        "yesno": {"labels": [], "preds": []},
        "factoid": {"labels": [], "preds": []},
        "summary": {
            "bleu": [], "rouge": [], "fuzzy": [],
            "rouge2_recall": [], "rouge2_f1": [],
            "rougeSu4_recall": [], "rougeSu4_f1": []
        },
        "list": {"labels": [], "preds": []}
    }

    for pred in predictions:
        gold = next((g for g in gold_data if g["id"] == pred["id"]), None)
        if not gold:
            continue
        gold_filtered, pred_filtered = extract_common_fields(gold, pred)
        if not gold_filtered or not pred_filtered:
            continue

        q_type = gold.get("type", "").lower()
        pred_ans = normalize_text(pred_filtered.get("exact_answer", ""))
        gold_ans = normalize_text(gold_filtered.get("exact_answer", ""))

        if q_type == "yesno":
            metrics[q_type]["labels"].append(gold_ans)
            metrics[q_type]["preds"].append(pred_ans)

        elif q_type == "factoid":
            if not isinstance(gold_ans, list):
                gold_ans = [gold_ans]
            metrics[q_type]["labels"].append(gold_ans)
            metrics[q_type]["preds"].append([pred_ans])

        elif q_type == "summary":
            pred_ideal = normalize_text(pred_filtered.get("ideal_answer", ""))
            gold_ideal = gold_filtered.get("ideal_answer", "")
            if isinstance(gold_ideal, list):
                gold_ideal = [normalize_text(ans) for ans in gold_ideal]
                metrics["summary"]["bleu"].append(max(compute_bleu(ans, pred_ideal) for ans in gold_ideal))
                metrics["summary"]["rouge"].append(max(compute_rouge_l(ans, pred_ideal) for ans in gold_ideal))
                metrics["summary"]["fuzzy"].append(max(fuzzy_match(ans, pred_ideal) for ans in gold_ideal))
                for ans in gold_ideal:
                    ext = compute_extended_rouge(ans, pred_ideal)
                    metrics["summary"]["rouge2_recall"].append(ext["rouge2_recall"])
                    metrics["summary"]["rouge2_f1"].append(ext["rouge2_f1"])
                    metrics["summary"]["rougeSu4_recall"].append(ext["rougeSu4_recall"])
                    metrics["summary"]["rougeSu4_f1"].append(ext["rougeSu4_f1"])
            else:
                gold_ideal = normalize_text(gold_ideal)
                metrics["summary"]["bleu"].append(compute_bleu(gold_ideal, pred_ideal))
                metrics["summary"]["rouge"].append(compute_rouge_l(gold_ideal, pred_ideal))
                metrics["summary"]["fuzzy"].append(fuzzy_match(gold_ideal, pred_ideal))
                ext = compute_extended_rouge(gold_ideal, pred_ideal)
                metrics["summary"]["rouge2_recall"].append(ext["rouge2_recall"])
                metrics["summary"]["rouge2_f1"].append(ext["rouge2_f1"])
                metrics["summary"]["rougeSu4_recall"].append(ext["rougeSu4_recall"])
                metrics["summary"]["rougeSu4_f1"].append(ext["rougeSu4_f1"])

        elif q_type == "list":
            pred_docs = [normalize_text(doc[0]) for doc in (pred_filtered.get("exact_answer") or [])][:k]
            gold_docs = [normalize_text(doc[0]) for doc in gold_filtered.get("exact_answer", [])]
            metrics["list"]["labels"].append(sorted(gold_docs))
            metrics["list"]["preds"].append(sorted(pred_docs))

    result = {}

    # Yes/No metrics
    labels = metrics["yesno"]["labels"]
    preds = metrics["yesno"]["preds"]
    result["Accuracy (Yes/No)"] = sum([l == p for l, p in zip(labels, preds)]) / len(labels) if labels else 0.0
    result["F1 Yes"] = compute_f1_yes_no(labels, preds, "yes")
    result["F1 No"] = compute_f1_yes_no(labels, preds, "no")
    result["Macro F1 (Yes/No)"] = compute_macro_f1(labels, preds)

    # Factoid metrics
    result["Strict Accuracy (Factoid)"] = compute_strict_accuracy(metrics["factoid"]["preds"], metrics["factoid"]["labels"])
    result["Lenient Accuracy (Factoid)"] = compute_lenient_accuracy(metrics["factoid"]["preds"], metrics["factoid"]["labels"])
    result["MRR (Factoid)"] = compute_mrr(metrics["factoid"]["preds"], metrics["factoid"]["labels"])

    # Summary metrics
    result.update({
        "BLEU (Summary)": sum(metrics["summary"]["bleu"]) / len(metrics["summary"]["bleu"]) if metrics["summary"]["bleu"] else 0.0,
        "ROUGE-L (Summary)": sum(metrics["summary"]["rouge"]) / len(metrics["summary"]["rouge"]) if metrics["summary"]["rouge"] else 0.0,
        "Fuzzy Match (Summary)": sum(metrics["summary"]["fuzzy"]) / len(metrics["summary"]["fuzzy"]) if metrics["summary"]["fuzzy"] else 0.0,
        "R-2 (Rec)": sum(metrics["summary"]["rouge2_recall"]) / len(metrics["summary"]["rouge2_recall"]) if metrics["summary"]["rouge2_recall"] else 0.0,
        "R-2 (F1)": sum(metrics["summary"]["rouge2_f1"]) / len(metrics["summary"]["rouge2_f1"]) if metrics["summary"]["rouge2_f1"] else 0.0,
        "R-1 (Rec)": sum(metrics["summary"]["rougeSu4_recall"]) / len(metrics["summary"]["rougeSu4_recall"]) if metrics["summary"]["rougeSu4_recall"] else 0.0,
        "R-1 (F1)": sum(metrics["summary"]["rougeSu4_f1"]) / len(metrics["summary"]["rougeSu4_f1"]) if metrics["summary"]["rougeSu4_f1"] else 0.0,
    })

    # List metrics
    mlb_list = MultiLabelBinarizer()
    # Gộp cả label và prediction trước khi fit
    mlb_list.fit(metrics["list"]["labels"] + metrics["list"]["preds"])

    y_true_list = mlb_list.transform(metrics["list"]["labels"])
    y_pred_list = mlb_list.transform(metrics["list"]["preds"])

    precision_list, recall_list, f1_list, _ = precision_recall_fscore_support(
        y_true_list, y_pred_list, average="macro", zero_division=0
    )

    result["Mean Precision (List)"] = precision_list
    result["Recall (List)"] = recall_list
    result["F-Measure (List)"] = f1_list

    return result


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results_to_excel(result_dict, retrieval_mode, embed_model, llm_model, output_path="evaluation_results.xlsx"):
    df = pd.DataFrame([{
        "retrieval_mode": retrieval_mode,
        "embed_model": embed_model,
        "llm_model": llm_model,
        **result_dict
    }])

    if os.path.exists(output_path):
        existing_df = pd.read_excel(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    cols = ["embed_model", "llm_model", "retrieval_mode"] + [col for col in df.columns if col not in {"embed_model", "llm_model", "retrieval_mode"}]
    df = df[cols]

    df.to_excel(output_path, index=False)
    print(f"[✔] Đã lưu kết quả vào {output_path}")


if __name__ == "__main__":
    retrieval_modes = ["dense", "hybrid", "bm25"]
    model_n = model.embed_model_name
    llm = model.llm_model_name
    for method in retrieval_modes:
        pred_path = f"testcase/predictions_{method}.json"
        gold_path = "testcase/answers1.json"
        predictions = load_json(pred_path)
        gold = load_json(gold_path)

        results = evaluate_output(predictions, gold)
        print(f"{'Mode':<7} | {'Embed Model':<20} | {'LLM':<15} | {'Metric':<30} : {'Score':>7}")
        print("-" * 90)
        for metric, score in results.items():
            print(f"{method:<7} | {model_n:<20} | {llm:<15} | {metric:<30} : {score:.4f}")
        save_results_to_excel(results, method,model_n,llm)
