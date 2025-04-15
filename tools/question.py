import json


dir = "data/test.json"
with open(dir) as f:
    data = json.load(f)

questions = []
gold_answers = []

for item in data:
    q = {
        "question_id": item["id"],
        "question": item["body"],
        "question_type": item["type"]
    }
    questions.append(q)

    gold = {
        "question_id": item["id"],
        "ideal_answer": item.get("ideal_answer", ""),
        "exact_answer": item.get("exact_answer", []),
        "documents": item.get("documents", [])
    }
    gold_answers.append(gold)

with open("questions.json", "w") as f:
    json.dump(questions, f, indent=2)

with open("answers_gold.json", "w") as f:
    json.dump(gold_answers, f, indent=2)
