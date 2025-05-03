import json
import os

# Thư mục chứa file .py hiện tại (folder A)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thư mục cha (project root), nơi chứa cả A, B, testcase
project_root = os.path.dirname(current_dir)

# Đường dẫn tới file JSON gốc (trong folder B)
json_path = os.path.join(project_root, "data", "question1.json")

# Đường dẫn tới folder testcase (tự tạo nếu chưa có)
output_dir = os.path.join(project_root, "testcase")
os.makedirs(output_dir, exist_ok=True)

# Đọc dữ liệu
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Không tìm thấy file: {json_path}")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Nếu file là list thì giữ nguyên, nếu là dict chứa "questions" thì lấy ra
if isinstance(data, dict) and "questions" in data:
    questions_raw = data["questions"]
elif isinstance(data, list):
    questions_raw = data
else:
    raise ValueError("❌ Dữ liệu không đúng định dạng mong đợi (list hoặc dict có 'questions')")

questions = []
gold_answers = []

for item in questions_raw:
    questions.append({
        "question_id": item["id"],
        "question": item["body"],
        "question_type": item["type"]
    })
    gold_answers.append(item)

# Đường dẫn file output
out_questions_path = os.path.join(output_dir, "questions1.json")
out_answers_path = os.path.join(output_dir, "answers1.json")

# Ghi ra file
with open(out_questions_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

with open(out_answers_path, "w", encoding="utf-8") as f:
    json.dump(gold_answers, f, indent=2, ensure_ascii=False)

print(f"✅ Đã lưu file vào: {output_dir}")
