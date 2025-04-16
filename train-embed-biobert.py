import json, os
from rag import *

res = []
current_dir = os.path.dirname(os.path.abspath(__file__))  # thư mục chứa file .py
json_path = os.path.join(current_dir, 'data', 'questions.json')
output_path = "output/result.txt"
os.makedirs("output", exist_ok=True)

# with open (json_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# for ques in data:
#     tmp = qa_sys(ques)
#     res.append(tmp)

# print(res)

# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(res, f, indent=5, ensure_ascii=False)

# print(f"Write {len(res)} results into {output_path}")

item = {
    "question_id": "5c6aef167c78d69471000023",
    "question": "Is there a deep-learning algorithm for protein solubility prediction?",
    "question_type": "yesno"
}

result = qa_sys(item)
print(json.dumps(result, indent=5, ensure_ascii=False))

client.close()