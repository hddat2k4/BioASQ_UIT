import json
import os
from rag import run_batch_qa, qa_sys, client

res = []
current_dir = os.path.dirname(os.path.abspath(__file__))  # thư mục chứa file .py
json_path = os.path.join(current_dir, 'testcase', 'questions1.json')

def feed_llm(isMultiple=False, retrieval_mode="dense", top_k=10):
    filename = f"predictions_{retrieval_mode}.json"
    output_path = os.path.join(current_dir, 'testcase', filename)

    if isMultiple:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = run_batch_qa(data, retrieval_mode=retrieval_mode, top_k=top_k)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=5, ensure_ascii=False)

        print(f"\n✅ Write {len(results)} results into {output_path}")

    else:
        item = {
            "question_id": "5c6aef167c78d69471000023",
            "question": "Is there a deep-learning algorithm for protein solubility prediction?",
            "question_type": "yesno"
        }

        result = qa_sys(item, retrieval_mode=retrieval_mode, top_k=top_k)
        print(json.dumps(result, indent=5, ensure_ascii=False))

    client.close()

# ✅ True nếu muốn trả lời tất cả câu hỏi trong testcase/questions1.json
# ❌ False nếu chỉ muốn test 1 câu hỏi mẫu
feed_llm(isMultiple=True, retrieval_mode="hybrid", top_k=10)

# --- Mở rộng nếu cần chạy cả 3 chế độ:
# if __name__ == "__main__":
#     modes = ["dense", "bm25", "hybrid"]
#     for mode in modes:
#         feed_llm(isMultiple=True, retrieval_mode=mode)
