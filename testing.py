import json, os
from rag import *

res = []
current_dir = os.path.dirname(os.path.abspath(__file__))  # thư mục chứa file .py
json_path = os.path.join(current_dir, 'testcase', 'questions.json')

def feed_llm(isMultiple=False, retrieval_mode="dense"):
    filename = f"predictions_{retrieval_mode}.json"
    output_path = os.path.join(current_dir, 'testcase', filename)

    if isMultiple:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for ques in data:
            tmp = qa_sys(ques, retrieval_mode=retrieval_mode)
            res.append(tmp)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=5, ensure_ascii=False)

        print(f"✅ Write {len(res)} results into {output_path}")

    else:
        item = {
            "question_id": "5c6aef167c78d69471000023",
            "question": "Is there a deep-learning algorithm for protein solubility prediction?",
            "question_type": "yesno"
        }

        result = qa_sys(item, retrieval_mode=retrieval_mode)
        print(json.dumps(result, indent=5, ensure_ascii=False))
    client.close()

##### True neu muon tra loi full 16 cau hoi trong testcase/questions.json, false neu chi muon tra loi 1 cau hoi mau
feed_llm(isMultiple=True, retrieval_mode="bm25")