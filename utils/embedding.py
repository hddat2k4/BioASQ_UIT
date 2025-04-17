import os
import json
import pickle
import gzip
import uuid
from typing import List, Dict, Any

from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import torch

import model

# Configuration
input_folder = ""     # Path to folder containing pubmed25nXXXX.json
output_folder = ""    # Path to save output files
start = 939
end = 941
batch_size = 200

# Load model
embedding_model_name = model.embed_model_name
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer(embedding_model_name, device=device)


def embed_and_save_to_pkl(
    input_folder: str,
    output_folder: str,
    embedding_model,
    start: int,
    end: int,
    batch_size: int = 200
) -> None:
    """
    Đọc các file JSON từ input_folder (pubmed25nXXXX.json),
    tách ở mức câu, embedding và lưu ra file .pkl để đẩy lên Weaviate.
    """
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(start, end + 1), desc="Embedding files"):
        file_name = f"pubmed25n{i:04d}.json"
        file_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f"embeddings_{i:04d}.pkl.gz")

        if not os.path.exists(file_path):
            print(f"⚠️ Không tìm thấy: {file_path}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Lỗi đọc {file_name}: {e}")
            continue

        all_embeddings: List[Dict[str, Any]] = []
        texts, ids, metas = [], [], []

        for article in tqdm(data, desc=f"🧠 Processing {file_name}", leave=False):
            pmid = str(article.get("pmid", "")).strip()
            title = article.get("title", "").strip()
            abstract = article.get("abstract", "").strip()

            full_text = f"{title} {abstract}".strip()
            if not full_text:
                continue

            sentences = sent_tokenize(full_text)
            for idx, sentence in enumerate(sentences):
                uid = f"{pmid}_s{idx}" if pmid else str(uuid.uuid4())

                texts.append(sentence)
                ids.append(uid)
                metas.append({
                    "pmid": pmid,
                    "chunk": idx,
                    "title": title,
                    "abstract": abstract
                })

                if len(texts) >= batch_size:
                    vectors = embedding_model.encode(texts)
                    all_embeddings.extend([
                        {
                            "uuid": u,
                            "page_content": doc,
                            "metadata": meta,
                            "vector": vec
                        }
                        for u, doc, meta, vec in zip(ids, texts, metas, vectors)
                    ])
                    texts, ids, metas = [], [], []

        # Xử lý phần còn dư
        if texts:
            vectors = embedding_model.encode(texts)
            all_embeddings.extend([
                {
                    "uuid": u,
                    "page_content": doc,
                    "metadata": meta,
                    "vector": vec
                }
                for u, doc, meta, vec in zip(ids, texts, metas, vectors)
            ])

        with gzip.open(output_path, "wb") as f_out:
            pickle.dump(all_embeddings, f_out)

        print(f"✅ Đã lưu: {output_path} ({len(all_embeddings)} câu)")

    print("🎉 Hoàn tất embedding toàn bộ.")


# Run the embedding pipeline
embed_and_save_to_pkl(input_folder, output_folder, embedding_model, start, end, batch_size)