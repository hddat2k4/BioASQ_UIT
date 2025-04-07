import torch
import os
from tqdm import tqdm
import json
from nltk import tokenize
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter




# Chia batch
def batched(iterable, n=166):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


# Cấu hình hàm embedding theo chuẩn của langchain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, show_progress_bar=True, convert_to_numpy=True).tolist()


model_name = "BAAI/bge-small-en-v1.5"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
raw_model = SentenceTransformer(model_name, device=device)
embeddings = SentenceTransformerEmbeddings(raw_model)

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)



vector_store = Chroma(
    collection_name="pubmed_test_bge",
    embedding_function=embeddings,
    persist_directory="./chromadb", 
)



dir = 'pubmed_json_2025'

for i in tqdm(range(12, 15), desc="Indexing pubmed JSON files"):
    file = f"pubmed25n{i:04d}.json"
    path = os.path.join(dir, file)

    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    all_ids, all_texts, all_metas = [], [], []

    # Duyệt từng bài báo
    for article in data:
        full_text = f"{article['title']}\n{article['abstract']}"
        chunks = splitter.split_text(full_text)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{article['pmid']}_{idx}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append({
                "title": article["title"],
                "pmid": article["pmid"]
            })

    # Đẩy lên Chroma theo batch
    for batch_texts, batch_ids, batch_metas in tqdm(
        zip(batched(all_texts), batched(all_ids), batched(all_metas)),
        total=(len(all_texts) // 166) + 1,
        desc=f"→ Pushing {file}"
    ):
        vector_store.add_texts(
            texts=batch_texts,
            metadatas=batch_metas,
            ids=batch_ids
        )

print("✅ Hoàn tất indexing các snippet vào ChromaDB.")
