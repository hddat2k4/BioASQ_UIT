import torch, os, json, nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import uuid

from pymilvus import Collection, MilvusException, connections, db, utility
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_milvus import Milvus


# --- Kết nối Milvus và reset database ---
db_name = "pubmed"

try:
    connections.connect(uri="http://localhost:19530", token="root:Milvus")
    existing_databases = db.list_database()

    if db_name in existing_databases:
        print(f"Database '{db_name}' already exists.")

        db.using_database(db_name)
        collections = utility.list_collections()
        for collection_name in collections:
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"✅ Collection '{collection_name}' đã bị xoá.")

        db.drop_database(db_name)
        print(f"🗑️ Database '{db_name}' đã bị xoá.")
        connections.disconnect("default")
        connections.connect(uri="http://localhost:19530", token="root:Milvus")

    db.create_database(db_name)
    db.using_database(db_name)
    print(f"✅ Database '{db_name}' đã được tạo mới.")

except MilvusException as e:
    print(f"❌ Có lỗi xảy ra: {e}")


# --- Khởi tạo embedding model ---
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


# --- Kết nối Milvus vector store ---
URI = "http://localhost:19530"

vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="pubmed",
    connection_args={"uri": URI, "token": "root:Milvus", "db_name": "pubmed"},
    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
    consistency_level="Strong",
    drop_old=False,
)


# --- Tiền xử lý và indexing ---
dir = 'pubmed_json_2025'

for i in tqdm(range(12, 14), desc="Indexing pubmed JSON files"):
    file = f"pubmed25n{i:04d}.json"
    path = os.path.join(dir, file)

    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    ids = []

    for article in data:
        title = article.get("title", "").strip()
        abstract = article.get("abstract", "").strip()
        if not title and not abstract:
            continue

        full_text = f"{title}\n{abstract}".strip()
        chunks = sent_tokenize(full_text)

        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4()) 
            doc = Document(
                page_content=chunk,
                metadata={
                    "title": title,
                    "pmid": article["pmid"],
                    "has_abstract": bool(abstract),
                    "chunk" : idx
                }
            )
            documents.append(doc)
            ids.append(chunk_id)

    print(f"📥 Số document từ {file}: {len(documents)}")
    vector_store.add_documents(documents=documents, ids=ids)

print("✅ Hoàn tất indexing các snippet vào Milvus.")
