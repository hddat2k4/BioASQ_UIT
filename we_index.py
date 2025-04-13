import torch, os, json, nltk, uuid
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# --- Import thư viện cần thiết ---
import weaviate
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Weaviate  # Sử dụng vector store của Weaviate từ LangChain
from sentence_transformers import SentenceTransformer

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

# --- Kết nối đến Weaviate vector store ---
# Giả sử Weaviate đang chạy tại http://localhost:8080 (điều này phù hợp với cấu hình Docker của bạn)
client = weaviate.Client("http://localhost:8080")

# --- Tùy chọn: Kiểm tra và tạo schema (class) nếu chưa tồn tại ---
# Chúng ta định nghĩa một schema tên "Document" với các trường "page_content" và "metadata"
schema = client.schema.get()
if not any(cls["class"] == "Document" for cls in schema.get("classes", [])):
    document_schema = {
        "class": "Document",
        "properties": [
            {"name": "page_content", "dataType": ["text"]},
            # Nếu muốn lưu metadata dưới dạng text (chuỗi JSON) thì cần thêm trường này,
            # hoặc bạn có thể lưu các thuộc tính riêng biệt nếu muốn.
            {"name": "metadata", "dataType": ["text"]},
        ],
        # "vectorizer": "none" có nghĩa là bạn sẽ cung cấp vector embeddings bên ngoài (không dùng vectorizer của Weaviate)
        "vectorizer": "none"
    }
    client.schema.create_class(document_schema)
    print("✅ Đã tạo schema 'Document' trong Weaviate.")

# --- Khởi tạo vector store sử dụng Weaviate ---
# index_name ở đây chính là tên của class/schema ("Document")
# text_key là trường chứa nội dung chính của tài liệu ("page_content")
# embedding_function là hàm chuyển đổi text thành vector (ở đây sử dụng embeddings.embed_query)
vector_store = Weaviate(
    client,
    index_name="Document",
    text_key="page_content",
    embedding_function=embeddings.embed_query
)

# --- Tiền xử lý và indexing ---
# Giả sử thư mục 'pubmed_json_2025' chứa các file JSON cần index
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
                    "pmid": article.get("pmid", ""),
                    "has_abstract": bool(abstract),
                    "chunk": idx
                }
            )
            documents.append(doc)
            ids.append(chunk_id)

    print(f"📥 Số document từ {file}: {len(documents)}")
    vector_store.add_documents(documents=documents, ids=ids)

print("✅ Hoàn tất indexing các snippet vào Weaviate.")
