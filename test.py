from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()

# Load lại từ Chroma đã lưu
embedding = SentenceTransformerEmbeddings(SentenceTransformer("BAAI/bge-small-en-v1.5"))
vector_store = Chroma(
    collection_name="pubmed_test_bge",
    embedding_function=embedding,
    persist_directory="./chromadb"  # Trùng với nơi đã lưu
)

# Test truy vấn
results = vector_store.similarity_search("Pizotifen as an antidepressant.", k=5)
for doc in results:
    print(doc)

