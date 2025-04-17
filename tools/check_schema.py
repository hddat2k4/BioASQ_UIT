import weaviate
from sentence_transformers import SentenceTransformer

client = weaviate.connect_to_local()

# Lấy collection bạn muốn truy vấn
collection = client.collections.get("PubMedAbstract")

# Encode dense vector
query = "List clinical symptoms of the MECOM-associated syndrome"
embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
vector = embed_model.encode(query).tolist()

# --- BM25 search ---
print("\n🔍 BM25:")
results = collection.query.bm25(
    query=query,
    limit=5,
    return_metadata=["score"]
)
for o in results.objects:
    print("-", o.properties.get("title", "<no title>"))

# --- Dense search ---
print("\n🧠 Dense:")
results = collection.query.near_vector(
    near_vector=vector,
    limit=5,
    return_metadata=["score"]
)
for o in results.objects:
    print("-", o.properties.get("title", "<no title>"))

# --- Hybrid search ---
print("\n⚖️ Hybrid:")
results = collection.query.hybrid(
    query=query,
    vector=vector,
    alpha=0.5,
    limit=5,
    return_metadata=["score"]
)
for o in results.objects:
    print("-", o.properties.get("title", "<no title>"))
client.close()