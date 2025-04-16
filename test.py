from rag import *

"""
Dùng để test thử việc lấy n tài liệu đầu tiên trong collection trên Weaviate
"""

def get_documents(n=5):
    collection = client.collections.get("PubMedAbstract")
    results = collection.query.fetch_objects(limit=n)

    if not results.objects:
        return []

    documents = []
    for obj in results.objects:
        documents.append({
            "id": obj.uuid,
            "pmid": obj.properties.get("pmid"),
            "chunk": obj.properties.get("chunk"),
            "page_content": obj.properties.get("page_content")
        })
    return documents

# --- Gọi và in kết quả ---
docs = get_documents(n=5)
if docs:
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print("PMID:", doc["pmid"])
        print("Chunk:", doc["chunk"])
        print("Content:", doc["page_content"])
else:
    print("Không tìm thấy tài liệu.")

client.close()