import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
    VectorDistances, 
)

# Kết nối tới Weaviate local
client = weaviate.connect_to_local()
names = "gist"

# Xoá collection cũ nếu cần
client.collections.delete(names)  #"PubMedAbstract"
# # # Tạo schema mới có hỗ trợ hybrid (BM25 + vector)

client.collections.create(
    name=names,
    description="PubMed abstract entries",
    inverted_index_config=Configure.inverted_index(bm25_b=0.75, bm25_k1=1.2),
    vectorizer_config=Configure.Vectorizer.none(),
    vector_index_config=Configure.VectorIndex.flat(distance_metric=VectorDistances.COSINE),
    properties=[
        Property(name="page_content", data_type=DataType.TEXT),
        Property(name="pmid", data_type=DataType.TEXT),
        Property(name="title", data_type=DataType.TEXT),
        Property(name="abstract", data_type=DataType.TEXT),
        Property(name="chunk", data_type=DataType.INT),
    ]
)

print(f"✅ Created schema {names} successfully.")
client.close()