import weaviate
import weaviate.classes as wvc

# Kết nối tới Weaviate local
client = weaviate.connect_to_local()

client.collections.delete("PubMedAbstract")

client.collections.create(
    name="PubMedAbstract",
    description="PubMed abstract entries",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    properties=[
        wvc.config.Property(name="page_content", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="pmid", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="abstract", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="chunk", data_type=wvc.config.DataType.INT),
    ]
)

print("Đã tạo schema 'PubMedAbstract' thành công.")
client.close()


