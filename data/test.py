from weaviate import Client

client = Client("http://localhost:8080")

if client.is_ready():
    print("✅ Weaviate is ready!")
else:
    print("❌ Weaviate is NOT ready.")
