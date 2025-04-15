import torch, os
from dotenv import load_dotenv
from BioASQ.oldtest.classifier import question_type
from pymilvus import connections, db, Collection
from langchain_milvus import Milvus
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI



## Kết nối với API Gemini
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Gemini chat text
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# Kết nối tới Milvus server
connections.connect(uri="http://localhost:19530", token="root:Milvus")

# Kết nối tới database và collection được tạo trước đó
db.using_database("pubmed")
collection = Collection("pubmed")
collection.load()


# Khởi tạo hàm embedding

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


## Kết nối tới vector database Milvus
URI = "http://localhost:19530"

vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="pubmed",
    connection_args={"uri": URI, "token": "root:Milvus", "db_name": "pubmed"},
    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
    consistency_level="Strong",
    drop_old=False,
)



## Khai báo dạng câu prompt, query 
from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Optional, Literal


QuestionType = Literal["yesno", "factoid", "summary", "list"]



class State(TypedDict):
    question: str                          # Câu hỏi từ người dùng
    retrieved_docs: List[Document]        # Danh sách Document từ Chroma/PubMed
    answer: Optional[str]                 # Câu trả lời do LLM sinh ra
    pmids: Optional[List[str]]            # Danh sách PMID của các bài báo
    snippets: Optional[List[str]]         # Các đoạn trích phù hợp (context ngắn)
    titles: Optional[List[str]]           # Tiêu đề của các tài liệu liên quan
    type: QuestionType



from langchain_core.prompts import PromptTemplate
## tách riêng, để json, bỏ type (chú ý dạng factoid)
## chú ý dạng đầu ra
template = """
You are a biomedical assistant.

Use the following article titles, snippets from PubMed to answer the question below.
Based on the question type given below, generate an appropriate answer:


- For **yes/no** questions, respond with "yes" or "no", and provide a brief explanation or supporting evidence from the snippets.
- For **factoid** questions, respond with a short, factual answer — typically a named entity such as a person, place, date, number, or specific biomedical term.
- For **list** questions, respond with a concise, comma-separated list of relevant biomedical terms or entities.
- For **summary** questions, respond with a brief paragraph (2–3 sentences) summarizing the key points from the snippets.


Titles:
{titles}

Snippets:
{snippets}

Type:
{type}


Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(template)

# print(collection.schema)

# ques = "What bacterium is associated with gastroenteritis in children according to studies in London and Jamaica?"
# print(f"Question: {ques}") 
# results = vector_store.similarity_search(ques, k=10)
# for i, doc in enumerate(results):
#     print(f"\nResult {i+1}")
#     print(f"Content: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")



def retrieve(state: State) -> State:
    # Truy xuất từ vector store
    retrieved_docs = vector_store.similarity_search(state["question"], k=10)
    # Trích thông tin cần từ metadata + nội dung
    snippets = [doc.page_content for doc in retrieved_docs]
    titles = [doc.metadata.get("title", "") for doc in retrieved_docs]
    type = question_type(state["question"])


    # Trả về state mới (hoặc cập nhật lại state cũ)
    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "snippets": snippets,
        "titles": titles,
        "type" : type
    }


def generate(state: State):
    # docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "snippets": state['snippets'], "titles": state["titles"], "type": state["type"]})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])

graph_builder.add_edge(START, "retrieve")

graph = graph_builder.compile()

# for step in graph.stream(
#     {"question": "What bacterium is associated with gastroenteritis in children according to studies in London and Jamaica?"},
#     stream_mode="updates",
# ):
#     print(f"{step}\n\n----------------\n")

ques = "What bacterium is associated with gastroenteritis in children according to studies in London and Jamaica?"
print(f"Question: {ques}") 
result = graph.invoke({"question": ques})
print("Retrieved Docs:")
for doc in result["retrieved_docs"]:
    pmid = doc.metadata.get("pmid","")
    chunk_num = doc.metadata.get("chunk","")
    title = doc.metadata.get("title", "")
    snippet = doc.page_content

    print({
        "pmid": pmid,
        "chunk_num": chunk_num,
        "title": title,
        "snippet": snippet
    })

print("Answer: ",result["answer"])


