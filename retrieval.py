import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Gemini chat text
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# wrapper như bạn đã làm
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





from langchain_core.prompts import PromptTemplate

template = """
You are a biomedical assistant.

Use the following article titles and snippets from PubMed to answer the question below.

- For **yes/no** questions, respond with only "yes" or "no".
- For **factoid** questions, respond with a short, factual answer — typically a named entity such as a person, place, date, number, or specific biomedical term.

Titles:
{titles}

Snippets:
{snippets}



Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(template)



from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Optional


class State(TypedDict):
    question: str                          # Câu hỏi từ người dùng
    retrieved_docs: List[Document]        # Danh sách Document từ Chroma/PubMed
    answer: Optional[str]                 # Câu trả lời do LLM sinh ra
    pmids: Optional[List[str]]            # Danh sách PMID của các bài báo
    snippets: Optional[List[str]]         # Các đoạn trích phù hợp (context ngắn)
    titles: Optional[List[str]]           # Tiêu đề của các tài liệu liên quan


def analyze_query(state: State):
    return {"query": {"query": state["question"]}}

def retrieve(state: State) -> State:
    # Truy xuất từ vector store
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)

    # Trích thông tin cần từ metadata + nội dung
    snippets = [doc.page_content for doc in retrieved_docs]
    titles = [doc.metadata.get("title", "") for doc in retrieved_docs]

    # Trả về state mới (hoặc cập nhật lại state cũ)
    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "snippets": snippets,
        "titles": titles,
    }


def generate(state: State):
    # docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "snippets": state['snippets'], "titles": state["titles"]})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])

graph_builder.add_edge(START, "analyze_query")

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
    pmid, chunk_num = doc.id.split("_")

    title = doc.metadata.get("title", "")
    snippet = doc.page_content

    print({
        "pmid": pmid,
        "chunk_num": chunk_num,
        "title": title,
        "snippet": snippet
    })

print("Answer: ",result["answer"])