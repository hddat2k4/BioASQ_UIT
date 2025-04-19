# -- Main QA system --
import weaviate, torch, os, json, re
from rm3 import expand_query
from prompt import get_prompt
from dotenv import load_dotenv
from utils import model
from itertools import cycle, islice
from langchain_weaviate.vectorstores import WeaviateVectorStore
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# # -- Load LLM --
# llm = ChatGoogleGenerativeAI(
#     model=model.llm_model_name,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

api_keys = os.getenv("GOOGLE_API_KEYS").split(",")
api_key_iterator = cycle(api_keys)

def get_next_api_key():
    return next(api_key_iterator)

# Retry LLM wrapper: thử lần lượt các key cho đến khi thành công
def get_llm_with_retry(model_name, max_tries=None):
    max_tries = max_tries or len(api_keys)
    for key in islice(api_key_iterator, max_tries):
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
            )
            # Gọi thử để kiểm tra key (hoặc bỏ nếu muốn lười kiểm)
            return llm
        except Exception as e:
            print(f"[⚠️] API key {key[:6]}... bị lỗi: {e}")
            continue
    raise RuntimeError("❌ Tất cả API key đều bị lỗi hoặc bị giới hạn.")

# -- Embedding wrapper --
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        assert isinstance(text, str), f"Expected string for query, got {type(text)}"
        return self.model.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()

# -- Load model --
model_name = model.embed_model_name
device = 'cuda' if torch.cuda.is_available() else 'cpu'
raw_model = SentenceTransformer(model_name, device=device)
embeddings = SentenceTransformerEmbeddings(raw_model)

# -- Connect Weaviate --
client = weaviate.connect_to_local()
vectorstore = WeaviateVectorStore(
    client=client,
    #index_name="PubMedAbstract",
    index_name = "Pubmedfull",
    text_key="page_content",
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# -- Helper: extract snippets --
def extract_snippets(docs):
    snippets = []
    doc_id = set()
    for doc in docs:
        meta = doc.metadata
        pmid = meta.get("pmid", "")
        chunk_index = meta.get("chunk", 0)

        snippets.append({
            "pmid": pmid,
            "text": doc.page_content,
            "chunk": chunk_index,
            #"score": score
        })
        # if score>=0.4:
        #     doc_id.add(pmid)
        doc_id.add(pmid)
    return snippets, list(doc_id)

# -- Helper: clean output of LLM --
def clean_output(text):
    if isinstance(text, dict): 
        return json.dumps(text)
    
    match = re.search(r'{.*}', text, re.DOTALL)
    if match:
        return match.group(0)  # chuỗi JSON
    raise ValueError("Không tìm thấy JSON hợp lệ.")

# -- Helper: extract context for prompt --
def extract_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# -- Prompt chain creator --
def get_prompt_chain(question_type):
    template = get_prompt(question_type)
    llm = get_llm_with_retry(model.llm_model_name)
    return template | llm | StrOutputParser()

def retrieve_docs_sdk(query, mode="bm25", top_k=10, alpha=None):
    #collection = client.collections.get("PubMedAbstract")
    collection = client.collections.get("Pubmedfull")
    vector = embeddings.embed_query(query)

    if mode == "bm25":
        res = collection.query.bm25(
            query=query,
            limit=top_k,
            return_metadata=["score"]
        )
    elif mode == "hybrid":
        if alpha is None:
            alpha = 0.5
        res = collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=alpha,
            limit=top_k,
            return_metadata=["score"]
        )
    else:
        raise ValueError("Only 'bm25' and 'hybrid' are supported here.")

    results = []
    for o in res.objects:
        results.append({
            "pmid": o.properties.get("pmid"),
            "text": o.properties.get("page_content"),
            "chunk": o.properties.get("chunk", 0),
            "score": o.metadata.get("score", 0.0)
        })
    return results


# -- Main QA system --
def qa_sys(item, retrieval_mode="dense", top_k=10, n_terms=7, lambda_param=0.6):
    question = item["question"]
    q_type = item["question_type"]
    qid = item["question_id"]

    # --- Lấy tài liệu ban đầu theo retrieval mode ---
    if retrieval_mode == "dense":
        docs = vectorstore.similarity_search(question, alpha=1, k=top_k)

    elif retrieval_mode == "bm25":
        docs = vectorstore.similarity_search(question,alpha = 0, k=top_k)
        expanded_q = expand_query(question, docs, n_terms=n_terms, lambda_param=0.6)
        docs = vectorstore.similarity_search(question,alpha = 0, k=top_k)

    elif retrieval_mode == "hybrid":
        # Bước 1: Lấy tài liệu initial theo dense
        dense_docs = vectorstore.similarity_search(question, alpha=1, k=top_k)

        # Bước 2: Mở rộng câu hỏi bằng RM3
        expanded_q = expand_query(question, dense_docs, n_terms=n_terms, lambda_param=0.6)

        # Bước 3: Truy xuất lại bằng hybrid
        docs = vectorstore.similarity_search(query=expanded_q, alpha=0.4, k=top_k)  # alpha có thể điều chỉnh cho hybrid

    else:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    # --- Trích xuất đoạn văn và context ---
    snippets, doc_id = extract_snippets(docs)
    context = extract_context(docs)

    # --- Xây dựng đầu vào prompt ---
    chain_input = {
        "question": question,
        "context": context
    }

    rag_chain = get_prompt_chain(q_type)
    response = rag_chain.invoke(chain_input)

    # --- Xử lý kết quả ---
    try:
        if isinstance(response, str):
            cleaned = clean_output(response)
            parsed = json.loads(cleaned)
        else:
            parsed = response                  
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = {
            "exact_answer": None,
            "ideal_answer": str(response)      
        }

    return {
        "id": qid,
        "body": question,
        "type": q_type,
        "documents": doc_id,
        "snippets": snippets,
        "exact_answer": parsed.get("exact_answer"),
        "ideal_answer": parsed.get("ideal_answer")
    }



## -- Retrieve docs --
def retrieve_docs(item):
    question = item["question"]
    qid = item["question_id"]
    q_type = item["question_type"]

    res = vectorstore.similarity_search_with_score(question, k=15)
    docs = [r[0] for r in res]
    scores = [r[1] for r in res]


    retrieved_docs = []
    for doc, score in zip(docs, scores):
        meta = doc.metadata
        pmid = meta.get("pmid", "")
        chunk_index = meta.get("chunk", 0)
        retrieved_docs.append({
            "pmid": pmid,
            "text": doc.page_content,
            "chunk": chunk_index,
            "score": score
        })

    return {
        "question_id": qid,
        "question": question,
        "question_type": q_type,
        "retrieved_docs": retrieved_docs
    }



## -- Retrieve docs --
def retrieve_docs(item):
    question = item["question"]
    qid = item["question_id"]
    q_type = item["question_type"]

    res = vectorstore.similarity_search_with_score(question, k=15)
    docs = [r[0] for r in res]
    scores = [r[1] for r in res]


    retrieved_docs = []
    for doc, score in zip(docs, scores):
        meta = doc.metadata
        pmid = meta.get("pmid", "")
        chunk_index = meta.get("chunk", 0)
        retrieved_docs.append({
            "pmid": pmid,
            "text": doc.page_content,
            "chunk": chunk_index,
            "score": score
        })

    return {
        "question_id": qid,
        "question": question,
        "question_type": q_type,
        "retrieved_docs": retrieved_docs
    }