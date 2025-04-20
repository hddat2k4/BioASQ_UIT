# -- Main QA system --
import torch, os, json, re
from rm3 import expand_query
from prompt import get_prompt
from dotenv import load_dotenv
from utils import model
from itertools import cycle, islice
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

api_keys = os.getenv("GOOGLE_API_KEYS").split(",")
api_key_iterator = cycle(api_keys)

def get_next_api_key():
    return next(api_key_iterator)

def get_llm_with_retry(model_name, max_tries=None):
    max_tries = max_tries or len(api_keys)
    for key in islice(api_key_iterator, max_tries):
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
            )
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
import weaviate
from weaviate.auth import Auth
from weaviate.config import AdditionalConfig, Timeout

client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
    additional_config=AdditionalConfig(
        timeout=Timeout(
            init=30,    # timeout khi khởi tạo kết nối
            query=120,   # timeout cho các truy vấn (near_vector, hybrid, v.v.)
            insert=120  # timeout khi insert / batch add
        )
    )
)
collection = client.collections.get("bgebase")   #Pubmedfull

# -- SDK Retrieval Function --
def retrieve_docs_sdk(query, mode="bm25", top_k=10, alpha=None):
    vector = embeddings.embed_query(query)

    if mode == "bm25":
        res = collection.query.bm25(
            query=query,
            limit=top_k,
            return_metadata=["score"],
        )
    elif mode == "hybrid":
        if alpha is None:
            alpha = 0.5
        res = collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=alpha,
            limit=top_k,
            return_metadata=["score"],
        )
    elif mode == "dense":
        res = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=["score"],
        )
    else:
        raise ValueError("Only 'dense', 'bm25' and 'hybrid' are supported.")

    results = []
    for o in res.objects:
        results.append(Document(
            page_content=o.properties.get("page_content", ""),
            metadata={
                "pmid": o.properties.get("pmid"),
                "chunk": o.properties.get("chunk", 0),
                "score": getattr(o.metadata, "score", 0.0)
            }
        ))
    return results

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
            "chunk": chunk_index
        })
        doc_id.add(pmid)
    return snippets, list(doc_id)

# -- Helper: clean output of LLM --
def clean_output(text):
    if isinstance(text, dict): 
        return json.dumps(text)
    match = re.search(r'{.*}', text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("Không tìm thấy JSON hợp lệ.")

# -- Prompt chain creator --
def get_prompt_chain(question_type):
    template = get_prompt(question_type)
    llm = get_llm_with_retry(model.llm_model_name)
    return template | llm | StrOutputParser()

# -- Main QA system --
def qa_sys(item, retrieval_mode="dense", top_k=10, n_terms=7, lambda_param=0.6):
    question = item["question"]
    q_type = item["question_type"]
    qid = item["question_id"]

    # --- Truy xuất tài liệu bằng SDK ---
    if retrieval_mode == "bm25":
        docs = retrieve_docs_sdk(question, mode="bm25", top_k=top_k)
        expanded_q = expand_query(question, docs, n_terms=n_terms, lambda_param=lambda_param)
        docs = retrieve_docs_sdk(expanded_q, mode="bm25", top_k=top_k)
    elif retrieval_mode == "hybrid":
        dense_docs = retrieve_docs_sdk(question, mode="dense", top_k=top_k)
        expanded_q = expand_query(question, dense_docs, n_terms=n_terms, lambda_param=lambda_param)
        docs = retrieve_docs_sdk(expanded_q, mode="hybrid", top_k=top_k, alpha=0.4)
    else:
        docs = retrieve_docs_sdk(question, mode="dense", top_k=top_k)

    # --- Prompt context & snippets ---
    snippets, doc_id = extract_snippets(docs)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain_input = {
        "question": question,
        "context": context
    }

    rag_chain = get_prompt_chain(q_type)
    response = rag_chain.invoke(chain_input)

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

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback

def run_batch_qa(questions, retrieval_mode="dense", top_k=10, batch_size=5, max_retry=3):
    results = []
    total = len(questions)
    
    def process_one(item):
        qid = item["question_id"]
        for attempt in range(max_retry):
            try:
                answer = qa_sys(item, retrieval_mode=retrieval_mode, top_k=top_k)
                print(f"✅ QID {qid} done.")
                return answer
            except Exception as e:
                print(f"⚠️ Retry {attempt+1}/{max_retry} for QID {qid} — {e}")
                traceback.print_exc()
                time.sleep(1)
        print(f"❌ QID {qid} failed after {max_retry} retries.")
        return {
            "id": item["question_id"],
            "body": item["question"],
            "type": item["question_type"],
            "documents": [],
            "snippets": [],
            "exact_answer": None,
            "ideal_answer": None
        }

    for i in range(0, total, batch_size):
        batch = questions[i:i + batch_size]
        print(f"\n🚀 Processing batch {i+1}–{i+len(batch)} of {total}")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_item = {executor.submit(process_one, item): item for item in batch}
            for future in as_completed(future_to_item):
                results.append(future.result())
        time.sleep(1)  # nghỉ giữa các batch

    return results

