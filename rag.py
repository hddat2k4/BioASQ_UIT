import weaviate, torch, os, json, re
from prompt import get_prompt
from dotenv import load_dotenv
from langchain_weaviate.vectorstores import WeaviateVectorStore
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -- Load LLM --
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-exp-03-25",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

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
model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
raw_model = SentenceTransformer(model_name, device=device)
embeddings = SentenceTransformerEmbeddings(raw_model)

# -- Connect Weaviate --
client = weaviate.connect_to_local()
vectorstore = WeaviateVectorStore(
    client=client,
    index_name="PubMedAbstract",
    text_key="page_content",
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# -- Helper: extract snippets --
def extract_snippets(docs,scores):
    snippets = []
    doc_id = set()
    for doc, score in zip(docs, scores):
        meta = doc.metadata
        pmid = meta.get("pmid", "")
        chunk_index = meta.get("chunk", 0)

        snippets.append({
            "pmid": pmid,
            "text": doc.page_content,
            "chunk": chunk_index,
            "score": score
        })
        if score>=0.5:
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
    return template | llm | StrOutputParser()

# -- Main QA system --
def qa_sys(item):
    question = item["question"]
    q_type = item["question_type"]
    qid = item["question_id"]

    # Get documents from retriever manually with score
    res = vectorstore.similarity_search_with_score(question, k=15)
    docs = [r[0] for r in res]
    scores = [r[1] for r in res]
    snippets, doc_id = extract_snippets(docs, scores)
    context = extract_context(docs)

    # Build prompt input
    chain_input = {
        "question": question,
        "context": context
    }

    # Get LLM answer
    rag_chain = get_prompt_chain(q_type)
    response = rag_chain.invoke(chain_input)

    # Parse string response into dict (assumes valid JSON)
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
        "snippets": [i for i in snippets if i['score']>=0.5],   # use this one if you wanna get snippet by score
        # "snippets" : snippets, 
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
