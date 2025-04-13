from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from graph import graph  # <-- import graph từ file của bạn

app = FastAPI()


# ---------------------
# POST: body-based input
# ---------------------
class QuestionRequest(BaseModel):
    question: str


@app.post("/qa")
def qa_post(request: QuestionRequest):
    state = graph.invoke({"question": request.question})
    return format_response(request.question, state)


# ---------------------
# GET: query param input
# ---------------------
@app.get("/qa")
def qa_get(question: str = Query(..., description="Question to be answered")):
    state = graph.invoke({"question": question})
    return format_response(question, state)


# ---------------------
# Trả kết quả dưới dạng JSON
# ---------------------
def format_response(question: str, state: dict) -> dict:
    # retrieved = [
    #     {
    #         "pmid": doc.metadata.get("pmid", "N/A"),
    #         "chunk_num": doc.metadata.get("chunk", "N/A"),
    #         "title": doc.metadata.get("title", "N/A"),
    #         "snippet": doc.page_content.strip()
    #     }
    #     for doc in state.get("retrieved_docs", [])
    # ]

    return {
        "question": question,
        "question_type": state.get("type", "N/A"),
        "answer": state.get("answer", "No answer"),
        # "retrieved_docs": retrieved,
        # "titles": state.get("titles", []),
        # "pmids": state.get("pmids", []),
    }
