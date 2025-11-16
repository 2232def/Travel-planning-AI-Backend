from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from app.services.workflow import run_workflow
load_dotenv()

# from ..app.services.retriever import QdrantRetriever
from app.services.llm import generate_query_or_respond

router = APIRouter(prefix="/api")

class AskRequest(BaseModel):
    question: str
    k: int = 5

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/ask")
def ask(request: AskRequest):
    """
    Ask a travel-related question using the agentic RAG workflow.
    
    The workflow:
    1. Decides if retrieval is needed
    2. Retrieves relevant documents from Qdrant
    3. Grades document relevance
    4. Rewrites question if needed
    5. Generates final answer
    """
    try:
        answer = run_workflow(request.question, request.k)
        return {"answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))