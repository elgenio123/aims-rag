"""HTTP API around RagPipeline. Run: uvicorn src.api.app:app --host 0.0.0.0 --port 8000"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.rag import RagPipeline
from src.utils import setup_logging

setup_logging()
pipeline: Optional[RagPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RagPipeline()
    yield


app = FastAPI(title="AIMS Cameroon RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)


class TraceItem(BaseModel):
    doc_id: Optional[str]
    source_url: Optional[str]
    score: Optional[float]


class AskResponse(BaseModel):
    answer: str
    trace: List[TraceItem]
    used_chunks: int


@app.get("/health")
def health():
    try:
        chunks = pipeline.indexer.count()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ok", "chunks": chunks}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be blank")
    try:
        return pipeline.answer(question, top_k=req.top_k)
    except Exception as e:
        logger.exception(f"Failed to answer question: {e}")
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable. Please try again.")
