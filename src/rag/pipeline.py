"""LangChain RAG pipeline using Mistral via direct API or OpenRouter.

Ensures strict context grounding through a system message.
"""
from typing import List, Dict
from loguru import logger
from langchain.schema import Document as LCDocument, SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from config import (
    MISTRAL_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_X_TITLE,
)
from src.embedder import Indexer

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "If the answer is not present, reply: 'The information is not available in the current documents.'"
)

QA_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "System: " + SYSTEM_PROMPT + "\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

class RagPipeline:
    def __init__(self):
        # Defer LLM initialization until actually needed (query time)
        self.llm = None
        self.indexer = Indexer()
        try:
            self.indexer.load()
        except Exception:
            logger.warning("Vector index not found; build it first.")

    def _ensure_llm(self):
        if self.llm is not None:
            return
        # Prefer OpenRouter if configured; otherwise use direct Mistral
        if OPENROUTER_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
            except Exception as e:
                raise RuntimeError(
                    "langchain-openai is required for OpenRouter. Install it and retry."
                ) from e
            default_headers = {}
            if OPENROUTER_HTTP_REFERER:
                default_headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
            if OPENROUTER_X_TITLE:
                default_headers["X-Title"] = OPENROUTER_X_TITLE
            self.llm = ChatOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                model=OPENROUTER_MODEL,
                default_headers=default_headers or None,
            )
            logger.info(f"Using OpenRouter model: {OPENROUTER_MODEL}")
        else:
            if not MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY is not set and OPENROUTER_API_KEY not provided")
            from langchain_mistralai import ChatMistralAI
            self.llm = ChatMistralAI(api_key=MISTRAL_API_KEY)
            logger.info("Using direct Mistral API")

    def build_index(self, chunks: List[Dict]):
        self.indexer.add(chunks)
        self.indexer.save()
        logger.info("Vector index built and saved.")

    def _format_context(self, metas: List[Dict]) -> str:
        parts = []
        for m in metas:
            src = m.get('source_url', '')
            did = m.get('doc_id', '')
            cat = m.get('category', '')
            text = m.get('text', '')
            parts.append(f"[doc_id={did} | category={cat} | source={src}]\n{text}")
        return "\n\n".join(parts)

    def answer(self, question: str, top_k: int = 4) -> Dict:
        self._ensure_llm()
        results = self.indexer.search(question, top_k=top_k)
        context = self._format_context(results)
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        logger.info(f"Retrieved {len(results)} chunks for question")
        # Use structured messages to enforce system constraints
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        resp = self.llm.invoke(messages)
        answer_text = resp.content if hasattr(resp, 'content') else str(resp)
        trace = [{
            'doc_id': r.get('doc_id'),
            'source_url': r.get('source_url'),
            'score': r.get('score'),
        } for r in results]
        return {
            'answer': answer_text,
            'trace': trace,
            'used_chunks': len(results),
        }
