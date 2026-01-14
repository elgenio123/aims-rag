"""Persistent storage for documents."""
from pathlib import Path
import json
from typing import List
from loguru import logger
from config import DOCUMENTS_PATH
from .document import Document

class DocumentStorage:
    def __init__(self, base_path: str = DOCUMENTS_PATH):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_document(self, doc: Document):
        """Save a single document as JSON."""
        path = self.base_path / f"{doc.doc_id}.json"
        with path.open('w', encoding='utf-8') as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved document {doc.doc_id} -> {path}")

    def load_document(self, doc_id: str) -> Document:
        """Load a document by ID."""
        path = self.base_path / f"{doc_id}.json"
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return Document(**data)

    def list_documents(self) -> List[Path]:
        """List all document JSON files."""
        return list(self.base_path.glob('*.json'))

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists."""
        return (self.base_path / f"{doc_id}.json").exists()
