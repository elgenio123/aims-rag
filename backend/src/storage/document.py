"""Document model and storage utilities."""
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from config import INSTITUTION, VALID_CATEGORIES

class Document(BaseModel):
    doc_id: str = Field(..., description="Unique document ID")
    title: str = Field(..., description="Page or section title")
    source_url: str = Field(..., description="Original URL")
    institution: str = Field(default=INSTITUTION, description="Institution name")
    category: str = Field(..., description="Content category")
    raw_text: str = Field(..., description="Cleaned, human-readable content")
    scrape_timestamp: str = Field(..., description="ISO 8601 timestamp")

    def validate_category(self):
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {self.category}")

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
