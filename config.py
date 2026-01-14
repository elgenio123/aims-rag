"""Configuration management for AIMS RAG pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Mistral API (direct)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# OpenRouter (OpenAI-compatible) configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "")

# Vector Database
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(VECTOR_DB_DIR))

# Document Storage
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", str(DOCUMENTS_DIR))

# Scraping Configuration
SCRAPE_DELAY_SECONDS = float(os.getenv("SCRAPE_DELAY_SECONDS", "2.0"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "5"))
RESPECT_ROBOTS_TXT = os.getenv("RESPECT_ROBOTS_TXT", "true").lower() == "true"
USER_AGENT = "AIMS-RAG-Bot/1.0 (Educational Purpose)"

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "75"))

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")  # options: sentence-transformers, fastembed

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "pipeline.log"))

# Institution
INSTITUTION = "AIMS Cameroon"

# Valid categories
VALID_CATEGORIES = [
    "Admissions",
    "Academics",
    "Scholarships",
    "Faculty",
    "Student Life",
    "Institutional",
    "Research",
    "Administration",
    "FAQs",
    "Policies"
]
