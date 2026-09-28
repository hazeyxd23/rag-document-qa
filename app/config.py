"""Application settings, loaded from environment variables and the .env file.

Every other module gets config via get_settings() instead of reading
os.environ directly, so there is one place that defines names, types,
and defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# rag-document-qa/ (this file is rag-document-qa/app/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unrelated variables in .env are not an error
    )

    # --- Generation ---
    llm_provider: Literal["openai", "ollama"] = "openai"
    # SecretStr hides the value in logs/reprs; call .get_secret_value() to use it.
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_timeout_seconds: float = 60.0

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking (characters) ---
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Retrieval ---
    top_k: int = 5

    # --- Paths ---
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    index_dir: Path = PROJECT_ROOT / "data" / "index"

    @model_validator(mode="after")
    def check_values(self) -> "Settings":
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            # overlap >= size would mean the chunker never moves forward
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be positive")
        return self


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it everywhere."""
    return Settings()
