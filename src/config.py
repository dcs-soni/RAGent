"""
Configuration Module
====================
Centralizes all settings using Pydantic for validation.

Developer Thinking:
    Instead of scattering magic numbers and env vars across files, we define
    a single Settings class. This gives us:
    - Type validation at startup (fail fast if GOOGLE_API_KEY is missing)
    - IDE autocomplete everywhere (settings.CHUNK_SIZE, not os.getenv("CHUNK_SIZE"))
    - Single source of truth for all tunable parameters
"""

from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    GOOGLE_API_KEY: str = ""

    LLM_MODEL: str = "gemini-2.0-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    RETRIEVAL_K: int = 4
    RELEVANCE_THRESHOLD: float = 0.7

    MAX_RETRIES: int = 3

    DOCS_DIR: str = "docs"
    CHROMA_DB_DIR: str = "chroma_db"

    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_PROJECT: str = "ragent_prod"
    LANGCHAIN_API_KEY: str = ""

    @model_validator(mode="after")
    def validate_api_key(self) -> "Settings":
        """Fail fast if GOOGLE_API_KEY is not set — prevents confusing errors later."""
        if not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "your-google-api-key-here":
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Get a free key from https://aistudio.google.com/apikey "
                "and add it to your .env file."
            )
        return self

    @property
    def docs_path(self) -> Path:
        """Resolved absolute path to the documents directory."""
        return Path(self.DOCS_DIR).resolve()

    @property
    def chroma_path(self) -> Path:
        """Resolved absolute path to the ChromaDB directory."""
        return Path(self.CHROMA_DB_DIR).resolve()


settings = Settings()

