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
from pydantic import field_validator, model_validator
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
    HTTP_TIMEOUT_SECONDS: int = 30
    ENABLE_WEB_SEARCH: bool = False
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_FILENAME_LENGTH: int = 128

    DOCS_DIR: str = "docs"
    CHROMA_DB_DIR: str = "chroma_db"
    DOCUMENT_REGISTRY_FILENAME: str = ".documents.json"
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_PROJECT: str = "ragent_prod"
    LANGCHAIN_API_KEY: str = ""

    @field_validator(
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_K",
        "MAX_RETRIES",
        "HTTP_TIMEOUT_SECONDS",
        "MAX_UPLOAD_SIZE_MB",
        "MAX_FILENAME_LENGTH",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Configuration values must be greater than zero")
        return value

    @field_validator("BACKEND_CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        cleaned = [origin.strip().rstrip("/") for origin in origins if origin.strip()]
        if not cleaned:
            raise ValueError("BACKEND_CORS_ORIGINS must contain at least one origin")
        return cleaned

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    def require_google_api_key(self) -> None:
        """Validate the Google API key only when model-backed features are used."""
        if not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "your-google-api-key-here":
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Get a key from https://aistudio.google.com/apikey "
                "and add it to your .env file."
            )

    @property
    def docs_path(self) -> Path:
        """Resolved absolute path to the documents directory."""
        return Path(self.DOCS_DIR).resolve()

    @property
    def chroma_path(self) -> Path:
        """Resolved absolute path to the ChromaDB directory."""
        return Path(self.CHROMA_DB_DIR).resolve()

    @property
    def document_registry_path(self) -> Path:
        """Registry file that tracks stable document IDs and ingestion state."""
        return self.docs_path / self.DOCUMENT_REGISTRY_FILENAME

    @property
    def max_upload_size_bytes(self) -> int:
        """Upload limit converted from megabytes to bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()

