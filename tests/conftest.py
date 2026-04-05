import os

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("API_KEY", "test-key-for-auth")

from src.api import app
from src.config import settings
from src.rate_limiter import rate_limiter


settings.API_KEY = os.environ["API_KEY"]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_security_limits():
    original_values = {
        "API_KEY": settings.API_KEY,
        "MAX_UPLOAD_SIZE_MB": settings.MAX_UPLOAD_SIZE_MB,
        "MAX_UPLOAD_REQUEST_OVERHEAD_KB": settings.MAX_UPLOAD_REQUEST_OVERHEAD_KB,
        "UPLOAD_READ_CHUNK_SIZE_KB": settings.UPLOAD_READ_CHUNK_SIZE_KB,
        "MAX_CHAT_REQUEST_SIZE_KB": settings.MAX_CHAT_REQUEST_SIZE_KB,
        "CHAT_RATE_LIMIT_REQUESTS": settings.CHAT_RATE_LIMIT_REQUESTS,
        "CHAT_RATE_LIMIT_WINDOW_SECONDS": settings.CHAT_RATE_LIMIT_WINDOW_SECONDS,
        "DOCUMENT_WRITE_RATE_LIMIT_REQUESTS": settings.DOCUMENT_WRITE_RATE_LIMIT_REQUESTS,
        "DOCUMENT_WRITE_RATE_LIMIT_WINDOW_SECONDS": settings.DOCUMENT_WRITE_RATE_LIMIT_WINDOW_SECONDS,
        "INGEST_RATE_LIMIT_REQUESTS": settings.INGEST_RATE_LIMIT_REQUESTS,
        "INGEST_RATE_LIMIT_WINDOW_SECONDS": settings.INGEST_RATE_LIMIT_WINDOW_SECONDS,
    }
    rate_limiter.reset()
    yield
    rate_limiter.reset()
    for key, value in original_values.items():
        setattr(settings, key, value)
