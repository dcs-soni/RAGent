from src.config import settings


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.API_KEY}"}


def test_chat_rate_limit_returns_429(client):
    settings.CHAT_RATE_LIMIT_REQUESTS = 1
    settings.CHAT_RATE_LIMIT_WINDOW_SECONDS = 60

    payload = {"question": "hello", "thread_id": "thread-1"}
    first_response = client.post("/chat", json=payload, headers=_auth_headers())
    second_response = client.post("/chat", json=payload, headers=_auth_headers())

    assert first_response.status_code == 400
    assert second_response.status_code == 429
    assert second_response.json() == {"detail": "Rate limit exceeded. Please retry later."}
    assert "Retry-After" in second_response.headers


def test_document_write_rate_limit_returns_429(client):
    settings.DOCUMENT_WRITE_RATE_LIMIT_REQUESTS = 1
    settings.DOCUMENT_WRITE_RATE_LIMIT_WINDOW_SECONDS = 60

    first_response = client.delete("/documents/missing-doc", headers=_auth_headers())
    second_response = client.delete("/documents/missing-doc", headers=_auth_headers())

    assert first_response.status_code == 404
    assert second_response.status_code == 429
    assert second_response.json() == {"detail": "Rate limit exceeded. Please retry later."}


def test_ingest_rate_limit_returns_429(client):
    settings.INGEST_RATE_LIMIT_REQUESTS = 1
    settings.INGEST_RATE_LIMIT_WINDOW_SECONDS = 60

    first_response = client.post("/ingest", headers=_auth_headers())
    second_response = client.post("/ingest", headers=_auth_headers())

    assert first_response.status_code == 400
    assert second_response.status_code == 429
    assert second_response.json() == {"detail": "Rate limit exceeded. Please retry later."}


def test_oversized_upload_returns_413(client):
    settings.MAX_UPLOAD_SIZE_MB = 1
    settings.MAX_UPLOAD_REQUEST_OVERHEAD_KB = 256

    oversized_pdf = b"%PDF-" + (b"A" * (settings.max_upload_size_bytes + 1))
    response = client.post(
        "/documents",
        files={"file": ("large.pdf", oversized_pdf, "application/pdf")},
        headers=_auth_headers(),
    )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_oversized_chat_body_returns_413(client):
    settings.MAX_CHAT_REQUEST_SIZE_KB = 1

    response = client.post(
        "/chat",
        json={"question": "A" * 2048, "thread_id": "thread-1"},
        headers=_auth_headers(),
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large for this endpoint."}


def test_missing_content_length_returns_411(client):
    def chunked_body():
        yield b'{"question": "A", "thread_id": "thread-1"}'
    
    response = client.post(
        "/chat",
        content=chunked_body(),
        headers=_auth_headers(),
    )
    assert response.status_code == 411
    assert "Length Required" in response.json()["detail"]


def test_rate_limit_respects_x_forwarded_for(client):
    # Reset internal rate limiter for this test
    from src.api import rate_limiter
    rate_limiter.reset()

    settings.CHAT_RATE_LIMIT_REQUESTS = 1
    settings.CHAT_RATE_LIMIT_WINDOW_SECONDS = 60
    
    payload = {"question": "hello", "thread_id": "thread-1"}
    
    # User 1 initial
    client.post("/chat", json=payload, headers={**_auth_headers(), "X-Forwarded-For": "192.168.1.1, 10.0.0.1"})
    
    # User 1 retry (should be blocked)
    resp_user1_blocked = client.post("/chat", json=payload, headers={**_auth_headers(), "X-Forwarded-For": "192.168.1.1"})
    assert resp_user1_blocked.status_code == 429
    
    # User 2 initial (should be allowed despite User 1's exhaustion)
    resp_user2_allowed = client.post("/chat", json=payload, headers={**_auth_headers(), "X-Forwarded-For": "10.0.0.5"})
    assert resp_user2_allowed.status_code != 429
