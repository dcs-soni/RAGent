from src.config import settings

def test_health_check_unauthenticated_and_minimal(client):
    """Verify /health can be called without an API key and does not leak metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "graph_ready" in data
    # Ensure sensitive metadata is stripped
    assert "documents" not in data
    assert "ingestion" not in data

def test_protected_routes_without_auth(client):
    """Verify sensitive routes return 403 when no key is provided."""
    endpoints = [
        ("GET", "/documents"),
        ("POST", "/documents"),
        ("POST", "/upload"),
        ("PUT", "/documents/test-doc-id"),
        ("DELETE", "/documents/test-doc-id"),
        ("POST", "/ingest"),
        ("GET", "/ingest/status"),
        ("GET", "/ingest/jobs/test-job-id"),
        ("POST", "/chat"),
        ("GET", "/system/status"),
    ]
    for method, path in endpoints:
        if method == "GET":
            response = client.get(path)
        elif method == "POST":
            response = client.post(path, json={"question": "hello", "thread_id": "123"} if path == "/chat" else None)
        elif method == "PUT":
            response = client.put(path)
        elif method == "DELETE":
            response = client.delete(path)
        
        # 403 is returned by fastapi.security.HTTPBearer when token is absent/invalid
        assert response.status_code == 403, f"Expected 403 for {method} {path}, got {response.status_code}"

def test_protected_routes_with_invalid_auth(client):
    """Verify sensitive routes return 403 when wrong key is provided."""
    headers = {"Authorization": "Bearer wrong-key"}
    response = client.get("/documents", headers=headers)
    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid or missing API Key"}

def test_system_status_with_valid_auth(client):
    """Verify /system/status works with valid auth and returns full metadata."""
    headers = {"Authorization": f"Bearer {settings.API_KEY}"}
    response = client.get("/system/status", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "ingestion" in data
