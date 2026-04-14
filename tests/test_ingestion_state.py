from src.config import settings
from unittest.mock import patch

def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.API_KEY}"}

def test_mutation_lockout_during_ingestion(client, mocker):
    """Verify that API immediately drops mutations with HTTP 409 when ingestion is globally running."""
    mocker.patch("src.services.document_service.is_ingestion_running", return_value=True)

    response = client.post(
        "/documents",
        files={"file": ("test.pdf", b"%PDF-dummy", "application/pdf")},
        headers=_auth_headers()
    )
    assert response.status_code == 409
    assert "disabled while ingestion is running" in response.json().get("detail", "")

    response = client.put(
        "/documents/fake-id",
        files={"file": ("test.pdf", b"%PDF-dummy", "application/pdf")},
        headers=_auth_headers()
    )
    assert response.status_code == 409

    response = client.delete("/documents/fake-id", headers=_auth_headers())
    assert response.status_code == 409

def test_concurrent_ingestion_returns_409(client, mocker):
    """Verify that the ingestion_service blocks concurrent ingestion jobs."""
    mocker.patch("src.services.ingestion_service.ingestion_running", True)
    
    response = client.post("/ingest", headers=_auth_headers())
    assert response.status_code == 409
    assert "Ingestion is already running" in response.json().get("detail", "")
