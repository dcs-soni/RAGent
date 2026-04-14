from io import BytesIO
from pypdf import PdfWriter
from src.config import settings
import pytest

def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.API_KEY}"}

def _create_minimal_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()

def test_invalid_pdf_validation(client):
    """Verify that pypdf properly blocks malformed payloads posing as PDFs."""
    bad_pdf_bytes = b"%PDF-1.4\n%Spoofed content without valid cross-reference tables"
    response = client.post(
        "/documents",
        files={"file": ("invalid.pdf", bad_pdf_bytes, "application/pdf")},
        headers=_auth_headers(),
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Malformed or corrupted" in detail or "no pages" in detail or "not a valid PDF" in detail

@pytest.fixture
def isolated_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "REGISTRY_FILE", str(tmp_path / "registry.json"))
    monkeypatch.setattr(settings, "METADATA_DIR", str(tmp_path / "metadata"))
    (tmp_path / "metadata").mkdir(exist_ok=True)
    yield tmp_path

def test_valid_pdf_upload(client, isolated_docs):
    """Verify that a legitimate, uncorrupted PDF successfully passes structural validation."""
    pdf_bytes = _create_minimal_pdf()
    response = client.post(
        "/documents",
        files={"file": ("valid.pdf", pdf_bytes, "application/pdf")},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    assert "document" in response.json()
