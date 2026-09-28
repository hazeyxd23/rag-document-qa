"""HTTP-level tests with a fake embedder and fake LLM, so no model download or API key is needed."""

import re
import zlib

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.generate import REFUSAL, Generation, GenerationError


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims: same words -> similar vectors. Deterministic and instant."""

    model_name = "fake-embedder"
    dim = 64
    max_tokens = 10_000

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % self.dim] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeProvider:
    def __init__(self):
        self.reply = "The capital of France is Paris [1]."
        self.fail = False

    def generate(self, system: str, user: str) -> Generation:
        if self.fail:
            raise GenerationError("OpenAI request failed: AuthenticationError: key sk-secret-123 invalid")
        return Generation(self.reply, "fake-llm", prompt_tokens=321, completion_tokens=9)


def write_pdf(path, page_texts):
    doc = pymupdf.open()
    for text in page_texts:
        doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=10)
    doc.save(path)
    doc.close()


@pytest.fixture
def settings(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    write_pdf(pdf_dir / "facts.pdf", [
        "Photosynthesis converts light into chemical energy in plants.",
        "The capital of France is Paris. Paris lies on the Seine.",
    ])
    return Settings(pdf_dir=pdf_dir, index_dir=tmp_path / "index", llm_provider="ollama")


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def client(settings, provider):
    app = create_app(settings=settings, embedder=FakeEmbedder(), provider=provider)
    with TestClient(app) as c:  # the context manager runs the lifespan (startup) hook
        yield c


def test_before_ingest_health_reports_no_index_and_query_is_503(client):
    health = client.get("/health").json()
    assert health["index_loaded"] is False and health["num_chunks"] == 0
    response = client.post("/query", json={"question": "What is the capital of France?"})
    assert response.status_code == 503
    assert "POST /ingest" in response.json()["detail"]


def test_ingest_then_query_returns_answer_with_page_citation(client, settings):
    ingest = client.post("/ingest")
    assert ingest.status_code == 200
    assert ingest.json()["files"] == ["facts.pdf"] and ingest.json()["num_chunks"] == 2
    assert (settings.index_dir / "manifest.json").exists()
    assert client.get("/health").json()["index_loaded"] is True

    response = client.post("/query", json={"question": "What is the capital of France?", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert [(c["source"], c["page"]) for c in body["citations"]] == [("facts.pdf", 2)]
    assert len(body["retrieved"]) == 2
    assert body["usage"] == {"model": "fake-llm", "prompt_tokens": 321, "completion_tokens": 9}
    assert set(body["timings"]) == {"retrieval_ms", "generation_ms", "total_ms"}
    assert "X-Response-Time-ms" in response.headers


def test_refusal_is_reported_with_no_citations(client, provider):
    client.post("/ingest")
    provider.reply = REFUSAL
    body = client.post("/query", json={"question": "Who won the 2018 World Cup?"}).json()
    assert body["answered"] is False and body["citations"] == []


@pytest.mark.parametrize("payload", [
    {},                                    # missing question
    {"question": ""},
    {"question": "   "},                   # whitespace only
    {"question": "x" * 2001},
    {"question": "ok?", "top_k": 0},
    {"question": "ok?", "top_k": 50},
])
def test_invalid_query_is_422(client, payload):
    client.post("/ingest")
    assert client.post("/query", json=payload).status_code == 422


def test_llm_failure_is_502_without_leaking_details(client, provider):
    client.post("/ingest")
    provider.fail = True
    response = client.post("/query", json={"question": "What is the capital of France?"})
    assert response.status_code == 502
    assert "sk-secret" not in response.text


def test_ingest_with_no_pdfs_is_400(client, settings):
    for pdf in settings.pdf_dir.glob("*.pdf"):
        pdf.unlink()
    assert client.post("/ingest").status_code == 400


def test_corrupt_pdf_is_skipped(client, settings):
    (settings.pdf_dir / "broken.pdf").write_bytes(b"not really a pdf")
    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json()["files"] == ["facts.pdf"]


def test_saved_index_is_loaded_at_startup(client, settings, provider):
    client.post("/ingest")
    restarted = create_app(settings=settings, embedder=FakeEmbedder(), provider=provider)
    with TestClient(restarted) as c:
        assert c.get("/health").json()["num_chunks"] == 2
        assert c.post("/query", json={"question": "capital of France?"}).status_code == 200
