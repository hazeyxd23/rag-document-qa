import json

import numpy as np
import pytest

from app.ingest import Chunk
from app.store import CHUNKS_FILE, VectorStore, StoreError

DIM = 8
MODEL = "test-model"


def unit_vectors(n: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_chunks(n: int) -> list[Chunk]:
    return [Chunk(text=f"chunk {i}", source="doc.pdf", page=i // 3 + 1, chunk_index=i) for i in range(n)]


@pytest.fixture
def store():
    s = VectorStore(DIM, MODEL)
    s.add(unit_vectors(10), make_chunks(10))
    return s


def test_search_finds_exact_vector_first(store):
    vectors = unit_vectors(10)
    results = store.search(vectors[7], k=3)
    assert results[0].chunk.chunk_index == 7
    assert results[0].score == pytest.approx(1.0, abs=1e-5)  # identical vector -> cosine 1
    assert [r.score for r in results] == sorted([r.score for r in results], reverse=True)


def test_score_is_cosine_similarity(store):
    query = unit_vectors(1, seed=99)[0]
    top = store.search(query, k=1)[0]
    expected = float(unit_vectors(10)[top.chunk.chunk_index] @ query)
    assert top.score == pytest.approx(expected, abs=1e-5)


def test_k_larger_than_index_returns_only_real_results(store):
    results = store.search(unit_vectors(1, seed=5)[0], k=50)
    assert len(results) == 10  # FAISS's -1 padding is filtered out


def test_empty_store_search_returns_nothing():
    assert VectorStore(DIM, MODEL).search(unit_vectors(1)[0], k=5) == []


def test_add_rejects_mismatched_or_unnormalized_input():
    s = VectorStore(DIM, MODEL)
    with pytest.raises(ValueError):
        s.add(unit_vectors(3), make_chunks(2))  # counts differ
    with pytest.raises(ValueError):
        s.add(unit_vectors(3) * 2, make_chunks(3))  # not unit length
    assert len(s) == 0 and s.chunks == []  # nothing half-added


def test_save_load_roundtrip(store, tmp_path):
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, expected_model=MODEL)
    assert len(loaded) == 10
    assert loaded.chunks == store.chunks
    assert loaded.search(unit_vectors(10)[4], k=1)[0].chunk.chunk_index == 4


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(StoreError, match="No index found"):
        VectorStore.load(tmp_path, expected_model=MODEL)


def test_load_rejects_different_embedding_model(store, tmp_path):
    store.save(tmp_path)
    with pytest.raises(StoreError, match="different models"):
        VectorStore.load(tmp_path, expected_model="some-other-model")


def test_load_rejects_out_of_sync_metadata(store, tmp_path):
    store.save(tmp_path)
    chunks_path = tmp_path / CHUNKS_FILE
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks_path.write_text(json.dumps(chunks[:-1]), encoding="utf-8")  # drop one chunk
    with pytest.raises(StoreError, match="out of sync"):
        VectorStore.load(tmp_path, expected_model=MODEL)
