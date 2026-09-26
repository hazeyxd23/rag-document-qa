"""Uses the real embedding model (downloaded on first run), so it is slower than the other tests."""

import numpy as np
import pytest

from app.config import get_settings
from app.ingest import Chunk
from app.retrieve import Embedder, build_store, search

TEXTS = [
    "Paris is the capital and largest city of France.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The mitochondria produces ATP through cellular respiration.",
    "FAISS is a library for efficient similarity search of dense vectors.",
]


@pytest.fixture(scope="module")  # load the model once for all tests in this file
def embedder():
    return Embedder(get_settings().embedding_model)


@pytest.fixture(scope="module")
def store(embedder):
    chunks = [Chunk(text=t, source="facts.pdf", page=i + 1, chunk_index=i) for i, t in enumerate(TEXTS)]
    return build_store(chunks, embedder)


def test_embeddings_are_unit_length(embedder):
    vectors = embedder.embed(["one", "two words"])
    assert vectors.shape == (2, embedder.dim)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("question, expected_page", [
    ("What is France's capital city?", 1),               # paraphrase, little word overlap
    ("How do plants make energy from sunlight?", 2),
    ("Which organelle generates ATP?", 3),
    ("library for nearest neighbour vector lookup", 4),
])
def test_search_matches_by_meaning(embedder, store, question, expected_page):
    results = search(question, embedder, store, k=2)
    assert results[0].chunk.page == expected_page
    assert results[0].score > results[1].score
