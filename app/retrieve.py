"""Embedding and retrieval: turn text into vectors and find the chunks closest to a question.

The Embedder loads the model once in __init__ and is meant to be created once
per process and reused. Loading takes seconds; embedding a question takes ~10-20 ms.

Command line:
    python -m app.retrieve build                 # chunk + embed data/pdfs, save the index
    python -m app.retrieve search "question"     # search the saved index
"""

import logging
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingest import Chunk
from app.store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)  # the slow part: loads weights from disk
        self.dim = self.model.get_embedding_dimension()
        self.max_tokens = self.model.max_seq_length  # longer inputs are silently truncated

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of unit-length vectors."""
        vectors = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,  # unit length, so L2 distance ranks like cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text)["input_ids"])  # includes [CLS] and [SEP]


def build_store(chunks: list[Chunk], embedder: Embedder) -> VectorStore:
    """Embed all chunks and put them in a new VectorStore."""
    too_long = [c for c in chunks if embedder.count_tokens(c.text) > embedder.max_tokens]
    if too_long:
        logger.warning("%d of %d chunks exceed %d tokens and will be truncated when embedded; "
                       "consider a smaller CHUNK_SIZE", len(too_long), len(chunks), embedder.max_tokens)

    store = VectorStore(dim=embedder.dim, model_name=embedder.model_name)
    if chunks:
        store.add(embedder.embed([c.text for c in chunks]), chunks)
    return store


def search(question: str, embedder: Embedder, store: VectorStore, k: int) -> list[SearchResult]:
    """Embed the question and return the k most similar chunks, best first."""
    query_vector = embedder.embed([question])[0]
    return store.search(query_vector, k)


if __name__ == "__main__":
    from app.config import get_settings
    from app.ingest import load_pdfs

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Hugging Face logs every HTTP check it makes while loading the model; hide that noise.
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    command = sys.argv[1] if len(sys.argv) > 1 else ""

    t = time.perf_counter()
    embedder = Embedder(settings.embedding_model)
    print(f"model loaded in {time.perf_counter() - t:.1f}s")

    if command == "build":
        chunks = load_pdfs(settings.pdf_dir, settings.chunk_size, settings.chunk_overlap)
        t = time.perf_counter()
        store = build_store(chunks, embedder)
        store.save(settings.index_dir)
        print(f"embedded {len(store)} chunks in {time.perf_counter() - t:.1f}s, saved to {settings.index_dir}")

    elif command == "search" and len(sys.argv) > 2:
        store = VectorStore.load(settings.index_dir, expected_model=settings.embedding_model)
        t = time.perf_counter()
        results = search(sys.argv[2], embedder, store, settings.top_k)
        print(f"search took {1000 * (time.perf_counter() - t):.0f} ms\n")
        for rank, r in enumerate(results, 1):
            print(f"{rank}. [{r.score:.3f}] {r.chunk.source} p.{r.chunk.page} #{r.chunk.chunk_index}")
            print(f"   {r.chunk.text[:200]}...\n")

    else:
        print(__doc__)
        sys.exit(1)
