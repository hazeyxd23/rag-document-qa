"""Vector store: a FAISS index plus the chunk metadata for each vector.

FAISS only stores vectors. IndexFlatL2 identifies each vector by its insertion
position (0, 1, 2, ...), and search returns those positions. We map a position
back to its text and citation through self.chunks, where chunks[i] belongs to
vector i. That mapping is only valid if both are built together, in the same
order, so they are only ever added, saved, and loaded together.

This module knows nothing about embedding models; it works with raw vectors.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from app.ingest import Chunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


class StoreError(Exception):
    """The saved index is missing, inconsistent, or built with a different model."""


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, higher = more similar (1.0 = identical direction)


class VectorStore:
    def __init__(self, dim: int, model_name: str):
        self.dim = dim
        self.model_name = model_name  # recorded so we never mix vectors from two models
        self.index = faiss.IndexFlatL2(dim)  # exact (brute-force) search
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Add vectors and their chunks together, so position i always means the same chunk."""
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vectors but {len(chunks)} chunks")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        # search() converts L2 distance to cosine similarity, which assumes unit-length vectors.
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
            raise ValueError("vectors must be L2-normalized")

        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        """Return the k chunks closest to query_vector, best first."""
        if len(self) == 0:
            return []
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        distances, ids = self.index.search(query, k)  # both shaped (1, k)

        results = []
        for distance, idx in zip(distances[0], ids[0]):
            if idx == -1:  # FAISS pads with -1 when the index has fewer than k vectors
                continue
            # For unit vectors, squared L2 distance = 2 - 2*cos, so cos = 1 - d/2.
            results.append(SearchResult(chunk=self.chunks[idx], score=float(1 - distance / 2)))
        return results

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / INDEX_FILE))
        with open(index_dir / CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)
        # Written last: if saving is interrupted, the counts won't match and load() refuses it.
        manifest = {"model_name": self.model_name, "dim": self.dim, "num_chunks": len(self.chunks)}
        with open(index_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, index_dir: Path, expected_model: str) -> "VectorStore":
        manifest_path = index_dir / MANIFEST_FILE
        if not manifest_path.exists():
            raise StoreError(f"No index found in {index_dir}. Run ingestion first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest["model_name"] != expected_model:
            raise StoreError(
                f"Index was built with '{manifest['model_name']}' but the configured model is "
                f"'{expected_model}'. Vectors from different models can't be compared; re-run ingestion."
            )

        store = cls(dim=manifest["dim"], model_name=manifest["model_name"])
        store.index = faiss.read_index(str(index_dir / INDEX_FILE))
        with open(index_dir / CHUNKS_FILE, encoding="utf-8") as f:
            store.chunks = [Chunk(**c) for c in json.load(f)]

        counts = (store.index.ntotal, len(store.chunks), manifest["num_chunks"])
        if len(set(counts)) != 1 or store.index.d != manifest["dim"]:
            raise StoreError(
                f"Index files in {index_dir} are out of sync (vectors, chunks, manifest = {counts}). "
                "Re-run ingestion."
            )
        return store
