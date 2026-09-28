"""HTTP API: POST /ingest, POST /query, GET /health.

Heavy objects (embedding model, LLM client, FAISS index) are created once in the
lifespan hook and kept on app.state; every request reuses them.

Run:
    uvicorn app.api:app --port 8001
Then open http://localhost:8001/docs for interactive documentation.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from app.config import Settings, get_settings
from app.generate import GenerationError, LLMProvider, generate_answer, make_provider
from app.ingest import load_pdfs
from app.retrieve import Embedder, build_store, search
from app.store import SearchResult, StoreError, VectorStore

logger = logging.getLogger("rag.api")


# --- Request / response models: validated at the boundary, documented in /docs ---

class QueryRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    top_k: int | None = Field(default=None, ge=1, le=20, description="Defaults to TOP_K from config")


class Citation(BaseModel):
    source: str
    page: int
    chunk_index: int
    score: float
    text: str


class Timings(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class Usage(BaseModel):
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


class QueryResponse(BaseModel):
    answer: str
    answered: bool = Field(description="False when the documents don't contain the answer")
    citations: list[Citation] = Field(description="Retrieved chunks the answer cites")
    retrieved: list[Citation] = Field(description="All chunks given to the model, best first")
    timings: Timings
    usage: Usage


class IngestResponse(BaseModel):
    files: list[str]
    num_chunks: int
    seconds: float


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_chunks: int
    embedding_model: str
    llm_provider: str


# --- Routes ---

router = APIRouter()


def to_citation(result: SearchResult) -> Citation:
    c = result.chunk
    return Citation(source=c.source, page=c.page, chunk_index=c.chunk_index,
                    score=round(result.score, 4), text=c.text)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    store = state.store
    return HealthResponse(
        status="ok",
        index_loaded=store is not None,
        num_chunks=len(store) if store is not None else 0,
        embedding_model=state.embedder.model_name,
        llm_provider=state.settings.llm_provider,
    )


# Plain `def` (not async): embedding is slow CPU work, so FastAPI runs this in a
# thread pool and the event loop stays free to serve other requests.
@router.post("/ingest", response_model=IngestResponse)
def ingest(request: Request) -> IngestResponse:
    state = request.app.state
    settings: Settings = state.settings

    if not state.ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "An ingest is already running")
    try:
        start = time.perf_counter()
        chunks = load_pdfs(settings.pdf_dir, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(400, f"No PDFs with extractable text in {settings.pdf_dir.name}/")
        new_store = build_store(chunks, state.embedder)
        new_store.save(settings.index_dir)
        # Swap in only once the new index is complete; in-flight queries keep the old one.
        state.store = new_store
        seconds = time.perf_counter() - start
    finally:
        state.ingest_lock.release()

    files = sorted({c.source for c in chunks})
    logger.info("ingest files=%d chunks=%d seconds=%.1f", len(files), len(chunks), seconds)
    return IngestResponse(files=files, num_chunks=len(chunks), seconds=round(seconds, 2))


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    state = request.app.state
    store = state.store  # read once: an ingest may swap it while this request runs
    if store is None:
        raise HTTPException(503, "No index loaded. Call POST /ingest first.")
    k = body.top_k or state.settings.top_k

    t0 = time.perf_counter()
    results = search(body.question, state.embedder, store, k)
    t1 = time.perf_counter()
    try:
        answer = generate_answer(body.question, results, state.provider)
    except GenerationError as e:
        logger.error("generation failed: %s", e)  # details stay in the server log
        raise HTTPException(502, "The language model request failed. See server logs.") from e
    t2 = time.perf_counter()

    timings = Timings(retrieval_ms=round((t1 - t0) * 1000, 1),
                      generation_ms=round((t2 - t1) * 1000, 1),
                      total_ms=round((t2 - t0) * 1000, 1))
    g = answer.generation
    # Log sizes and outcomes, not the question text itself (it may contain personal data).
    logger.info("query chars=%d k=%d retrieval_ms=%.0f generation_ms=%.0f "
                "prompt_tokens=%s completion_tokens=%s answered=%s cited=%d",
                len(body.question), k, timings.retrieval_ms, timings.generation_ms,
                g.prompt_tokens, g.completion_tokens, answer.answered, len(answer.citations))

    return QueryResponse(
        answer=answer.text,
        answered=answer.answered,
        citations=[to_citation(r) for r in answer.citations],
        retrieved=[to_citation(r) for r in answer.retrieved],
        timings=timings,
        usage=Usage(model=g.model, prompt_tokens=g.prompt_tokens, completion_tokens=g.completion_tokens),
    )


# --- App assembly ---

async def log_requests(request: Request, call_next):
    """Log every request's method, path, status, and latency; expose latency as a header."""
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    logger.info("%s %s -> %d in %.1f ms", request.method, request.url.path, response.status_code, ms)
    return response


def load_store_if_present(settings: Settings, model_name: str) -> VectorStore | None:
    """A missing or stale index isn't fatal: the API starts and /query returns 503 until /ingest."""
    try:
        store = VectorStore.load(settings.index_dir, expected_model=model_name)
    except StoreError as e:
        logger.warning("%s", e)
        return None
    logger.info("loaded index with %d chunks", len(store))
    return store


def create_app(settings: Settings | None = None, embedder: Embedder | None = None,
               provider: LLMProvider | None = None) -> FastAPI:
    """Build the app. Tests pass lightweight fakes; production uses the real components."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once at startup, before any request is served.
        s = settings or get_settings()
        app.state.settings = s
        app.state.provider = provider or make_provider(s)  # bad LLM config fails startup
        app.state.embedder = embedder or Embedder(s.embedding_model)  # the slow part (seconds)
        app.state.store = load_store_if_present(s, app.state.embedder.model_name)
        app.state.ingest_lock = threading.Lock()
        logger.info("ready: provider=%s embedding_model=%s", s.llm_provider, app.state.embedder.model_name)
        yield

    app = FastAPI(title="RAG Document Q&A", lifespan=lifespan,
                  description="Answers questions about PDFs using retrieval, with page-level citations.")
    app.middleware("http")(log_requests)
    app.include_router(router)
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = create_app()
