"""Retrieval evaluation: recall@k and MRR over a page-labeled question set.

Each question in questions.jsonl lists the (source, page) pairs that contain its
answer. A retrieved chunk counts as relevant if its page is one of them. Labels
are pages rather than chunk ids, so one eval set works for any chunking config.
Questions with no relevant pages are unanswerable; they are skipped for retrieval
metrics and used to check refusals when --with-llm is given.

Usage:
    python -m eval.run_eval                                  # current CHUNK_SIZE/CHUNK_OVERLAP
    python -m eval.run_eval --configs 400:75 800:150 1200:225  # compare chunkings
    python -m eval.run_eval --with-llm                       # also answer every question with the LLM
"""

import argparse
import json
import logging
import time
from pathlib import Path

from app.config import get_settings
from app.ingest import load_pdfs
from app.retrieve import Embedder, build_store, search
from app.store import VectorStore

QUESTIONS_FILE = Path(__file__).parent / "questions.jsonl"
KS = (1, 3, 5, 10)

Page = tuple[str, int]  # (source filename, page number)


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def relevant_pages(question: dict) -> set[Page]:
    return {(r["source"], r["page"]) for r in question["relevant"]}


def recall_at_k(retrieved: list[Page], relevant: set[Page], k: int) -> float:
    """Fraction of the relevant pages that appear in the top k results."""
    return len(relevant & set(retrieved[:k])) / len(relevant)


def reciprocal_rank(retrieved: list[Page], relevant: set[Page]) -> float:
    """1/rank of the first relevant result (1.0 if first), 0 if none was retrieved."""
    for rank, page in enumerate(retrieved, 1):
        if page in relevant:
            return 1 / rank
    return 0.0


def evaluate_retrieval(questions: list[dict], embedder: Embedder, store: VectorStore) -> dict:
    """Average recall@k for each k in KS, and MRR@max(KS), over answerable questions."""
    answerable = [q for q in questions if q["relevant"]]
    totals = {f"recall@{k}": 0.0 for k in KS} | {"mrr": 0.0}
    misses = []
    for q in answerable:
        results = search(q["question"], embedder, store, k=max(KS))
        retrieved = [(r.chunk.source, r.chunk.page) for r in results]
        relevant = relevant_pages(q)
        for k in KS:
            totals[f"recall@{k}"] += recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)
        totals["mrr"] += rr
        if rr < 1 / 5:  # no relevant page in the top 5
            misses.append((q["id"], q["question"], sorted(p for _, p in relevant), [p for _, p in retrieved[:5]]))

    metrics = {name: value / len(answerable) for name, value in totals.items()}
    return {"metrics": metrics, "num_questions": len(answerable), "misses": misses}


def evaluate_generation(questions: list[dict], embedder: Embedder, store: VectorStore, k: int) -> dict:
    """Ask the configured LLM every question; check refusals and whether citations hit labeled pages."""
    from app.generate import generate_answer, make_provider

    provider = make_provider(get_settings())
    rows = []
    for q in questions:
        start = time.perf_counter()
        answer = generate_answer(q["question"], search(q["question"], embedder, store, k), provider)
        cited = {(c.chunk.source, c.chunk.page) for c in answer.citations}
        rows.append({
            "id": q["id"],
            "answerable": bool(q["relevant"]),
            "answered": answer.answered,
            "citation_hit": bool(cited & relevant_pages(q)),
            "seconds": time.perf_counter() - start,
            "tokens": (answer.generation.prompt_tokens or 0) + (answer.generation.completion_tokens or 0),
            "text": answer.text,
        })

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if r["answered"]]
    return {
        "answered_rate": len(answered) / len(answerable),
        "citation_hit_rate": sum(r["citation_hit"] for r in answered) / max(len(answered), 1),
        "refusal_rate_unanswerable": sum(not r["answered"] for r in unanswerable) / max(len(unanswerable), 1),
        "avg_seconds": sum(r["seconds"] for r in rows) / len(rows),
        "avg_tokens": sum(r["tokens"] for r in rows) / len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", metavar="SIZE:OVERLAP",
                        help="chunking configs to compare, e.g. 400:75 800:150 (default: current .env)")
    parser.add_argument("--with-llm", action="store_true",
                        help="also run generation with the configured LLM (costs API tokens)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)  # hide per-build warnings; results are printed below
    settings = get_settings()
    questions = load_questions()
    configs = ([tuple(int(x) for x in c.split(":")) for c in args.configs] if args.configs
               else [(settings.chunk_size, settings.chunk_overlap)])

    embedder = Embedder(settings.embedding_model)  # loaded once, reused for every config
    header = "| chunk size / overlap | chunks | " + " | ".join(f"recall@{k}" for k in KS) + " | MRR@10 |"
    rows, reports = [], []
    for size, overlap in configs:
        # A fresh in-memory index per config; the saved index in data/index is never touched.
        chunks = load_pdfs(settings.pdf_dir, size, overlap)
        store = build_store(chunks, embedder)
        report = evaluate_retrieval(questions, embedder, store)
        m = report["metrics"]
        rows.append(f"| {size} / {overlap} | {len(chunks)} | "
                    + " | ".join(f"{m[f'recall@{k}']:.2f}" for k in KS) + f" | {m['mrr']:.2f} |")
        reports.append((size, overlap, store, report))

    print(f"\nRetrieval on {reports[0][3]['num_questions']} answerable questions "
          f"({len(questions)} total), embedding model {settings.embedding_model}\n")
    print(header)
    print("|" + "---|" * (len(KS) + 3))
    print("\n".join(rows))

    for size, overlap, _, report in reports:
        print(f"\nMisses at {size}/{overlap} (no labeled page in top 5):")
        for qid, question, want, got in report["misses"]:
            print(f"  {qid} want p.{want} got p.{got}  {question}")

    if args.with_llm:
        size, overlap, store, _ = next((r for r in reports if r[:2] == (settings.chunk_size, settings.chunk_overlap)),
                                       reports[0])
        gen = evaluate_generation(questions, embedder, store, settings.top_k)
        print(f"\nGeneration with {settings.llm_provider} (chunks {size}/{overlap}, top_k={settings.top_k}):")
        print(f"  answered (answerable questions):     {gen['answered_rate']:.0%}")
        print(f"  citation hits a labeled page:        {gen['citation_hit_rate']:.0%} of answered")
        print(f"  refused (unanswerable questions):    {gen['refusal_rate_unanswerable']:.0%}")
        print(f"  avg latency {gen['avg_seconds']:.1f}s, avg tokens {gen['avg_tokens']:.0f}")
        for r in gen["rows"]:
            flag = "" if r["answered"] == r["answerable"] else "   <-- " + ("refused" if r["answerable"] else "ANSWERED")
            print(f"  {r['id']}: {r['text'][:110]!r}{flag}")


if __name__ == "__main__":
    main()
