"""PDF ingestion: extract text per page, clean it, and split it into overlapping chunks.

Each chunk keeps the metadata needed to cite it later: source filename,
1-based page number, and its position (chunk_index) within the document.

Run directly to inspect how a PDF gets chunked:
    python -m app.ingest data/pdfs/some.pdf
"""

import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

# Characters that end a sentence; we prefer to cut a chunk right after one.
SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")


@dataclass
class Chunk:
    text: str
    source: str  # PDF filename, e.g. "manual.pdf"
    page: int  # 1-based page number, as a human would cite it
    chunk_index: int  # position of this chunk within its document (0, 1, 2, ...)


def clean_text(text: str) -> str:
    """Undo PDF line-wrapping artifacts so the text reads as normal prose."""
    # NFKC turns typographic ligatures like "ﬁ" (one char) into plain "fi",
    # so "ﬁne-tuning" matches "fine-tuning" for the tokenizer.
    text = unicodedata.normalize("NFKC", text)
    # "retriev-\nal" -> "retrieval" (hyphen at a line break followed by a lowercase letter).
    # Known limitation: a real compound split at a line end, "knowledge-\nintensive",
    # also gets joined; telling them apart would need a dictionary.
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    # Collapse newlines, tabs and repeated spaces into single spaces.
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, cleaned_text) for every page that has text."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text = clean_text(page.get_text())
            page_number = page.number + 1  # PyMuPDF is 0-based
            if not text:
                logger.warning("%s page %d has no extractable text (scanned?), skipping",
                               pdf_path.name, page_number)
                continue
            pages.append((page_number, text))
    return pages


def _find_break(text: str, min_end: int, max_end: int) -> int:
    """Pick where a chunk should end, somewhere in text[min_end:max_end].

    Preference: right after the last sentence end, else at the last space,
    else a hard cut at max_end.
    """
    window = text[min_end:max_end]

    last_sentence_end = None
    for match in SENTENCE_END.finditer(window):
        last_sentence_end = match.end()
    if last_sentence_end is not None:
        return min_end + last_sentence_end

    last_space = window.rfind(" ")
    if last_space != -1:
        return min_end + last_space + 1

    return max_end


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most chunk_size characters that overlap by ~chunk_overlap."""
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        max_end = start + chunk_size
        if max_end >= len(text):
            end = len(text)
        else:
            # Only look for a break in the back half of the window (and past the
            # overlap) so chunks are never tiny and the loop always moves forward.
            min_end = start + max(chunk_size // 2, chunk_overlap + 1)
            end = _find_break(text, min_end, max_end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        # Step back by the overlap, then forward to the start of a word.
        start = end - chunk_overlap
        next_space = text.find(" ", start, end)
        if next_space != -1:
            start = next_space + 1

    return chunks


def chunk_pdf(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Extract and chunk one PDF. Chunks never cross page boundaries, so each has one page."""
    chunks = []
    for page_number, page_text in extract_pages(pdf_path):
        for piece in split_text(page_text, chunk_size, chunk_overlap):
            chunks.append(Chunk(text=piece, source=pdf_path.name, page=page_number,
                                chunk_index=len(chunks)))
    return chunks


def load_pdfs(pdf_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every PDF in a directory, in filename order."""
    chunks = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc_chunks = chunk_pdf(pdf_path, chunk_size, chunk_overlap)
        logger.info("%s: %d chunks", pdf_path.name, len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks


if __name__ == "__main__":
    from app.config import get_settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.pdf_dir

    if target.is_dir():
        result = load_pdfs(target, settings.chunk_size, settings.chunk_overlap)
    else:
        result = chunk_pdf(target, settings.chunk_size, settings.chunk_overlap)

    if not result:
        print(f"No chunks produced from {target}")
        sys.exit(1)

    lengths = [len(c.text) for c in result]
    print(f"\n{len(result)} chunks from {len({c.source for c in result})} file(s), "
          f"{len({(c.source, c.page) for c in result})} page(s)")
    print(f"chunk length: min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)}")
    for c in result[:3]:
        print(f"\n--- {c.source} p.{c.page} #{c.chunk_index} ({len(c.text)} chars) ---")
        print(c.text)
