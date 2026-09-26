import pymupdf
import pytest

from app.ingest import chunk_pdf, clean_text, split_text

# 1000 unique words, so we can tell exactly which words each chunk contains.
WORDS = [f"w{i}" for i in range(1000)]
TEXT = " ".join(WORDS)


def word_ids(chunk: str) -> list[int]:
    return [int(w[1:]) for w in chunk.split()]


# --- clean_text ---

def test_clean_text_joins_hyphenated_line_breaks_and_collapses_whitespace():
    raw = "Dense retriev-\nal works  well.\n\nNext\tparagraph here."
    assert clean_text(raw) == "Dense retrieval works well. Next paragraph here."


def test_clean_text_expands_ligatures():
    # "ﬁ" is the single-character "fi" ligature that LaTeX PDFs often contain
    assert clean_text("ﬁne-tuning") == "fine-tuning"


def test_clean_text_keeps_normal_hyphens():
    assert clean_text("state-of-the-art") == "state-of-the-art"


# --- split_text ---

def test_empty_and_short_text():
    assert split_text("", 100, 20) == []
    assert split_text("short text", 100, 20) == ["short text"]


def test_chunks_respect_max_size():
    chunks = split_text(TEXT, 200, 40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_never_split_words_and_cover_all_text():
    chunks = split_text(TEXT, 200, 40)
    seen = set()
    for c in chunks:
        ids = word_ids(c)  # would raise if a chunk contained a broken word like "w1" from "w12"
        assert ids == list(range(ids[0], ids[-1] + 1)), "chunk words must be contiguous"
        seen.update(ids)
    assert seen == set(range(len(WORDS)))


def test_consecutive_chunks_overlap():
    chunks = split_text(TEXT, 200, 40)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert word_ids(nxt)[0] <= word_ids(prev)[-1], "next chunk should start inside the previous one"


def test_zero_overlap_means_no_shared_words():
    chunks = split_text(TEXT, 200, 0)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert word_ids(nxt)[0] == word_ids(prev)[-1] + 1


def test_prefers_to_cut_at_sentence_ends():
    text = " ".join(f"Sentence number {i} is here." for i in range(50))
    chunks = split_text(text, 150, 30)
    assert all(c.endswith(".") for c in chunks)


def test_text_without_spaces_is_hard_cut():
    chunks = split_text("x" * 250, 100, 10)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) == 3


def test_large_overlap_still_terminates():
    chunks = split_text(TEXT, 100, 99)
    assert chunks[-1].endswith(WORDS[-1])


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text(TEXT, 100, 100)


# --- chunk_pdf (on a generated PDF) ---

@pytest.fixture
def sample_pdf(tmp_path):
    """3-page PDF: short page 1, blank page 2, long page 3."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page_texts = [
        "The capital of France is Paris.",
        "",
        " ".join(f"Fact number {i} about the third page." for i in range(60)),
    ]
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=9)
    doc.save(path)
    doc.close()
    return path


def test_chunk_pdf_metadata(sample_pdf):
    chunks = chunk_pdf(sample_pdf, chunk_size=300, chunk_overlap=50)

    assert all(c.source == "sample.pdf" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    pages = [c.page for c in chunks]
    assert pages == sorted(pages), "chunks should be in document order"
    assert set(pages) == {1, 3}, "blank page 2 should be skipped, numbering is 1-based"

    assert chunks[0].text == "The capital of France is Paris."
    assert pages.count(3) > 1, "long page should produce several chunks"
    assert all(len(c.text) <= 300 for c in chunks)
