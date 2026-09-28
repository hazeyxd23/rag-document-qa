import pytest

from eval.run_eval import load_questions, recall_at_k, reciprocal_rank

A, B, C, D = ("doc.pdf", 1), ("doc.pdf", 2), ("doc.pdf", 3), ("doc.pdf", 4)


def test_recall_at_k_counts_fraction_of_relevant_pages_found():
    retrieved = [C, A, D, B]
    assert recall_at_k(retrieved, {A}, k=1) == 0.0
    assert recall_at_k(retrieved, {A}, k=2) == 1.0
    assert recall_at_k(retrieved, {A, B}, k=3) == 0.5  # found A but not B
    assert recall_at_k(retrieved, {A, B}, k=4) == 1.0


def test_recall_ignores_duplicate_chunks_from_the_same_page():
    assert recall_at_k([A, A, A], {A, B}, k=3) == 0.5


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank([A, B], {A}) == 1.0
    assert reciprocal_rank([C, D, B, A], {A, B}) == pytest.approx(1 / 3)
    assert reciprocal_rank([C, D], {A}) == 0.0


def test_question_file_is_well_formed():
    questions = load_questions()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    for q in questions:
        assert q["question"].strip()
        for r in q["relevant"]:
            assert r["source"].endswith(".pdf") and r["page"] >= 1
    assert any(not q["relevant"] for q in questions), "keep some unanswerable questions for refusal checks"
