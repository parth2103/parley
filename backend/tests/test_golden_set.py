"""Unit tests for the golden evaluation dataset and deterministic scoring logic."""

import pytest
from backend.rag.corpus import CORPUS
from eval.scenarios.golden_set import GOLDEN_SET, GoldenItem


def test_golden_set_size_and_balance():
    assert len(GOLDEN_SET) >= 25, f"Expected at least 25 items, got {len(GOLDEN_SET)}"
    assert len(GOLDEN_SET) <= 40, f"Expected at most 40 items, got {len(GOLDEN_SET)}"

    answerable = [item for item in GOLDEN_SET if item.is_answerable]
    unanswerable = [item for item in GOLDEN_SET if not item.is_answerable]

    assert len(answerable) >= 20, f"Expected at least 20 answerable, got {len(answerable)}"
    assert len(unanswerable) >= 4, f"Expected 4-6 unanswerable, got {len(unanswerable)}"
    assert len(unanswerable) <= 6


def test_golden_set_target_chunks_exist():
    valid_chunk_ids = {c.chunk_id for c in CORPUS}
    for item in GOLDEN_SET:
        if item.is_answerable:
            assert len(item.target_chunk_ids) > 0, f"Item {item.id} has no target chunks"
            for cid in item.target_chunk_ids:
                assert cid in valid_chunk_ids, f"Item {item.id} references invalid chunk {cid}"
        else:
            assert len(item.target_chunk_ids) == 0, f"Unanswerable {item.id} should have empty target chunks"
            assert item.refusal_required is True


def test_golden_set_ids_unique():
    ids = [item.id for item in GOLDEN_SET]
    assert len(ids) == len(set(ids)), "Golden set item IDs must be unique"


def test_golden_set_retrieval_recall_top3():
    from backend.rag.retriever import BM25HashedNgramRetriever
    retriever = BM25HashedNgramRetriever(CORPUS)
    answerable = [item for item in GOLDEN_SET if item.is_answerable]
    hits = 0
    for item in answerable:
        retrieved = [chunk for chunk, _score in retriever.retrieve(item.question, top_k=3)]
        chunk_ids = {c.chunk_id for c in retrieved}
        if any(target in chunk_ids for target in item.target_chunk_ids):
            hits += 1

    recall = hits / len(answerable)
    # Recall is discriminative (off ceiling 1.0 due to conversational vocabulary gaps), but >= 0.90
    assert 0.90 <= recall < 1.0, f"Expected discriminative Recall@3 in [0.90, 1.0), got {recall:.4f}"


def test_deterministic_scorer_groundedness():
    from eval.harness.text_eval import DeterministicScorer

    item = GOLDEN_SET[0]  # GS-01: animal strike deductible $500
    retrieved = [c for c in CORPUS if c.chunk_id == "CHUNK-AUTO-01"]

    # Grounded response with correct number
    corr, grnd, halls, is_ref, ref_acc = DeterministicScorer.score_item(
        item, "The standard deductible for animal strikes is $500.", retrieved
    )
    assert corr is True
    assert grnd is True
    assert len(halls) == 0

    # Hallucinated number ($250 is not in CHUNK-AUTO-01 or question)
    corr, grnd, halls, is_ref, ref_acc = DeterministicScorer.score_item(
        item, "The standard deductible is $250.", retrieved
    )
    assert corr is False
    assert grnd is False
    assert "250" in halls

