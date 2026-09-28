"""Unit tests for fictional policy corpus and BM25 + hashed n-gram retriever."""

import pytest
from backend.rag.corpus import CORPUS
from backend.rag.retriever import BM25HashedNgramRetriever, HybridRetriever, tokenize


def test_corpus_structure_and_chunk_ids():
    assert len(CORPUS) >= 12
    ids = [c.chunk_id for c in CORPUS]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
    for c in CORPUS:
        assert c.chunk_id.startswith("CHUNK-")
        assert c.doc_id.startswith("DOC-")
        assert len(c.content) > 20


def test_tokenizer():
    tokens = tokenize("Comprehensive auto policy POL-4401 carries a $500 deductible.")
    assert "comprehensive" in tokens
    assert "pol-4401" in tokens
    assert "$500" in tokens
    assert "deductible" in tokens


def test_hybrid_retrieval_returns_top_k():
    retriever = HybridRetriever()
    results = retriever.retrieve("deductible for auto damage", top_k=3)
    assert len(results) == 3
    top_chunk, score = results[0]
    assert score > 0
    assert top_chunk.chunk_id.startswith("CHUNK-AUTO")


def test_trap_disambiguation_standard_vs_glass_deductible():
    retriever = HybridRetriever()
    # Standard deductible query should rank CHUNK-AUTO-01 above CHUNK-AUTO-02
    res_std = retriever.retrieve("What is the standard comprehensive deductible on policy POL-4401?", top_k=5)
    ids_std = [c.chunk_id for c, _ in res_std]
    assert ids_std[0] == "CHUNK-AUTO-01"

    # Glass deductible query should rank CHUNK-AUTO-02 above CHUNK-AUTO-01
    res_glass = retriever.retrieve("How much is the deductible for repairing a chipped windshield?", top_k=5)
    ids_glass = [c.chunk_id for c, _ in res_glass]
    assert ids_glass[0] == "CHUNK-AUTO-02"


def test_bm25_hashed_ngram_retriever_alias():
    retriever = BM25HashedNgramRetriever()
    results = retriever.retrieve("towing roadside assistance", top_k=3)
    assert len(results) == 3
    assert results[0][0].chunk_id == "CHUNK-AUTO-03"

