"""Boundary tests missing from the starter grading suite; no paid services needed."""

from unittest.mock import Mock

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.m1_chunking import chunk_hierarchical, chunk_structure_aware
from src.m2_search import DenseSearch, SearchResult, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, save_report
from src.pipeline import run_query


def test_hierarchy_bounds_and_cross_document_ids():
    parents, children = chunk_hierarchical("word " * 500, 200, 50, {"source": "a"})
    other, _ = chunk_hierarchical("word " * 500, 200, 50, {"source": "b"})
    assert all(len(p.text) <= 200 for p in parents)
    assert all(len(c.text) <= 50 for c in children)
    assert {p.metadata["parent_id"] for p in parents}.isdisjoint(
        p.metadata["parent_id"] for p in other
    )


def test_markdown_fence_does_not_create_section():
    chunks = chunk_structure_aware("# Real\n```md\n# Fake\n```\n\n|a|b|\n|1|2|")
    assert len(chunks) == 1
    assert "# Fake" in chunks[0].text


def test_rrf_preserves_distinct_sources_and_counts_rank_once():
    a = SearchResult("same", 9.0, {"source": "a"}, "bm25")
    b = SearchResult("same", 2.0, {"source": "b"}, "dense")
    merged = reciprocal_rank_fusion([[a, a], [a, b]])
    assert len(merged) == 2
    assert merged[0].score == pytest.approx(2 / 61)


def test_dense_index_query_and_replace():
    dense = DenseSearch.__new__(DenseSearch)
    dense.client = QdrantClient(":memory:")
    dense._encoder = Mock()
    dense._encoder.encode.side_effect = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0, 0.0]),
        np.array([[0.0, 1.0]]),
    ]
    dense.index([{"text": "A", "metadata": {"chunk_id": "a"}}, {"text": "B"}])
    assert dense.search("query", top_k=1)[0].metadata["chunk_id"] == "a"
    dense.index([{"text": "C"}])
    assert dense.client.count("lab18_production").count == 1


def test_single_rerank_score():
    reranker = CrossEncoderReranker()
    reranker._model = Mock()
    reranker._model.predict.return_value = np.float32(0.5)
    assert reranker.rerank("q", [{"text": "a"}])[0].rerank_score == 0.5
    assert reranker.rerank("q", [{"text": "a"}], top_k=0) == []


def test_missing_key_report_is_explicit_and_keeps_samples(monkeypatch, tmp_path):
    import config

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    result = evaluate_ragas(["q"], ["a"], [["c"]], ["g"])
    assert result["scores_valid"] is False
    assert len(result["per_question"]) == 1
    save_report(result, [], str(tmp_path / "report.json"))
    import json

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["num_questions"] == 1
    assert report["per_question"][0]["contexts"] == ["c"]


def test_input_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_ragas(["q"], [], [], [])


def test_query_expands_unique_parents(monkeypatch):
    from src import pipeline

    parents, children = chunk_hierarchical(
        "source evidence " * 30, 200, 50, {"source": "a"}
    )
    search = Mock()
    search.parent_map = {p.metadata["parent_id"]: p for p in parents}
    search.query_timings = []
    search.search.return_value = [
        SearchResult(c.text, 1.0, {**c.metadata, "parent_id": c.parent_id}, "hybrid")
        for c in children
    ]
    reranker = CrossEncoderReranker()
    reranker._model = Mock()
    reranker._model.predict.return_value = np.arange(len(children), 0, -1)
    monkeypatch.setattr(pipeline, "generate_answer", lambda q, contexts: ("a", "test"))
    _, contexts = run_query("q", search, reranker)
    assert len(contexts) == len(parents)
    assert parents[0].text in contexts[0]


def test_semantic_splits_on_cosine_boundary(monkeypatch):
    import src.m1_chunking as chunking

    encoder = Mock()
    encoder.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(chunking, "_semantic_encoder", lambda: encoder)
    result = chunking.chunk_semantic("A. B. C.", threshold=0.5)
    assert [chunk.text for chunk in result] == ["A. B.", "C."]
    assert chunking.chunk_semantic("") == []
    encoder.encode.assert_called_once()


def test_enrichment_one_request_and_source_identity(monkeypatch, tmp_path):
    import src.m5_enrichment as enrichment

    completion = Mock()
    completion.choices = [
        Mock(
            message=Mock(
                content='{"summary":"S","questions":["Q?"],"context":"C","metadata":{"source":"fake","chunk_id":"fake"}}'
            )
        )
    ]
    client = Mock()
    client.chat.completions.create.return_value = completion
    import openai

    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=client))
    monkeypatch.setattr(enrichment, "OPENAI_API_KEY", "test-placeholder")
    chunks = [
        {
            "text": "unique-regression-source-identity",
            "metadata": {"source": "real", "chunk_id": "real"},
        }
    ]
    monkeypatch.setattr(
        enrichment, "__file__", str(tmp_path / "src" / "m5_enrichment.py")
    )
    result = enrichment.enrich_chunks(chunks)[0]
    assert client.chat.completions.create.call_count == 1
    assert result.auto_metadata["source"] == result.auto_metadata["chunk_id"] == "real"
    assert result.original_text in result.enriched_text
