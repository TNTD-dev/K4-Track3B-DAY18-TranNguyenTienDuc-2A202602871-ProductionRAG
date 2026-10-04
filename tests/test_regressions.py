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


@pytest.mark.parametrize("content", [None, "   "])
def test_empty_generation_response_uses_explicit_fallback(monkeypatch, content):
    import openai

    from src import pipeline

    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content=content, tool_calls=None))
    ]
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(pipeline, "OPENAI_API_KEY", "test-key")
    answer, status = pipeline.generate_answer("query", ["source evidence"])
    assert answer == "source evidence"
    assert status == "extractive_fallback"


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("0.1 + 0.2", "0.3"),
        ("4200000 * 0.03 * (16 - 10) / 30", "25200"),
        ("-12 + 3 * (7 - 2)", "3"),
    ],
)
def test_decimal_calculation(expression, expected):
    from src.calculation import calculate

    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression", ["__import__('os').system('exit 1')", "2 ** 100", "1 / 0", "True + 1"]
)
def test_calculator_rejects_non_arithmetic_and_invalid_operations(expression):
    from src.calculation import calculate

    with pytest.raises((ValueError, ArithmeticError)):
        calculate(expression)


def test_generation_executes_calculator_and_returns_final_answer(monkeypatch):
    import json

    import openai
    from openai.types.chat import ChatCompletionMessage

    from src import pipeline

    request = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "calculation-1",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": json.dumps({"expression": "72 / 6"}),
                },
            }
        ],
    )
    final = ChatCompletionMessage(role="assistant", content="Kết quả là 12.")
    client = Mock()
    client.chat.completions.create.side_effect = [
        Mock(choices=[Mock(message=request)]),
        Mock(choices=[Mock(message=final)]),
    ]
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(pipeline, "OPENAI_API_KEY", "test-key")
    answer, status = pipeline.generate_answer("chia tổng cho các nhóm", ["nguồn"])
    assert answer == "Kết quả là 12."
    assert status == "llm_calculated"
    assert client.chat.completions.create.call_count == 2
    tool_message = client.chat.completions.create.call_args.kwargs["messages"][-1]
    assert tool_message["tool_call_id"] == "calculation-1"
    assert json.loads(tool_message["content"]) == {"result": "12"}


@pytest.mark.parametrize(
    "elapsed, grace, expected", [(45, 30, "108000"), (20, 30, "0"), (30, 30, "0")]
)
def test_overdue_fee_excludes_grace_period(elapsed, grace, expected):
    from src.calculation import overdue_fee

    result = overdue_fee(7200000, 0.03, elapsed, grace)
    assert result["estimated_fee"] == expected
    assert int(result["overdue_days"]) == max(0, elapsed - grace)
    assert result["assumption"]


def test_overdue_fee_rejects_invalid_rate():
    from src.calculation import overdue_fee

    with pytest.raises(ValueError):
        overdue_fee(7200000, 3, 45, 30)


@pytest.mark.parametrize("money", ["7,2 triệu", "7.200.000 VNĐ", "7200 nghìn"])
def test_grounded_fee_uses_source_rule_and_question_values(money):
    from src.calculation import grounded_fee_answer

    contexts = [
        "[Nguồn: example.md]\nKhoản tạm ứng phải thanh toán trong vòng **30 ngày**. Khoản tạm ứng chưa thanh toán sau 30 ngày tính phí **3%/tháng**."
    ]
    answer = grounded_fee_answer(
        f"Tạm ứng {money}, sau 45 ngày thanh toán, phí bao nhiêu?", contexts
    )
    assert "**108000 VNĐ**" in answer
    assert "15 ngày" in answer
    assert "giả định tính toán" in answer
    assert "example.md" in answer


def test_grounded_fee_declines_ambiguous_or_unsupported_evidence():
    from src.calculation import grounded_fee_answer

    question = "Tạm ứng 7 triệu, sau 45 ngày thanh toán, phí bao nhiêu?"
    assert (
        grounded_fee_answer(
            question, ["[Nguồn: x.md]\nKhoản tạm ứng tính phí 3%/tháng."]
        )
        is None
    )
    assert (
        grounded_fee_answer(
            question,
            [
                "[Nguồn: x.md]\nKhoản tạm ứng sau 30 ngày tính phí 3%/tháng, sau 60 ngày tính phí 4%/tháng."
            ],
        )
        is None
    )
    assert (
        grounded_fee_answer(
            "Tạm ứng 7 triệu đi công tác, sau 45 ngày thanh toán?", ["Khoản tạm ứng"]
        )
        is None
    )


def test_policy_version_filter_keeps_historical_evidence():
    from src.retrieval import policy_metadata, valid_policy

    a = policy_metadata(
        {
            "text": "# Example (Phiên bản 2020)\nNgày hiệu lực: 01/06/2020",
            "metadata": {"source": "a"},
        }
    )
    b = policy_metadata(
        {
            "text": "# Example (Phiên bản 2022)\nNgày hiệu lực: 01/06/2022",
            "metadata": {"source": "b"},
        }
    )
    assert a["policy_family"] == b["policy_family"]
    versions = {a["policy_family"]: [a["effective_date"], b["effective_date"]]}
    old, new = SearchResult("a", 1, a, "hybrid"), SearchResult("b", 1, b, "hybrid")
    assert not valid_policy(old, "Hiện hành?", versions)
    assert valid_policy(new, "Hiện hành?", versions)
    assert valid_policy(old, "Năm 2021?", versions)
    assert not valid_policy(new, "Năm 2021?", versions)


def test_multi_part_query_covers_both_facets(monkeypatch):
    from src import pipeline
    from src.m3_rerank import RerankResult

    a, b = [
        SearchResult(text, 1, {"source": source}, "hybrid")
        for text, source in [("first", "a"), ("second", "b")]
    ]
    search = Mock(parent_map={}, query_timings=[])
    search.search.side_effect = [[a], [a], [b]]
    reranker = Mock()
    reranker.rerank.side_effect = lambda q, docs, top_k: [
        RerankResult(d["text"], 1, 1, d["metadata"], i + 1)
        for i, d in enumerate(docs[:top_k])
    ]
    monkeypatch.setattr(
        pipeline, "generate_answer", lambda q, contexts: ("answer", "test")
    )
    _, contexts = pipeline.run_query(
        "Được hỗ trợ bao nhiêu và điều kiện là gì?", search, reranker
    )
    assert len(contexts) == 2
    assert any("[Nguồn: a]" in c for c in contexts)
    assert any("[Nguồn: b]" in c for c in contexts)
    assert search.search.call_count == 3
