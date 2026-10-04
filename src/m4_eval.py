from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation."""
    import math

    from config import OPENAI_API_KEY

    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("Evaluation inputs must have equal lengths")
    metrics = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    rows = []
    error = None
    try:
        if not questions:
            raise ValueError("No evaluation samples")
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured; RAGAS scores unavailable"
            )
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        import copy
        from pathlib import Path

        relevance = copy.deepcopy(answer_relevancy)
        judge = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, request_timeout=60, max_retries=1
        )
        # Translate the generic RAGAS examples, preserving schema and scoring formula.
        # Native 0.1.x adapt() can cache fenced JSON that fails Prompt validation.
        relevance.question_generation.examples = [
            {
                "answer": "Albert Einstein sinh ra ở Đức.",
                "context": "Albert Einstein là nhà vật lý lý thuyết sinh ra ở Đức.",
                "output": {
                    "question": "Albert Einstein sinh ra ở đâu?",
                    "noncommittal": 0,
                },
            },
            {
                "answer": "Nó có thể đổi màu da theo nhiệt độ môi trường.",
                "context": "Một loài ếch mới ở rừng Amazon có khả năng đổi màu da theo nhiệt độ môi trường.",
                "output": {
                    "question": "Loài ếch mới được phát hiện có khả năng đặc biệt gì?",
                    "noncommittal": 0,
                },
            },
            {
                "answer": "Everest",
                "context": "Đỉnh núi cao nhất tính từ mực nước biển nằm ở Himalaya.",
                "output": {
                    "question": "Ngọn núi cao nhất trên Trái Đất là gì?",
                    "noncommittal": 0,
                },
            },
            {
                "answer": "Tôi không biết tính năng của điện thoại ra mắt năm 2023 vì không có thông tin sau 2022.",
                "context": "Năm 2023 ra mắt điện thoại có pin dùng được một tháng.",
                "output": {
                    "question": "Điện thoại ra mắt năm 2023 có tính năng đột phá nào?",
                    "noncommittal": 1,
                },
            },
        ]
        relevance.question_generation.examples.append(
            {
                "answer": "Không. Thư viện không mở cửa vào Chủ nhật.",
                "context": "Thư viện mở cửa từ thứ Hai đến thứ Bảy và đóng cửa Chủ nhật.",
                "output": {
                    "question": "Thư viện có mở cửa vào Chủ nhật không?",
                    "noncommittal": 0,
                },
            }
        )
        relevance.question_generation.language = "vietnamese"
        relevance.question_generation.instruction += (
            " Generate the question in Vietnamese. An explicit yes/no answer or a prohibition "
            "grounded in context is committal, not evasive. Do not mark an answer noncommittal "
            "merely because it contains a negation (such as 'không')."
        )
        prompt_path = (
            Path(__file__).resolve().parents[1] / "reports/evaluator_prompt_vi.json"
        )
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            relevance.question_generation.json(ensure_ascii=False, indent=2)
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, relevance, context_precision, context_recall],
            llm=judge,
            embeddings=OpenAIEmbeddings(
                model="text-embedding-3-small", request_timeout=60, max_retries=1
            ),
            run_config=RunConfig(timeout=240, max_retries=2, max_workers=2),
            raise_exceptions=True,
        )
        frame = result.to_pandas()
        frame.to_json(
            Path(__file__).resolve().parents[1] / "reports/evaluation_raw_latest.json",
            orient="records", force_ascii=False, indent=2,
        )
        for _, row in frame.iterrows():
            values = [float(row[name]) for name in metrics]
            if not all(math.isfinite(value) and -1e-9 <= value <= 1 + 1e-9 for value in values):
                raise ValueError(
                    f"RAGAS invalid metrics for {row['question']!r}: "
                    f"{dict(zip(metrics, values))}"
                )
            # Clamp only floating-point roundoff at the mathematical bounds.
            values = [min(1.0, max(0.0, value)) for value in values]
            rows.append(
                EvalResult(
                    row["question"],
                    row["answer"],
                    list(row["contexts"]),
                    row["ground_truth"],
                    *values,
                )
            )
    except Exception as exc:  # noqa: BLE001 — external service boundary, fallback is recorded
        error = f"{type(exc).__name__}: {exc}"
        print(f"  RAGAS evaluation unavailable: {error}")
        rows = [
            EvalResult(q, a, c, gt, 0.0, 0.0, 0.0, 0.0)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
    return {
        **{
            name: sum(getattr(r, name) for r in rows) / len(rows) if rows else 0.0
            for name in metrics
        },
        "per_question": rows,
        "evaluation_status": "failed" if error else "success",
        "evaluation_error": error,
        "scores_valid": error is None,
        "evaluator": "ragas",
        "judge_model": "gpt-4o-mini",
        "answer_relevancy_language": "vietnamese",
        "evaluation_run_config": {"timeout": 240, "max_retries": 2, "max_workers": 2},
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    tree = {
        "context_recall": (
            "Missing relevant evidence",
            "Check chunk boundaries, hybrid retrieval, multi-hop coverage",
        ),
        "context_precision": (
            "Irrelevant or obsolete evidence",
            "Rerank and filter by effective policy version",
        ),
        "faithfulness": (
            "Answer contains unsupported claims",
            "Require evidence citations and abstention; temperature=0",
        ),
        "answer_relevancy": (
            "Answer does not address the question",
            "Answer each requested sub-question explicitly",
        ),
    }
    failures = []
    for result in eval_results:
        scores = {metric: getattr(result, metric) for metric in tree}
        worst = min(scores, key=scores.get)
        diagnosis, fix = tree[worst]
        failures.append(
            {
                "question": result.question,
                "expected": result.ground_truth,
                "answer": result.answer,
                "contexts": result.contexts,
                "worst_metric": worst,
                "score": scores[worst],
                "average_score": sum(scores.values()) / 4,
                "diagnosis": diagnosis,
                "suggested_fix": fix,
                "error_tree": "Output wrong → evidence complete? → evidence relevant/current? → answer grounded? → answer on-topic?",
                "diagnosis_is_hypothesis": True,
            }
        )
    return sorted(failures, key=lambda item: item["average_score"])[: max(0, bottom_n)]


def save_report(
    results: dict, failures: list[dict], path: str = "reports/ragas_report.json"
):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    from datetime import datetime, timezone

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(r) for r in results.get("per_question", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
