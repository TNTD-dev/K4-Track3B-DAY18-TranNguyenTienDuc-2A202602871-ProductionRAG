"""Controlled baseline: basic chunks + dense only; shared generation and evaluation."""

import time

from config import EMBEDDING_MODEL, NAIVE_COLLECTION
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import evaluate_ragas, load_test_set, save_report
from src.pipeline import ROOT, generate_answer


def main():
    chunks = [
        {"text": c.text, "metadata": c.metadata}
        for doc in load_documents()
        for c in chunk_basic(doc["text"], metadata=doc["metadata"])
    ]
    search = DenseSearch()
    start = time.perf_counter()
    search.index(chunks, collection=NAIVE_COLLECTION)
    index_ms = (time.perf_counter() - start) * 1000
    questions, answers, evidence, references, timings = [], [], [], [], []
    for item in load_test_set():
        start = time.perf_counter()
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [
            f"[Nguồn: {r.metadata.get('source', 'unknown')}]\n{r.text}" for r in results
        ]
        search_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        answer, status = generate_answer(item["question"], contexts)
        timings.append(
            {
                "question": item["question"],
                "dense_search_ms": search_ms,
                "generation_ms": (time.perf_counter() - start) * 1000,
                "generation_status": status,
            }
        )
        questions.append(item["question"])
        answers.append(answer)
        evidence.append(contexts)
        references.append(item["ground_truth"])
        print(f"Baseline [{len(questions)}/20] {item['question']}", flush=True)
    results = evaluate_ragas(questions, answers, evidence, references)
    results.update(
        {
            "index_ms": index_ms,
            "query_timings": timings,
            "retrieval_backend": search.backend,
            "pipeline": "basic+dense",
            "generation_model": "gpt-4o-mini",
            "embedding_model": EMBEDDING_MODEL,
        }
    )
    save_report(results, [], str(ROOT / "reports/naive_baseline_report.json"))
    return results


if __name__ == "__main__":
    main()
