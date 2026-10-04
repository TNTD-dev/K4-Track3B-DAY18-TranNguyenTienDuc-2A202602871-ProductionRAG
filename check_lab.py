"""Validate real submission evidence; return nonzero for incomplete deliverables."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def validate():
    errors = []
    required = [
        *(
            f"src/m{i}_{name}.py"
            for i, name in enumerate(
                ("chunking", "search", "rerank", "eval", "enrichment"), 1
            )
        ),
        "src/pipeline.py",
        "analysis/failure_analysis.md",
        "reports/ragas_report.json",
        "reports/naive_baseline_report.json",
        "reports/latency_report.json",
    ]
    for file in required:
        if not (ROOT / file).is_file():
            errors.append(f"Missing {file}")
    reflections = [
        p
        for p in (ROOT / "analysis/reflections").glob("reflection_*.md")
        if p.name != "reflection_TEMPLATE.md"
    ]
    if not (ROOT / "analysis/reflections/reflection_TranNguyenTienDuc.md").is_file():
        errors.append("Missing reflection with the required student filename")
    if not reflections:
        errors.append("Missing personal reflection")
    for path in [ROOT / "analysis/failure_analysis.md", *reflections]:
        if path.is_file() and re.search(
            r"\[Họ|\[Tên|copy template|\(copy template\)", path.read_text()
        ):
            errors.append(f"Unfilled template: {path.name}")
    for path in (ROOT / "src").glob("*.py"):
        if "# TODO" in path.read_text():
            errors.append(f"Unimplemented marker: {path.name}")
    expected_samples = json.loads((ROOT / "test_set.json").read_text())
    for file in ("ragas_report.json", "naive_baseline_report.json"):
        try:
            report = json.loads((ROOT / "reports" / file).read_text())
            aggregate = report["aggregate"]
            if (
                not aggregate.get("scores_valid")
                or aggregate.get("evaluation_status") != "success"
            ):
                errors.append(f"{file}: real RAGAS evaluation not successful")
            if report["num_questions"] != 20 or len(report["per_question"]) != 20:
                errors.append(f"{file}: expected all 20 samples")
            if any(
                not isinstance(aggregate.get(m), (int, float))
                or not 0 <= aggregate[m] <= 1
                for m in METRICS
            ):
                errors.append(f"{file}: invalid metric values")
            for index, row in enumerate(report["per_question"]):
                if not all(
                    key in row
                    for key in (
                        "question",
                        "answer",
                        "contexts",
                        "ground_truth",
                        *METRICS,
                    )
                ):
                    errors.append(f"{file}: missing fields in sample {index + 1}")
                    continue
                if not isinstance(row["contexts"], list) or not all(
                    isinstance(c, str) for c in row["contexts"]
                ):
                    errors.append(f"{file}: invalid contexts in sample {index + 1}")
                if any(
                    type(row[m]) not in (int, float)
                    or not math.isfinite(row[m])
                    or not 0 <= row[m] <= 1
                    for m in METRICS
                ):
                    errors.append(
                        f"{file}: invalid per-question metrics in sample {index + 1}"
                    )
                if index >= len(expected_samples) or any(
                    row[key] != expected_samples[index][key]
                    for key in ("question", "ground_truth")
                ):
                    errors.append(f"{file}: sample {index + 1} does not match test set")
            if len(report["per_question"]) == 20:
                for metric in METRICS:
                    average = sum(row[metric] for row in report["per_question"]) / 20
                    if not math.isclose(aggregate[metric], average, abs_tol=1e-9):
                        errors.append(f"{file}: inconsistent aggregate {metric}")
            if file == "ragas_report.json":
                if len(report.get("failures", [])) != 5:
                    errors.append("Expected 5 measured failure analyses")
                bottom = sorted(
                    report["per_question"], key=lambda row: sum(row[m] for m in METRICS)
                )[:5]
                if [f.get("question") for f in report.get("failures", [])] != [
                    r["question"] for r in bottom
                ]:
                    errors.append("Failure rows do not match measured bottom-5")
                analysis = (ROOT / "analysis/failure_analysis.md").read_text()
                if len(re.findall(r"^### #\d+\.", analysis, re.MULTILINE)) != 5 or any(
                    row["question"] not in analysis for row in bottom
                ):
                    errors.append(
                        "Written analysis does not cover all five measured failures"
                    )
                print(
                    f"Metrics >= 0.70: {sum(aggregate[m] >= 0.70 for m in METRICS)}/4"
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{file}: {exc}")
    try:
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=ROOT,
            timeout=300,
            check=False,
        )
        if tests.returncode:
            errors.append("pytest failed")
        lint = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src/"],
            cwd=ROOT,
            timeout=30,
            check=False,
        )
        if lint.returncode:
            errors.append("ruff failed")
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(str(exc))
    for error in errors:
        print(f"❌ {error}")
    print(
        "🚀 Bài lab sẵn sàng để nộp!"
        if not errors
        else f"❌ Bài lab chưa hợp lệ: {len(errors)} lỗi"
    )
    return not errors


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
