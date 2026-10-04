"""Repeat the frozen RAGAS judge on unchanged predictions; retain every run."""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.m4_eval import EvalResult, evaluate_ragas, failure_analysis, save_report

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def main(repeats=3):
    if repeats < 2:
        raise ValueError("At least two judge runs are required")
    paths = ["naive_baseline_report.json", "ragas_report.json"]
    original = {
        name: json.loads((ROOT / "reports" / name).read_text()) for name in paths
    }
    runs = {name: [report] for name, report in original.items()}
    out = ROOT / "reports" / "final_judge_runs"
    out.mkdir(exist_ok=True)
    for name, report in original.items():
        if (
            not report["aggregate"].get("scores_valid")
            or report["aggregate"].get("judge_repeats", 1) != 1
        ):
            raise ValueError(
                "Start from one complete successful evaluation per pipeline"
            )
        (out / f"run_1_{name}").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
    for repeat in range(2, repeats + 1):
        for name in paths:
            rows = original[name]["per_question"]
            start = time.perf_counter()
            result = evaluate_ragas(
                [r["question"] for r in rows],
                [r["answer"] for r in rows],
                [r["contexts"] for r in rows],
                [r["ground_truth"] for r in rows],
            )
            result["evaluation_ms"] = (time.perf_counter() - start) * 1000
            target = out / f"run_{repeat}_{name}"
            save_report(result, [], str(target))
            if not result["scores_valid"]:
                raise RuntimeError(f"Judge run failed: {target}. No mean published.")
            runs[name].append(json.loads(target.read_text()))
            print(f"{name} judge run {repeat}/{repeats} complete", flush=True)
    for name in paths:
        report = original[name]
        combined = []
        for index, first in enumerate(report["per_question"]):
            combined.append(
                EvalResult(
                    first["question"],
                    first["answer"],
                    first["contexts"],
                    first["ground_truth"],
                    *(
                        mean(run["per_question"][index][metric] for run in runs[name])
                        for metric in METRICS
                    ),
                )
            )
        result = {
            **report["aggregate"],
            "per_question": combined,
            **{
                metric: mean(getattr(row, metric) for row in combined)
                for metric in METRICS
            },
            "judge_repeats": repeats,
            "evaluation_method": "equal_mean_of_frozen_RAGAS_judge_runs",
            "judge_run_scores": [
                {metric: run["aggregate"][metric] for metric in METRICS}
                for run in runs[name]
            ],
            "judge_score_dispersion": {
                metric: {
                    "min": min(run["aggregate"][metric] for run in runs[name]),
                    "max": max(run["aggregate"][metric] for run in runs[name]),
                    "std": pstdev(run["aggregate"][metric] for run in runs[name]),
                }
                for metric in METRICS
            },
            "evaluation_ms": sum(
                run["aggregate"].get("evaluation_ms", 0) for run in runs[name]
            ),
            "evaluation_note": f"Same saved predictions/contexts scored {repeats} times; all successful judge runs included equally. Baseline initial judge time was not recorded; its evaluation_ms sums additional runs only.",
        }
        failures = failure_analysis(combined, 5) if name == "ragas_report.json" else []
        save_report(result, failures, str(ROOT / "reports" / name))
        print(name, {metric: result[metric] for metric in METRICS}, flush=True)
    latency = ROOT / "reports/latency_report.json"
    data = json.loads(latency.read_text())
    production = json.loads((ROOT / "reports/ragas_report.json").read_text())
    data["evaluation_ms"] = production["aggregate"]["evaluation_ms"]
    data["judge_repeats"] = repeats
    latency.write_text(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    main(parser.parse_args().repeats)
