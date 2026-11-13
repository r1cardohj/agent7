"""Render JSONL evaluation results as Markdown, with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

INVALID_STATUSES = frozenset({"infrastructure_error", "verify_error", "runner_error"})
STRATEGY_ORDER = ("fixed-fast", "fixed-strong", "auto-tier", "auto-full")


def load_results(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "results.jsonl"
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "task_id" not in value or "strategy" not in value:
            raise ValueError(f"invalid result at {path}:{number}")
        records.append(value)
    if not records:
        raise ValueError(f"no results in {path}")
    return records


def _valid(record: dict[str, Any]) -> bool:
    return record.get("status") not in INVALID_STATUSES


def _tokens(record: dict[str, Any]) -> int:
    return sum(int(record.get(key, 0) or 0) for key in (
        "prompt_tokens", "completion_tokens", "score_prompt_tokens", "score_completion_tokens",
    ))


def _median(records: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(record.get(key, 0) or 0) for record in records)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def paired_success_delta(
    records: list[dict[str, Any]], candidate: str, baseline: str, *, samples: int = 2000,
) -> dict[str, Any]:
    """Cluster bootstrap by task; repeats within a task are not independent samples."""
    grouped: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["strategy"] in {candidate, baseline} and _valid(record):
            grouped[record["task_id"]][(record["strategy"], int(record["repeat"]))] = record
    deltas: list[float] = []
    for runs in grouped.values():
        repeats = {repeat for strategy, repeat in runs if strategy == candidate}
        repeats &= {repeat for strategy, repeat in runs if strategy == baseline}
        if not repeats:
            continue
        candidate_mean = statistics.mean(
            bool(runs[(candidate, repeat)]["success"]) for repeat in repeats
        )
        baseline_mean = statistics.mean(
            bool(runs[(baseline, repeat)]["success"]) for repeat in repeats
        )
        deltas.append(candidate_mean - baseline_mean)
    if not deltas:
        return {"tasks": 0, "delta_pp": None, "low_pp": None, "high_pp": None}
    rng = random.Random(0)
    bootstraps = [
        statistics.mean(rng.choice(deltas) for _ in deltas)
        for _ in range(samples)
    ]
    return {
        "tasks": len(deltas),
        "delta_pp": 100 * statistics.mean(deltas),
        "low_pp": 100 * _percentile(bootstraps, 0.025),
        "high_pp": 100 * _percentile(bootstraps, 0.975),
    }


def _strategy_rows(records: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    names = list(dict.fromkeys(record["strategy"] for record in records))
    names.sort(key=lambda name: (STRATEGY_ORDER.index(name) if name in STRATEGY_ORDER else 99, name))
    for name in names:
        rows = [record for record in records if record["strategy"] == name]
        valid = [record for record in rows if _valid(record)]
        successes = sum(bool(record["success"]) for record in valid)
        yield {
            "strategy": name,
            "runs": len(rows),
            "valid": len(valid),
            "successes": successes,
            "success_rate": 100 * successes / len(valid) if valid else None,
            "invalid": len(rows) - len(valid),
            "wall": _median(valid, "wall_seconds") if valid else None,
            "score": _median(valid, "score_seconds") if valid else None,
            "tokens": statistics.median(_tokens(record) for record in valid) if valid else None,
            "failovers": sum(int(record.get("failovers", 0)) for record in rows),
        }


def _number(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "# agent7 auto evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Success is based on independent verification. Infrastructure and harness failures "
        "are shown separately and excluded from success-rate denominators.",
        "",
        "| Strategy | Runs | Valid | Passed | Success % | Invalid | Median wall s | "
        "Median score s | Median tokens | Failovers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _strategy_rows(records):
        lines.append(
            f"| {row['strategy']} | {row['runs']} | {row['valid']} | {row['successes']} | "
            f"{_number(row['success_rate'])} | {row['invalid']} | {_number(row['wall'])} | "
            f"{_number(row['score'], 2)} | {_number(row['tokens'], 0)} | {row['failovers']} |"
        )
    lines += ["", "Tokens include execution and auto-scoring input/output tokens.", ""]

    categories = sorted({record.get("category", "unknown") for record in records})
    strategies = [row["strategy"] for row in _strategy_rows(records)]
    lines += [
        "## Success by category",
        "",
        "| Category | " + " | ".join(strategies) + " |",
        "| --- | " + " | ".join("---:" for _ in strategies) + " |",
    ]
    for category in categories:
        cells = []
        for strategy in strategies:
            rows = [record for record in records if record["strategy"] == strategy
                    and record.get("category") == category and _valid(record)]
            cells.append(f"{sum(bool(row['success']) for row in rows)}/{len(rows)}" if rows else "n/a")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    if any(record["strategy"] == "auto-full" for record in records):
        lines += [
            "",
            "## Paired success delta: auto-full minus baseline",
            "",
            "Percentile bootstrap over tasks (2,000 resamples, fixed seed); repeats are "
            "averaged within each task. An interval spanning zero is inconclusive.",
            "",
            "| Baseline | Paired tasks | Delta pp | 95% interval pp |",
            "| --- | ---: | ---: | ---: |",
        ]
        for baseline in strategies:
            if baseline == "auto-full":
                continue
            delta = paired_success_delta(records, "auto-full", baseline)
            interval = "n/a" if delta["tasks"] == 0 else (
                f"[{_number(delta['low_pp'])}, {_number(delta['high_pp'])}]"
            )
            lines.append(
                f"| {baseline} | {delta['tasks']} | {_number(delta['delta_pp'])} | {interval} |"
            )
    lines += [
        "",
        "## Reading this report",
        "",
        "- Compare `auto-full` with both fixed models and the simpler `auto-tier` ablation.",
        "- Inspect `artifacts/*/verify.txt` and `changes.patch` before trusting outliers.",
        "- A routing-score or intent agreement is not evidence of downstream task quality.",
        "- Raw tokens are retained so cost can be calculated separately with dated provider prices.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results.jsonl or its run directory")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    try:
        rendered = render_report(load_results(args.results))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"error: {error}\n")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
