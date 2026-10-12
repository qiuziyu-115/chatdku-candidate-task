#!/usr/bin/env python3
"""Run the agent over a fixed question set and record how it cited.

The harness deliberately does not import the code it measures. It has its own
small regex for reading [n] numbers, and it reads the agent's extra output
defensively. That is what lets the same script run against an unmodified
revision and still produce a baseline to compare against.

Usage, from the repository root:

    uv run python eval/run_eval.py --out eval/results/before.jsonl
    uv run python eval/run_eval.py --out eval/results/after.jsonl
    uv run python eval/compare.py eval/results/before.jsonl eval/results/after.jsonl

Every question costs several model calls. This is not free, and it is not
deterministic.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Kept local on purpose, so the harness does not depend on the code under test.
MARKER = re.compile(r"\[(\d{1,3})\]")


def load_questions(path: Path, limit: int | None) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.strip().startswith("#"):
            questions.append(json.loads(line))
    return questions[:limit] if limit else questions


def retrieval_hit(trajectory: str, expected_source: str | None) -> bool | None:
    """Did the evidence mention the document we expected?

    Returns None for the out-of-scope questions, which have no expected document.
    Those are left out of the hit rate rather than counted as misses.
    """
    if not expected_source:
        return None
    return expected_source.casefold() in (trajectory or "").casefold()


def measure(prediction, response: str) -> dict:
    """Collect what can be measured, with or without citation support."""
    numbers = [int(value) for value in MARKER.findall(response)]
    row = {
        "citation_count": len(numbers),
        "has_source_list": "**Sources**" in response,
    }
    made_up = getattr(prediction, "invalid_labels", None)
    if made_up is not None:
        row["made_up_numbers"] = list(made_up)
    return row


def summarize(rows: list[dict], out: Path) -> None:
    print("\n=== summary ===")
    print(f"questions                   : {len(rows)}")
    allowed = [row for row in rows if row["gate_decision"] == "ALLOW"]
    print(f"gate ALLOW                  : {len(allowed)}/{len(rows)}")
    hits = [row["retrieval_hit"] for row in rows if row["retrieval_hit"] is not None]
    if hits:
        print(f"expected document retrieved : {sum(hits)}/{len(hits)}")
    print(f"answers with a citation     : {sum(1 for r in rows if r['citation_count'] > 0)}/{len(rows)}")
    print(f"answers with a source list  : {sum(1 for r in rows if r['has_source_list'])}/{len(rows)}")
    if any("made_up_numbers" in row for row in rows):
        made_up = sum(1 for row in rows if row["made_up_numbers"])
        print(f"answers with a made-up cite : {made_up}/{len(rows)}")
    print(f"written to                  : {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--out", type=Path, required=True, help="Where to write the JSONL results.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument("--no-rewrite", action="store_true", help="Skip query rewriting.")
    args = parser.parse_args()

    if not args.questions.exists():
        print(f"Question file not found: {args.questions}", file=sys.stderr)
        return 2

    # Import the agent first. It loads NumPy before DSPy on purpose, because
    # DSPy's lazy-import proxy breaks NumPy's own module setup otherwise.
    from chatdku.agent import Agent

    import dspy

    # DSPy caches model responses on disk by default. A cached answer is not a
    # fresh measurement and it makes the timings meaningless, so turn it off.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    questions = load_questions(args.questions, args.limit)
    if not questions:
        print(f"No questions found in {args.questions}", file=sys.stderr)
        return 2

    agent = Agent(rewrite_query=not args.no_rewrite)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.out.open("w", encoding="utf-8") as sink:
        for index, item in enumerate(questions, start=1):
            agent.reset()
            started = time.perf_counter()
            prediction = agent(item["question"])
            elapsed = time.perf_counter() - started

            response = prediction.response
            row = {
                "n": index,
                "id": item.get("id"),
                "topic": item.get("topic"),
                "question": item["question"],
                "expected_source": item.get("expected_source"),
                "gate_decision": prediction.gate_decision,
                "retrieval_hit": retrieval_hit(prediction.trajectory, item.get("expected_source")),
                "seconds": round(elapsed, 2),
                "response": response,
                **measure(prediction, response),
            }
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()

            print(
                f"[{index}/{len(questions)}] {row['id']:<7} "
                f"gate={row['gate_decision']:<12} "
                f"citations={row['citation_count']} "
                f"made_up={row.get('made_up_numbers', 'n/a')} "
                f"{row['seconds']}s"
            )

    summarize(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
