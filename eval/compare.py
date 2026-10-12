#!/usr/bin/env python3
"""Compare two result files produced by run_eval.py.

Usage:

    uv run python eval/compare.py eval/results/before.jsonl eval/results/after.jsonl

A metric that only exists on one side is printed as "n/a" rather than as zero,
because "not measured" and "measured as zero" are different claims.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(rows: list[dict]) -> dict:
    hits = [row["retrieval_hit"] for row in rows if row.get("retrieval_hit") is not None]
    has_made_up_field = any("made_up_numbers" in row for row in rows)
    times = [row["seconds"] for row in rows if "seconds" in row]
    return {
        "questions": len(rows),
        "gate ALLOW": sum(1 for row in rows if row.get("gate_decision") == "ALLOW"),
        "expected document retrieved": f"{sum(hits)}/{len(hits)}" if hits else "n/a",
        "answers with a citation": sum(1 for row in rows if row.get("citation_count", 0) > 0),
        "answers with a source list": sum(1 for row in rows if row.get("has_source_list")),
        "answers with a made-up cite": (
            sum(1 for row in rows if row.get("made_up_numbers")) if has_made_up_field else "n/a"
        ),
        "mean seconds per question": round(statistics.mean(times), 1) if times else "n/a",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    missing = [str(path) for path in (args.before, args.after) if not path.exists()]
    if missing:
        raise SystemExit(f"Missing result file(s): {', '.join(missing)}")

    before = stats(load(args.before))
    after = stats(load(args.after))

    width = max(len(key) for key in before) + 2
    print(f"{'metric':<{width}}{'before':>14}{'after':>14}")
    print("-" * (width + 28))
    for key in before:
        print(f"{key:<{width}}{str(before[key]):>14}{str(after[key]):>14}")

    print()
    print("Reminder: 'n/a' means not measured, which is not the same as zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
