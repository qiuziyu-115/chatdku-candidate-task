#!/usr/bin/env python3
"""Measure retrieval quality on its own, with no language model involved.

Embeddings run locally, so this costs nothing and is repeatable. It answers a
different question from ``run_eval.py``: not "did the model answer well?" but
"did the right document come back at all?". Separating the two matters, because
a bad answer can come from bad retrieval or from a model that ignored good
retrieval, and the paid evaluation cannot tell those apart.

Usage:

    uv run python eval/retrieval_only.py
    uv run python eval/retrieval_only.py --top-k 20     # does a wider window help?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from chatdku.config import get_settings
from chatdku.ingestion.pipeline import qdrant_path
from chatdku.tools import KeywordRetriever, VectorRetriever
from chatdku.tools.retrieval import document_from_payload

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")


def load_questions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sources(documents) -> set[str]:
    return {
        str(document.metadata.get("source_path") or document.metadata.get("file_name") or "?")
        for document in documents
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=None, help="Defaults to CHATDKU_RETRIEVAL_TOP_K.")
    args = parser.parse_args()

    settings = get_settings()
    top_k = args.top_k or settings.retrieval_top_k

    client = QdrantClient(path=str(qdrant_path(settings)))
    try:
        total = client.count(settings.vector_collection).count
        records, _ = client.scroll(
            collection_name=settings.vector_collection, with_payload=True, with_vectors=False, limit=100_000
        )
        per_source: dict[str, int] = {}
        for record in records:
            # The payload is LlamaIndex-encoded, so reuse the same decoder the
            # keyword retriever uses rather than reading keys off the payload.
            document = document_from_payload(str(record.id), record.payload, 0.0)
            name = str(
                document.metadata.get("source_path")
                or document.metadata.get("file_name")
                or "?"
            )
            per_source[name] = per_source.get(name, 0) + 1
    finally:
        client.close()

    print("--- index composition ---")
    print(f"total chunks : {total}")
    for name, count in sorted(per_source.items(), key=lambda item: -item[1]):
        print(f"  {count:>5} ({count / total:6.1%})  {name}")

    vector = VectorRetriever(settings)
    keyword = KeywordRetriever(settings)

    print(f"\n--- retrieval hit rate at top_k={top_k} (no model calls) ---")
    rows = []
    for item in load_questions(args.questions):
        question = item["question"]
        expected = item.get("expected_source")
        vector_hits = sources(vector.query(question, top_k))
        keyword_hits = sources(keyword.query(question, top_k))
        row = {
            "id": item.get("id"),
            "expected": expected,
            "vector": (expected in vector_hits) if expected else None,
            "keyword": (expected in keyword_hits) if expected else None,
            "union": (expected in (vector_hits | keyword_hits)) if expected else None,
        }
        rows.append(row)
        if expected:
            mark = lambda value: "hit " if value else "MISS"  # noqa: E731
            print(
                f"  {row['id']:<7} vector={mark(row['vector'])} "
                f"keyword={mark(row['keyword'])} union={mark(row['union'])}  {question[:58]}"
            )
        else:
            print(f"  {row['id']:<7} (out of scope; no expected document)            {question[:58]}")

    scored = [row for row in rows if row["expected"]]
    if scored:
        print("\n--- summary over answerable questions ---")
        for key in ("vector", "keyword", "union"):
            hits = sum(1 for row in scored if row[key])
            print(f"  {key:<8}: {hits}/{len(scored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
