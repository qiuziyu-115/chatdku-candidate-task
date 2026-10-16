# Evaluation harness

A small before-and-after comparison. It answers one question: **does the agent
now produce answers whose sources a reader can actually look up, and how often
does it try to cite something that does not exist?**

## Start with the free script

`retrieval_only.py` exercises both retrieval tools directly. Embeddings run
locally, so it costs nothing and involves no model calls.

```bash
uv run python eval/retrieval_only.py
```

It prints how the index is made up and whether the expected document comes back.
If retrieval is broken, a paid run cannot tell you anything useful, so run this
first.

## Then the paid comparison

Each question costs several model calls (query rewrite, up to three planner
steps, the gate, synthesis). Running all 12 questions twice is roughly 100 calls.
Cheap on a small model, but not free and not deterministic.

```bash
# 1. Baseline: the unmodified revision. eval/ first appears in the feature
#    commit, so borrow just that directory. Write the output outside the repo,
#    otherwise switching branches would discard it.
git checkout a60f11d
git checkout 07d06ec -- eval/
uv run python eval/run_eval.py --out /tmp/before.jsonl

# 2. Back on the feature revision, with the changes applied.
git checkout main
uv run python eval/run_eval.py --out /tmp/after.jsonl

# 3. Compare.
uv run python eval/compare.py /tmp/before.jsonl /tmp/after.jsonl
```

Add `--limit 3` to spend less. Add `--no-rewrite` to skip query rewriting, which
saves one call per question.

## What is recorded for each question

| Field | Meaning | Available on |
| --- | --- | --- |
| `gate_decision` | ALLOW / CLARIFY / INSUFFICIENT from the program gate | both |
| `retrieval_hit` | Whether the expected document was retrieved at all. A control metric: it measures retrieval, which these changes do not touch. | both |
| `citation_count` | How many `[n]` numbers the answer contains | both |
| `has_source_list` | Whether the answer ends with a program-generated source list | both |
| `made_up_numbers` | Numbers cited that were never issued by a retriever | branch only |

`made_up_numbers` needs the citation registry, so it is `n/a` on the baseline.
`compare.py` prints `n/a` instead of `0`, because "not measured" and "measured as
zero" are different claims.

## The question set

`questions.jsonl` has twelve entries in three groups:

- **8 questions from the Advising FAQ.** That document's headings are literally
  questions, so they are a natural set rather than invented ones.
- **2 questions from the undergraduate bulletin.** These facts also appear in the
  FAQ, so the two documents compete for the same retrieval slots and
  `retrieval_hit` is not trivially 100%.
- **2 out-of-scope probes** (tuition for a future intake, dining hall hours).
  Neither document should answer these. They are where the gate should say
  INSUFFICIENT, and where an ungrounded model is most likely to make up a
  citation.

## Reading the result honestly

The likely outcome is not "everything improved". A small model may not follow the
`[n]` format, which can trigger the fallback often. That is a real finding rather
than a bug in the harness: it would mean prompt wording alone does not make a
small model cite its sources, and the programmatic check is doing the work. Any
such result belongs in the limitations section of the README.
