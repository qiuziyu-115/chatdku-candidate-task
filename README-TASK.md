# ChatDKU candidate task — implementation notes

What I set up, what I changed, and what is still wrong. The upstream `README.md`,
`docs/ingestion.md` and `docs/agent.md` are unchanged.

The supplied document set is **not committed**. `data/` is git-ignored upstream
and stays that way.

## Setup and how to run it

```bash
git -c http.version=HTTP/1.1 clone https://github.com/Edge-Intelligence-Lab/ChatDKU-Public.git
cd ChatDKU-Public && uv sync
cp .env.example .env            # set CHATDKU_LLM_MODEL / _BASE_URL / _API_KEY

mkdir -p data/documents         # copy the two supplied documents in
uv run agent --check-embeddings    # prints a vector dimension, e.g. 384
uv run chatdku-ingest              # add --replace to rebuild an existing index
uv run agent "When is the deadline to withdraw from a class?"

uv run pytest                      # no model, no network, no index needed
uv run python eval/retrieval_only.py   # free, no model calls
```

Eight setup problems are in `docs/setup-issues.md`. Three cost real time: the
clone fails with an HTTP/2 framing error, the HuggingFace hub stalls on network
checks even when the model is already cached, and `.docx` ingestion fails because
`docx2txt` was never declared.

Embeddings ran locally on CPU, so no GPU was needed. The chat model was a hosted
OpenAI-compatible API.

## What I changed

Both changes come from one gap. ChatDKU promises "accurate, verifiable
information", but nothing in the code checked that a claim could be traced back to
a source:

- `tools/retrieval.py` printed `Source: <file path>` with no stable id, so eight
  chunks of a 435-page PDF looked identical to the model.
- `modules/prompts.py` never asked for a source.
- `agent.py` ran the program gate *before* synthesis, so nothing ever saw the
  finished answer. A made-up citation could not be caught.

### 1. Give every retrieved passage a number

**Problem.** Passages reach the model with no identity, and the prompt does not
ask for sources. A reader cannot tell which sentence came from which document. If
the model names a source anyway, that name is free text the program cannot check.

**Change.** New file `citations.py`. Each retrieved chunk gets a number like
`[1]`. The registry is cleared each turn and shared by both tools, so a number
always means the same passage. `format_documents` puts the number into the
observation, the prompt asks for inline numbers, and the agent appends the source
list itself rather than asking the model to write it.

**Checked by** `tests/test_citations.py` and `test_retrieval_format.py`: numbering
starts at 1 and follows the order passages were found; the same chunk retrieved
twice keeps one number; a chunk found by both tools has one identity; numbering
restarts each turn; the source list never names a passage that was not retrieved;
without a registry the old output is unchanged byte for byte.

### 2. Check that the numbers in the answer are real

**Problem.** A prompt is a request, not a guarantee. A small model will cite a
number that was never issued, or write an answer with no citation at all. The gate
cannot catch either case, because it runs before the answer exists.

**Change.** New file `verification.py`, four plain functions that read the
finished answer and compare its numbers with the ones actually issued. When
`CHATDKU_REQUIRE_CITATIONS` is on (the default) and the gate allowed the answer,
the agent throws the answer away if either check fails and falls back to the
gate's safe response.

Why throw the whole answer away instead of deleting the bad number? Given
`The deadline is March 1 [99].`, deleting the number leaves `The deadline is
March 1.` — a confident claim with nothing behind it, and now nothing marks it as
doubtful. The model claimed support that does not exist, so the answer should not
be published rather than quietly edited.

**Checked by** `tests/test_verification.py` and `test_agent_citations.py`: a
made-up number is detected; an answer containing one is discarded, not edited; a
discarded answer does not inherit a source list; an allowed answer with no
citation is also discarded; with enforcement off, made-up numbers are still
removed; malformed brackets like `[]` and `[1234]` do not crash anything.

### Small fix: the missing `.docx` dependency

`loader.py` and `docs/ingestion.md` both say `.docx` is supported, but
`pyproject.toml` never declared `docx2txt` — `llama-index-readers-file` only
installs it as an optional extra. So a clean checkout cannot ingest the Word
document. I declared it.

This is not one of the two improvements. It had to be fixed before the document
set could be ingested at all, and it is listed separately so the contribution is
not overstated.

## How I verified it

**Unit tests — free, and the main reason to trust any of this.**

```
uv run pytest  ->  44 passed in about 2 seconds
```

The count is not the point. The point is that the whole citation mechanism is
testable with no model, no network, no API key and no index. If checking a change
needed paid model calls, it would not get checked often enough to trust.

**Retrieval on its own — also free, and already run.** `eval/retrieval_only.py`
uses local embeddings only.

- The index holds 844 chunks: 808 (95.7%) bulletin, 36 (4.3%) FAQ. `manifest.json`
  says 436, but that is counted before chunking, because the PDF reader emits one
  record per page.
- Retrieval found the expected document **10 times out of 10**, for both tools and
  their union, at `top_k=8` and again at `top_k=20`.

So retrieval is not the bottleneck: any weakness in a final answer comes from how
the model uses evidence it already has, which is the layer these changes target.
That run also backed a design choice below — one turn calling both tools returned
16 passages that were only 13 different chunks, so without a shared registry three
of them would each have been given two numbers.

**The paid before-and-after evaluation — run.** Twelve questions, twice, with
DSPy's response cache switched off so every answer is a fresh measurement. The
harness has its own regex for reading `[n]` and reads the agent's extra output
carefully, so the same script ran against an unmodified revision to produce the
baseline. `eval/README.md` has the commands.

| | before | after |
| --- | --- | --- |
| answers with a citation | 0 / 12 | **10 / 12** |
| answers with a source list | 0 / 12 | **10 / 12** |
| answers with a made-up citation | not measurable | **0 / 12** |
| expected document retrieved | 8 / 10 | 8 / 10 |
| gate ALLOW | 10 / 12 | 10 / 12 |
| mean seconds per question | 32.2 | 32.4 |

Three things are worth reading out of that.

**It works.** Before, not one answer in twelve carried a citation, so nothing was
verifiable in the sense the product claims. After, every answer the gate allowed
has both inline citations and a source list.

**The model does follow the format.** That was the assumption the free local
checks could not test, because it depends on the model rather than on the code.
Zero answers out of twelve contained a number that was never issued.

**The control metric did not move.** `retrieval_hit` stayed at 8/10 and the gate
agreed on the same ten questions both times, so the citation columns did not
improve because different documents happened to be retrieved. That is what makes
the comparison worth something rather than two unrelated runs.

The two answers without citations are the two the gate refused, which is correct.
An earlier run of this same evaluation did not hold retrieval constant and moved
6/10 to 7/10 with nothing changed; that is how noisy a twelve-question sample is,
and it is worth knowing before reading anything into the numbers.

### Running it found a bug that no unit test caught

This is the argument for doing the evaluation at all rather than trusting the free
checks.

The first version of the fallback returned `gate.safe_response` whenever an answer
had to be withheld. That reads as correct and is wrong. `ProgramGate` only fills
that field in when it *refuses* an answer. When it decides ALLOW, the field is an
empty string by design.

So the first time the fallback actually fired, the user was shown **nothing at
all** — an empty reply. Every unit test passed, because the tests supplied their
own gate stub with a message in it. It only appeared when the real gate, the real
model and the real fallback path came together.

The fix is a `_withheld_response` helper that supplies its own wording when the
gate has none, plus three regression tests that use an allowing gate with a blank
`safe_response`. In the second run the fallback never fired, so this fix rests on
those tests rather than on the evaluation. The free checks and the paid run find
different kinds of problem, and this is a concrete example.

## Design decisions worth explaining

**One registry per turn, shared by both tools.** The planner can search up to
three times in a turn. If each tool numbered its own results from `[1]`, then
`[1]` in the first observation and `[1]` in the second would be different
passages, and a `[1]` in the answer could not be worked out. The measurement above
shows this is not hypothetical.

**One number per chunk, not per file.** A number per file would be useless: the
808 bulletin chunks are 808 different passages. The source list groups numbers by
file afterwards, so the reader still sees one line per document.

**The check runs after synthesis, because the gate runs before it.** The gate
judges whether the *evidence* could support an answer. It never sees the answer.
"Does every citation here point to something real?" can only be asked about
finished text.

**`format_documents` takes the registry as an optional argument.** A required
parameter would have broken every existing caller for no reason. Keeping it
optional is what makes the old-output regression test mean something.

## What is still wrong

**A valid citation is not a correct one.** The check proves that `[3]` points to a
passage that really was retrieved. It does **not** prove the passage supports the
sentence. A model can attach a real number to a claim the passage does not
support, and nothing here would notice. Catching that needs entailment checking, a
much bigger job. Until then, "verifiable" here means the citation points somewhere
real, not that it is the right place.

**Throwing answers away is expensive.** One mis-formatted number, or no citation
at all, loses the whole response and the user is told the answer cannot be
confirmed. That fits ChatDKU's posture — an unsupported claim about a deadline is
worse than admitting uncertainty — but it is aggressive. `CHATDKU_REQUIRE_CITATIONS=false`
measures the other side. In the run above it fired zero times in ten allowed
answers, so the cost is real in principle but was not paid in this sample.

**Numbers only mean something inside their own turn.** Tool memory carries
observations from earlier turns into later ones, where the numbers no longer point
anywhere. The synthesizer only sees the current turn, so the risk is small, but a
summary quoting an old `[2]` would mislead.

**The corpus is unbalanced, but I did not see it cause a problem.** 95.7% of the
index is the bulletin, yet retrieval still scored 10/10. So this could be a
problem, but it is not one I observed. A 36-chunk FAQ is easy to retrieve; a
bigger set might not be.

**The two tools are never combined.** A cosine similarity and a BM25 score are not
comparable, so the tools are still called separately and merged by the planner
reading them. That is the next improvement, not attempted here.

## A note on AI coding tools

They were used to read the existing code, draft the new modules and tests, write
this document, and diagnose the setup failures. Every change should therefore be
treated as unexplained until someone has walked through it. The design decisions
above and the tests are the list of things to be able to defend and to modify
live; the tests run in seconds and can be broken on purpose to check that each one
is really understood.

The judgement calls were not delegated: which gaps to close, how big each change
should be, keeping `format_documents` backwards compatible, discarding a whole
answer over one bad citation, and everything in the section above.

## Files

```
src/chatdku/citations.py       new — the number registry
src/chatdku/verification.py    new — the four checking functions
tests/                         44 tests, no model or network needed
eval/                          question set and the before/after harness
docs/setup-issues.md           what went wrong during setup
```

Also changed: `agent.py`, `config.py`, `modules/prompts.py`,
`tools/retrieval.py`, `tools/vector_retriever.py`, `tools/keyword_retriever.py`,
`pyproject.toml`, `.env.example`.

Next, in order: add more questions and repeat runs before trusting any number
above, since twelve questions measured once is a direction and not a result;
combine the two retrievers, which already overlap on 3 of 16 passages; and move
from "the citation points somewhere real" to "the citation supports the
sentence".
