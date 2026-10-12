# Setup issues encountered

Running notes from setting up this repository on a clean machine (macOS 15.7.4,
Apple Silicon, Python 3.13.5, `uv` 0.12.21). The candidate task asks for
important setup problems to be documented, so this file records what actually
went wrong, why, and what fixed it — including the issues that were my own
environment's fault rather than the repository's.

## 1. `git clone` fails with an HTTP/2 framing error

**Symptom**

```
$ git clone https://github.com/Edge-Intelligence-Lab/ChatDKU-Public.git
Cloning into 'ChatDKU-Public'...
fatal: unable to access '...': Error in the HTTP2 framing layer
```

The repository itself is fine — `git ls-remote` against the same URL succeeds.

**Cause.** Git negotiates HTTP/2 with GitHub, and the connection is unstable on
this network. `ls-remote` only exchanges a few packets, so it usually survives;
a full clone transfers long enough to hit the failure.

**Fix.** Pin the transport to HTTP/1.1 for this clone:

```bash
git -c http.version=HTTP/1.1 clone https://github.com/Edge-Intelligence-Lab/ChatDKU-Public.git
```

To make it permanent for this machine, `git config --global http.version HTTP/1.1`.

## 2. HuggingFace hub stalls on network checks even though the model is cached

**Symptom.** The embedding health check reaches the model, then hangs:

```
'timed out' thrown while requesting HEAD https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/./config_sentence_transformers.json
WARNING huggingface_hub.utils._http: Retrying in 1s [Retry 1/5].
```

**Cause.** `huggingface_hub` verifies each file against the Hub before using the
local copy. On a slow or filtered connection every check costs five retries with
backoff. The model files were already present in `data/cache/models/`, so the
checks were pure overhead — and on a completely offline machine this would block
setup indefinitely.

**Fix.** Tell the hub not to phone home:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uv run chatdku-ingest
```

This is worth knowing for any deployment that pre-seeds the model cache: the
first run in an air-gapped environment will otherwise appear to hang.

## 3. A `uv` virtual environment is not relocatable

**Symptom.** Copying an existing `.venv` into the project directory and running
the CLI produces nonsense: the shell tries to execute Python source as shell
script, starting with ``import: command not found``.

**Cause.** Two things inside a `uv` venv hardcode the absolute path it was
created at:

- the editable install is a `.pth` file containing the old `src/` directory, and
- every console script (`agent`, `chatdku-ingest`, and 31 others) has a shebang
  pointing at the old interpreter.

**Fix.** Repoint both:

```bash
# 1. the editable install
printf '%s\n' "$PWD/src" > .venv/lib/python3.13/site-packages/_editable_impl_chatdku_public.pth

# 2. the shebangs (the `!` matters)
for f in .venv/bin/*; do
  head -1 "$f" | grep -q '^#!/old/path' && sed -i '' "1s|.*|#!$PWD/.venv/bin/python|" "$f"
done
```

**Honest alternative.** Doing this by hand is only worth it to avoid re-downloading
a 1.2 GB environment. On an unrestricted machine the correct move is simply
`uv sync`, which rebuilds the venv properly. This workaround is an artifact of a
sandboxed shell where `~/.cache/uv` was not writable (see 4).

## 4. `uv` cannot initialise its cache in a sandboxed shell

**Symptom**

```
error: Failed to initialize cache at `/Users/<user>/.cache/uv`
  cause: failed to open file `.../sdists-v9/.git`: Operation not permitted
```

**Fix.** Point the cache somewhere writable:

```bash
export UV_CACHE_DIR=/path/to/writable/dir
uv run --no-sync agent "question"
```

Not a repository problem, but it is the kind of thing that makes a first-time
setup look broken when it is not.

## 5. Ingestion refuses to overwrite an existing collection

**Symptom**

```
FileExistsError: An index already exists. Pass replace=True to rebuild 'chatdku'.
```

**Cause.** Deliberate. `chatdku-ingest` protects the existing index so a stray
run cannot silently destroy a large, slow-to-rebuild collection.

**Fix.** Rebuild explicitly after changing source documents, chunk settings, or
the embedding model:

```bash
uv run chatdku-ingest --replace
```

## 6. `Payload indexes have no effect in the local Qdrant.`

**Cause.** Expected, and already documented in `docs/ingestion.md`. Payload
indexes exist for filtering large server-hosted collections; the embedded
on-disk Qdrant used here does not need them.

**Fix.** None. It is a warning, not an error. Recorded here because it is the
first thing that looks alarming in the ingestion log.

## 7. `.docx` ingestion fails: `docx2txt` is an undeclared dependency

**Symptom**

```
$ uv run chatdku-ingest
INFO chatdku.ingestion.loader: Located 2 supported source file(s)
Traceback (most recent call last):
  ...
ImportError: docx2txt is required to read Microsoft Word files: `pip install docx2txt`
```

**Cause.** A real gap in the repository. `chatdku.ingestion.loader.SUPPORTED_SUFFIXES`
and `docs/ingestion.md` both advertise `.docx` support, but
`llama-index-readers-file` only brings in `docx2txt` as an optional extra, and
`pyproject.toml` does not declare it. A clean checkout therefore cannot ingest a
Word document despite claiming to support one.

**Fix.** Declared it in `pyproject.toml` as a runtime dependency. This is an
undeclared-dependency fix, not a new capability, so it stays within the scope of
"make the advertised feature work":

```toml
dependencies = [
    "docx2txt>=0.9",
    ...
]
```

The PDF reader path was already fine — `pypdf` ships with
`llama-index-readers-file`, which is why the failure appeared on the FAQ rather
than the bulletin. For the record, of the formats listed in `SUPPORTED_SUFFIXES`,
`html2text` is also absent, so `.html` support is in the same position.

## 8. The candidate document set is very unbalanced

**Observation.** The two supplied documents differ by more than an order of
magnitude:

| Document | Extracted text | Indexed chunks | Share of index |
| --- | --- | --- | --- |
| `Advising FAQ (12-19-24 Update).docx` | ~48,000 characters | 36 | 4.3% |
| `ug_bulletin_2025-2026.pdf` | ~994,000 characters, 435 pages | 808 | 95.7% |
| **Total** | | **844** | |

(Measured with `uv run python eval/retrieval_only.py`. Note that
`manifest.json` reports `document_count: 436` — that is the count of *loaded
records before chunking*, because the PDF reader emits one record per page. The
844 figure is the number of embedded chunks actually in the Qdrant collection.)

**Why it matters.** With `CHATDKU_RETRIEVAL_TOP_K=8` over 844 chunks, the
bulletin occupies about 96% of the index. A question whose answer lives in the
FAQ competes against a much larger document, and an answer that depends on a
specific bulletin section has only eight slots to find it.

**What was actually observed, rather than assumed.** The retrieval-only
measurement (`eval/retrieval_only.py`, no model calls) returns the expected
document in 10 of 10 answerable questions, for the vector tool, the keyword tool,
and their union — at `top_k=8` *and* at `top_k=20`. So on this question set the
imbalance did not cause a single miss, and it would be wrong to present it as a
demonstrated failure. It remains a structural risk: the FAQ is only 36 chunks, so
a larger or more specialised document set could easily behave differently.

The measurement is recorded here rather than a conclusion, because "the corpus is
unbalanced" and "the corpus is unbalanced and it is hurting retrieval" are
different claims and only the first one is established.

**Also worth noting:** the PDF has a real text layer, so no OCR was needed. That
was not guaranteed — a scanned bulletin would have required OCR before ingestion
could produce anything searchable, and `docs/ingestion.md` covers that case.
