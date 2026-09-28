# Agent

The CLI agent uses the embedded Qdrant collection created by ingestion. It is a local harness: it has no backend API, tracing service, reranker, or external application database.

## Configure the language model

Set an OpenAI-compatible model endpoint in `.env`:

```dotenv
CHATDKU_LLM_MODEL=Qwen/Qwen3-4B
CHATDKU_LLM_BASE_URL=http://localhost:8000/v1
# CHATDKU_LLM_API_KEY=
```

An API key is optional for local servers. Set `CHATDKU_LLM_API_KEY` when the endpoint requires one. This repository does not provide a language-model endpoint, API key, or source documents.

Build an index before starting the agent. The retrieval tools raise a clear error if the configured Qdrant collection does not exist.

## Run it

Start an interactive session:

```bash
uv run agent
```

Ask a single question and exit:

```bash
uv run agent "What is ChatDKU?"
```

By default the agent rewrites a question using conversation and tool memory before retrieval. Use the original wording directly when debugging a query:

```bash
uv run agent --no-rewrite "ChatDKU"
```

## Retrieval and response flow

1. The query rewriter incorporates relevant conversation and tool history.
2. The planner can make up to three retrieval calls.
3. `vector_search` takes `semantic_query` for conceptual similarity search.
4. `keyword_search` takes `keyword_query` for exact names, phrases, and codes.
5. The program gate rejects answers whose retrieved evidence is missing, irrelevant, or too broad.
6. The synthesizer creates a response from the approved evidence.

The planner receives each tool's exact argument schema and is instructed not to invent aliases such as `query`, `keywords`, or `search_term`.

## Inspect context and tool calls

Enable agent diagnostics with:

```bash
CHATDKU_LOG_LEVEL=DEBUG uv run agent
```

For every question, the agent logs the context inputs given to query rewriting, planning, and response synthesis. It also logs the planner trajectory, tool arguments, retrieved observations, gate decision, and final response. The debug context contains user messages and retrieved document text, so use it only in a trusted terminal.

The CLI keeps `httpx`, `httpcore`, and LlamaIndex's own debug output quiet so the useful ChatDKU context logs remain readable.
