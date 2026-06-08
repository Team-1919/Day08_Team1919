# Coding Agent Rules

These rules apply to all code, tests, scripts, and documentation under this
`group_project/` directory.

## Unified RAG Stack

Use this stack consistently when implementing or modifying the group project:

- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Vector store: local FAISS via `faiss-cpu`
- FAISS index path: `data/faiss_index/`
- BM25 library: `rank-bm25`
- BM25 corpus path: `data/bm25_corpus.pkl`
- LLM: OpenAI `gpt-4o-mini`
- LLM fallback: deterministic mock response when `OPENAI_API_KEY` is missing
- API key source: `.env`, read from `OPENAI_API_KEY`

## Implementation Rules

- Treat all data paths as relative to `group_project/` unless a caller provides
  an explicit path.
- Do not introduce another embedding model, vector database, BM25 backend, or LLM
  model unless the user explicitly asks for a stack change.
- Load environment variables with `python-dotenv` before creating an OpenAI
  client.
- Never hard-code API keys. Read `OPENAI_API_KEY` from `.env` or the process
  environment.
- If `OPENAI_API_KEY` is empty or unavailable, generation code must still run by
  returning a mock answer that clearly indicates mock mode.
