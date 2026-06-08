"""Task 10 - Generation with citation."""

from __future__ import annotations

import re

from .task9_retrieval_pipeline import retrieve


# top_k=5 gives enough evidence without overloading a small prompt.
TOP_K = 5
# top_p and temperature are documented for an LLM-backed version; the local
# generator below is deterministic for repeatable tests and demos.
TOP_P = 0.9
TEMPERATURE = 0.3


SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim must include a citation in [Source, Year] format.
If evidence is insufficient, say 'I cannot verify this information'."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Put the best chunk first and the second-best near the end to reduce
    lost-in-the-middle effects. Example: [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2].
    """
    if len(chunks) <= 2:
        return list(chunks)

    reordered = list(chunks[0::2])
    reordered.extend(reversed(chunks[1::2]))
    return reordered


def format_context(chunks: list[dict]) -> str:
    """Format chunks with source labels for citation-aware prompting."""
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {idx}")
        doc_type = metadata.get("type", "unknown")
        chunk_index = metadata.get("chunk_index", "N/A")
        parts.append(
            f"[Document {idx} | Source: {source} | Type: {doc_type} | Chunk: {chunk_index}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citations.

    The local generator summarizes retrieved evidence instead of calling an LLM,
    which keeps the assignment runnable without API keys.
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    answer = _compose_answer(query, reordered)
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


def _compose_answer(query: str, chunks: list[dict]) -> str:
    snippets: list[str] = []
    for chunk in chunks[:3]:
        sentence = _best_sentence(chunk.get("content", ""), query)
        if not sentence:
            continue
        snippets.append(f"{sentence} {_citation(chunk)}")

    if not snippets:
        return "I cannot verify this information"

    intro = f"Dựa trên các tài liệu đã truy xuất cho câu hỏi: \"{query}\":"
    return intro + "\n\n" + "\n\n".join(f"- {snippet}" for snippet in snippets)


def _best_sentence(content: str, query: str) -> str:
    query_terms = set(_tokens(query))
    sentences = re.split(r"(?<=[.!?])\s+|\n+", content)
    best = ""
    best_score = -1
    for sentence in sentences:
        clean = re.sub(r"\s+", " ", sentence).strip()
        if len(clean) < 25:
            continue
        score = len(query_terms & set(_tokens(clean)))
        if score > best_score:
            best = clean
            best_score = score
    if not best:
        best = re.sub(r"\s+", " ", content).strip()[:280]
    return best[:450]


def _citation(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "Unknown source")
    year = metadata.get("year")
    if not year:
        year_match = re.search(r"(20\d{2}|19\d{2})", source)
        year = year_match.group(1) if year_match else "N/A"
    label = source.replace(".md", "").replace(".pdf", "").replace("-", " ")
    return f"[{label}, {year}]"


def _tokens(text: str) -> list[str]:
    import unicodedata

    normalized = unicodedata.normalize("NFD", text.lower())
    no_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.findall(r"\w+", no_accents, flags=re.UNICODE)


if __name__ == "__main__":
    result = generate_with_citation("Hinh phat ma tuy?")
    print(result["answer"])
