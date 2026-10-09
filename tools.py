from typing import Optional

from langchain_core.tools import tool

from rag import (
    retrieve_with_scores_indexed, expand_with_neighbors,
    web_search as _web_search, DocumentIndex,
)
from rag.reranker import rerank_scores, RERANK_MIN_SCORE

_active_index: Optional[DocumentIndex] = None


def set_active_index(index: Optional[DocumentIndex]) -> None:
    global _active_index
    _active_index = index


@tool
def search_document(query: str) -> str:
    """Search the student's uploaded course document for information
    relevant to the query. Use this when their question might be
    answered by material in the source document — especially content
    outside what you were already given for the current section."""
    if _active_index is None or _active_index.is_empty():
        return "No document is loaded for this session."
    hits = retrieve_with_scores_indexed(_active_index, query, k=3)
    if not hits:
        return "No relevant content found in the document for this query."
    # Pull in each hit's immediate neighbors too — a multi-chunk section
    # (e.g. "Data Models in DBMS") shouldn't get cut off just because its
    # continuation scored slightly lower than an unrelated chunk elsewhere.
    # min_score gates neighbors so an unrelated-but-adjacent chunk (e.g. a
    # history/timeline chunk sitting next to a definition) isn't pulled in
    # just because it's nearby.
    expanded = expand_with_neighbors(_active_index, hits, window=1, min_score=0.3)
    # Best-first: expand_with_neighbors returns page order (right for teaching
    # a lecture section in flow — planner.assign_context keeps it), but for a
    # doubt the most relevant chunk should come first. Measured on the DBMS
    # golden set (same chunks): ranking precision 0.68 -> 0.90, useful chunk
    # first in 17/18 questions instead of 5/18.
    expanded = sorted(expanded, key=lambda h: (-h[1], h[2]))
    results = [chunk for chunk, score, position in expanded]
    # Reranker filter (rag/reranker.py): drop chunks the cross-encoder scores
    # as clearly unrelated and send the rest by reranker score. Same chunk
    # set otherwise; if the reranker isn't available, nothing changes.
    scores = rerank_scores(query, results)
    if scores is not None:
        kept = sorted((i for i, s in enumerate(scores) if s >= RERANK_MIN_SCORE),
                      key=lambda i: (-scores[i], i))
        if not kept:
            return "No relevant content found in the document for this query."
        results = [results[i] for i in kept]
    return "\n\n---\n\n".join(results)


@tool
def search_web(query: str) -> str:
    """Search the web for current, external, or additional information
    the course document doesn't cover. Use this for things outside the
    document's scope, or anything that benefits from up-to-date
    information."""
    results = _web_search(query, k=3)
    if not results:
        return "No web results found."
    return "\n\n---\n\n".join(results)