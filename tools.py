from typing import Optional

from langchain_core.tools import tool

from rag import (
    retrieve_with_scores_indexed, expand_with_neighbors,
    web_search as _web_search, DocumentIndex,
)

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
    results = [chunk for chunk, score, position in expanded]
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