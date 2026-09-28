"""
rag/retrieval.py — neighbor/section expansion on top of plain top-k
retrieval.

Plain top-k semantic search treats every chunk as an independent, isolated
unit. That's fine when one chunk fully answers a query, but a real section
(e.g. "Data Models in DBMS", which spans several sub-headings in a typical
source document) usually spans multiple chunks — only one or two of which
score highest for any single query. Grabbing just the top-k in that case
returns one relevant chunk plus whatever unrelated chunks happened to
score next best, instead of the rest of the SAME section.

This module adds one extra step: for every chunk that scores well, also
pull in its immediate neighbors (by position in the document — how
chunk_elements lays them out) as likely continuations of the same
material, then dedupe and hand back a document-ordered context block.
This is deliberately NOT the same as raising top_k — raising top_k pulls
in more DIFFERENT topics; expansion pulls in more of the SAME topic around
an already-relevant hit.

A neighbor is only ever a GUESS that it continues the same material — it's
included because it's adjacent, not because it independently matched the
query. min_score gates that guess: a neighbor whose inherited (discounted)
score would fall below min_score is dropped instead of kept, so expansion
still helps with genuine continuations (the next chunk of the same
section) without also pulling in whatever unrelated content happens to
sit next to a hit (e.g. a history/timeline chunk sitting right before a
definition chunk). Direct hits themselves are never filtered by min_score
here — that's the caller's job before expansion, same as before.
"""

from typing import List, Optional, Tuple

from .ingest import is_garbled_text

Hit = Tuple[str, float, int]  # (chunk_text, score, position)


def expand_with_neighbors(
    index, hits: List[Hit], window: int = 1, min_score: Optional[float] = None
) -> List[Hit]:
    """For each (chunk, score, position) hit, also include the chunks at
    position-window..position+window (clipped to the document's bounds).
    A neighbor inherits a discounted version of its parent hit's score
    (0.9x) unless it's ALSO an independent hit, in which case its own
    (higher) score wins.

    If min_score is given, a neighbor is only kept when its inherited
    score is >= min_score — this is what stops an unrelated-but-adjacent
    chunk (good for "continue the same section", bad when the neighbor is
    actually about something else) from being pulled in just because it
    happens to sit next to a real hit. Original hits are always kept
    regardless of min_score, since they already passed whatever threshold
    the caller applied before calling this.

    Returns hits deduped by position and sorted by position — document
    order, not score — so the merged context reads as a coherent passage
    instead of being shuffled by relevance.
    """
    chunks = getattr(index, "chunks", None)
    if not chunks or not hits:
        return hits

    by_position = {}
    for chunk, score, pos in hits:
        if pos not in by_position or score > by_position[pos][0]:
            by_position[pos] = (score, chunk)

    for chunk, score, pos in hits:
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            npos = pos + offset
            if npos < 0 or npos >= len(chunks):
                continue
            neighbor_score = score * 0.9
            if min_score is not None and neighbor_score < min_score:
                continue
            existing = by_position.get(npos)
            if existing is None or neighbor_score > existing[0]:
                by_position[npos] = (neighbor_score, chunks[npos])

    return sorted(
        [(chunk, score, pos) for pos, (score, chunk) in by_position.items()],
        key=lambda h: h[2],
    )


def filter_low_quality(hits: List[Hit]) -> List[Hit]:
    """Defense in depth: drop any hit whose text still looks like garbled
    character-by-character extraction, in case something garbled made it
    into the index despite ingest-time filtering (e.g. an older cached
    index built before this check existed)."""
    return [h for h in hits if not is_garbled_text(h[0])]


def retrieve_with_expansion(
    index,
    retrieve_fn,
    query: str,
    k: int = 5,
    min_score: float = 0.0,
    window: int = 1,
    max_chunks: Optional[int] = None,
) -> List[Hit]:
    """Full pipeline: initial retrieval -> neighbor/section expansion ->
    quality filtering -> optional trim -> final, document-ordered context.

    retrieve_fn must be an *_indexed retrieval function — e.g.
    rag.retrieve_with_scores_indexed — returning (chunk, score, position)
    triples, since expansion needs each hit's position to find neighbors.

    min_score gates BOTH the initial hits and, via expand_with_neighbors,
    any neighbor pulled in around them — an unrelated-but-adjacent chunk
    no longer rides in for free just by being nearby.
    """
    hits = retrieve_fn(index, query, k=k)
    hits = [h for h in hits if h[1] >= min_score]
    if not hits:
        return []

    expanded = expand_with_neighbors(index, hits, window=window, min_score=min_score)
    expanded = filter_low_quality(expanded)

    if max_chunks and len(expanded) > max_chunks:
        # Trim the weakest-scoring entries first, then restore document
        # order for what's left, so the final context still reads coherently.
        expanded = sorted(expanded, key=lambda h: h[1], reverse=True)[:max_chunks]
        expanded = sorted(expanded, key=lambda h: h[2])

    return expanded


def retrieve_text_with_expansion(index, retrieve_fn, query: str, **kwargs) -> List[str]:
    """Convenience wrapper returning just the chunk texts, in document
    order — for call sites that don't need scores/positions themselves."""
    return [c for c, s, p in retrieve_with_expansion(index, retrieve_fn, query, **kwargs)]