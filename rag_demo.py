"""
rag_demo.py — try the rag package against one of your own documents
before we wire it into the Planner/Explainer agents.

Run:
    python rag_demo.py path/to/your/notes.pdf              # real embeddings + FAISS (default)
    python rag_demo.py path/to/your/notes.pdf --tfidf       # TF-IDF, offline, no download

Then type queries (e.g. a topic or section title you'd expect the
document to cover) and see what gets retrieved. Type an empty line to
quit.

Shows two views per query:
  - "Top matches, ranked by relevance" — score-sorted, for judging
    whether retrieval actually found the right material. Each result is
    tagged (direct hit) or (neighbor, inherited score) so a chunk that
    only showed up because it sits next to a real hit — and whose score
    is just 0.9x the real hit's score, not independently earned — doesn't
    get mistaken for a genuine match.
  - "Expanded context block, in document order" — what the
    Planner/Explainer would actually receive as grounding for a section:
    the same chunks, but in reading order instead of score order.
"""

import sys
from rag import (
    build_index_vector,
    retrieve_with_scores_indexed_vector,
    build_index_tfidf,
    retrieve_with_scores_indexed_tfidf,
    web_search,
)
from rag.retrieval import expand_with_neighbors, filter_low_quality


def main():
    args = [a for a in sys.argv[1:] if a != "--tfidf"]
    use_tfidf = "--tfidf" in sys.argv

    if len(args) < 1:
        print("Usage: python rag_demo.py <path to .pdf/.md/.txt/.pptx> [--tfidf]")
        sys.exit(1)

    path = args[0]

    if use_tfidf:
        print("(using TF-IDF backend — offline, literal word matching)")
        build_index = build_index_tfidf
        retrieve_with_scores_indexed = retrieve_with_scores_indexed_tfidf
        weak_threshold, solid_hint = 0.1, ">0.3 solid, 0.1-0.3 weak, <0.1 poor"
    else:
        print("(using vector/FAISS backend — real embeddings, needs a one-time model download)")
        build_index = build_index_vector
        retrieve_with_scores_indexed = retrieve_with_scores_indexed_vector
        weak_threshold, solid_hint = 0.3, ">0.5 solid, 0.3-0.5 weak, <0.3 poor"

    print(f"Indexing {path}...")
    try:
        index = build_index(path)
    except RuntimeError as e:
        print(f"\n[rag_demo error] {e}")
        sys.exit(1)

    if index.is_empty():
        print("No text could be extracted from this document.")
        sys.exit(1)

    print(f"Indexed {len(index.chunks)} chunks.\n")
    print("Type a query (e.g. a topic you'd expect this document to cover).")
    print(f"Scores shown are cosine similarity: {solid_hint}.")
    print("Empty line to quit.\n")

    while True:
        query = input("> ").strip()
        if not query:
            break

        # Direct hits first — this is the real relevance signal.
        initial_hits = retrieve_with_scores_indexed(index, query, k=3)
        direct_positions = {pos for _, _, pos in initial_hits}

        # Then expand to neighbors (document-adjacent chunks, 0.9x
        # inherited score) and drop anything that still looks garbled.
        expanded = expand_with_neighbors(index, initial_hits, window=1, min_score=weak_threshold)
        expanded = filter_low_quality(expanded)

        max_chunks = 7
        if len(expanded) > max_chunks:
            expanded = sorted(expanded, key=lambda h: h[1], reverse=True)[:max_chunks]

        best_score = max((score for _, score, _ in expanded), default=0.0)

        if expanded:
            ranked = sorted(expanded, key=lambda h: h[1], reverse=True)
            print(f"\nTop matches, ranked by relevance (best score: {best_score:.3f}):")
            for i, (chunk, score, position) in enumerate(ranked, 1):
                tag = "direct hit" if position in direct_positions else "neighbor, inherited score"
                snippet = chunk[:180] + ("..." if len(chunk) > 180 else "")
                print(f"  [{i}] score={score:.3f}  position={position}  ({tag})")
                print(f"      {snippet}")

            by_position = sorted(expanded, key=lambda h: h[2])
            print("\nExpanded context block, in document order (what the Explainer would receive):")
            for chunk, score, position in by_position:
                tag = "direct" if position in direct_positions else "neighbor"
                snippet = chunk[:180] + ("..." if len(chunk) > 180 else "")
                print(f"  [pos {position}, {tag}] {snippet}")
        else:
            print("\nNo chunks retrieved for this query.")

        if best_score < weak_threshold:
            print("\n(best match is weak — this is what would trigger a web search fallback)")
            web_results = web_search(query, k=3)
            if web_results:
                for i, r in enumerate(web_results, 1):
                    print(f"  [web {i}] {r[:180]}{'...' if len(r) > 180 else ''}")
            else:
                print("  (no web results either — would fall back to general knowledge)")
        print()


if __name__ == "__main__":
    main()