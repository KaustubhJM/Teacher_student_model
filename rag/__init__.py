"""
rag/ — document ingestion + retrieval + web search, as a standalone
package with no dependency on any of the teaching/voice files.

Public API additions in this version (all additive — nothing below removes
or changes the signature of anything that existed before):

    from rag import load_document_structured, chunk_elements, is_garbled_text
    from rag import retrieve_with_scores_indexed, expand_with_neighbors
    from rag import retrieve_with_expansion, retrieve_text_with_expansion
    from rag import load_youtube_structured, is_youtube_source, get_video_title

See ingest.py, retrieval.py, and youtube_ingest.py's module docstrings for
what each does. build_index (both backends) now transparently accepts a
YouTube URL in place of a file path — see vector_index.py/document_index.py.
"""

from .ingest import (
    load_document, chunk_text,
    load_document_structured, chunk_elements, is_garbled_text,
)
from .youtube_ingest import (
    load_youtube, load_youtube_structured,
    is_youtube_source, extract_video_id, get_video_title,
)
from .vector_index import (
    VectorIndex, build_index as build_index_vector,
    retrieve as retrieve_vector, retrieve_with_scores as retrieve_with_scores_vector,
    retrieve_with_scores_indexed as retrieve_with_scores_indexed_vector,
)
from .document_index import (
    DocumentIndex, build_index as build_index_tfidf,
    retrieve as retrieve_tfidf, retrieve_with_scores as retrieve_with_scores_tfidf,
    retrieve_with_scores_indexed as retrieve_with_scores_indexed_tfidf,
)
from .websearch import web_search
from .retrieval import (
    expand_with_neighbors, filter_low_quality,
    retrieve_with_expansion, retrieve_text_with_expansion,
)

# Default backend: real embeddings + FAISS. Use the _tfidf names directly
# if you specifically want the no-download offline version.
build_index = build_index_vector
retrieve = retrieve_vector
retrieve_with_scores = retrieve_with_scores_vector
retrieve_with_scores_indexed = retrieve_with_scores_indexed_vector

__all__ = [
    "load_document",
    "chunk_text",
    "load_document_structured",
    "chunk_elements",
    "is_garbled_text",
    "load_youtube",
    "load_youtube_structured",
    "is_youtube_source",
    "extract_video_id",
    "get_video_title",
    "VectorIndex",
    "DocumentIndex",
    "build_index",
    "retrieve",
    "retrieve_with_scores",
    "retrieve_with_scores_indexed",
    "build_index_vector",
    "retrieve_vector",
    "retrieve_with_scores_vector",
    "retrieve_with_scores_indexed_vector",
    "build_index_tfidf",
    "retrieve_tfidf",
    "retrieve_with_scores_tfidf",
    "retrieve_with_scores_indexed_tfidf",
    "web_search",
    "expand_with_neighbors",
    "filter_low_quality",
    "retrieve_with_expansion",
    "retrieve_text_with_expansion",
]