"""
rag/document_index.py — TF-IDF retrieval backend (offline, no model
download). Mirrors vector_index.py's metadata/position support so neighbor
expansion works identically regardless of which backend is active.

build_index accepts either a local file path or a YouTube URL —
is_youtube_source() decides which loader runs — same as vector_index.py.
No on-disk caching exists for this backend (TF-IDF fit is cheap), so
there's no mtime/cache-key concern to handle for YouTube sources here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import load_document_structured, chunk_elements
from .youtube_ingest import is_youtube_source, load_youtube_structured


@dataclass
class DocumentIndex:
    """A document's chunks plus their TF-IDF representation. metadatas[i]
    describes chunks[i] — see ingest.chunk_elements()."""
    chunks: List[str] = field(default_factory=list)
    vectorizer: Optional[TfidfVectorizer] = None
    matrix = None  # sparse TF-IDF matrix, one row per chunk
    metadatas: List[dict] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


def build_index(path: str, chunk_size: int = 200, overlap: int = 40) -> DocumentIndex:
    """Load a source (a local file OR a YouTube URL) and build a
    searchable TF-IDF index over its structure-aware, metadata-carrying
    chunks."""
    if is_youtube_source(path):
        elements = load_youtube_structured(path)
    else:
        elements = load_document_structured(path)
    chunk_dicts = chunk_elements(elements, chunk_size=chunk_size, overlap=overlap)
    chunks = [c["text"] for c in chunk_dicts]
    index = DocumentIndex(chunks=chunks, metadatas=chunk_dicts)
    if not chunks:
        return index
    index.vectorizer = TfidfVectorizer(stop_words="english")
    index.matrix = index.vectorizer.fit_transform(chunks)
    return index


def retrieve_with_scores_indexed(index: DocumentIndex, query: str, k: int = 3) -> List[Tuple[str, float, int]]:
    """Like retrieve_with_scores, but also returns each chunk's position
    in index.chunks, needed by rag.retrieval.expand_with_neighbors."""
    if index.is_empty() or index.vectorizer is None:
        return []
    query_vec = index.vectorizer.transform([query])
    scores = cosine_similarity(query_vec, index.matrix)[0]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(index.chunks[i], float(scores[i]), i) for i in ranked[:k]]


def retrieve_with_scores(index: DocumentIndex, query: str, k: int = 3) -> List[Tuple[str, float]]:
    """Like retrieve(), but returns (chunk, score) pairs with NO score
    filtering — useful for seeing actual match quality."""
    return [(c, s) for c, s, _ in retrieve_with_scores_indexed(index, query, k)]


def retrieve(index: DocumentIndex, query: str, k: int = 3, min_score: float = 0.05) -> List[str]:
    """Return up to k chunks most relevant to `query`, dropping anything
    below min_score."""
    return [c for c, s in retrieve_with_scores(index, query, k) if s >= min_score]