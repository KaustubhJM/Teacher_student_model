"""
rag/vector_index.py — real semantic search: sentence-transformer
embeddings + FAISS.

build_index accepts either a local file path or a YouTube URL
(is_youtube_source() tells them apart) — routed to load_document_structured()
or load_youtube_structured() respectively, then both flow through the same
chunk_elements() so nothing downstream (Planner, Explainer, retrieval) needs
to know or care which kind of source it came from.

On-disk caching: a normal file is keyed by its absolute path + mtime, so
editing the source invalidates the cache automatically. A YouTube source
has no mtime to check, so it's keyed by video ID instead and trusted until
the cache folder is deleted manually — a transcript essentially never
changes after upload, so this doesn't need the same invalidation guard.

Changes from the original version (see ingest.py's module docstring for
the full picture): build_index now goes through load_document_structured()
+ chunk_elements() instead of load_document()+chunk_text(), so each chunk
carries metadata (chunk_id, page, element types, position) that
retrieve_with_scores_indexed() and rag/retrieval.py's neighbor expansion
rely on. The cache key includes a format version so old on-disk caches
(built before chunks carried metadata) are simply ignored and rebuilt
rather than causing a KeyError on load.
"""

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .ingest import load_document_structured, chunk_elements
from .youtube_ingest import is_youtube_source, extract_video_id, load_youtube_structured
from .vision_describe import shortfall_count, vision_available_now

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # lazy singleton — the download/load only happens once per process

# Bump this if the chunk/metadata format changes again — old cache folders
# under a stale version are simply orphaned (ignored) and rebuilt, instead
# of crashing when the new code reads an old meta.json missing new fields.
_CACHE_VERSION = "v8-code"


_model_lock = threading.Lock()  # preload thread + main thread must not load it twice


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            import socket
            from sentence_transformers import SentenceTransformer

            previous_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10)
            try:
                try:
                    # Already downloaded -> load from disk only. Without this,
                    # every run first checks huggingface.co online: measured
                    # ~7s on a good connection, 69s on a slow one, vs 0.1-0.2s.
                    _model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
                except Exception:
                    _model = SentenceTransformer(_MODEL_NAME)  # first run: download
            except Exception as e:
                raise RuntimeError(
                    f"Couldn't load/download the embedding model ({_MODEL_NAME}). "
                    "This needs internet access on first run to fetch it from "
                    "Hugging Face (~80MB, cached locally after that — every run "
                    "after the first is instant and offline). Check your "
                    f"internet connection and try again. (underlying error: {e})"
                ) from e
            finally:
                socket.setdefaulttimeout(previous_timeout)
    return _model


def preload_model_in_background() -> None:
    """Start loading the embedding model (~10s) in a daemon thread so it
    overlaps with other startup work instead of stalling the first
    retrieval (assign_context). If it fails, the next real _get_model()
    call simply tries again and raises the usual error."""
    def _load():
        try:
            _get_model()
        except Exception:
            pass
    threading.Thread(target=_load, daemon=True).start()


@dataclass
class VectorIndex:
    """A document's chunks plus a FAISS index over their embeddings.
    metadatas[i] describes chunks[i]: {"chunk_id", "page", "types",
    "position"} — see ingest.chunk_elements()."""
    chunks: List[str] = field(default_factory=list)
    index: Optional[object] = None  # faiss.IndexFlatIP
    metadatas: List[dict] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ---------------------------------------------------------------------------
# On-disk persistence
# ---------------------------------------------------------------------------

def _cache_paths(source_path: str, chunk_size: int, overlap: int, cache_dir: str):
    # Folder = readable prefix + hash, e.g. "yt-0zFfcEr1e9U-6c5b4cd8b1f0" or
    # "unit-1-artificial-intelligence-1-a1b2c3d4e5f6". The hash alone decides
    # identity (same inputs as before); the prefix is only for humans.
    if is_youtube_source(source_path):
        video_id = extract_video_id(source_path)
        key_input = f"youtube::{video_id}::{chunk_size}::{overlap}::{_CACHE_VERSION}"
        prefix = f"yt-{video_id}"
    else:
        key_input = f"{os.path.abspath(source_path)}::{chunk_size}::{overlap}::{_CACHE_VERSION}"
        stem = os.path.splitext(os.path.basename(source_path))[0]
        prefix = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:40].rstrip("-") or "doc"
    key = hashlib.md5(key_input.encode()).hexdigest()[:12]
    folder = os.path.join(cache_dir, f"{prefix}-{key}")
    return folder, os.path.join(folder, "index.faiss"), os.path.join(folder, "meta.json")


# Cache folders this process already (re)built — an INCOMPLETE index is
# retried at most once per run (classroom calls build_index twice at start).
_attempted_this_process: set = set()


def save_index(vindex: VectorIndex, folder: str, index_path: str, meta_path: str,
                source_path: str, chunk_size: int, overlap: int, incomplete: bool = False) -> None:
    import faiss
    os.makedirs(folder, exist_ok=True)
    faiss.write_index(vindex.index, index_path)
    is_yt = is_youtube_source(source_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_path": source_path if is_yt else os.path.abspath(source_path),
            # No mtime for a URL — youtube cache validity is handled by
            # video-ID keying instead, see _cache_is_valid.
            "source_mtime": None if is_yt else os.path.getmtime(source_path),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": vindex.chunks,
            "metadatas": vindex.metadatas,
            # True if vision was temporarily unavailable while indexing —
            # see _cache_is_valid (old caches without this key = complete).
            "incomplete": incomplete,
        }, f)


def load_index(index_path: str, meta_path: str) -> VectorIndex:
    import faiss
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return VectorIndex(chunks=meta["chunks"], index=index, metadatas=meta.get("metadatas", []))


def _cache_is_valid(meta_path: str, source_path: str, chunk_size: int, overlap: int) -> bool:
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if is_youtube_source(source_path):
        # Video-ID-based cache key (see _cache_paths) already guarantees
        # this folder belongs to this exact video — no mtime to compare.
        return meta.get("chunk_size") == chunk_size and meta.get("overlap") == overlap

    valid = (
        meta.get("source_mtime") == os.path.getmtime(source_path)
        and meta.get("chunk_size") == chunk_size
        and meta.get("overlap") == overlap
    )
    if valid and meta.get("incomplete"):
        # Built while vision was temporarily unavailable. Rebuild once vision
        # works again (OCR + successful vision results are cached, so only
        # the missing items call the API); while it's still unavailable,
        # reuse the incomplete cache rather than re-indexing every run.
        folder = os.path.dirname(meta_path)
        if folder not in _attempted_this_process and vision_available_now():
            print("(cached index is INCOMPLETE and vision is available again — re-indexing to fill it in)")
            return False
    return valid


# ---------------------------------------------------------------------------
# Build / retrieve
# ---------------------------------------------------------------------------

def build_index(path: str, chunk_size: int = 200, overlap: int = 40,
                 cache_dir: Optional[str] = ".rag_cache") -> VectorIndex:
    """Load a source (a local file OR a YouTube URL — is_youtube_source()
    decides which loader runs), chunk it (structure-aware, metadata-
    carrying), embed every chunk, and build a FAISS index — or load a
    cached one from disk if this exact source was already indexed before."""
    folder = index_path = meta_path = None
    if cache_dir:
        folder, index_path, meta_path = _cache_paths(path, chunk_size, overlap, cache_dir)
        if _cache_is_valid(meta_path, path, chunk_size, overlap):
            print(f"(loaded cached vector index from {folder})")
            return load_index(index_path, meta_path)

    import faiss

    shortfall_before = shortfall_count()
    if is_youtube_source(path):
        elements = load_youtube_structured(path)
    else:
        elements = load_document_structured(path)
    incomplete = shortfall_count() > shortfall_before
    chunk_dicts = chunk_elements(elements, chunk_size=chunk_size, overlap=overlap)
    chunks = [c["text"] for c in chunk_dicts]
    vindex = VectorIndex(chunks=chunks, metadatas=chunk_dicts)
    if not chunks:
        return vindex

    model = _get_model()
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Inner product on L2-normalized vectors == cosine similarity —
    # this is the standard way to do cosine search with FAISS.
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    vindex.index = faiss_index

    if cache_dir:
        _attempted_this_process.add(folder)
        save_index(vindex, folder, index_path, meta_path, path, chunk_size, overlap, incomplete=incomplete)
        print(f"(cached vector index to {folder})")
        if incomplete:
            n = shortfall_count() - shortfall_before
            print(f"(note: {n} figure/page(s) couldn't use vision right now — the index is marked "
                  "INCOMPLETE and will be rebuilt automatically on a later run once vision is available)")

    return vindex


def retrieve_with_scores_indexed(vindex: VectorIndex, query: str, k: int = 3) -> List[Tuple[str, float, int]]:
    """Like retrieve_with_scores, but also returns each chunk's position
    in vindex.chunks (== its document order), needed by
    rag.retrieval.expand_with_neighbors for neighbor/section expansion."""
    if vindex.is_empty() or vindex.index is None:
        return []
    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    query_vec = np.asarray(query_vec, dtype="float32")

    k = min(k, len(vindex.chunks))
    scores, indices = vindex.index.search(query_vec, k)
    return [
        (vindex.chunks[i], float(scores[0][pos]), int(i))
        for pos, i in enumerate(indices[0])
        if i != -1
    ]


def retrieve_with_scores(vindex: VectorIndex, query: str, k: int = 3) -> List[Tuple[str, float]]:
    """Return up to k (chunk, score) pairs, best first, no filtering.
    Scores are cosine similarity: with real embeddings, >0.5 tends to mean
    a solid match, 0.3-0.5 weak/related, below 0.3 usually unrelated —
    calibrate against your own documents with rag_demo.py."""
    return [(c, s) for c, s, _ in retrieve_with_scores_indexed(vindex, query, k)]


def retrieve(vindex: VectorIndex, query: str, k: int = 3, min_score: float = 0.3) -> List[str]:
    """Return up to k chunks scoring at or above min_score."""
    return [chunk for chunk, score in retrieve_with_scores(vindex, query, k) if score >= min_score]