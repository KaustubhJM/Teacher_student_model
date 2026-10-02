"""
rag/reranker.py — cross-encoder relevance scores for the doubt tool
(tools.search_document).

A cross-encoder reads the question and a chunk TOGETHER, so it separates
useful from useless chunks far better than embedding cosine similarity
(measured on the golden sets: median score +0.54 useful vs -5.40 useless;
cosine 0.55 vs 0.44 with heavy overlap).

search_document keeps only chunks scoring >= RERANK_MIN_SCORE. -8 is a
conservative cut: on both golden sets (DBMS, TC) it lost no recall at all
(incl. multi-hop) while P@k rose 0.65 -> 0.78 / 0.50 -> 0.75 and words
sent fell 16-30%. A stricter -2.1 cut lost multi-hop recall (TC 0.97 ->
0.73). See CLAUDE.md "Retriever eval".

Optional by design: if the model can't be loaded (offline before its first
download, broken install), rerank_scores() returns None and the caller
keeps today's behaviour (all chunks, embedding best-first).
"""

import threading
from typing import List, Optional

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~90 MB, CPU ~0.5 s per 6 chunks
RERANK_MIN_SCORE = -8.0

_model = None
_unavailable = False
_lock = threading.Lock()  # preload thread + doubt call must not load it twice


def _get_reranker():
    global _model, _unavailable
    with _lock:
        if _model is None and not _unavailable:
            import socket
            from sentence_transformers import CrossEncoder

            previous_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10)
            try:
                try:
                    # Already downloaded -> disk only (skips the online HF check,
                    # same reason as vector_index._get_model).
                    _model = CrossEncoder(RERANKER_MODEL, max_length=512, local_files_only=True)
                except Exception:
                    _model = CrossEncoder(RERANKER_MODEL, max_length=512)  # first run: download
            except Exception as e:
                _unavailable = True
                print(f"(reranker unavailable — doubt search keeps all chunks unfiltered: {e})")
            finally:
                socket.setdefaulttimeout(previous_timeout)
    return _model


def preload_reranker_in_background() -> None:
    """Load the reranker in a daemon thread at startup so the first doubt
    doesn't wait for it (~6 s from disk). It first waits for the embedding
    model (vector_index._get_model holds a lock), so the two loads don't
    compete for CPU during the tuned startup — embeddings are needed first
    (assign_context), the reranker only at the first doubt."""
    def _load():
        try:
            from .vector_index import _get_model
            _get_model()
        except Exception:
            pass
        _get_reranker()
    threading.Thread(target=_load, daemon=True).start()


def rerank_scores(query: str, chunks: List[str]) -> Optional[List[float]]:
    """Relevance score per chunk (higher = more relevant), or None when the
    reranker isn't available — the caller then keeps all chunks."""
    if not chunks:
        return None
    model = _get_reranker()
    if model is None:
        return None
    try:
        return [float(s) for s in model.predict([(query, c) for c in chunks])]
    except Exception as e:
        print(f"(reranker failed — keeping all chunks: {e})")
        return None
