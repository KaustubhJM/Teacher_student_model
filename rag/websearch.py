"""
rag/websearch.py — free, no-API-key web search.

Framed as a general-purpose tool (not just a "fallback") since the
plan is for the future Explainer agent to call this itself when it
decides a doubt needs current/external information, same as it would
call document retrieval. Whether it's used as a fallback or as a
tool the agent picks on its own is a decision made by whatever calls
this — this function just does the search.
"""

from typing import List


def web_search(query: str, k: int = 3) -> List[str]:
    """Search the web and return up to k short text snippets (title +
    excerpt, not full pages) — enough to ground an explanation, not a
    replacement for real research."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=k))
        return [f"{r.get('title', '')}: {r.get('body', '')}" for r in results]
    except Exception as e:
        print(f"(web search unavailable: {e})")
        return []
