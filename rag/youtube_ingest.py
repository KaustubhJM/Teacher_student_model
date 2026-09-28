"""
rag/youtube_ingest.py — turn a YouTube video into the same ordered,
structured elements ingest.py produces for other sources.

Fetches the video's transcript (manual captions preferred over
auto-generated, then falls back to translating whatever exists) via
youtube-transcript-api — free, no API key, no audio download. Groups the
raw caption lines (usually just a few words/seconds each) into readable
text segments before returning them, since chunk_elements() (see
ingest.py) already handles packing structural units into chunks —
feeding it one element per 3-second caption line would fragment context
far too aggressively.

Returned elements match ingest.py's shape exactly:
    {"type": "text", "content": str, "page": None, "top": float}
"top" holds the segment's start timestamp in seconds. Note: chunk_elements()
only reads "page" and "type" off each element today, not "top" — so this
timestamp currently does NOT survive into the final chunk metadata. It's
kept here so that's a future addition (threading timestamps through for
"jump to this moment in the video" citations) rather than lost data.

Two entry points, mirroring ingest.py:
    load_youtube(url_or_id) -> str                     (flat text)
    load_youtube_structured(url_or_id) -> List[dict]    (ordered elements)

is_youtube_source(path_or_url) lets callers (build_index in both
vector_index.py and document_index.py) tell a YouTube URL apart from a
normal file path, so document_path can be either one transparently.

Note on youtube-transcript-api versions: v1.0+ replaced the old static
YouTubeTranscriptApi.list_transcripts()/get_transcript() methods with an
instance-based API (YouTubeTranscriptApi().list()/.fetch()), and changed
fetched entries from plain dicts to FetchedTranscriptSnippet objects
(access via .text/.start/.duration, or .to_raw_data() for the old dict
shape). _fetch_transcript() below tries the new API first and falls back
to the old one, so this works whichever version is installed.

Run standalone:
    python youtube_ingest.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
"""

import re
import sys
from typing import List, Optional

# Scheme optional; any subdomain (www., m., music.) — e.g. a link copied
# from a phone is m.youtube.com, and a pasted "youtube.com/watch?v=..."
# often has no https://.
_YOUTUBE_URL_PATTERN = re.compile(
    r"^(https?://)?([a-z0-9-]+\.)?(youtube\.com|youtu\.be|youtube-nocookie\.com)/", re.IGNORECASE
)

# Non-speech caption artifacts auto-captions commonly insert inline —
# stripped so they don't get treated as lecture content downstream (by
# the Planner's outline generation or the Explainer's grounding).
_NOISE_PATTERN = re.compile(r"\[(music|applause|laughter|inaudible|silence)\]", re.IGNORECASE)
# Leading speaker-change marker some auto-caption tracks use (">> Name:"
# or bare ">>") — stripped from the start of a line only, so a genuine
# ">" appearing mid-sentence (rare, but possible in spoken math/code) is
# left alone.
_SPEAKER_MARKER_PATTERN = re.compile(r"^>>\s*")


def is_youtube_source(path_or_url: str) -> bool:
    """True if this looks like a YouTube URL rather than a local file
    path. Matches youtube.com / youtu.be links (any subdomain, scheme
    optional) —
    NOT a bare 11-char video ID, since a bare ID is indistinguishable
    from an arbitrary local filename stem. Callers that want to accept a
    bare ID directly should do so explicitly, not through this check."""
    return bool(_YOUTUBE_URL_PATTERN.match(path_or_url.strip()))


def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL (watch, youtu.be, embed, shorts) or a
    bare 11-character video ID and returns just the ID."""
    url_or_id = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    match = re.search(r"(?:v=|/videos/|embed/|shorts/|live/|youtu\.be/)([A-Za-z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)

    raise ValueError(f"Couldn't extract a YouTube video ID from: {url_or_id!r}")


def _fetch_title(video_id: str) -> Optional[str]:
    """Best-effort video title via YouTube's oEmbed endpoint — no API key
    needed. Returns None on failure; title is a nice-to-have default
    lecture topic, not required for transcript extraction."""
    try:
        import urllib.request
        import json as _json
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return data.get("title")
    except Exception:
        return None


def _fetch_via_new_api(video_id: str, languages: List[str]) -> List[dict]:
    """youtube-transcript-api >= 1.0: instance-based, .list()/.fetch(),
    entries are FetchedTranscriptSnippet objects."""
    from youtube_transcript_api import YouTubeTranscriptApi

    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)

    try:
        transcript = transcript_list.find_manually_created_transcript(languages)
    except Exception:
        try:
            transcript = transcript_list.find_generated_transcript(languages)
        except Exception:
            available = next(iter(transcript_list), None)
            if available is None:
                raise RuntimeError(f"No transcript at all available for video {video_id}.")
            transcript = available.translate("en")

    fetched = transcript.fetch()
    # .to_raw_data() gives back the old {"text", "start", "duration"} dict
    # shape regardless of snippet object internals — keeps everything
    # below this point version-agnostic.
    return fetched.to_raw_data()


def _fetch_via_old_api(video_id: str, languages: List[str]) -> List[dict]:
    """youtube-transcript-api < 1.0: static list_transcripts()/
    get_transcript(), entries already plain dicts."""
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    try:
        transcript = transcript_list.find_manually_created_transcript(languages)
    except Exception:
        try:
            transcript = transcript_list.find_generated_transcript(languages)
        except Exception:
            available = next(iter(transcript_list), None)
            if available is None:
                raise RuntimeError(f"No transcript at all available for video {video_id}.")
            transcript = available.translate("en")

    return transcript.fetch()


def _fetch_transcript(video_id: str, languages: Optional[List[str]] = None) -> List[dict]:
    """Returns raw caption entries: [{"text", "start", "duration"}, ...].
    Prefers a manually-created transcript over auto-generated (more
    accurate); falls back to auto-generated, then to machine-translating
    whatever transcript exists if nothing in `languages` is available at
    all. Tries the new (>=1.0) API first, falls back to the old one —
    see module docstring."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise RuntimeError(
            "youtube-transcript-api is required for YouTube ingestion. "
            "Install with: pip install youtube-transcript-api"
        )

    languages = languages or ["en", "en-US", "en-GB"]
    uses_new_api = hasattr(YouTubeTranscriptApi, "list") and not hasattr(YouTubeTranscriptApi, "list_transcripts")

    try:
        if uses_new_api:
            return _fetch_via_new_api(video_id, languages)
        return _fetch_via_old_api(video_id, languages)
    except Exception as e:
        raise RuntimeError(
            f"Couldn't fetch a transcript for video {video_id}. It may have "
            "captions disabled, be private/unavailable, or not exist. "
            f"(underlying error: {e})"
        ) from e


def _clean_caption_text(text: str) -> str:
    """Strips non-speech caption artifacts (music/applause/silence tags,
    leading speaker-change markers) and collapses whitespace. Returns ''
    if nothing meaningful is left (e.g. a caption line that was only
    "[Music]") so the caller can skip it entirely."""
    text = _NOISE_PATTERN.sub("", text)
    text = _SPEAKER_MARKER_PATTERN.sub("", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _group_into_segments(entries: List[dict], target_words: int = 150) -> List[dict]:
    """Groups raw caption lines into larger, readable segments of roughly
    `target_words` words each — mirrors ingest.py's philosophy of packing
    natural units instead of leaving fragments too small to carry meaning
    on their own. Each segment keeps the START timestamp of its first
    (non-empty, cleaned) caption line."""
    if not entries:
        return []

    segments = []
    current_texts, current_start, current_words = [], None, 0

    for entry in entries:
        text = _clean_caption_text(entry["text"])
        if not text:
            continue
        if current_words >= target_words and current_texts:
            segments.append({"text": " ".join(current_texts), "start": current_start})
            current_texts, current_words = [], 0
        if not current_texts:
            current_start = entry["start"]
        current_texts.append(text)
        current_words += len(text.split())

    if current_texts:
        segments.append({"text": " ".join(current_texts), "start": current_start})

    return segments


def load_youtube_structured(url_or_id: str, languages: Optional[List[str]] = None) -> List[dict]:
    """Fetch a YouTube video's transcript and return it as ordered,
    structured elements — same shape ingest.py's other loaders produce,
    so it flows straight into chunk_elements()/build_index() with no
    special-casing needed downstream."""
    video_id = extract_video_id(url_or_id)
    entries = _fetch_transcript(video_id, languages=languages)
    segments = _group_into_segments(entries)

    return [
        {"type": "text", "content": seg["text"], "page": None, "top": seg["start"]}
        for seg in segments
    ]


def load_youtube(url_or_id: str, languages: Optional[List[str]] = None) -> str:
    """Back-compat convenience: flat transcript text, no timestamps —
    mirrors ingest.py's load_document()."""
    elements = load_youtube_structured(url_or_id, languages=languages)
    return "\n\n".join(e["content"] for e in elements)


def get_video_title(url_or_id: str) -> Optional[str]:
    """Best-effort video title, for deriving a default lecture topic the
    same way plan_demo.py derives one from a filename."""
    return _fetch_title(extract_video_id(url_or_id))


# ---------------------------------------------------------------------------
# Standalone check
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python youtube_ingest.py <youtube url or video id>")
        sys.exit(1)

    url_or_id = sys.argv[1]
    title = get_video_title(url_or_id)
    if title:
        print(f"Title: {title}")

    elements = load_youtube_structured(url_or_id)
    print(f"\nExtracted {len(elements)} segments.\n")
    for i, el in enumerate(elements[:5], 1):
        mins, secs = divmod(int(el["top"]), 60)
        preview = el["content"][:150] + ("..." if len(el["content"]) > 150 else "")
        print(f"[{i}] ({mins:02d}:{secs:02d}) {preview}")
    if len(elements) > 5:
        print(f"... and {len(elements) - 5} more segments")


if __name__ == "__main__":
    main()