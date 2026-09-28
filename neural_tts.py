"""
neural_tts.py — Microsoft Edge neural voices via edge-tts (free, no API
key, needs internet).

Tried FIRST by teacher_natural.speak() and voice_io.speak_interruptible().
Whenever this returns None (not installed, offline, service error), the
caller falls back to the existing offline pyttsx3 voice (tts_utils.py), so
speech never breaks because of this module. After the first real failure
it disables itself for the rest of the run, so an offline session doesn't
wait on a network timeout before every sentence.

Latency: edge-tts takes ~1-2s per request, and the classroom speaks one
sentence at a time — without prefetching that's a pause before every
sentence. prefetch() starts synthesizing upcoming sentences in background
threads while the current one plays; get_audio() then returns instantly
from the cache (an interrupted sentence re-spoken also hits the cache).
"""

import asyncio
import io
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np

# Switch voice here. Samples tried: en-US-AvaNeural, en-US-JennyNeural,
# en-US-AriaNeural, en-GB-SoniaNeural, en-IN-NeerjaNeural.
NEURAL_VOICE = "en-US-AvaNeural"
NEURAL_RATE = "-5%"   # slightly slower than default, for a softer pace

try:
    import edge_tts
    import soundfile as sf
    import sounddevice as sd
    USE_NEURAL_TTS = True
except ImportError:
    USE_NEURAL_TTS = False

_CACHE_MAX = 64
_disabled = False
_lock = threading.Lock()
_cache: "OrderedDict[str, Future]" = OrderedDict()
_pool = ThreadPoolExecutor(max_workers=2)


async def _fetch_mp3(text: str) -> bytes:
    buf = bytearray()
    communicate = edge_tts.Communicate(
        text, NEURAL_VOICE, rate=NEURAL_RATE, connect_timeout=5, receive_timeout=15,
    )
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    return bytes(buf)


def _synthesize(text: str) -> Tuple[np.ndarray, int]:
    # Runs in a worker thread, so asyncio.run gets its own event loop.
    mp3 = asyncio.run(_fetch_mp3(text))
    samples, samplerate = sf.read(io.BytesIO(mp3), dtype="int16")
    return samples, samplerate


def _future(text: str) -> Future:
    with _lock:
        fut = _cache.get(text)
        if fut is None:
            fut = _pool.submit(_synthesize, text)
            _cache[text] = fut
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
        else:
            _cache.move_to_end(text)
        return fut


def _speakable(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def available() -> bool:
    return USE_NEURAL_TTS and not _disabled


def prefetch(texts: List[str]) -> None:
    """Start synthesizing these in the background (in order). No-op when
    the neural voice isn't available."""
    if not available():
        return
    for t in texts:
        if _speakable(t):
            _future(t)


def get_audio(text: str) -> Optional[Tuple[np.ndarray, int]]:
    """(samples, samplerate) for `text`, or None — caller must then fall
    back to pyttsx3."""
    global _disabled
    if not available() or not _speakable(text):
        return None
    try:
        return _future(text).result(timeout=30)
    except Exception as e:
        if isinstance(e, edge_tts.exceptions.NoAudioReceived):
            return None  # this one text only (e.g. odd symbols) — keep neural on
        _disabled = True
        print(f"(neural voice unavailable — using the offline voice for the rest of this run: {e!r})")
        return None


def play(samples: np.ndarray, samplerate: int,
         should_stop: Optional[Callable[[], bool]] = None) -> Tuple[bool, float]:
    """Play audio, blocking. Polls should_stop every 30ms and cuts playback
    the moment it returns True. Returns (interrupted, fraction_played)."""
    duration = max(len(samples) / samplerate, 1e-6)
    start = time.time()
    sd.play(samples, samplerate)
    while sd.get_stream().active:
        if should_stop is not None and should_stop():
            sd.stop()
            return True, min((time.time() - start) / duration, 1.0)
        time.sleep(0.03)
    return False, 1.0
