"""
voice_io.py — standalone voice input prototype: barge-in detection + STT.

This is deliberately NOT wired into teacher_natural.py yet. Get this
file working on your own machine first (it needs a real microphone,
which this build sandbox doesn't have), THEN we plug listen_for_barge_in()
and speak_interruptible() into check_in / teach_section in place of
input() / speak().

What it does:
  1. speak_interruptible(text) — speaks text aloud (pyttsx3) while a
     background thread listens to your mic. The moment you start
     talking, playback stops almost instantly — this is the actual
     barge-in, not a simulated one.
  2. record_until_silence() — after an interrupt (or whenever you want
     to ask something), records from the mic until you stop talking
     for a bit, then returns the audio.
  3. transcribe() — sends that audio to Google's free speech-to-text
     endpoint (via the `speech_recognition` library) and returns text.

Requires a real microphone and speakers, so it can only be verified on
your machine, not in this sandbox.

Run:
    python voice_io.py
(speaks a sample paragraph — try interrupting it out loud partway through)
"""

import io
import wave
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import speech_recognition as sr
from tts_utils import fresh_engine
import neural_tts


SAMPLE_RATE = 16000
BLOCK_DURATION = 0.03  # 30ms blocks — small enough for near-instant detection
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

# Tune these two if barge-in triggers too easily (lower them) or not at
# all (raise them) — it depends on your mic's sensitivity and room noise.
RMS_THRESHOLD = 300          # amplitude level counted as "someone's talking"
SUSTAINED_BLOCKS_NEEDED = 3  # ~90ms of continuous voice before we trust it
SILENCE_BLOCKS_TO_STOP = 20  # ~600ms of silence marks the end of your turn
MAX_RECORD_SECONDS = 15


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# 1. Background barge-in monitor — runs while the teacher is talking
# ---------------------------------------------------------------------------

class BargeInMonitor:
    """Watches the mic in the background and sets `triggered` the moment
    it hears sustained speech. Meant to run only while speak_interruptible()
    is actively talking."""

    def __init__(self, threshold: int = RMS_THRESHOLD,
                 sustained_blocks: int = SUSTAINED_BLOCKS_NEEDED,
                 debug: bool = False):
        self.threshold = threshold
        self.sustained_blocks = sustained_blocks
        self.debug = debug
        self.triggered = threading.Event()
        self._loud_streak = 0
        self._debug_block_count = 0
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        if self.triggered.is_set():
            return
        level = _rms(indata[:, 0])
        if self.debug:
            self._debug_block_count += 1
            if self._debug_block_count % 10 == 0:  # ~every 300ms, not every 30ms
                print(f"[mic level] {level:.0f}  (threshold: {self.threshold})")
        if level > self.threshold:
            self._loud_streak += 1
            if self._loud_streak >= self.sustained_blocks:
                self.triggered.set()
        else:
            self._loud_streak = 0

    def start(self):
        self.triggered.clear()
        self._loud_streak = 0
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=BLOCK_SIZE, callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            raise RuntimeError(
                "Couldn't open a microphone. Check Windows Settings > "
                "Sound > Input, make sure a mic is set as default and "
                "isn't disabled/muted, then try again. "
                f"(underlying error: {e})"
            ) from e

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ---------------------------------------------------------------------------
# 2. Speaking that can be cut off mid-sentence
# ---------------------------------------------------------------------------

def speak_interruptible(text: str, debug: bool = False) -> tuple[bool, int]:
    """Speak `text` aloud. Returns (interrupted, resume_index):
      - interrupted: True if the student talked over it before it finished
      - resume_index: character offset into `text` where playback actually
        stopped. 0 if not interrupted (nothing to resume) or len(text) if
        it finished naturally. Use text[resume_index:] to continue the
        SAME content from exactly where it was cut off, rather than
        skipping the rest of it or re-saying it from scratch.

    Uses pyttsx3's per-word callback (which reports the character offset
    of each word as it's spoken) to both check the mic monitor and know
    precisely how far playback got.

    Pass debug=True to print live mic levels (~every 300ms) while
    speaking — useful for figuring out what RMS_THRESHOLD should
    actually be for your mic and room.
    """
    monitor = BargeInMonitor(debug=debug)

    # Neural voice first (fetched before the mic opens, so synthesis time
    # isn't spent listening); on any failure fall through to pyttsx3 below.
    audio = neural_tts.get_audio(text)
    if audio is not None:
        monitor.start()
        try:
            print(text)
            was_interrupted, fraction = neural_tts.play(*audio, should_stop=monitor.triggered.is_set)
            return was_interrupted, (int(len(text) * fraction) if was_interrupted else len(text))
        except Exception:
            pass
        finally:
            monitor.stop()

    monitor.start()
    interrupted = {"flag": False}
    last_location = {"pos": 0}

    engine = fresh_engine()

    def on_word(name, location, length):
        last_location["pos"] = location
        if monitor.triggered.is_set() and not interrupted["flag"]:
            interrupted["flag"] = True
            engine.stop()

    engine.connect("started-word", on_word)
    print(text)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    monitor.stop()

    resume_index = last_location["pos"] if interrupted["flag"] else len(text)
    return interrupted["flag"], resume_index


# ---------------------------------------------------------------------------
# 3. Recording your interruption/question after playback stops
# ---------------------------------------------------------------------------

def record_until_silence(
    silence_blocks_to_stop: int = SILENCE_BLOCKS_TO_STOP,
    max_seconds: float = MAX_RECORD_SECONDS,
    start_timeout: Optional[float] = None,
) -> np.ndarray:
    """Records mic audio starting immediately, stopping after you've been
    quiet for a bit. Returns a 1-D int16 numpy array.

    start_timeout (optional): if nobody starts speaking within this many
    seconds, stop and return an EMPTY array (nothing to transcribe). Once
    speech starts, recording continues until silence or max_seconds. None
    (the default) = the original behaviour, used by every older caller."""
    blocks = []
    silence_streak = 0
    heard_any_speech = False
    max_blocks = int(max_seconds / BLOCK_DURATION)
    start_blocks = int(start_timeout / BLOCK_DURATION) if start_timeout is not None else None

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=BLOCK_SIZE
        ) as stream:
            for n in range(max_blocks):
                block, _ = stream.read(BLOCK_SIZE)
                blocks.append(block[:, 0].copy())
                level = _rms(block[:, 0])
                if level > RMS_THRESHOLD:
                    heard_any_speech = True
                    silence_streak = 0
                else:
                    silence_streak += 1
                if heard_any_speech and silence_streak >= silence_blocks_to_stop:
                    break
                if start_blocks is not None and not heard_any_speech and n + 1 >= start_blocks:
                    return np.array([], dtype=np.int16)   # nobody started talking
    except Exception as e:
        raise RuntimeError(
            "Couldn't open a microphone to record. Check Windows Settings "
            "> Sound > Input and try again. "
            f"(underlying error: {e})"
        ) from e

    return np.concatenate(blocks) if blocks else np.array([], dtype=np.int16)


def audio_array_to_wav_bytes(samples: np.ndarray, samplerate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def transcribe(samples: np.ndarray, samplerate: int = SAMPLE_RATE) -> str:
    """Transcribe recorded audio using Google's free speech recognition
    endpoint (no API key needed, but does need internet)."""
    if samples.size == 0:
        return ""
    recognizer = sr.Recognizer()
    wav_bytes = audio_array_to_wav_bytes(samples, samplerate)
    audio_data = sr.AudioData(
        wave.open(io.BytesIO(wav_bytes)).readframes(len(samples)),
        samplerate, 2,
    )
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""  # heard something, couldn't make out words
    except sr.RequestError as e:
        print(f"(speech recognition service error: {e})")
        return ""


# ---------------------------------------------------------------------------
# Standalone demo — proves the barge-in + capture + transcribe loop works
# ---------------------------------------------------------------------------

def demo(debug: bool = False):
    sample_lecture = (
        "Alright, let's talk about Newton's first law. An object at rest "
        "stays at rest, and an object in motion stays in motion, unless "
        "acted on by an outside force. This is often called the law of "
        "inertia, and it's the foundation for everything else we'll cover "
        "about how forces affect motion. Think about a hockey puck sliding "
        "across smooth ice — it barely slows down, because there's very "
        "little friction acting against it."
    )
    print("(try interrupting out loud partway through)")
    if debug:
        print("(debug mode — printing mic levels as it speaks)\n")
    was_interrupted, resume_index = speak_interruptible(sample_lecture, debug=debug)

    if was_interrupted:
        print(f"\n[barge-in detected at character {resume_index}/{len(sample_lecture)} "
              "— listening for your question]")
        print(f"[remaining, unspoken text]: ...{sample_lecture[resume_index:]!r}")
        audio = record_until_silence()
        text = transcribe(audio)
        print(f"[you said]: {text or '(could not understand audio)'}")
    else:
        print("\n(played all the way through — no interruption detected)")


if __name__ == "__main__":
    import sys
    debug_mode = "--debug" in sys.argv
    try:
        demo(debug=debug_mode)
    except RuntimeError as e:
        print(f"\n[voice_io error] {e}")
