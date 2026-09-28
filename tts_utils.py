"""
tts_utils.py — one fix, used everywhere pyttsx3 is used.

pyttsx3.init() caches engines internally in a weakref.WeakValueDictionary
keyed by driver name:

    def init(driverName=None, debug=False):
        try:
            eng = _activeEngines[driverName]
        except KeyError:
            eng = Engine(driverName, debug)
            _activeEngines[driverName] = eng
        return eng

That means calling pyttsx3.init() again does NOT reliably give you a new
engine — it gives you back the same cached one UNLESS the previous one
has already been garbage collected, which is timing-dependent and not
something you control. On Windows, reusing the same SAPI5 engine across
multiple say()+runAndWait() calls is a known source of the engine's
internal loop going silently "busy" and swallowing later calls — which
is exactly the "some lines just don't get spoken" symptom.

fresh_engine() below forces a truly new engine every time by clearing
the cache entry before calling init(), removing the race entirely.
"""

import pyttsx3

# Teacher voice: first installed match wins (Windows SAPI5 names). Zira =
# soft US female, Hazel = UK female. If none are installed, the system
# default voice is kept, so speech never fails because of this.
PREFERRED_VOICES = ["Zira", "Hazel"]


def _select_voice(engine) -> None:
    try:
        voices = engine.getProperty("voices")
        for wanted in PREFERRED_VOICES:
            for v in voices:
                if wanted.lower() in v.name.lower():
                    engine.setProperty("voice", v.id)
                    return
    except Exception:
        pass


def fresh_engine(rate: int = 150):
    """Return a guaranteed brand-new pyttsx3 engine, bypassing its
    internal engine cache."""
    if None in pyttsx3._activeEngines:
        del pyttsx3._activeEngines[None]
    engine = pyttsx3.init()
    _select_voice(engine)
    engine.setProperty("rate", rate)
    return engine
