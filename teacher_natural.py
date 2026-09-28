"""
teacher_natural.py — the "sounds like a real tuition teacher" layer.

Reuses the graph plumbing from teacher.py (state schema shape, LLM call,
interrupt/resume mechanics) and layers three things on top:

  1. A persona prompt that talks like a tutor in the room with you —
     refers to you by name, uses spoken (not written) phrasing, and
     closes each section with a natural comprehension check instead of
     a robotic "press enter to continue."
  2. Natural, varied resume language after you ask a question mid-lecture
     ("Alright, let's pick back up..." / "Okay — back to it —"), so
     resuming doesn't feel like a program restarting.
  3. Actual text-to-speech output, so the teacher is heard, not just
     printed. Uses pyttsx3 (fully offline, works with no API key) by
     default; swap `speak()` for a cloud TTS client (ElevenLabs,
     Cartesia) later for a more natural voice — the rest of the file
     doesn't need to change.

Run:
    export GROQ_API_KEY=your_key_here     # optional, falls back to stub
    python teacher_natural.py "Newton's Laws of Motion" Kaustubh
"""

import os
import sys
import json
import random
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Reuse everything that doesn't need to change: the LLM call, whether a
# real key is configured, and the "offline stub" fallback behaviour.
from teacher import call_llm, USE_REAL_LLM
import neural_tts


# ---------------------------------------------------------------------------
# Speech output — isolated in one function so swapping providers later
# means changing this function only, nothing else in the file.
# ---------------------------------------------------------------------------

try:
    import pyttsx3
    from tts_utils import fresh_engine
    # Just a probe to confirm pyttsx3 + a driver are available at all.
    _probe_engine = fresh_engine()
    del _probe_engine
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


def speak(text: str, out_path: Optional[str] = None) -> None:
    """Speak `text` aloud. If out_path is given, also save the audio there.

    Swap this function's body for a cloud TTS call (ElevenLabs, Cartesia,
    etc.) when you want a more natural voice than the offline espeak
    engine — everything else in this file calls speak() and doesn't care
    how the audio is produced.

    Uses tts_utils.fresh_engine(), which forces a genuinely new pyttsx3
    engine every call (see tts_utils.py for why this matters — pyttsx3's
    own init() secretly caches engines and doesn't reliably give you a
    new one just because you call it again).
    """
    print(text)  # always show the text too, useful for logs/debugging
    if out_path is None:
        # Neural voice first; on any failure fall through to pyttsx3 below.
        audio = neural_tts.get_audio(text)
        if audio is not None:
            try:
                neural_tts.play(*audio)
                return
            except Exception:
                pass
    if not TTS_AVAILABLE:
        return
    try:
        engine = fresh_engine()
        if out_path:
            engine.save_to_file(text, out_path)
            engine.runAndWait()
        else:
            engine.say(text)
            engine.runAndWait()
        engine.stop()
    except Exception:
        # No audio output device available (e.g. headless environment) —
        # text above already printed, so just keep going silently.
        pass


# Natural, varied ways to resume after answering a question — picked at
# random so it doesn't sound like the same canned line every time.
RESUME_PHRASES = [
    "Alright, let's move on to the next part.",
    "Okay, let's keep going.",
    "Right, moving on.",
    "Cool, let's continue.",
    "Good — let's build on that.",
]


# ---------------------------------------------------------------------------
# State — same shape as teacher.py's TeachState plus a student_name field.
# (Kept as its own TypedDict, not a subclass, because LangGraph reads
# schema fields directly off the class rather than via inheritance.)
# ---------------------------------------------------------------------------

class NaturalTeachState(TypedDict):
    topic: str
    student_name: str
    outline: List[str]
    idx: int
    transcript: List[str]
    _pending_question: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def build_outline_prompt(state: NaturalTeachState) -> str:
    return (
        f"You are planning a short spoken lecture on '{state['topic']}' "
        f"for a student named {state['student_name']}.\n"
        "Break it into 4-6 section titles, simplest concept first, "
        "building up. Return ONLY a JSON list of section title strings, "
        "nothing else."
    )


def plan_outline(state: NaturalTeachState) -> NaturalTeachState:
    raw = call_llm(build_outline_prompt(state))
    try:
        outline = json.loads(raw)
    except json.JSONDecodeError:
        outline = [line.strip("- ") for line in raw.splitlines() if line.strip()]

    speak(
        f"Hey {state['student_name']}, today we're covering "
        f"{state['topic']}. We'll go through it in "
        f"{len(outline)} parts, and just jump in anytime you've got a "
        f"question."
    )
    print(f"\n(Plan: {' -> '.join(outline)})\n")
    return {"outline": outline, "idx": 0, "transcript": []}


def build_teach_prompt(state: NaturalTeachState, section: str) -> str:
    is_first = state["idx"] == 0
    is_last = state["idx"] == len(state["outline"]) - 1
    return (
        f"You are a warm, experienced human tuition teacher speaking "
        f"OUT LOUD to one student named {state['student_name']}, "
        f"currently on the section '{section}' of a lecture on "
        f"'{state['topic']}'. Prior sections already covered: "
        f"{state['outline'][:state['idx']]}.\n\n"
        "Write what you would actually SAY, not an essay:\n"
        "- Conversational spoken phrasing, contractions, natural rhythm — "
        "not bullet points or written prose\n"
        f"- Address {state['student_name']} by name at least once, "
        "naturally (not forced)\n"
        "- 3-5 sentences explaining the concept clearly, connecting to "
        "what came before if relevant\n"
        "- End with ONE short, natural comprehension check addressed to "
        f"{state['student_name']} — phrase it differently than a generic "
        "'does this make sense', vary it based on the content "
        "(e.g. asking them to picture something, or predict what happens "
        "next)\n"
        f"{'- This is the opening section, so ease into it.' if is_first else ''}"
        f"{'- This is the last section, so give it a sense of wrapping up.' if is_last else ''}"
    )


def teach_section(state: NaturalTeachState) -> NaturalTeachState:
    section = state["outline"][state["idx"]]
    text = call_llm(build_teach_prompt(state, section))
    speak(f"\n--- {section} ---\n{text}")
    return {"transcript": state["transcript"] + [f"[TEACHER] {section}: {text}"]}


def check_in(state: NaturalTeachState) -> Command:
    """Pause point — currently text input, designed to be swapped for a
    streaming STT + VAD (voice activity detection) trigger later. The
    interrupt/resume mechanics stay identical either way."""
    question = interrupt("(press Enter to continue, or type a question) > ")
    if question and question.strip():
        return Command(goto="respond_to_student", update={"_pending_question": question})
    return Command(goto="advance", update={})


def build_response_prompt(state: NaturalTeachState, section: str, student_input: str) -> str:
    return (
        f"You are a warm human tuition teacher speaking OUT LOUD to "
        f"{state['student_name']}. You were just teaching the section "
        f"'{section}' of a lecture on '{state['topic']}', and possibly "
        "ended it with a comprehension check question of your own. "
        f"{state['student_name']} just said: \"{student_input}\"\n\n"
        "This could be any of:\n"
        "- a genuine question they're asking YOU\n"
        "- their answer/attempt at the comprehension check you posed\n"
        "- just an observation or comment, not a question at all\n\n"
        "Read what they said and respond the way a real teacher would for "
        "whichever it actually is:\n"
        "- if it's their answer to your check: tell them clearly whether "
        "it's right, gently correct or refine it if it's off or "
        "incomplete, and add one sentence of insight\n"
        "- if it's a genuine question: answer it directly\n"
        "- if it's just a comment: acknowledge it naturally and build on "
        "it if relevant\n"
        "Do NOT default to treating it as a question if it isn't one. "
        f"Respond in 2-4 conversational sentences, addressing "
        f"{state['student_name']} naturally. Do not say goodbye or wrap "
        "up — you're about to keep teaching."
    )


def respond_to_student(state: NaturalTeachState) -> NaturalTeachState:
    student_input = state.get("_pending_question", "")
    section = state["outline"][state["idx"]]
    response = call_llm(build_response_prompt(state, section, student_input))
    speak(f"\n[{state['student_name']} said] {student_input}")
    speak(response)
    speak(random.choice(RESUME_PHRASES))
    return {"transcript": state["transcript"] + [f"[STUDENT] {student_input}", f"[TEACHER] {response}"]}


def advance(state: NaturalTeachState) -> NaturalTeachState:
    return {"idx": state["idx"] + 1}


def route_after_advance(state: NaturalTeachState) -> str:
    return "teach_section" if state["idx"] < len(state["outline"]) else END


# ---------------------------------------------------------------------------
# Graph wiring — identical shape to teacher.py's build_graph, just bound
# to the natural-teaching nodes and the richer state schema.
# ---------------------------------------------------------------------------

def build_natural_graph():
    g = StateGraph(NaturalTeachState)
    g.add_node("plan_outline", plan_outline)
    g.add_node("teach_section", teach_section)
    g.add_node("check_in", check_in)
    g.add_node("respond_to_student", respond_to_student)
    g.add_node("advance", advance)

    g.set_entry_point("plan_outline")
    g.add_edge("plan_outline", "teach_section")
    g.add_edge("teach_section", "check_in")
    g.add_edge("respond_to_student", "advance")
    g.add_conditional_edges("advance", route_after_advance, ["teach_section", END])

    return g.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    student_name = args[-1] if len(args) > 1 else "there"
    topic = " ".join(args[:-1]) if len(args) > 1 else (" ".join(args) or "Newton's Laws of Motion")

    graph = build_natural_graph()
    config = {"configurable": {"thread_id": "session-natural-1"}}

    if not USE_REAL_LLM:
        print("(No GROQ_API_KEY found — running with an offline stub LLM.)")
    if not TTS_AVAILABLE:
        print("(pyttsx3/espeak not available — running text-only.)")

    state_input = {"topic": topic, "student_name": student_name}

    while True:
        graph.invoke(state_input, config=config)
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        try:
            user_reply = input("> ").strip()
        except EOFError:
            break
        state_input = Command(resume=user_reply)

    print("\n=== Lecture complete ===")


if __name__ == "__main__":
    main()
