"""
teacher_voice.py — the fully voice-driven teacher.

Combines two pieces that were each tested standalone:
  - teacher_natural.py: the tutor persona prompts (imported directly,
    not duplicated)
  - voice_io.py: real barge-in detection + speech-to-text

No text input anywhere in this file — you only ever talk.

Two listening moments per section:
  1. WHILE the teacher is speaking (mid-sentence barge-in) — if you talk
     over it, playback stops immediately.
  2. AFTER the teacher finishes a section normally — a short listening
     window (a few seconds) in case you want to ask something before
     it moves on. Stay quiet and it just continues.

Run:
    export GROQ_API_KEY=your_key_here
    python teacher_voice.py "Newton's Laws of Motion" Kaustubh
"""

import sys
import re
import random
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from teacher import call_llm, USE_REAL_LLM
from teacher_natural import (
    build_outline_prompt,
    build_teach_prompt,
    build_response_prompt,
    RESUME_PHRASES,
    speak as plain_speak,  # no mic-monitoring — for short, non-interruptible lines
)
from voice_io import speak_interruptible, record_until_silence, transcribe

import json


# How long to wait quietly after a section finishes before assuming the
# student has no question and moving on. Keep this short — it's a grace
# window, not a real recording (if they do start talking, record_until_silence
# takes over and records for as long as needed).
POST_SECTION_LISTEN_SECONDS = 4

# When the section ended with a comprehension check (see
# comprehension_check.py): how long to wait for the student to START
# answering. Once they speak, the answer is recorded until they pause
# (up to voice_io.MAX_RECORD_SECONDS) — not cut off like the 4 s window.
CHECK_ANSWER_WAIT_SECONDS = 8


class VoiceTeachState(TypedDict):
    topic: str
    student_name: str
    outline: List[str]
    idx: int
    transcript: List[str]


def plan_outline(state: VoiceTeachState) -> VoiceTeachState:
    raw = call_llm(build_outline_prompt(state))
    try:
        outline = json.loads(raw)
    except json.JSONDecodeError:
        outline = [line.strip("- ") for line in raw.splitlines() if line.strip()]

    plain_speak(
        f"Hey {state['student_name']}, today we're covering "
        f"{state['topic']}. We'll go through it in "
        f"{len(outline)} parts, and just jump in anytime you've got a "
        f"question."
    )
    print(f"\n(Plan: {' -> '.join(outline)})\n")
    return {"outline": outline, "idx": 0, "transcript": []}


def _listen_briefly() -> str:
    """A short post-section listening window. Returns transcribed text,
    or '' if the student stayed quiet."""
    audio = record_until_silence(max_seconds=POST_SECTION_LISTEN_SECONDS)
    if audio.size == 0:
        return ""
    return transcribe(audio)


def _listen_for_answer() -> str:
    """The student's answer to the section's comprehension check: waits up
    to CHECK_ANSWER_WAIT_SECONDS for them to start, then records the whole
    answer. Returns transcribed text, or '' if they stayed quiet."""
    audio = record_until_silence(start_timeout=CHECK_ANSWER_WAIT_SECONDS)
    if audio.size == 0:
        return ""
    return transcribe(audio)


def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter — good enough for the LLM's lecture-style
    prose. Used so interruptions resume at a clean sentence boundary
    instead of a raw character offset, which can land mid-clause and
    sound like a disconnected fragment when replayed."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


MAX_INTERRUPTIONS_PER_SECTION = 6  # safety cap against pathological loops


def _handle_student_input(state: VoiceTeachState, section: str, student_input: str) -> str:
    """Answers/responds to what the student said, speaks the response,
    and returns a natural transition line. Does not touch idx/transcript
    itself — caller folds those in.

    The "student said X" echo is printed, not spoken — no real teacher
    repeats your question back verbatim before answering it.
    """
    response = call_llm(build_response_prompt(state, section, student_input))
    print(f"\n[{state['student_name']} said] {student_input}")
    plain_speak(response)
    plain_speak(random.choice(RESUME_PHRASES))
    return response


def teach_and_listen(state: VoiceTeachState) -> VoiceTeachState:
    """Teaches one section fully before ever advancing, one sentence at
    a time. If a sentence gets interrupted partway through, it answers
    the question, then RE-SPEAKS THAT SAME SENTENCE FROM THE START
    (never a mid-clause fragment) before moving on to the next one.
    Only once every sentence has been delivered cleanly does it do a
    brief post-section listen and advance to the next section."""
    section = state["outline"][state["idx"]]
    text = call_llm(build_teach_prompt(state, section))
    sentences = split_into_sentences(text)
    transcript = state["transcript"] + [f"[TEACHER] {section}: {text}"]

    print(f"\n--- {section} ---")  # console only — not spoken

    i = 0
    interruption_count = 0
    while i < len(sentences):
        if interruption_count >= MAX_INTERRUPTIONS_PER_SECTION:
            break  # safety valve — stop retrying, move on regardless

        interrupted, _ = speak_interruptible(sentences[i])
        if not interrupted:
            i += 1
            continue

        interruption_count += 1
        audio = record_until_silence()
        student_input = transcribe(audio)
        if student_input:
            response = _handle_student_input(state, section, student_input)
            transcript += [f"[STUDENT] {student_input}", f"[TEACHER] {response}"]
        # i does NOT advance — the interrupted sentence gets retried
        # fully from its start on the next loop iteration.

    # Section is now fully delivered — one last brief window in case
    # there's a final question, THEN move on to the next section.
    student_input = _listen_briefly()
    if student_input:
        response = _handle_student_input(state, section, student_input)
        transcript += [f"[STUDENT] {student_input}", f"[TEACHER] {response}"]

    return {"transcript": transcript, "idx": state["idx"] + 1}


def route_after_section(state: VoiceTeachState) -> str:
    return "teach_and_listen" if state["idx"] < len(state["outline"]) else END


def build_voice_graph():
    g = StateGraph(VoiceTeachState)
    g.add_node("plan_outline", plan_outline)
    g.add_node("teach_and_listen", teach_and_listen)

    g.set_entry_point("plan_outline")
    g.add_edge("plan_outline", "teach_and_listen")
    g.add_conditional_edges(
        "teach_and_listen", route_after_section, ["teach_and_listen", END]
    )

    return g.compile(checkpointer=MemorySaver())


def main():
    args = sys.argv[1:]
    student_name = args[-1] if len(args) > 1 else "there"
    topic = " ".join(args[:-1]) if len(args) > 1 else (" ".join(args) or "Newton's Laws of Motion")

    if not USE_REAL_LLM:
        print("(No GROQ_API_KEY found — running with an offline stub LLM.)")

    graph = build_voice_graph()
    config = {"configurable": {"thread_id": "session-voice-1"}}
    graph.invoke({"topic": topic, "student_name": student_name}, config=config)

    print("\n=== Lecture complete ===")


if __name__ == "__main__":
    main()
