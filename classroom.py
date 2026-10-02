import os
import sys
import sqlite3
import time
import random
from typing import Dict, List, Optional, Tuple

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig

from teacher import USE_REAL_LLM
from teacher_natural import speak as plain_speak, RESUME_PHRASES
from teacher_voice import (
    VoiceTeachState,
    split_into_sentences,
    MAX_INTERRUPTIONS_PER_SECTION,
    _listen_briefly,
)
from voice_io import speak_interruptible, record_until_silence, transcribe
from neural_tts import prefetch as prefetch_speech
from planner import build_outline_from_document, assign_context
from explainer import BeatStream, answer_doubt
from rag import build_index, DocumentIndex
from rag.ingest import is_pdf_file, missing_pdf_pages
from rag.vector_index import preload_model_in_background
from rag.reranker import preload_reranker_in_background
from tools import set_active_index
from quiz import generate_quiz, print_quiz, USE_REAL_GEMINI, QuizUnavailable
import persistence

CHECKPOINT_DB_PATH = os.environ.get("CLASSROOM_CHECKPOINT_DB_PATH", "classroom_checkpoints.db")


class ClassroomState(VoiceTeachState):
    document_path: Optional[str]
    section_context: Dict[str, List[str]]
    quiz: List[dict]
    doubt_history: List[Dict[str, str]]  # every Q/A doubt asked this lecture, in order


def plan_lecture(state: ClassroomState) -> ClassroomState:
    document_path = state.get("document_path")
    index = build_index(document_path) if document_path else DocumentIndex()

    set_active_index(index)

    outline = build_outline_from_document(state["topic"], index)

    # Greeting BEFORE assign_context: it only needs the outline, and while
    # it's being spoken the embedding model (preloaded in the background
    # from main()) finishes loading — so assign_context's ~14s model load
    # no longer happens in silence before the greeting.
    source_note = " based on the material you gave me" if not index.is_empty() else ""
    plain_speak(
        f"Hey {state['student_name']}, today we're covering "
        f"{state['topic']}{source_note}. We'll go through it in "
        f"{len(outline)} parts, and just jump in anytime you've got a "
        f"question."
    )
    print(f"\n(Plan: {' -> '.join(outline)})\n")

    section_context = assign_context(outline, index)

    return {
        "outline": outline, "section_context": section_context,
        "idx": 0, "transcript": [], "doubt_history": [],
    }


def _handle_doubt(state: ClassroomState, section: str, student_input: str,
                   doubt_history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    response = answer_doubt(state, section, student_input, doubt_history=doubt_history)
    print(f"\n[{state['student_name']} said] {student_input}")
    resume_phrase = random.choice(RESUME_PHRASES)
    prefetch_speech([response, resume_phrase])
    plain_speak(response)
    plain_speak(resume_phrase)
    updated_history = doubt_history + [{"question": student_input, "answer": response}]
    return response, updated_history


def teach_and_listen(state: ClassroomState, config: RunnableConfig) -> ClassroomState:
    """Same interruption-retry shape as before, but now also the natural
    'crash/restart resume point': the checkpointer persists state after
    this node returns, so a process that dies (or is deliberately closed
    for the day) mid-lecture picks back up at the start of whichever
    section it last completed, not from scratch. doubt_history is
    threaded through the same way transcript already is, so a doubt
    asked in section 2 is still remembered when answering one in
    section 4 — or after resuming tomorrow."""
    section = state["outline"][state["idx"]]
    context_chunks = state["section_context"].get(section, [])
    # Beats are generated in the background; beat 1 is spoken as soon as
    # it's ready while beats 2-3 are still being generated (see
    # explainer.BeatStream — hides Groq rate-limit waits behind speech).
    beat_stream = BeatStream(state, section, context_chunks)
    exchanges: List[str] = []   # [STUDENT]/[TEACHER] doubt lines, in order
    doubt_history = state.get("doubt_history", [])

    print(f"\n--- {section} ---")

    interruption_count = 0
    for beat_text in beat_stream:
        if interruption_count >= MAX_INTERRUPTIONS_PER_SECTION:
            break

        # Neural voice: synthesize this beat's sentences in the background
        # so there's no network pause before each one (no-op if unavailable).
        prefetch_speech(split_into_sentences(beat_text))

        sentences = split_into_sentences(beat_text)
        i = 0
        while i < len(sentences):
            if interruption_count >= MAX_INTERRUPTIONS_PER_SECTION:
                break

            interrupted, _ = speak_interruptible(sentences[i])
            if not interrupted:
                i += 1
                continue

            interruption_count += 1
            audio = record_until_silence()
            student_input = transcribe(audio)
            if student_input:
                response, doubt_history = _handle_doubt(state, section, student_input, doubt_history)
                exchanges += [f"[STUDENT] {student_input}", f"[TEACHER] {response}"]

    student_input = _listen_briefly()
    if student_input:
        response, doubt_history = _handle_doubt(state, section, student_input, doubt_history)
        exchanges += [f"[STUDENT] {student_input}", f"[TEACHER] {response}"]

    # Same transcript as before: the section's full beats text first, then
    # the doubts in order. all_beats() waits for any beat still generating.
    beats_text = beat_stream.all_beats()
    transcript = state["transcript"] + [f"[TEACHER] {section}: {' '.join(beats_text)}"] + exchanges

    persistence.touch(config["configurable"]["thread_id"])

    return {"transcript": transcript, "idx": state["idx"] + 1, "doubt_history": doubt_history}


def route_after_section(state: ClassroomState) -> str:
    return "teach_and_listen" if state["idx"] < len(state["outline"]) else "generate_quiz_node"


def generate_quiz_node(state: ClassroomState, config: RunnableConfig) -> ClassroomState:
    """Runs ONLY after every section has been taught — never mid-lecture.
    Also marks the session 'completed' in the progress store, so a
    later run with the same student+topic starts a fresh attempt
    instead of silently reusing this finished thread.

    If generate_quiz raises QuizUnavailable, it propagates out of the
    graph on purpose: the checkpoint stays parked at THIS node, so
    re-running the same command later resumes straight into the quiz
    without re-teaching the lecture. main() catches it and exits cleanly."""
    plain_speak(
        f"Alright {state['student_name']}, that's the lecture — give me a "
        "moment, I'm putting together a quick quiz to check what stuck."
    )
    questions = generate_quiz(
        topic=state["topic"],
        outline=state["outline"],
        section_context=state["section_context"],
        transcript=state["transcript"],
        document_path=state.get("document_path"),
    )
    print_quiz(questions)

    persistence.mark_completed(
        config["configurable"]["thread_id"],
        quiz_note=f"{len(questions)} questions generated",
    )

    return {"quiz": questions}


def build_classroom_graph():
    g = StateGraph(ClassroomState)
    g.add_node("plan_lecture", plan_lecture)
    g.add_node("teach_and_listen", teach_and_listen)
    g.add_node("generate_quiz_node", generate_quiz_node)

    g.set_entry_point("plan_lecture")
    g.add_edge("plan_lecture", "teach_and_listen")
    g.add_conditional_edges(
        "teach_and_listen", route_after_section, ["teach_and_listen", "generate_quiz_node"]
    )
    g.add_edge("generate_quiz_node", END)

    # check_same_thread=False: fine here, this process only ever drives
    # one graph.invoke() at a time against this connection.
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return g.compile(checkpointer=checkpointer)


def _fetch_youtube_title(url: str) -> Optional[str]:
    """YouTube's public oEmbed endpoint returns the video title with no
    API key. Returns None on any failure (offline, private video, etc.)."""
    import json
    import urllib.parse
    import urllib.request
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    endpoint = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe="")
    try:
        with urllib.request.urlopen(endpoint, timeout=8) as resp:
            return json.load(resp).get("title") or None
    except Exception:
        return None


def _derive_topic_from_filename(path: str) -> str:
    if "youtube.com" in path or "youtu.be" in path:
        # The URL's last segment is just the video ID + query string
        # (e.g. '8Tvqjhkk6Vo?list=LL'), so ask YouTube for the real title.
        return _fetch_youtube_title(path) or "this video"
    if path.startswith(("http://", "https://")):
        return "this web page"
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("_", " ").replace("-", " ").title()


# Share of a PDF's pages that must yield text: below WARN -> loud warning;
# below ASK -> also ask before teaching (the lecture would otherwise be
# taught from general knowledge while looking like it used the document).
COVERAGE_WARN = 0.8
COVERAGE_ASK = 0.5


def _page_ranges(pages: List[int]) -> str:
    """[2, 3, 4, 7] -> '2-4, 7'"""
    parts, start = [], None
    for i, p in enumerate(pages):
        if start is None:
            start = p
        if i == len(pages) - 1 or pages[i + 1] != p + 1:
            parts.append(f"{start}-{p}" if p != start else str(p))
            start = None
    return ", ".join(parts)


def _source_coverage_ok(document_path: str) -> bool:
    """Checks how much of a PDF actually made it into the index before a
    fresh lecture starts. Returns False only if the student declines to
    continue. The index built here is cached, so plan_lecture's own
    build_index() call just reloads it."""
    if not is_pdf_file(document_path):
        return True
    index = build_index(document_path)
    total, missing = missing_pdf_pages(document_path, index.metadatas)
    covered = total - len(missing)
    if total == 0 or covered / total >= COVERAGE_WARN:
        return True
    print(f"\n(WARNING: only {covered}/{total} pages of this PDF produced any text. "
          f"Pages {_page_ranges(missing)} look image-only and couldn't be read "
          "(no OCR/vision available). Those parts would be taught from general "
          "knowledge, not from your document.)")
    if covered / total >= COVERAGE_ASK:
        return True
    try:
        reply = input("Continue anyway? [y/N] ").strip().lower()
    except EOFError:
        reply = ""
    return reply in ("y", "yes")


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print('Usage: python classroom.py <document_path> <student_name> ["optional topic override"]')
        sys.exit(1)

    document_path = args[0]
    student_name = args[1]

    # Load the embedding model (~10s) in the background while the rest of
    # startup runs, instead of stalling the first retrieval. Same for the
    # doubt tool's reranker, so the first doubt doesn't wait for it.
    preload_model_in_background()
    preload_reranker_in_background()

    topic = args[2] if len(args) > 2 else _derive_topic_from_filename(document_path)

    if not USE_REAL_LLM:
        print("(No GROQ_API_KEY found — running with an offline stub LLM.)")
    if not USE_REAL_GEMINI:
        print("(No GOOGLE_API_KEY found — the end-of-lecture quiz will use an offline stub instead of real Gemini generation.)")

    thread_id = persistence.make_thread_id(student_name, topic)
    existing = persistence.get_session(thread_id)

    if existing is not None and existing["status"] == "completed":
        # Already finished this exact topic in a past run — start a
        # fresh attempt under its own thread instead of silently
        # reusing (and re-serving the old quiz from) the finished one.
        thread_id = f"{thread_id}-retry-{int(time.time())}"

    persistence.start_or_resume_session(thread_id, student_name, topic, document_path)
    config = {"configurable": {"thread_id": thread_id}}

    graph = build_classroom_graph()
    snapshot = graph.get_state(config)

    try:
        if snapshot.values and snapshot.next:
            print(f"(Welcome back, {student_name} — resuming '{topic}' right where you left off.)")
            # plan_lecture (the only place set_active_index runs) is skipped on
            # resume, so re-register the document index for search_document.
            saved_path = snapshot.values.get("document_path")
            if saved_path and "teach_and_listen" in snapshot.next:
                set_active_index(build_index(saved_path))
            graph.invoke(None, config=config)
        else:
            if not _source_coverage_ok(document_path):
                print("(Stopped before teaching — the lecture was not started.)")
                sys.exit(3)
            graph.invoke(
                {"topic": topic, "student_name": student_name, "document_path": document_path},
                config=config,
            )
    except QuizUnavailable as e:
        plain_speak(
            f"Sorry {student_name}, I couldn't put the quiz together right now — "
            "the quiz models are out of quota. Your lecture is saved."
        )
        print(f"\n(Quiz unavailable: {e})")
        print("(Re-run the exact same command later — it will skip straight to the quiz, no re-teaching.)")
        sys.exit(2)

    print("\n=== Lecture complete ===")


if __name__ == "__main__":
    main()