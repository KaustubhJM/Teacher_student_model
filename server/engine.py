"""The classroom steps for the website, built from the SAME functions the
CLI graph (classroom.py) uses — nothing here changes how a lecture is
planned, taught, answered or quizzed:

    plan      build_index -> build_outline_from_document -> assign_context
    teach     explainer.iter_section_beats (3 beats per section)
    doubt     explainer.answer_doubt (search_document / search_web tools)
    quiz      quiz.generate_quiz

The difference is only who drives it: the browser asks for one step at a
time and speaks the text itself, instead of classroom.py's loop speaking
through the laptop."""
import os
import random
import re
import threading
import time
from typing import Dict, List, Optional

from rag import build_index, DocumentIndex
from rag.ingest import is_pdf_file, missing_pdf_pages
from rag.youtube_ingest import is_youtube_source, get_video_title
from planner import build_outline_from_document, assign_context
from explainer import iter_section_beats, answer_doubt
from tools import set_active_index
from quiz import generate_quiz, QuizUnavailable
from comprehension_check import check_answer_input, remembered_check_reply

from . import store

STUDENT_NAME = os.environ.get("CLASSROOM_STUDENT_NAME", "Kaustubh")
EMOJIS = ["📘", "🧪", "🧮", "🌍", "⚙️", "🎨", "🧬", "💡"]

# How long one beats request waits for the next beat before returning
# (the browser simply asks again). Beat generation itself is not limited.
BEAT_POLL_SECONDS = 25


def split_into_sentences(text: str) -> List[str]:
    """Same rule as teacher_voice.split_into_sentences (the CLI speaks
    sentence by sentence; the website does too)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def derive_title(kind: str, name: str) -> str:
    """Default class title = the CLI's default topic rule."""
    if kind == "upload":
        stem = os.path.splitext(os.path.basename(name))[0]
        return stem.replace("_", " ").replace("-", " ").title()
    if kind == "text":
        return " ".join(name.split()[:6])
    return name


def new_class(title: str, sources: List[dict], document_path: Optional[str],
              student_name: str = STUDENT_NAME, title_from_source: bool = False) -> dict:
    cls = {
        "id": f"class-{int(time.time() * 1000)}",
        "title": title,
        "emoji": random.choice(EMOJIS),
        "createdAt": time.strftime("%Y-%m-%d"),
        "sources": sources,
        "outline": [],
        "sectionsDone": 0,
        "position": {"s": 0, "i": 0},  # next sentence to speak (section, sentence) — resume point
        "status": "planning",      # planning -> new -> in_progress -> completed | failed
        "planStep": 0,
        "error": None,
        "quizScore": None,
        "_created": time.time(),
        "_document_path": document_path,
        "_student_name": student_name,
        "_title_from_source": title_from_source,
        "_section_context": {},
        "_beats": {},              # section index (str) -> list of beat texts (complete sections only)
        "_exchanges": {},          # section index (str) -> [STUDENT]/[TEACHER] lines
        "_transcript": [],
        "_doubt_history": [],
        "_doubt_log": [],          # [{s, i, question, answer}] — where each doubt was asked (captions on resume)
        "_quiz": None,
    }
    store.save(cls)
    return cls


# ---------------------------------------------------------------------------
# Planning (background thread, started by POST /api/classes)
# ---------------------------------------------------------------------------

_indexes: Dict[str, object] = {}
_index_lock = threading.Lock()


def _get_index(cls: dict):
    """The class's document index (cached in memory; build_index itself
    reloads from .rag_cache on disk after a server restart)."""
    with _index_lock:
        index = _indexes.get(cls["id"])
        if index is None:
            path = cls.get("_document_path")
            index = build_index(path) if path else DocumentIndex()
            _indexes[cls["id"]] = index
        return index


def _source_detail(cls: dict, index) -> Optional[str]:
    path = cls.get("_document_path")
    if not path:
        return None
    if index.is_empty():
        return "No readable text found — taught from general knowledge"
    if is_pdf_file(path):
        total, missing = missing_pdf_pages(path, index.metadatas)
        detail = f"{total - len(missing)}/{total} pages read"
        if total and (total - len(missing)) / total < 0.8:
            detail += " — the rest looked image-only"
        return detail
    return None


def plan_class(class_id: str) -> None:
    cls = store.get(class_id)
    if cls is None:
        return
    try:
        path = cls.get("_document_path")
        if path and cls.get("_title_from_source") and is_youtube_source(path):
            title = get_video_title(path)
            if title:
                cls = store.update(class_id, title=title)

        index = _get_index(cls)
        detail = _source_detail(cls, index)
        if detail and cls["sources"]:
            def set_detail(c):
                c["sources"][0]["detail"] = detail
            cls = store.update(class_id, fn=set_detail)

        store.update(class_id, planStep=1)
        outline = build_outline_from_document(cls["title"], index)

        store.update(class_id, planStep=2)
        section_context = assign_context(outline, index)

        store.update(class_id, outline=outline, _section_context=section_context,
                     status="new", planStep=3)
    except Exception as e:
        store.update(class_id, status="failed", error=f"{type(e).__name__}: {e}")


def start_planning(class_id: str) -> None:
    threading.Thread(target=plan_class, args=(class_id,), daemon=True).start()


# ---------------------------------------------------------------------------
# Teaching: beats per section, generated in the background (like BeatStream)
# ---------------------------------------------------------------------------

def _lecture_state(cls: dict, idx: int) -> dict:
    """The same state fields classroom.ClassroomState carries."""
    return {
        "topic": cls["title"],
        "student_name": cls["_student_name"],
        "outline": cls["outline"],
        "idx": idx,
        "transcript": cls["_transcript"],
        "document_path": cls.get("_document_path"),
        "section_context": cls["_section_context"],
        "doubt_history": cls["_doubt_history"],
    }


class _BeatJob:
    def __init__(self, cls: dict, idx: int):
        self.class_id, self.idx = cls["id"], idx
        self.beats: List[str] = []
        self.done = False
        self.error: Optional[str] = None
        self._cond = threading.Condition()
        section = cls["outline"][idx]
        chunks = cls["_section_context"].get(section, [])
        threading.Thread(target=self._run, args=(_lecture_state(cls, idx), section, chunks),
                         daemon=True).start()

    def _run(self, state, section, chunks):
        try:
            for text in iter_section_beats(state, section, chunks):
                with self._cond:
                    self.beats.append(text)
                    self._cond.notify_all()

            def keep(c):
                c["_beats"][str(self.idx)] = list(self.beats)
            store.update(self.class_id, fn=keep)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        with self._cond:
            self.done = True
            self._cond.notify_all()
        with _jobs_lock:
            if _jobs.get((self.class_id, self.idx)) is self:
                del _jobs[(self.class_id, self.idx)]

    def wait(self, have: int, timeout: float) -> None:
        with self._cond:
            self._cond.wait_for(lambda: len(self.beats) > have or self.done, timeout)


_jobs: Dict[tuple, _BeatJob] = {}
_jobs_lock = threading.Lock()


class TeachingError(RuntimeError):
    pass


def get_beats(class_id: str, idx: int, have: int) -> dict:
    """Beats of section `idx` after the first `have`, each split into
    sentences. Waits up to BEAT_POLL_SECONDS for at least one new beat.
    done=True -> the section has no more beats."""
    cls = store.get(class_id)
    saved = cls["_beats"].get(str(idx))
    if saved is not None:
        return {"beats": [split_into_sentences(b) for b in saved[have:]], "done": True}

    with _jobs_lock:
        job = _jobs.get((class_id, idx))
        if job is None:
            job = _jobs[(class_id, idx)] = _BeatJob(cls, idx)
    job.wait(have, BEAT_POLL_SECONDS)

    new = job.beats[have:]
    if job.error and not new:
        raise TeachingError(job.error)
    return {"beats": [split_into_sentences(b) for b in new], "done": job.done and not job.error}


def section_done(class_id: str, idx: int) -> dict:
    """The browser finished speaking section idx: record it in the
    transcript exactly like teach_and_listen (all beats, then the doubts)."""
    def advance(c):
        if idx != c["sectionsDone"]:
            return  # already recorded (or out of order) — idempotent
        section = c["outline"][idx]
        beats = c["_beats"].get(str(idx), [])
        c["_transcript"] = (c["_transcript"] + [f"[TEACHER] {section}: {' '.join(beats)}"]
                            + c["_exchanges"].get(str(idx), []))
        c["sectionsDone"] = idx + 1
        c["status"] = "completed" if idx + 1 >= len(c["outline"]) else "in_progress"
        _move_position(c, idx + 1, 0)
    return store.update(class_id, fn=advance)


def _pos(c: dict) -> tuple:
    p = c.get("position") or {"s": c.get("sectionsDone", 0), "i": 0}  # records made before "position"
    return (p["s"], p["i"])


def _move_position(c: dict, s: int, i: int) -> None:
    """Only forward — a late request can't move the resume point back."""
    if (s, i) > _pos(c):
        c["position"] = {"s": s, "i": i}


def save_progress(class_id: str, s: int, i: int) -> dict:
    """The browser finished speaking sentence i-1 of section s (so the next
    one is i). Saved after every sentence, so leaving mid-section resumes
    at the same sentence — and the class counts as started."""
    def move(c):
        if c["status"] == "new":
            c["status"] = "in_progress"
        if s < len(c["outline"]):
            _move_position(c, s, i)
    return store.update(class_id, fn=move)


def history(class_id: str) -> List[dict]:
    """What was already taught, as caption items for the website (same
    shapes as useLecture's items): every spoken sentence up to the resume
    point, with each doubt placed where it was asked."""
    c = store.get(class_id)
    end_s, end_i = _pos(c)
    doubts: Dict[tuple, List[dict]] = {}
    for d in c.get("_doubt_log", []):
        doubts.setdefault((d["s"], d["i"]), []).append(d)

    def doubt_items(s, i):
        out = []
        for d in doubts.pop((s, i), []):
            out += [{"kind": "student", "text": d["question"]}, {"kind": "teacher", "text": d["answer"]}]
        return out

    items: List[dict] = []
    for s in range(min(end_s + 1, len(c["outline"]))):
        sentences = [x for b in c["_beats"].get(str(s), []) for x in split_into_sentences(b)]
        last = len(sentences) if s < end_s else min(end_i, len(sentences))
        for i in range(last):
            items += doubt_items(s, i)          # asked before sentence i was (re)spoken
            items.append({"kind": "sentence", "s": s, "i": i, "text": sentences[i]})
        items += doubt_items(s, last)
    for key in sorted(doubts):                  # anything left (e.g. after the lecture ended)
        items += doubt_items(*key)
    return items


# ---------------------------------------------------------------------------
# Doubts
# ---------------------------------------------------------------------------

# tools.search_document reads ONE global active index, so doubts are
# answered one at a time (single-user app).
_doubt_lock = threading.Lock()


def ask_doubt(class_id: str, idx: int, question: str, sentence: int = 0,
              check_question: Optional[str] = None) -> str:
    """A doubt — or, with check_question, the student's reply to the
    section's comprehension check (`question` is then the reply)."""
    cls = store.get(class_id)
    idx = max(0, min(idx, len(cls["outline"]) - 1))
    section = cls["outline"][idx]
    # Reply to the end-of-section check: the teacher gets the exact question
    # and is told to explain the right answer (comprehension_check.py).
    llm_input = check_answer_input(check_question, question) if check_question else question
    with _doubt_lock:
        set_active_index(_get_index(cls))
        answer = answer_doubt(_lecture_state(cls, idx), section, llm_input,
                              doubt_history=cls["_doubt_history"])
    # Memory of earlier doubts keeps which check was answered (short form).
    remembered = remembered_check_reply(check_question, question) if check_question else question

    lines = [f"[STUDENT] {remembered}", f"[TEACHER] {answer}"]

    def record(c):
        c["_doubt_history"] = c["_doubt_history"] + [{"question": remembered, "answer": answer}]
        c["_doubt_log"] = c.get("_doubt_log", []) + [
            {"s": idx, "i": sentence, "question": question, "answer": answer}]
        if c["status"] == "new":
            c["status"] = "in_progress"
        if idx < c["sectionsDone"]:      # section already in the transcript (e.g. after the lecture)
            c["_transcript"] = c["_transcript"] + lines
        else:
            c["_exchanges"][str(idx)] = c["_exchanges"].get(str(idx), []) + lines
    store.update(class_id, fn=record)
    return answer


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

_quiz_lock = threading.Lock()


def _answer_index(correct: str, options: List[str]) -> Optional[int]:
    """quiz.py asks for "the exact text of the correct option"; also accept
    a bare letter ("B", "B)") or an option that starts with the answer."""
    c = correct.strip().casefold()
    for j, opt in enumerate(options):
        if opt.strip().casefold() == c:
            return j
    m = re.fullmatch(r"\(?([a-h])[\).:]?", c)
    if m and ord(m.group(1)) - 97 < len(options):
        return ord(m.group(1)) - 97
    for j, opt in enumerate(options):
        o = opt.strip().casefold()
        if c and (o.startswith(c) or c.startswith(o)):
            return j
    return None


def to_web_quiz(questions: List[dict]) -> List[dict]:
    out = []
    for q in questions:
        options = [str(o) for o in q.get("options", [])]
        j = _answer_index(str(q.get("correct_answer", "")), options)
        if q.get("question") and len(options) >= 2 and j is not None:
            out.append({"q": q["question"], "options": options, "answer": j,
                        "why": q.get("explanation", "")})
    return out


def get_quiz(class_id: str) -> List[dict]:
    """Generated once per class, then cached (raises QuizUnavailable)."""
    with _quiz_lock:
        cls = store.get(class_id)
        if cls.get("_quiz"):
            return cls["_quiz"]
        questions = to_web_quiz(generate_quiz(
            topic=cls["title"],
            outline=cls["outline"],
            section_context=cls["_section_context"],
            transcript=cls["_transcript"],
            document_path=cls.get("_document_path"),
        ))
        if not questions:
            raise QuizUnavailable("the quiz came back without usable questions")
        store.update(class_id, _quiz=questions)
        return questions


def forget(class_id: str) -> None:
    with _index_lock:
        _indexes.pop(class_id, None)
