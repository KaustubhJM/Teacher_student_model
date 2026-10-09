"""FastAPI app for the website. Run from the project root:

    uvicorn server.app:app --port 8000

(the Vite dev server in web/ forwards /api/* here)."""
import os
import re
import shutil
import sys

# The rag caches (.rag_cache/.ocr_cache/.vision_cache) are relative paths —
# share them with the CLI by always working from the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from contextlib import asynccontextmanager  # noqa: E402
from typing import Optional  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from teacher import USE_REAL_LLM  # noqa: E402
from quiz import USE_REAL_GEMINI, QuizUnavailable  # noqa: E402
from rag.vector_index import preload_model_in_background  # noqa: E402
from rag.reranker import preload_reranker_in_background  # noqa: E402

from . import engine, store  # noqa: E402

# File types rag/ingest.py understands (same list as the website's picker).
ALLOWED_EXTS = {".pdf", ".docx", ".pptx", ".md", ".txt", ".csv", ".xlsx", ".html", ".json",
                ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_MB = 100
YT_RE = re.compile(r"^(https?://)?([\w-]+\.)?(youtube\.com|youtu\.be)/\S+$", re.I)

@asynccontextmanager
async def _lifespan(app):
    # Same as classroom.main(): load the embedding model and reranker in the
    # background so the first plan/doubt doesn't wait for them.
    preload_model_in_background()
    preload_reranker_in_background()
    yield


app = FastAPI(title="AI Classroom backend", lifespan=_lifespan)


def _get_or_404(class_id: str) -> dict:
    cls = store.get(class_id)
    if cls is None:
        raise HTTPException(404, "Class not found")
    return cls


def _require_ready(cls: dict) -> None:
    if cls["status"] in ("planning", "failed"):
        raise HTTPException(409, f"Class is {cls['status']}")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    # This server only answers /api/...; the website itself is Vite's.
    return ("<body style='font-family:sans-serif;background:#131314;color:#e3e3e3;padding:40px'>"
            "<h2>AI Classroom backend is running ✅</h2>"
            "<p>This is the API only. Open the website at "
            "<a style='color:#a8c7fa' href='http://localhost:5173'>http://localhost:5173</a> "
            "(start it with <code>cd web &amp;&amp; npm run dev</code>).</p></body>")


@app.get("/api/health")
def health():
    return {"ok": True, "llm": USE_REAL_LLM, "gemini": USE_REAL_GEMINI}


@app.get("/api/classes")
def list_classes():
    return [store.public(c) for c in store.list_all()]


@app.get("/api/classes/{class_id}")
def get_class(class_id: str):
    return store.public(_get_or_404(class_id))


def _safe_name(name: str) -> str:
    name = os.path.basename(name or "upload")
    return re.sub(r"[^\w.\- ()]+", "_", name).strip() or "upload"


@app.post("/api/classes")
def create_class(kind: str = Form(...), title: str = Form(""), url: str = Form(""),
                 text: str = Form(""), topic: str = Form(""),
                 student_name: str = Form(engine.STUDENT_NAME),
                 file: Optional[UploadFile] = File(None)):
    title = title.strip()
    document_path = None

    if kind == "upload":
        if file is None or not file.filename:
            raise HTTPException(400, "Choose a file first.")
        name = _safe_name(file.filename)
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(400, f"Unsupported file type '{ext}'.")
        source = {"type": "pdf" if ext == ".pdf" else "text", "name": name, "detail": ""}
        default_title = engine.derive_title("upload", name)
    elif kind == "youtube":
        url = url.strip()
        if not YT_RE.match(url):
            raise HTTPException(400, "Paste a YouTube link, like https://youtu.be/…")
        document_path = url
        source = {"type": "youtube", "name": url, "detail": "Transcript"}
        default_title = "YouTube lesson"   # replaced by the real video title while planning
    elif kind == "text":
        words = len(text.split())
        if words < 20:
            raise HTTPException(400, "Paste at least a paragraph of text.")
        source = {"type": "text", "name": "Pasted text", "detail": f"{words} words"}
        default_title = engine.derive_title("text", text)
    elif kind == "topic":
        if not topic.strip():
            raise HTTPException(400, "Type a topic to learn.")
        source = {"type": "topic", "name": topic.strip(),
                  "detail": "No document — taught from general knowledge"}
        default_title = topic.strip()
    else:
        raise HTTPException(400, f"Unknown source kind '{kind}'.")

    cls = engine.new_class(title or default_title, [source], document_path,
                           student_name=student_name.strip() or engine.STUDENT_NAME,
                           title_from_source=not title)

    # Files are saved under server/data/uploads/<class id>/ and ingested from there.
    if kind in ("upload", "text"):
        folder = os.path.join(store.UPLOAD_DIR, cls["id"])
        os.makedirs(folder, exist_ok=True)
        if kind == "upload":
            document_path = os.path.join(folder, source["name"])
            with open(document_path, "wb") as out:
                shutil.copyfileobj(file.file, out, length=1024 * 1024)
            if os.path.getsize(document_path) > MAX_UPLOAD_MB * 1024 * 1024:
                shutil.rmtree(folder, ignore_errors=True)
                store.delete(cls["id"])
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
        else:
            document_path = os.path.join(folder, "pasted_text.txt")
            with open(document_path, "w", encoding="utf-8") as out:
                out.write(text)
        cls = store.update(cls["id"], _document_path=document_path)

    engine.start_planning(cls["id"])
    return store.public(cls)


@app.delete("/api/classes/{class_id}")
def delete_class(class_id: str):
    _get_or_404(class_id)
    store.delete(class_id)
    engine.forget(class_id)
    shutil.rmtree(os.path.join(store.UPLOAD_DIR, class_id), ignore_errors=True)
    return {"ok": True}


@app.post("/api/classes/{class_id}/restart")
def restart_class(class_id: str):
    """Replay from section 1: progress, transcript and doubts reset; the
    plan, generated beats and quiz are kept (no tokens spent again)."""
    cls = _get_or_404(class_id)
    _require_ready(cls)
    cls = store.update(class_id, sectionsDone=0, status="new", quizScore=None,
                       position={"s": 0, "i": 0}, _transcript=[], _doubt_history=[],
                       _doubt_log=[], _exchanges={})
    return store.public(cls)


class Progress(BaseModel):
    section: int
    sentence: int   # index of the NEXT sentence to speak in that section


@app.post("/api/classes/{class_id}/progress")
def save_progress(class_id: str, p: Progress):
    cls = _get_or_404(class_id)
    _require_ready(cls)
    return store.public(engine.save_progress(class_id, max(0, p.section), max(0, p.sentence)))


@app.get("/api/classes/{class_id}/history")
def class_history(class_id: str):
    """Captions of what was already taught (for resuming on the website)."""
    cls = _get_or_404(class_id)
    _require_ready(cls)
    return {"items": engine.history(class_id)}


@app.get("/api/classes/{class_id}/sections/{idx}/beats")
def section_beats(class_id: str, idx: int, have: int = 0):
    cls = _get_or_404(class_id)
    _require_ready(cls)
    if not 0 <= idx < len(cls["outline"]):
        raise HTTPException(404, "No such section")
    try:
        return engine.get_beats(class_id, idx, max(0, have))
    except engine.TeachingError as e:
        raise HTTPException(502, f"Couldn't prepare this part of the lecture: {e}")


@app.post("/api/classes/{class_id}/sections/{idx}/done")
def section_done(class_id: str, idx: int):
    cls = _get_or_404(class_id)
    _require_ready(cls)
    if not 0 <= idx < len(cls["outline"]):
        raise HTTPException(404, "No such section")
    return store.public(engine.section_done(class_id, idx))


class Doubt(BaseModel):
    section: int
    question: str
    sentence: int = 0   # where in the section it was asked (for captions on resume)


@app.post("/api/classes/{class_id}/doubts")
def ask_doubt(class_id: str, doubt: Doubt):
    cls = _get_or_404(class_id)
    _require_ready(cls)
    if not doubt.question.strip():
        raise HTTPException(400, "Empty question")
    return {"answer": engine.ask_doubt(class_id, doubt.section, doubt.question.strip(),
                                       sentence=max(0, doubt.sentence))}


class CheckReply(BaseModel):
    section: int
    question: str     # the comprehension check the teacher ended the section with
    reply: str        # what the student typed/said (anything, incl. "I don't know")
    sentence: int = 0


@app.post("/api/classes/{class_id}/check")
def answer_check(class_id: str, body: CheckReply):
    """The student's reply to the end-of-section comprehension check: the
    teacher judges it and explains the right answer if needed."""
    cls = _get_or_404(class_id)
    _require_ready(cls)
    if not body.reply.strip():
        raise HTTPException(400, "Empty reply")
    return {"answer": engine.ask_doubt(class_id, body.section, body.reply.strip(),
                                       sentence=max(0, body.sentence),
                                       check_question=body.question.strip() or None)}


@app.post("/api/classes/{class_id}/quiz")
def get_quiz(class_id: str):
    cls = _get_or_404(class_id)
    _require_ready(cls)
    try:
        return {"questions": engine.get_quiz(class_id)}
    except QuizUnavailable as e:
        raise HTTPException(503, f"The quiz models are out of quota right now — try again later. ({e})")


class Score(BaseModel):
    got: int
    total: int


@app.post("/api/classes/{class_id}/quiz/score")
def save_score(class_id: str, score: Score):
    _get_or_404(class_id)
    cls = store.update(class_id, quizScore={"got": score.got, "total": score.total})
    return store.public(cls)
