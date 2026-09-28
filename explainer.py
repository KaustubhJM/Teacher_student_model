import os
import re
import time
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from teacher import USE_REAL_LLM, call_llm
from teacher_natural import build_teach_prompt
from teacher_voice import VoiceTeachState
from tools import search_document, search_web

EXPLAINER_MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ROUNDS = 3

# How many prior doubt Q/A pairs (from THIS lecture) get replayed as real
# conversation history before the current question. Capped so a lecture
# with many interruptions doesn't slowly grow every subsequent doubt's
# prompt size without bound — 6 is generous for "what did you mean by
# that a minute ago" while staying well clear of the 8000 TPM ceiling
# this same model already hit elsewhere in the project.
MAX_DOUBT_HISTORY_TURNS = 6

# A section's 3 beat calls go out back to back, and together they can
# exceed Groq's 8000 tokens/minute free tier (~9-11k measured on a real
# lecture) even though each call alone is well under it. On a rate-limit
# error, wait the delay Groq suggests and retry — prompts are unchanged.
# A wait longer than MAX_RATE_LIMIT_WAIT_SECONDS (e.g. a per-DAY limit)
# is raised immediately, same as before.
MAX_RATE_LIMIT_RETRIES = 2
MAX_RATE_LIMIT_WAIT_SECONDS = 65

_explainer_llm = None
_explainer_llm_with_tools = None


def _get_explainer_llm():
    global _explainer_llm
    if _explainer_llm is None:
        from langchain_groq import ChatGroq
        _explainer_llm = ChatGroq(model=EXPLAINER_MODEL, temperature=0.4)
    return _explainer_llm


def _get_explainer_llm_with_tools():
    global _explainer_llm_with_tools
    if _explainer_llm_with_tools is None:
        _explainer_llm_with_tools = _get_explainer_llm().bind_tools([search_document, search_web])
    return _explainer_llm_with_tools


def build_grounded_teach_prompt(state: VoiceTeachState, section: str, context_chunks: List[str]) -> str:
    base_prompt = build_teach_prompt(state, section)

    if context_chunks:
        context_block = "\n".join(f"- {c}" for c in context_chunks)
        grounding = (
            "\n\nHere is source material from the student's own document for "
            f"this section:\n{context_block}\n\n"
            "Use this as your PRIMARY grounding — it's the material the "
            "student is actually studying, so stay consistent with it and "
            "teach it in your own words, don't quote it verbatim. But your "
            "job is to deliver a COMPLETE, accurate explanation of this "
            "section, not a strict summary of only these chunks: if this "
            "material is incomplete, has gaps, or you know something "
            "relevant it doesn't cover, fill in from your own knowledge "
            "too. Never cut an explanation short or leave it thin just "
            "because the provided material ran out.\n"
        )
    else:
        grounding = (
            "\n\n(No specific source material was found for this section — "
            "teach it fully from your own knowledge.)\n"
        )

    return base_prompt + grounding


# Each beat is one smaller, separate LLM call rather than one big call
# for the whole section. Two reasons: (1) gpt-oss-120b has the same
# 8000-tokens/minute free-tier ceiling that already crashed the Planner
# — a single long, detailed section risks hitting it the same way; (2)
# splitting genuinely produces MORE total detail than one call would,
# since each beat gets its own dedicated instruction rather than
# competing for room in one "3-5 sentences" prompt.
TEACHING_BEATS = [
    (
        "introduction",
        "Introduce this concept clearly and set up why it matters. "
        "3-5 sentences.",
    ),
    (
        "deep_dive",
        "Now go deeper — explain the mechanism, nuances, and walk "
        "through at least one concrete example in real detail. This is "
        "the most substantial part of the section, take your time: "
        "6-10 sentences.",
    ),
    (
        "connection",
        "Connect this to the bigger picture — how it relates to what's "
        "already been covered, or why it matters going forward — and "
        "end with ONE short, natural comprehension check. 3-5 "
        "sentences.",
    ),
]


def build_beat_prompt(state: VoiceTeachState, section: str, context_chunks: List[str],
                       beat_instruction: str, prior_beats_text: str, is_last_beat: bool) -> str:
    is_first_section = state["idx"] == 0
    is_last_section = state["idx"] == len(state["outline"]) - 1

    prompt = (
        f"You are a warm, experienced human tuition teacher speaking OUT "
        f"LOUD to one student named {state['student_name']}, currently on "
        f"the section '{section}' of a lecture on '{state['topic']}'. "
        f"Prior sections already covered: {state['outline'][:state['idx']]}.\n\n"
    )

    if prior_beats_text:
        prompt += (
            "You've already said this so far in THIS section — continue "
            f"naturally from here, don't repeat it:\n{prior_beats_text}\n\n"
        )

    prompt += (
        f"Right now: {beat_instruction}\n\n"
        "Write what you would actually SAY, not an essay: conversational "
        "spoken phrasing, contractions, natural rhythm — not bullet "
        f"points or written prose. Address {state['student_name']} by "
        "name naturally if it fits, but don't force it into every beat.\n"
    )

    if context_chunks:
        context_block = "\n".join(f"- {c}" for c in context_chunks)
        prompt += (
            f"\nSource material for this section (teach it in your own "
            "words, don't quote verbatim; fill any gaps with your own "
            f"knowledge — deliver COMPLETE, accurate content, not a "
            f"strict summary of only this):\n{context_block}\n"
        )

    if not prior_beats_text and is_first_section:
        prompt += "\nThis is the opening section of the whole lecture, so ease into it.\n"
    if is_last_beat and is_last_section:
        prompt += "\nThis is the last section, so give it a sense of wrapping up.\n"

    return prompt


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return ("ratelimit" in type(e).__name__.lower() or "error code: 429" in msg
            or "rate limit" in msg or "rate_limit" in msg)


def _suggested_wait_seconds(e: Exception) -> Optional[float]:
    # Groq's message: "Please try again in 7.66s" or "... in 1m2.5s"
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def _call_llm_with_rate_limit_retry(prompt: str) -> str:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return call_llm(prompt)
        except Exception as e:
            if attempt == MAX_RATE_LIMIT_RETRIES or not _is_rate_limit(e):
                raise
            wait = _suggested_wait_seconds(e) or 30
            if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise
            print(f"(Groq rate-limited — waiting {wait:.0f}s before continuing...)")
            time.sleep(wait + 1)


def iter_section_beats(state: VoiceTeachState, section: str, context_chunks: List[str]):
    """Yields a section's beats one at a time, in order — each beat is a
    separate, smaller LLM call that sees the beats before it."""
    beats_text: List[str] = []
    num_beats = len(TEACHING_BEATS)
    for i, (_, beat_instruction) in enumerate(TEACHING_BEATS):
        prior_beats_text = "\n\n".join(beats_text)
        prompt = build_beat_prompt(
            state, section, context_chunks, beat_instruction, prior_beats_text,
            is_last_beat=(i == num_beats - 1),
        )
        text = _call_llm_with_rate_limit_retry(prompt)
        beats_text.append(text)
        yield text


def teach_section_in_beats(state: VoiceTeachState, section: str, context_chunks: List[str]) -> List[str]:
    """Generates a section's teaching content as several shorter,
    sequential LLM calls (beats) instead of one large call. Returns the
    list of beat texts in order, ready to be spoken/interrupt-checked
    one at a time by the caller."""
    return list(iter_section_beats(state, section, context_chunks))


class BeatStream:
    """Generates a section's beats in a background thread so the teacher
    can start speaking beat 1 while beats 2-3 are still being generated.

    Why: measured on DBMS_U_5_ONE_SHOT.pdf, a dense section's 3 beat calls
    (~2,300-2,750 words each) exceed Groq's 8000 tokens/min, so beat 3
    waited ~37s on the rate limit — and all 3 were generated before any
    speech, i.e. ~45s of silence per section. Now that wait overlaps with
    beat 1/2 being spoken. Prompts and content are identical to
    teach_section_in_beats().

    Iterate it to get beats in order (blocks until each is ready); an
    error in generation is re-raised at the beat it happened.
    all_beats() waits for the rest and returns every beat — used for the
    transcript, so it records all beats exactly as before."""

    def __init__(self, state: VoiceTeachState, section: str, context_chunks: List[str]):
        import queue
        import threading
        self._queue = queue.Queue()
        self._beats: List[str] = []
        self._finished = False
        self._thread = threading.Thread(
            target=self._run, args=(dict(state), section, list(context_chunks)), daemon=True,
        )
        self._thread.start()

    def _run(self, state, section, context_chunks):
        try:
            for text in iter_section_beats(state, section, context_chunks):
                self._queue.put(("beat", text))
            self._queue.put(("done", None))
        except BaseException as e:  # surfaced to the caller at that beat
            self._queue.put(("error", e))

    def _next(self) -> Optional[str]:
        if self._finished:
            return None
        kind, value = self._queue.get()
        if kind == "beat":
            self._beats.append(value)
            return value
        self._finished = True
        if kind == "error":
            raise value
        return None

    def __iter__(self):
        while True:
            beat = self._next()
            if beat is None:
                return
            yield beat

    def all_beats(self) -> List[str]:
        while self._next() is not None:
            pass
        return list(self._beats)


def _doubt_system_prompt(state: VoiceTeachState, section: str) -> str:
    return (
        f"You are a warm, knowledgeable tuition teacher speaking OUT LOUD "
        f"to {state['student_name']}, currently teaching a lecture on "
        f"'{state['topic']}', specifically the section '{section}'. "
        f"{state['student_name']} just said something — it could be a "
        "genuine question, an answer to a comprehension check you posed, "
        "or just a comment.\n\n"
        "If this student has asked you anything earlier in this same "
        "lecture, those exchanges are included below as real prior "
        "conversation turns — use them to resolve references like "
        "'that', 'what you just said', or 'going back to my earlier "
        "question', the way a tutor who's been in the room the whole "
        "time naturally would. Don't ask them to repeat something they "
        "already told you.\n\n"
        "You have two tools available:\n"
        "- search_document: checks the student's own course document for "
        "relevant material. Prefer this when their question is plausibly "
        "covered by their course material, even if it's about a different "
        "part of the document than the current section.\n"
        "- search_web: checks the web for information the document likely "
        "doesn't cover — current events, broader context, or anything "
        "clearly outside the document's scope.\n\n"
        "Use a tool if it would genuinely improve your answer. Don't use "
        "one if you already know the answer confidently and it's not "
        "something the document would specifically cover — general "
        "knowledge questions don't need tools. You may use both tools, one, "
        "or neither, and you may call a tool more than once if the first "
        "result wasn't enough.\n\n"
        "Once you're ready, respond the way a real teacher would: if they "
        "answered a comprehension check, tell them clearly whether it's "
        "right and add insight; if it's a genuine question, answer it "
        "directly and completely; if it's a comment, acknowledge and build "
        "on it. Respond in 2-5 conversational sentences, addressing "
        f"{state['student_name']} naturally. Do not say goodbye or wrap up "
        "— you're about to keep teaching."
    )


def answer_doubt(state: VoiceTeachState, section: str, student_input: str,
                  doubt_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Answers one doubt, with real memory of earlier doubts from this
    same lecture. doubt_history is a list of {"question", "answer"}
    dicts in chronological order — the caller (classroom.py) owns and
    accumulates this across the whole lecture, the same way it already
    accumulates transcript."""
    if not USE_REAL_LLM:
        return "[offline stub — set GROQ_API_KEY for real tool-calling]"

    doubt_history = doubt_history or []

    llm = _get_explainer_llm_with_tools()
    tool_map = {"search_document": search_document, "search_web": search_web}
    messages = [SystemMessage(content=_doubt_system_prompt(state, section))]

    for turn in doubt_history[-MAX_DOUBT_HISTORY_TURNS:]:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))

    messages.append(HumanMessage(content=student_input))

    for _ in range(MAX_TOOL_ROUNDS):
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls:
            fn = tool_map.get(call["name"])
            result = fn.invoke(call["args"]) if fn else f"Unknown tool: {call['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final = _get_explainer_llm().invoke(messages)
    return final.content