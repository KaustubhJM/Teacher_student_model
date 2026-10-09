"""The comprehension check that ends every section (explainer's
"connection" beat: "end with ONE short, natural comprehension check").

Shared by the CLI (classroom.teach_and_listen) and the website backend
(server/engine.py), so both treat the student's reply the same way:

- find_check_question: which sentence of the last beat is the question.
  Measured 2026-10-10 on the saved DBMS + TC lectures: 11/11 sections end
  with exactly one, always among the last 3 sentences (once followed by
  a short "take a second to think" line).
- check_answer_input: what answer_doubt receives for the reply. Without
  the question itself the teacher can't judge the reply, and "I don't
  know" gives it nothing to explain.
- is_skip_reply: the student said only "skip" / "next" / "move on"."""
import re
from typing import List, Optional

SKIP_LINE = "No problem — let's move on."

# Whole-reply matches only (after lower-casing, dropping punctuation and
# apostrophes), so an answer that merely contains "next" is never skipped.
_SKIP_REPLIES = {
    "skip", "skip it", "skip this", "skip question", "skip the question", "skip this question",
    "next", "next question", "pass", "move on", "lets move on",
}


def find_check_question(sentences: List[str]) -> Optional[str]:
    """The last sentence ending in '?' among a beat's last 3 sentences."""
    for sentence in reversed(sentences[-3:]):
        if sentence.strip().endswith("?"):
            return sentence.strip()
    return None


def check_answer_input(check_question: str, reply: str) -> str:
    """The student's reply to the section's comprehension check, as given
    to answer_doubt: the exact question, how to respond, then the reply."""
    return (
        f'[Context: you ended this section with the comprehension check: '
        f'"{check_question}". The student is replying to it below. Tell them '
        "clearly whether the answer is right. If it's wrong, only partly right, "
        "or they say they don't know (or reply with something unrelated), explain "
        "the correct answer clearly and simply before we move on to the next part.]\n"
        f"{reply}"
    )


def remembered_check_reply(check_question: str, reply: str) -> str:
    """Short form kept in doubt memory + transcript (so later doubts and
    the quiz know which question this answered)."""
    return f'(my answer to your check "{check_question}") {reply}'


def is_skip_reply(reply: str) -> bool:
    text = reply.lower().replace("'", "").replace("’", "")
    text = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    return text in _SKIP_REPLIES
