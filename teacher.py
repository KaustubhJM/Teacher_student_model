"""
Phase 0: AI Teaching Classroom — text-based teach/pause/resume loop.

One agent (the Teacher) explains a topic section by section. After every
section you get a chance to ask a question in the terminal — hit Enter to
just continue, or type a question to get an answer before the lecture
resumes exactly where it left off.

This uses LangGraph's native `interrupt()` + checkpointer, which is the
same mechanism you'd later reuse for real voice barge-in: the graph
literally pauses execution and waits for you, instead of you faking it
with a while-loop.

Run:
    export GROQ_API_KEY=your_key_here
    python teacher.py "Binary Search Trees"

If GROQ_API_KEY is not set, a fake offline LLM is used instead so you can
see the state machine work end to end without hitting any API.
"""

import os
import sys
import json
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

# Load the variables from the .env file into the system environment
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ---------------------------------------------------------------------------
# LLM setup — real Groq model if a key is present, otherwise an offline stub
# so the flow is runnable and inspectable without any network access.
# ---------------------------------------------------------------------------

USE_REAL_LLM = bool(os.environ.get("GROQ_API_KEY"))

if USE_REAL_LLM:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

    def call_llm(prompt: str) -> str:
        return llm.invoke(prompt).content
else:
    def call_llm(prompt: str) -> str:
        """Offline stand-in so you can test the graph logic with no API key."""
        if "Return ONLY a JSON list" in prompt:
            return json.dumps([
                "What the topic is and why it matters",
                "Core mechanism / how it actually works",
                "A worked example",
                "Common pitfalls and edge cases",
            ])
        if "as a PhD-level teacher" in prompt:
            return (
                "[offline stub lecture text — set GROQ_API_KEY to get real "
                "explanations from the model]"
            )
        return "[offline stub answer — set GROQ_API_KEY for a real response]"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TeachState(TypedDict):
    topic: str
    outline: List[str]
    idx: int
    transcript: List[str]
    _pending_question: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan_outline(state: TeachState) -> TeachState:
    prompt = (
        f"You are planning a short lecture on '{state['topic']}'.\n"
        "Break it into 4-6 section titles, simplest concept first, building "
        "up. Return ONLY a JSON list of section title strings, nothing else."
    )
    raw = call_llm(prompt)
    try:
        outline = json.loads(raw)
    except json.JSONDecodeError:
        outline = [line.strip("- ") for line in raw.splitlines() if line.strip()]
    print(f"\n=== Lecture plan: {state['topic']} ===")
    for i, section in enumerate(outline, 1):
        print(f"  {i}. {section}")
    print()
    return {"outline": outline, "idx": 0, "transcript": []}


def teach_section(state: TeachState) -> TeachState:
    section = state["outline"][state["idx"]]
    prompt = (
        f"You are teaching, as a PhD-level teacher, the section "
        f"'{section}' of a lecture on '{state['topic']}'. "
        "Explain it clearly in 3-5 sentences, building on the prior "
        f"sections: {state['outline'][:state['idx']]}. "
        "Do not repeat the section title back."
    )
    text = call_llm(prompt)
    print(f"--- Section {state['idx'] + 1}/{len(state['outline'])}: {section} ---")
    print(text)
    return {"transcript": state["transcript"] + [f"[TEACHER] {section}: {text}"]}


def check_in(state: TeachState) -> Command:
    """The pause point. Real barge-in later plugs in here instead of input()."""
    question = interrupt(
        "Press Enter to continue, or type a question: "
    )
    if question and question.strip():
        return Command(goto="answer_question", update={"_pending_question": question})
    return Command(goto="advance", update={})


def answer_question(state: TeachState) -> TeachState:
    # The question text arrives via the Command sent on resume — see main().
    question = state.get("_pending_question", "")
    prompt = (
        f"The student just interrupted your lecture on '{state['topic']}' "
        f"(currently on section '{state['outline'][state['idx']]}') to ask: "
        f"\"{question}\". Answer directly and briefly, then they'll resume "
        "the lecture."
    )
    answer = call_llm(prompt)
    print(f"\n[YOU ASKED] {question}")
    print(f"[TEACHER] {answer}\n")
    print("--- resuming lecture ---")
    return {"transcript": state["transcript"] + [f"[Q] {question}", f"[A] {answer}"]}


def advance(state: TeachState) -> TeachState:
    return {"idx": state["idx"] + 1}


def route_after_advance(state: TeachState) -> str:
    return "teach_section" if state["idx"] < len(state["outline"]) else END


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(TeachState)
    g.add_node("plan_outline", plan_outline)
    g.add_node("teach_section", teach_section)
    g.add_node("check_in", check_in)
    g.add_node("answer_question", answer_question)
    g.add_node("advance", advance)

    g.set_entry_point("plan_outline")
    g.add_edge("plan_outline", "teach_section")
    g.add_edge("teach_section", "check_in")
    # check_in returns a Command that routes directly to answer_question or advance
    g.add_edge("answer_question", "check_in")
    g.add_conditional_edges("advance", route_after_advance, ["teach_section", END])

    return g.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Main loop — drives the interrupt/resume cycle from the terminal
# ---------------------------------------------------------------------------

def main():
    topic = " ".join(sys.argv[1:]) or "Binary Search Trees"
    graph = build_graph()
    config = {"configurable": {"thread_id": "session-1"}}

    if not USE_REAL_LLM:
        print("(No GROQ_API_KEY found — running with an offline stub LLM. "
              "Set GROQ_API_KEY to get real lecture content.)")

    state_input = {"topic": topic}

    while True:
        result = graph.invoke(state_input, config=config)

        # If the graph is paused at an interrupt, ask the user for input
        # and resume the graph with their answer.
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break  # graph finished (hit END)

        try:
            user_reply = input("> ").strip()
        except EOFError:
            break

        state_input = Command(resume=user_reply)

    print("\n=== Lecture complete ===")


if __name__ == "__main__":
    main()
