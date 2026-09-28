"""
plan_demo.py — try the Planner agent on one of your own documents
before we wire it into the teaching graph.

Run:
    python plan_demo.py path/to/notes.pdf "Newton's Laws of Motion"
    python plan_demo.py path/to/notes.pdf                              # topic derived from filename
"""

import sys
import os
from planner import plan_lecture, USE_REAL_LLM, PLANNER_MODEL


def _derive_topic_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("_", " ").replace("-", " ").title()


def main():
    if len(sys.argv) < 2:
        print('Usage: python plan_demo.py <document_path> ["optional topic override"]')
        sys.exit(1)

    document_path = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else _derive_topic_from_filename(document_path)

    if not USE_REAL_LLM:
        print("(No GROQ_API_KEY found — running with an offline stub LLM.)")
    else:
        print(f"(Planner using {PLANNER_MODEL})")

    print(f"Planning a lecture on '{topic}' from {document_path}...\n")
    try:
        plan = plan_lecture(topic, document_path)
    except RuntimeError as e:
        print(f"[plan_demo error] {e}")
        sys.exit(1)

    print(f"=== Outline ({len(plan['outline'])} sections) ===")
    for i, section in enumerate(plan["outline"], 1):
        print(f"  {i}. {section}")

    print("\n=== Per-section source material ===")
    for section in plan["outline"]:
        chunks = plan["section_context"][section]
        print(f"\n[{section}]")
        if chunks:
            for i, chunk in enumerate(chunks, 1):
                print(f"  ({i}) {chunk[:150]}{'...' if len(chunk) > 150 else ''}")
        else:
            print("  (no matching source material — Explainer would fall back to web search / general knowledge)")


if __name__ == "__main__":
    main()
