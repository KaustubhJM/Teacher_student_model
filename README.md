# AI Teaching Classroom — Phase 0

A text-based prototype of the teach → pause → answer → resume loop, built
with LangGraph. This is the foundation you'll later swap text I/O for
voice (STT/TTS) and add the Curious Student agent on top of.

## Why this design

The core mechanic — "stop the teacher mid-explanation, ask a question,
resume exactly where you left off" — is handled by LangGraph's native
`interrupt()` + a checkpointer (`MemorySaver`). The graph literally
pauses execution at the `check_in` node and persists its state; your
terminal input becomes the "resume" signal. This is the *same*
mechanism you'll reuse for voice barge-in later — you'll just replace
`input()` in `main()` with a speech-to-text stream and VAD (voice
activity detection) trigger instead of waiting on stdin.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # https://console.groq.com
python teacher.py "Binary Search Trees"
```

Don't have a key handy? Just run it without one — it falls back to an
offline stub LLM so you can see the pause/resume state machine work
without hitting any API.

## How it works

- `plan_outline` — asks the model to break the topic into 4-6 sections
- `teach_section` — generates and prints the lecture text for the
  current section
- `check_in` — the pause point. Calls `interrupt()`, which halts
  execution and waits for you. Empty input → continue; anything else →
  routed to `answer_question`
- `answer_question` — answers your question using the lecture context,
  then loops back to `check_in` (so you resume the *same* section
  before advancing)
- `advance` — moves to the next section, or ends the graph

## What to build next (Phase 1 / Phase 2)

- **Curious Student agent**: add a second node after `teach_section`
  that runs a cheap classifier prompt ("is this section conceptually
  tricky? yes/no") and, on "yes," generates a probing question from a
  separate "curious student" persona before your own `check_in` pause
- **Voice**: replace `input()` in `main()` with a streaming STT client
  (e.g. Deepgram/AssemblyAI) and TTS for the teacher's spoken output;
  the barge-in event becomes your resume trigger instead of a blank
  Enter press
- **Content ingestion**: replace the topic string with a RAG pipeline
  over an uploaded PDF or video transcript, and have `plan_outline` /
  `teach_section` pull from retrieved chunks instead of the model's own
  knowledge
