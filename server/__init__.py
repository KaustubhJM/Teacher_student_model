"""Web backend for the classroom website (web/). Run from the project root:

    uvicorn server.app:app --port 8000

Text-only: the browser does the voice (speech synthesis + recognition).
Reuses rag / planner / explainer / quiz unchanged; the CLI classroom.py
and its databases are not touched."""
