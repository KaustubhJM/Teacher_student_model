"""
history.py — quick CLI to see a student's past classroom sessions.

Run:
    python history.py Kaustubh
"""

import sys
import time
import persistence


def main():
    if len(sys.argv) < 2:
        print("Usage: python history.py <student_name>")
        sys.exit(1)

    student_name = sys.argv[1]
    rows = persistence.list_history(student_name)

    if not rows:
        print(f"No sessions found for {student_name}.")
        return

    print(f"\n=== Sessions for {student_name} ===\n")
    for r in rows:
        last_touched = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["updated_at"]))
        status = r["status"]
        note = f" — {r['quiz_note']}" if r["quiz_note"] else ""
        print(f"[{status:11}] {r['topic']:40} last touched {last_touched}{note}")


if __name__ == "__main__":
    main()