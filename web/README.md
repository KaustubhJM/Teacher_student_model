# AI Classroom — web frontend

React + Vite. Runs on **mock data** for now (no Python backend yet).

## Run it

```
cd web
npm install      # first time only
npm run dev      # → http://localhost:5173
```

`.npmrc` points npm at `registry.yarnpkg.com` (an official mirror) because
`registry.npmjs.org` is blocked on this network (connection reset).

Best in **Edge** (it has the Microsoft "Ava" neural voice) or Chrome
(voice doubts use the browser's speech recogniser — Chrome/Edge only).

## How it's organised

| File | What it is |
|---|---|
| `src/main.jsx`, `src/App.jsx` | Entry point and URL → page routing |
| `src/index.css` | All styling. Colours/sizes are tokens at the top (`--bg`, `--accent`…) |
| `src/pages/Home.jsx` | Home: "Continue where you left off" + "Recent classes" grid |
| `src/pages/ClassPage.jsx` | Class page: Sources (left), Lecture (centre), Teacher (right) |
| `src/components/NewClassDialog.jsx` | "Start a new class" (upload / YouTube / pasted text / topic) |
| `src/components/LecturePanel.jsx` | Live captions, doubt bubbles, type-or-talk bar |
| `src/components/TeacherPanel.jsx` | Teacher, status, play/pause/mute/raise-hand, quiz card |
| `src/components/TeacherAvatar.jsx` | The animated teacher (SVG); mouth driven by spoken words |
| `src/components/QuizView.jsx` | Post-lecture quiz |
| `src/lib/useLecture.js` | Lecture engine — browser twin of `classroom.teach_and_listen` |
| `src/lib/speech.js` | Browser text-to-speech + speech-to-text (preview only) |
| `src/data/mockData.js` | **Mock** classes, lecture script, quiz — replaced by API calls later |
| `src/data/classStore.js` | Class list (mock + ones you create, kept in localStorage) |

## What's mock vs real

Mock today → real later (backend phase):
- class list / outline / lecture text → FastAPI over the existing planner + explainer
- doubt answers → `explainer.answer_doubt`
- voice → edge-tts MP3s from the server; mouth from audio loudness (`mouthRef.current.level`)
- quiz → `quiz.py`
