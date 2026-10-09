// Where the list of classes comes from.
// - Backend running (uvicorn server.app:app): real classes from the API.
// - Backend not running: PREVIEW mode — the mock list plus classes created
//   in this browser (localStorage). Every preview class has preview: true,
//   so the pages can label it "not generated from your source".
import { MOCK_CLASSES } from './mockData.js'
import { api, backendOnline } from './api.js'

const KEY = 'classroom.createdClasses'

function readCreated() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '[]')
  } catch {
    return []
  }
}

function writeCreated(list) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list))
  } catch {
    // Storage blocked (private window etc.) — the class just won't survive a reload.
  }
}

const previewList = () => [...readCreated(), ...MOCK_CLASSES].map((c) => ({ ...c, preview: true }))

export { backendOnline }

export async function listClasses() {
  return (await backendOnline()) ? api.listClasses() : previewList()
}

export async function getClass(id) {
  if (await backendOnline()) {
    try {
      return await api.getClass(id)
    } catch {
      return null
    }
  }
  return previewList().find((c) => c.id === id) || null
}

const EMOJIS = ['📘', '🧪', '🧮', '🌍', '⚙️', '🎨', '🧬', '💡']

// draft = { kind, title, fallbackTitle, source, file, url, text, topic } from NewClassDialog
export async function createClass(draft) {
  if (await backendOnline()) {
    const form = new FormData()
    form.append('kind', draft.kind)
    form.append('title', draft.title) // empty → the server picks one (file name / video title / …)
    if (draft.kind === 'upload') form.append('file', draft.file)
    if (draft.kind === 'youtube') form.append('url', draft.url)
    if (draft.kind === 'text') form.append('text', draft.text)
    if (draft.kind === 'topic') form.append('topic', draft.topic)
    return api.createClass(form)
  }
  const cls = {
    id: `class-${Date.now()}`,
    title: draft.title || draft.fallbackTitle,
    emoji: EMOJIS[Math.floor(Math.random() * EMOJIS.length)],
    createdAt: new Date().toISOString().slice(0, 10),
    sources: draft.source ? [draft.source] : [],
    // The real planner makes 4–7 sections from the document.
    outline: ['Introduction', 'Core ideas', 'Worked example', 'How it all connects'],
    sectionsDone: 0,
    status: 'new',
    quizScore: null,
  }
  writeCreated([cls, ...readCreated()])
  return { ...cls, preview: true }
}

export async function deleteClass(cls) {
  if (!cls.preview) return api.deleteClass(cls.id)
  writeCreated(readCreated().filter((c) => c.id !== cls.id))
}

export function formatDate(iso) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}
