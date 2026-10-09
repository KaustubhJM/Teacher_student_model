// Calls to the Python backend (server/app.py). In development, Vite
// forwards every /api/... request to http://127.0.0.1:8000 (vite.config.js).

async function call(path, options = {}) {
  const res = await fetch(`/api${path}`, options)
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      msg = (await res.json()).detail || msg
    } catch {
      // not JSON — keep the status text
    }
    throw new Error(msg)
  }
  return res.json()
}

const json = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

// Is the backend running? Asked once per page load.
let online = null
export function backendOnline() {
  if (online === null) {
    const ctrl = new AbortController()
    const t = setTimeout(() => ctrl.abort(), 2000)
    online = fetch('/api/health', { signal: ctrl.signal })
      .then((r) => r.ok)
      .catch(() => false)
      .finally(() => clearTimeout(t))
  }
  return online
}

export const api = {
  listClasses: () => call('/classes'),
  getClass: (id) => call(`/classes/${id}`),
  // form = FormData with kind (upload | youtube | text | topic), title, file / url / text / topic
  createClass: (form) => call('/classes', { method: 'POST', body: form }),
  deleteClass: (id) => call(`/classes/${id}`, { method: 'DELETE' }),
  restartClass: (id) => call(`/classes/${id}/restart`, { method: 'POST' }),
  // Waits (up to ~25 s) for beats after the first `have`: { beats: [[sentence, …], …], done }
  beats: (id, section, have) => call(`/classes/${id}/sections/${section}/beats?have=${have}`),
  sectionDone: (id, section) => call(`/classes/${id}/sections/${section}/done`, { method: 'POST' }),
  // sentence = where in the section it was asked (so captions can be rebuilt on resume)
  askDoubt: (id, section, question, sentence) => call(`/classes/${id}/doubts`, json({ section, question, sentence })),
  // Reply to the comprehension check that ended a section → { answer }
  answerCheck: (id, section, question, reply, sentence) =>
    call(`/classes/${id}/check`, json({ section, question, reply, sentence })),
  // Resume point = the NEXT sentence to speak; saved after every sentence.
  saveProgress: (id, section, sentence) => call(`/classes/${id}/progress`, json({ section, sentence })),
  // Captions of what was already taught: { items: [{ kind, s, i, text }, …] }
  history: (id) => call(`/classes/${id}/history`),
  quiz: (id) => call(`/classes/${id}/quiz`, { method: 'POST' }),
  saveScore: (id, got, total) => call(`/classes/${id}/quiz/score`, json({ got, total })),
}
