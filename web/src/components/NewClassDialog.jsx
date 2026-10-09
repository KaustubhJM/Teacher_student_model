import { useRef, useState } from 'react'
import Icon from './Icon.jsx'

// File types the Python ingester (rag/ingest.py) understands.
const ACCEPT = '.pdf,.docx,.pptx,.md,.txt,.csv,.xlsx,.html,.json,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff'
const YT_RE = /^(https?:\/\/)?([\w-]+\.)?(youtube\.com|youtu\.be)\/\S+/i

const TABS = [
  { id: 'upload', icon: 'upload', label: 'Upload files' },
  { id: 'youtube', icon: 'smart_display', label: 'YouTube' },
  { id: 'text', icon: 'content_paste', label: 'Copied text' },
  { id: 'topic', icon: 'lightbulb', label: 'Just a topic' },
]

// Same rule as classroom._derive_topic_from_filename: "DBMS_U_5.pdf" → "Dbms U 5".
function titleFromFile(name) {
  return name
    .replace(/\.[^.]+$/, '')
    .replace(/[_-]+/g, ' ')
    .trim()
    .replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase())
}

export default function NewClassDialog({ onClose, onCreate }) {
  const [tab, setTab] = useState('upload')
  const [file, setFile] = useState(null)
  const [url, setUrl] = useState('')
  const [text, setText] = useState('')
  const [topic, setTopic] = useState('')
  const [title, setTitle] = useState('')
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const fileInput = useRef(null)

  const pickFile = (f) => {
    if (!f) return
    setFile(f)
    setError('')
  }

  const submit = () => {
    let source = null
    let fallbackTitle = ''
    if (tab === 'upload') {
      if (!file) return setError('Choose a file first.')
      const ext = file.name.split('.').pop().toLowerCase()
      source = { type: ext === 'pdf' ? 'pdf' : 'text', name: file.name, detail: `${(file.size / 1e6).toFixed(1)} MB` }
      fallbackTitle = titleFromFile(file.name)
    } else if (tab === 'youtube') {
      if (!YT_RE.test(url.trim())) return setError('Paste a YouTube link, like https://youtu.be/…')
      source = { type: 'youtube', name: url.trim(), detail: 'Transcript' }
      fallbackTitle = 'YouTube lesson'
    } else if (tab === 'text') {
      if (text.trim().split(/\s+/).length < 20) return setError('Paste at least a paragraph of text.')
      source = { type: 'text', name: 'Pasted text', detail: `${text.trim().split(/\s+/).length} words` }
      fallbackTitle = text.trim().split(/\s+/).slice(0, 6).join(' ')
    } else {
      if (!topic.trim()) return setError('Type a topic to learn.')
      source = { type: 'topic', name: topic.trim(), detail: 'No document — taught from general knowledge' }
      fallbackTitle = topic.trim()
    }
    setBusy(true)
    // The backend can still refuse (e.g. unsupported file) — show its message here.
    Promise.resolve(onCreate({ kind: tab, title: title.trim(), fallbackTitle, source, file, url: url.trim(), text, topic: topic.trim() }))
      .catch((e) => setError(e.message))
      .finally(() => setBusy(false))
  }

  return (
    <div className="overlay" onClick={onClose}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label="Start a new class" onClick={(e) => e.stopPropagation()}>
        <button className="icon-btn close" onClick={onClose} aria-label="Close">
          <Icon name="close" />
        </button>
        <h2>Start a new class from</h2>
        <div className="sub">your own study material</div>

        <div className="source-tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              className={`source-tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => {
                setTab(t.id)
                setError('')
              }}
            >
              <Icon name={t.icon} /> {t.label}
            </button>
          ))}
        </div>

        {tab === 'upload' &&
          (file ? (
            <div className="chosen-file">
              <Icon name="description" />
              <span style={{ flex: 1 }}>{file.name}</span>
              <button className="icon-btn" onClick={() => setFile(null)} aria-label="Remove file">
                <Icon name="close" />
              </button>
            </div>
          ) : (
            <div
              className={`dropzone ${drag ? 'drag' : ''}`}
              onClick={() => fileInput.current.click()}
              onDragOver={(e) => {
                e.preventDefault()
                setDrag(true)
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDrag(false)
                pickFile(e.dataTransfer.files[0])
              }}
            >
              <Icon name="upload" />
              <div>Drag and drop a file here, or click to choose one</div>
              <div className="hint">PDF, Word, PowerPoint, Excel, CSV, text, Markdown, HTML, JSON or an image</div>
              <input ref={fileInput} type="file" accept={ACCEPT} hidden onChange={(e) => pickFile(e.target.files[0])} />
            </div>
          ))}

        {tab === 'youtube' && (
          <div className="field">
            <label htmlFor="yt">YouTube link</label>
            <input id="yt" className="input" placeholder="https://www.youtube.com/watch?v=…" value={url} onChange={(e) => { setUrl(e.target.value); setError('') }} />
          </div>
        )}

        {tab === 'text' && (
          <div className="field">
            <label htmlFor="txt">Paste your notes</label>
            <textarea id="txt" className="textarea" placeholder="Paste notes, an article or a chapter…" value={text} onChange={(e) => { setText(e.target.value); setError('') }} />
          </div>
        )}

        {tab === 'topic' && (
          <div className="field">
            <label htmlFor="topic">What do you want to learn?</label>
            <input id="topic" className="input" placeholder="Backpropagation in neural networks" value={topic} onChange={(e) => { setTopic(e.target.value); setError('') }} />
          </div>
        )}

        <div className="field">
          <label htmlFor="title">Class title (optional)</label>
          <input id="title" className="input" placeholder="Taken from the file name if empty" value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>

        {error && <p className="field-error" style={{ marginTop: 12 }}>{error}</p>}

        <div className="dialog-actions">
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" onClick={submit} disabled={busy}>
            <Icon name="school" /> {busy ? 'Uploading…' : 'Start class'}
          </button>
        </div>
      </div>
    </div>
  )
}
