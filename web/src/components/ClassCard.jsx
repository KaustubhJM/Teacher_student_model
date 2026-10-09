import Icon from './Icon.jsx'
import { formatDate } from '../data/classStore.js'

export function ClassBadge({ cls }) {
  if (cls.status === 'completed') {
    return cls.quizScore ? (
      <span className="badge done-badge">
        <Icon name="check_circle" style={{ fontSize: 16 }} /> Quiz {cls.quizScore.got}/{cls.quizScore.total}
      </span>
    ) : (
      <span className="badge done-badge">Finished</span>
    )
  }
  if (cls.status === 'planning') return <span className="badge planning-badge">Preparing…</span>
  if (cls.status === 'failed') return <span className="badge failed-badge">Couldn't prepare</span>
  if (cls.status === 'in_progress') {
    return (
      <span className="badge progress-badge">
        {cls.sectionsDone}/{cls.outline.length} sections
      </span>
    )
  }
  return <span className="badge new-badge">Not started</span>
}

// One card in the "Recent classes" grid (like a NotebookLM notebook card).
export default function ClassCard({ cls, menuOpen, onOpen, onToggleMenu, onMenuAction }) {
  const n = cls.sources.length
  return (
    <div className="class-card" onClick={onOpen} role="link" tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && onOpen()}>
      <div className="emoji">{cls.emoji}</div>
      <button
        className="icon-btn menu-btn"
        aria-label="More options"
        onClick={(e) => {
          e.stopPropagation() // don't also open the class
          onToggleMenu()
        }}
      >
        <Icon name="more_vert" />
      </button>
      {menuOpen && (
        <div className="menu" onClick={(e) => e.stopPropagation()}>
          <button onClick={() => onMenuAction('open')}>
            <Icon name="open_in_new" /> Open
          </button>
          <button onClick={() => onMenuAction('restart')}>
            <Icon name="restart_alt" /> Restart class
          </button>
          <button className="danger" onClick={() => onMenuAction('delete')}>
            <Icon name="delete" /> Delete
          </button>
        </div>
      )}
      <div className="card-title">{cls.title}</div>
      <div className="card-meta">
        {formatDate(cls.createdAt)} · {n} source{n === 1 ? '' : 's'}
      </div>
      <div className="badges">
        <ClassBadge cls={cls} />
        {cls.preview && <span className="badge new-badge">Preview</span>}
      </div>
    </div>
  )
}
