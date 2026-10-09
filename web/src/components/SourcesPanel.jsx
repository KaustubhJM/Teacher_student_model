import Icon from './Icon.jsx'

const SOURCE_ICON = { pdf: 'picture_as_pdf', youtube: 'smart_display', text: 'article', topic: 'lightbulb' }

// Left panel: the class's sources, the outline with progress, and the doubts asked.
export default function SourcesPanel({ cls, sectionIdx, sectionsDone, started, doubts }) {
  return (
    <aside className="panel">
      <div className="panel-head">
        <span>Sources</span>
        <Icon name="left_panel_close" style={{ color: 'var(--muted)' }} />
      </div>
      <div className="panel-body">
        {cls.sources.map((src, k) => (
          <div className="source-item" key={k} title={src.name}>
            <div className={`source-icon ${src.type}`}>
              <Icon name={SOURCE_ICON[src.type] || 'description'} style={{ fontSize: 20 }} />
            </div>
            <div style={{ minWidth: 0, flex: 1 }}>
              <div className="name">{src.name}</div>
              <div className="detail">{src.detail}</div>
            </div>
          </div>
        ))}

        <div className="panel-label">Outline</div>
        {cls.outline.map((title, k) => {
          const done = k < sectionsDone
          const current = started && !done && k === sectionIdx
          return (
            <div key={k} className={`outline-item ${done ? 'done' : ''} ${current ? 'current' : ''}`}>
              {done ? <Icon name="check_circle" /> : current ? <Icon name="graphic_eq" /> : <span className="num">{k + 1}</span>}
              <span>{title}</span>
            </div>
          )
        })}

        {doubts.length > 0 && (
          <>
            <div className="panel-label">Your doubts ({doubts.length})</div>
            {doubts.map((d, k) => (
              <div className="doubt-item" key={k}>
                <Icon name="help" style={{ fontSize: 16, verticalAlign: -3, marginRight: 6, color: 'var(--muted)' }} />
                {d}
              </div>
            ))}
          </>
        )}
      </div>
    </aside>
  )
}
