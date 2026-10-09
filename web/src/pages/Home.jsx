import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import TopBar from '../components/TopBar.jsx'
import ClassCard from '../components/ClassCard.jsx'
import NewClassDialog from '../components/NewClassDialog.jsx'
import Icon from '../components/Icon.jsx'
import { backendOnline, createClass, deleteClass, listClasses } from '../data/classStore.js'
import { api } from '../data/api.js'
import PreviewBanner from '../components/PreviewBanner.jsx'

export default function Home() {
  const navigate = useNavigate()
  const [classes, setClasses] = useState([])
  const [live, setLive] = useState(null) // null = still checking for the backend
  const [search, setSearch] = useState('')
  const [menuFor, setMenuFor] = useState(null) // id of the card whose ⋮ menu is open
  const [params] = useSearchParams()
  const [dialogOpen, setDialogOpen] = useState(params.get('new') === '1') // "+ New class" from a class page
  const [notice, setNotice] = useState('')

  const reload = () =>
    listClasses()
      .then(setClasses)
      .catch((e) => setNotice(`Couldn't load your classes: ${e.message}`))

  useEffect(() => {
    backendOnline().then(setLive)
    reload()
  }, [])

  // While a class is still being planned, refresh the list every few seconds.
  const planning = classes.some((c) => c.status === 'planning')
  useEffect(() => {
    if (!planning) return
    const t = setInterval(reload, 3000)
    return () => clearInterval(t)
  }, [planning])

  // Click anywhere else → close the ⋮ menu.
  useEffect(() => {
    if (!menuFor) return
    const close = () => setMenuFor(null)
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [menuFor])

  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(''), 3000)
    return () => clearTimeout(t)
  }, [notice])

  const visible = classes.filter((c) => c.title.toLowerCase().includes(search.toLowerCase()))
  const resume = classes.find((c) => c.status === 'in_progress')

  const onMenuAction = (cls, action) => {
    setMenuFor(null)
    if (action === 'open') navigate(`/class/${cls.id}`)
    if (action === 'restart') {
      if (cls.preview) return setNotice('Restarting needs the backend running.')
      if (['planning', 'failed'].includes(cls.status)) return setNotice('This class has no lecture to restart yet.')
      api
        .restartClass(cls.id)
        .then(() => {
          setNotice(`“${cls.title}” will start again from section 1.`)
          reload()
        })
        .catch((e) => setNotice(e.message))
    }
    if (action === 'delete') {
      if (cls.preview && !cls.id.startsWith('class-')) return setNotice('Sample classes can’t be deleted in the preview.')
      if (window.confirm(`Delete “${cls.title}”? This can’t be undone.`)) {
        deleteClass(cls)
          .then(reload)
          .catch((e) => setNotice(e.message))
      }
    }
  }

  // Errors are shown inside the dialog (it catches what this throws).
  const onCreate = async (draft) => {
    const cls = await createClass(draft)
    navigate(`/class/${cls.id}?start=1`)
  }

  return (
    <>
      <TopBar search={search} onSearch={setSearch} />
      <main className="home">
        {live === false && <PreviewBanner />}
        {resume && !search && (
          <section className="home-section">
            <div className="home-section-head">
              <h2>Continue where you left off</h2>
            </div>
            <div className="continue-card" onClick={() => navigate(`/class/${resume.id}`)} role="link" tabIndex={0}>
              <div className="emoji">{resume.emoji}</div>
              <div className="info">
                <h3>{resume.title}</h3>
                <div className="meta">
                  Next up: {resume.outline[resume.sectionsDone]} · section {resume.sectionsDone + 1} of {resume.outline.length}
                </div>
                <div className="progress">
                  <div style={{ width: `${(resume.sectionsDone / resume.outline.length) * 100}%` }} />
                </div>
              </div>
              <button className="btn primary">
                <Icon name="play_arrow" filled /> Resume
              </button>
            </div>
          </section>
        )}

        <section className="home-section">
          <div className="home-section-head">
            <h2>{search ? `Classes matching “${search}”` : 'Recent classes'}</h2>
            <button className="btn primary" onClick={() => setDialogOpen(true)}>
              <Icon name="add" /> New class
            </button>
          </div>
          <div className="class-grid">
            {!search && (
              <button className="new-card" onClick={() => setDialogOpen(true)}>
                <span className="plus">
                  <Icon name="add" />
                </span>
                Create new class
              </button>
            )}
            {visible.map((cls) => (
              <ClassCard
                key={cls.id}
                cls={cls}
                menuOpen={menuFor === cls.id}
                onOpen={() => navigate(`/class/${cls.id}`)}
                onToggleMenu={() => setMenuFor(menuFor === cls.id ? null : cls.id)}
                onMenuAction={(a) => onMenuAction(cls, a)}
              />
            ))}
          </div>
          {search && visible.length === 0 && <p style={{ color: 'var(--muted)' }}>No classes match that search.</p>}
        </section>
      </main>
      {dialogOpen && <NewClassDialog onClose={() => setDialogOpen(false)} onCreate={onCreate} />}
      {notice && <div className="toast">{notice}</div>}
    </>
  )
}
