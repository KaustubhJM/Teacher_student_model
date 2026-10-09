import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import TopBar from '../components/TopBar.jsx'
import SourcesPanel from '../components/SourcesPanel.jsx'
import LecturePanel, { PlanningState } from '../components/LecturePanel.jsx'
import TeacherPanel from '../components/TeacherPanel.jsx'
import QuizView from '../components/QuizView.jsx'
import PreviewBanner from '../components/PreviewBanner.jsx'
import Icon from '../components/Icon.jsx'
import { getClass } from '../data/classStore.js'
import { useLecture } from '../lib/useLecture.js'

export default function ClassPage() {
  const { classId } = useParams()
  const [params] = useSearchParams()
  const [cls, setCls] = useState(undefined) // undefined = loading, null = not found

  // Load the class; while the backend is still planning it, check again every 2 s.
  useEffect(() => {
    let alive = true
    let timer = null
    const load = () =>
      getClass(classId).then((c) => {
        if (!alive) return
        setCls(c)
        if (c && c.status === 'planning') timer = setTimeout(load, 2000)
      })
    load()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [classId])

  if (cls === undefined) {
    return (
      <>
        <TopBar title="Loading…" />
        <div className="center-state" style={{ height: '60vh' }}>
          <Icon name="progress_activity" className="spin" style={{ fontSize: 40 }} />
        </div>
      </>
    )
  }

  if (cls && cls.status === 'planning') {
    return (
      <>
        <TopBar title={cls.title} />
        <div className="center-state" style={{ height: '70vh' }}>
          <PlanningState title={cls.title} step={cls.planStep} />
          <p style={{ fontSize: 15, color: 'var(--muted)' }}>
            Big or scanned PDFs can take a few minutes the first time (OCR). You can leave this page; planning keeps going.
          </p>
        </div>
      </>
    )
  }

  if (cls && cls.status === 'failed') {
    return (
      <>
        <TopBar title={cls.title} />
        <div className="center-state" style={{ height: '60vh' }}>
          <Icon name="error" style={{ fontSize: 48, color: 'var(--danger)' }} />
          <h2>Couldn't prepare this class</h2>
          <p>{cls.error}</p>
          <Link className="btn primary" to="/">
            Back to your classes
          </Link>
        </div>
      </>
    )
  }

  if (!cls) {
    return (
      <>
        <TopBar title="Class not found" />
        <div className="center-state" style={{ height: '60vh' }}>
          <h2>We couldn't find that class</h2>
          <Link className="btn primary" to="/">
            Back to your classes
          </Link>
        </div>
      </>
    )
  }
  // key = fresh lecture state whenever you switch to another class
  return <ClassView key={cls.id} cls={cls} autoStart={params.get('start') === '1'} />
}

function ClassView({ cls, autoStart }) {
  const navigate = useNavigate()
  const lecture = useLecture(cls, { autoStart })
  const [showQuiz, setShowQuiz] = useState(false)

  const doubts = lecture.items.filter((it) => it.kind === 'student').map((it) => it.text)
  const started = !['ready', 'planning'].includes(lecture.phase)

  return (
    <div className="class-page">
      <TopBar title={cls.title} onNewClass={() => navigate('/?new=1')} />
      <div className="class-layout">
        <SourcesPanel
          cls={cls}
          sectionIdx={lecture.sectionIdx}
          sectionsDone={lecture.sectionsDone}
          started={started}
          doubts={doubts}
        />
        <div className="center-col">
          {cls.preview && <PreviewBanner compact />}
          {showQuiz ? (
            <QuizView cls={cls} onBack={() => setShowQuiz(false)} />
          ) : (
            <LecturePanel cls={cls} lecture={lecture} onOpenQuiz={() => setShowQuiz(true)} />
          )}
        </div>
        <TeacherPanel lecture={lecture} onOpenQuiz={() => setShowQuiz(true)} />
      </div>
      {lecture.notice && <div className="toast">{lecture.notice}</div>}
    </div>
  )
}
