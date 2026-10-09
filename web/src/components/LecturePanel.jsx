import { useEffect, useRef, useState } from 'react'
import Icon from './Icon.jsx'
import { STUDENT_NAME, TEACHER_NAME } from '../data/mockData.js'

const PLAN_STEPS = ['Reading your material', 'Planning the outline', 'Picking the notes for each part', 'Getting ready to teach']

// `step` = the backend's real planning step (0 reading, 1 outline, 2 notes,
// 3 ready). Without it (preview), the steps just tick along on a timer.
export function PlanningState({ title, step: realStep }) {
  const [timerStep, setStep] = useState(0)
  useEffect(() => {
    if (realStep !== undefined) return
    const t = setInterval(() => setStep((s) => Math.min(s + 1, PLAN_STEPS.length - 1)), 1000)
    return () => clearInterval(t)
  }, [realStep])
  const step = realStep ?? timerStep
  return (
    <div className="center-state">
      <h2>Preparing “{title}”</h2>
      <div className="steps">
        {PLAN_STEPS.map((s, k) => (
          <div key={s} className={`step ${k < step ? 'done' : k === step ? 'active' : ''}`}>
            {k < step ? <Icon name="check_circle" /> : k === step ? <Icon name="progress_activity" className="spin" /> : <Icon name="radio_button_unchecked" />}
            {s}
          </div>
        ))}
      </div>
    </div>
  )
}

// Centre column: live captions of the lecture + doubts + the "ask" bar.
export default function LecturePanel({ cls, lecture, onOpenQuiz }) {
  const { phase, items, sections, pos, sectionIdx, currentSentence, interim } = lecture
  const [draft, setDraft] = useState('')
  const scroller = useRef(null)

  // Keep the newest line in view.
  useEffect(() => {
    const el = scroller.current
    if (el) el.scrollTop = el.scrollHeight
  }, [items.length, pos.s, pos.i, phase])

  const send = () => {
    if (!draft.trim()) return
    lecture.askDoubt(draft)
    setDraft('')
  }

  if (phase === 'planning') {
    return (
      <section className="panel lecture">
        <PlanningState title={cls.title} />
      </section>
    )
  }

  if (phase === 'ready') {
    const resuming = pos.s > 0 || pos.i > 0
    return (
      <section className="panel lecture">
        <div className="center-state">
          <div style={{ fontSize: 64 }}>{cls.emoji}</div>
          <h2>{resuming ? 'Ready to continue?' : 'Ready when you are'}</h2>
          <p>
            {!resuming
              ? `${TEACHER_NAME} will teach ${sections.length} parts. Interrupt anytime: press the mic and speak, or type your doubt.`
              : pos.i > 0
                ? `You stopped partway through “${cls.outline[pos.s]}”. ${TEACHER_NAME} will pick up right where you left off.`
                : `You stopped after “${cls.outline[pos.s - 1]}”. ${TEACHER_NAME} will pick up with “${cls.outline[pos.s]}”.`}
          </p>
          <button className="btn primary" onClick={lecture.start}>
            <Icon name="play_arrow" filled /> {resuming ? 'Resume class' : 'Start class'}
          </button>
          <p style={{ fontSize: 14, color: 'var(--muted)' }}>Turn your speakers on. Headphones work best with the mic.</p>
        </div>
      </section>
    )
  }

  const finished = phase === 'done' && pos.s >= sections.length
  const showCurrent = currentSentence && ['teaching', 'paused', 'listening', 'thinking', 'answering'].includes(phase)
  // Waiting for the backend to write the next part (beats are generated on demand).
  const preparing = phase === 'preparing' && pos.s < sections.length
  // Mark where a new section starts so we can draw a divider before it.
  const rows = []
  let lastSection = -1
  for (const it of items) {
    rows.push({ it, divider: it.kind === 'sentence' && it.s !== lastSection })
    if (it.kind === 'sentence') lastSection = it.s
  }

  return (
    <section className="panel lecture">
      <div className="lecture-head">
        <div className="kicker">
          {finished || (phase === 'done' && items.length === 0)
            ? 'Class finished'
            : `Section ${sectionIdx + 1} of ${sections.length}`}
        </div>
        <h1>{finished || (phase === 'done' && items.length === 0) ? cls.title : cls.outline[sectionIdx]}</h1>
      </div>

      <div className="transcript" ref={scroller}>
        {phase === 'done' && items.length === 0 && (
          <div className="center-state">
            <Icon name="task_alt" style={{ fontSize: 56, color: 'var(--success)' }} />
            <h2>You finished this class</h2>
            <p>
              {cls.quizScore
                ? `Your quiz score was ${cls.quizScore.got}/${cls.quizScore.total}. Retake it, or replay the lecture.`
                : 'Take the quiz to check what stuck.'}
            </p>
            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn primary" onClick={onOpenQuiz}>
                <Icon name="quiz" /> Take the quiz
              </button>
              <button className="btn" onClick={lecture.restart}>
                <Icon name="replay" /> Replay class
              </button>
            </div>
          </div>
        )}

        {rows.map(({ it, divider: needsDivider }, k) => {
          const divider = needsDivider ? (
            <div className="section-divider">
              {it.s + 1}. {cls.outline[it.s]}
            </div>
          ) : null
          if (it.kind === 'greeting' || it.kind === 'sentence') {
            return (
              <div key={k}>
                {divider}
                <p className="sentence spoken">{it.text}</p>
              </div>
            )
          }
          const student = it.kind === 'student'
          return (
            <div key={k} className={`bubble-row ${student ? 'student' : ''}`}>
              <div className={`bubble ${student ? 'student' : 'teacher'}`}>
                <div className="who">{student ? STUDENT_NAME : TEACHER_NAME}</div>
                {it.text}
              </div>
            </div>
          )
        })}

        {showCurrent && (
          <>
            {pos.s !== lastSection && (
              <div className="section-divider">
                {pos.s + 1}. {cls.outline[pos.s]}
              </div>
            )}
            <p className={`sentence ${phase === 'teaching' ? 'current' : 'spoken'}`}>{currentSentence}</p>
          </>
        )}

        {lecture.check && !['thinking', 'answering'].includes(phase) && (
          <div className="check-card" role="group" aria-label="Question from the teacher">
            <div className="check-kicker">
              <Icon name="quiz" /> Your turn — {TEACHER_NAME} asked:
            </div>
            <p className="check-question">{lecture.check.question}</p>
            <p className="check-hint">
              Type your answer below or press the mic and say it. Not sure? Just say so — she'll explain.
            </p>
            <div className="check-actions">
              <button className="btn primary" onClick={lecture.listen}>
                <Icon name={phase === 'listening' ? 'stop' : 'mic'} /> {phase === 'listening' ? 'Done talking' : 'Answer by voice'}
              </button>
              <button className="btn" onClick={lecture.skipCheck}>
                <Icon name="skip_next" /> Skip question
              </button>
            </div>
          </div>
        )}

        {preparing && (
          <>
            {pos.s !== lastSection && (
              <div className="section-divider">
                {pos.s + 1}. {cls.outline[pos.s]}
              </div>
            )}
            <p className="sentence preparing">
              <Icon name="progress_activity" className="spin" style={{ fontSize: 18, verticalAlign: -3, marginRight: 8 }} />
              {TEACHER_NAME} is preparing this part from your notes…
            </p>
          </>
        )}

        {finished && items.length > 0 && (
          <div className="center-state" style={{ flex: 'none', paddingTop: 32 }}>
            <Icon name="task_alt" style={{ fontSize: 48, color: 'var(--success)' }} />
            <h2>That's the whole class!</h2>
            <button className="btn primary" onClick={onOpenQuiz}>
              <Icon name="quiz" /> Take the quiz
            </button>
          </div>
        )}
      </div>

      <div className="composer-wrap">
        <div className="composer">
          <input
            placeholder={
              phase === 'listening'
                ? interim || (lecture.check ? 'Listening… say your answer' : 'Listening… speak your doubt')
                : lecture.check
                  ? 'Type your answer (or “I don’t know”)'
                  : 'Type a doubt, or press the mic and speak'
            }
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && send()}
            aria-label="Your doubt"
          />
          <button
            className={`icon-btn mic ${phase === 'listening' ? 'on' : ''}`}
            onClick={lecture.listen}
            aria-label={phase === 'listening' ? 'Stop listening' : 'Ask by voice'}
            title={phase === 'listening' ? 'Done talking' : 'Ask by voice'}
          >
            <Icon name={phase === 'listening' ? 'stop' : 'mic'} />
          </button>
          <button className="icon-btn send" onClick={send} disabled={!draft.trim()} aria-label="Send doubt">
            <Icon name="arrow_upward" />
          </button>
        </div>
        <p className="disclaimer">{TEACHER_NAME} can make mistakes, so double-check important facts against your notes.</p>
      </div>
    </section>
  )
}
