import { useEffect, useState } from 'react'
import Icon from './Icon.jsx'
import { MOCK_QUIZ } from '../data/mockData.js'
import { api } from '../data/api.js'

// Placeholder questions for classes without a mock quiz.
function genericQuiz(cls) {
  const o = cls.outline
  return [
    {
      q: 'Which part did this class start with?',
      options: [o[1], o[0], o[o.length - 1]].filter(Boolean),
      answer: 1,
      why: `The first section was “${o[0]}”.`,
    },
    {
      q: 'How many sections did the class have?',
      options: [String(o.length - 1), String(o.length), String(o.length + 2)],
      answer: 1,
      why: `The outline had ${o.length} sections.`,
    },
  ]
}

const previewQuiz = (cls) => (cls.id === 'dbms-u5' ? MOCK_QUIZ : genericQuiz(cls))

// Centre column after the lecture. Real class: quiz.py's questions, made
// by the server once (Gemini, else Groq) and then cached. Preview: mock.
export default function QuizView({ cls, onBack }) {
  const [questions, setQuestions] = useState(() => (cls.preview ? previewQuiz(cls) : null))
  const [loadError, setLoadError] = useState('')
  const [attempt, setAttempt] = useState(0) // bump = try loading again
  const [picked, setPicked] = useState({})
  const [checked, setChecked] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (cls.preview) return
    let alive = true
    api
      .quiz(cls.id)
      .then((r) => alive && setQuestions(r.questions))
      .catch((e) => alive && setLoadError(e.message))
    return () => {
      alive = false
    }
  }, [cls, attempt])

  if (!questions) {
    return (
      <section className="panel lecture">
        <div className="center-state">
          {loadError ? (
            <>
              <Icon name="error" style={{ fontSize: 48, color: 'var(--danger)' }} />
              <h2>The quiz isn't ready</h2>
              <p>{loadError}</p>
              <div style={{ display: 'flex', gap: 10 }}>
                <button
                  className="btn primary"
                  onClick={() => {
                    setLoadError('')
                    setAttempt((a) => a + 1)
                  }}
                >
                  <Icon name="refresh" /> Try again
                </button>
                <button className="btn" onClick={onBack}>
                  Back to the lecture
                </button>
              </div>
            </>
          ) : (
            <>
              <Icon name="progress_activity" className="spin" style={{ fontSize: 40 }} />
              <h2>Writing your quiz…</h2>
              <p>Questions come from your notes and what was taught. This can take up to a minute.</p>
            </>
          )}
        </div>
      </section>
    )
  }

  const score = questions.filter((q, k) => picked[k] === q.answer).length

  const check = () => {
    if (Object.keys(picked).length < questions.length) {
      setError(`Answer all ${questions.length} questions first.`)
      return
    }
    setChecked(true)
    if (!cls.preview) api.saveScore(cls.id, score, questions.length).catch(() => {})
  }

  return (
    <section className="panel lecture">
      <div className="lecture-head" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button className="icon-btn" onClick={onBack} aria-label="Back to the lecture">
          <Icon name="arrow_back" />
        </button>
        <div>
          <div className="kicker">Quiz · {questions.length} questions</div>
          <h1>{cls.title}</h1>
        </div>
      </div>
      <div className="quiz">
        {checked && (
          <div className="quiz-score">
            <Icon name="emoji_events" style={{ fontSize: 32 }} />
            You got {score} of {questions.length} right.
          </div>
        )}
        {questions.map((q, k) => (
          <div className="quiz-q" key={k}>
            <h3>
              {k + 1}. {q.q}
            </h3>
            {q.options.map((opt, j) => {
              let state = picked[k] === j ? 'picked' : ''
              if (checked && j === q.answer) state = 'right'
              else if (checked && picked[k] === j) state = 'wrong'
              return (
                <button
                  key={j}
                  className={`option ${state}`}
                  onClick={() => {
                    if (checked) return
                    setPicked({ ...picked, [k]: j })
                    setError('')
                  }}
                >
                  <Icon name={state === 'right' ? 'check_circle' : state === 'wrong' ? 'cancel' : picked[k] === j ? 'radio_button_checked' : 'radio_button_unchecked'} />
                  {opt}
                </button>
              )
            })}
            {checked && <p className="quiz-why">{q.why}</p>}
          </div>
        ))}
        {error && <p className="field-error">{error}</p>}
        <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
          {!checked ? (
            <button className="btn primary" onClick={check}>
              <Icon name="done_all" /> Check answers
            </button>
          ) : (
            <button
              className="btn"
              onClick={() => {
                setPicked({})
                setChecked(false)
              }}
            >
              <Icon name="refresh" /> Try again
            </button>
          )}
          <button className="btn" onClick={onBack}>
            Back to the lecture
          </button>
        </div>
      </div>
    </section>
  )
}
