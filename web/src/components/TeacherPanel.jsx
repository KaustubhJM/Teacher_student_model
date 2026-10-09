import Icon from './Icon.jsx'
import TeacherAvatar from './TeacherAvatar.jsx'
import { TEACHER_NAME } from '../data/mockData.js'

const STATUS = {
  ready: { mood: 'idle', cls: '', icon: 'schedule', text: 'Waiting to start' },
  planning: { mood: 'think', cls: 'think', icon: 'edit_note', text: 'Preparing the lecture…' },
  teaching: { mood: 'talk', cls: 'talk', icon: 'graphic_eq', text: 'Teaching' },
  answering: { mood: 'talk', cls: 'talk', icon: 'record_voice_over', text: 'Answering your doubt' },
  listening: { mood: 'listen', cls: 'listen', icon: 'hearing', text: 'Listening to you…' },
  thinking: { mood: 'think', cls: 'think', icon: 'manage_search', text: 'Checking your notes…' },
  preparing: { mood: 'think', cls: 'think', icon: 'edit_note', text: 'Preparing the next part…' },
  awaiting: { mood: 'listen', cls: 'listen', icon: 'quiz', text: 'Waiting for your answer' },
  paused: { mood: 'idle', cls: '', icon: 'pause', text: 'Paused' },
  done: { mood: 'idle', cls: '', icon: 'task_alt', text: 'Class finished' },
}

// Right panel: the talking teacher, her status, controls and the quiz card.
export default function TeacherPanel({ lecture, onOpenQuiz }) {
  const { phase, voiceOn, sections, sectionsDone, mouthRef } = lecture
  const st = STATUS[phase] || STATUS.ready
  const running = ['teaching', 'answering', 'thinking', 'listening', 'preparing'].includes(phase)
  const quizReady = phase === 'done'
  const pct = Math.round((sectionsDone / Math.max(sections.length, 1)) * 100)

  return (
    <aside className="panel teacher-panel">
      <div className="teacher-stage">
        <TeacherAvatar mouthRef={mouthRef} mood={st.mood} />
      </div>
      <div className="teacher-name">{TEACHER_NAME}</div>
      <div className={`status-pill ${st.cls}`}>
        <Icon name={st.icon} /> {st.text}
      </div>

      <div className="teacher-controls">
        {running ? (
          <button className="icon-btn" onClick={lecture.pause} aria-label="Pause" title="Pause">
            <Icon name="pause" filled />
          </button>
        ) : (
          <button
            className="icon-btn"
            onClick={phase === 'done' ? lecture.restart : lecture.start}
            aria-label={phase === 'done' ? 'Replay class' : 'Play'}
            title={phase === 'done' ? 'Replay class' : 'Play'}
            disabled={phase === 'planning'}
          >
            <Icon name={phase === 'done' ? 'replay' : 'play_arrow'} filled />
          </button>
        )}
        <button className={`icon-btn ${voiceOn ? '' : 'off'}`} onClick={lecture.toggleVoice} aria-label={voiceOn ? 'Mute voice' : 'Unmute voice'} title={voiceOn ? 'Mute voice (captions only)' : 'Unmute voice'}>
          <Icon name={voiceOn ? 'volume_up' : 'volume_off'} />
        </button>
        <button className="icon-btn" onClick={lecture.listen} aria-label="Ask by voice" title="Ask by voice">
          <Icon name="front_hand" />
        </button>
      </div>

      <div className="teacher-meta">
        <div className="row">
          <span>Progress</span>
          <span>
            {sectionsDone}/{sections.length} sections
          </span>
        </div>
        <div className="progress" style={{ maxWidth: 'none', marginTop: 8 }}>
          <div style={{ width: `${pct}%` }} />
        </div>
      </div>

      <div className={`quiz-card ${quizReady ? '' : 'locked'}`}>
        <Icon name={quizReady ? 'quiz' : 'lock'} />
        <div className="txt">
          Quiz
          <small>{quizReady ? 'Check what you learned' : 'Unlocks when the lecture ends'}</small>
        </div>
        {quizReady && (
          <button className="btn small primary" onClick={onOpenQuiz}>
            Start
          </button>
        )}
      </div>
    </aside>
  )
}
