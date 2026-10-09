// The lecture "engine" for the page — the browser twin of classroom.py's
// teach_and_listen loop: speak sentence by sentence; on a doubt, stop,
// answer, say a resume phrase, and restart the interrupted sentence.
//
// Where the words come from:
// - real class (backend running): each section's 3 beats are fetched from
//   the server when the section starts, and the first beat is spoken while
//   the next ones are still being written (like the CLI's BeatStream).
//   Doubts go to the real answer_doubt; finished sections are reported so
//   the server keeps the transcript and progress (resume works).
// - preview class: the placeholder script from mockData.js.
//
// React note: a "hook" (useSomething) is a function that keeps state for a
// component. Whenever we call a setX(...) function, React re-draws the
// parts of the page that use X.
import { useCallback, useEffect, useRef, useState } from 'react'
import { getLectureScript, mockDoubtAnswer, RESUME_PHRASES, STUDENT_NAME } from '../data/mockData.js'
import { api } from '../data/api.js'
import { listenOnce, recognitionErrorMessage, recognitionSupported, speak } from './speech.js'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// Each section ends with ONE comprehension check (explainer's "connection"
// beat). Find it in the section's last few sentences; null = none.
function findCheckQuestion(sentences) {
  for (const sent of [...(sentences || [])].slice(-3).reverse()) {
    if (sent.trim().endsWith('?')) return sent.trim()
  }
  return null
}

// Same line and same skip words as the CLI (comprehension_check.py).
const SKIP_LINE = "No problem — let's move on."
const SKIP_REPLIES = new Set([
  'skip', 'skip it', 'skip this', 'skip question', 'skip the question', 'skip this question',
  'next', 'next question', 'pass', 'move on', 'lets move on',
])
// Whole-reply match only, so an answer that merely contains "next" isn't a skip.
function isSkipReply(text) {
  const t = text.toLowerCase().replace(/['’]/g, '').replace(/[^\w\s]/g, ' ').split(/\s+/).filter(Boolean).join(' ')
  return SKIP_REPLIES.has(t)
}

// phase: ready | planning | preparing | teaching | awaiting | paused | listening | thinking | answering | done
// awaiting = the section ended with a question; waiting for the student's answer (or Skip).
export function useLecture(cls, { autoStart = false } = {}) {
  const live = Boolean(cls && !cls.preview)
  const numSections = cls ? cls.outline.length : 0

  // Sentences of each section loaded so far ([[...], [...], ...]).
  // Preview: everything is known up front.
  const [sections, setSections] = useState(() => {
    if (!cls) return []
    return live ? cls.outline.map(() => []) : getLectureScript(cls)
  })
  const sectionsRef = useRef(sections)
  const beatCountRef = useRef(cls ? cls.outline.map(() => 0) : []) // beats received per section
  const completeRef = useRef(cls ? cls.outline.map(() => !live) : []) // all beats received?
  const loadingRef = useRef({}) // section -> in-flight fetch promise

  // in_progress → resume at the saved sentence (preview/older records: the
  // next section); completed → positioned past the end.
  const startPos = !cls
    ? { s: 0, i: 0 }
    : cls.status === 'in_progress'
      ? cls.position ?? { s: cls.sectionsDone, i: 0 }
      : cls.status === 'completed'
        ? { s: numSections, i: 0 }
        : { s: 0, i: 0 }

  const [phase, setPhase] = useState(() => {
    if (!cls) return 'ready'
    if (cls.status === 'completed') return 'done'
    return autoStart ? 'planning' : 'ready'
  })
  const [items, setItems] = useState([]) // finished transcript lines + doubts
  const [pos, setPos] = useState(startPos) // current sentence
  const [voiceOn, setVoiceOn] = useState(true)
  const [interim, setInterim] = useState('')
  const [notice, setNotice] = useState('')
  const [check, setCheckState] = useState(null) // { s, question } while waiting for an answer
  const checkRef = useRef(null)
  const setCheck = (c) => {
    checkRef.current = c
    setCheckState(c)
  }

  // Refs = values the async loop reads without waiting for a re-render.
  const posRef = useRef(pos)
  const voiceRef = useRef(voiceOn)
  const tokenRef = useRef(0) // bumping it stops whatever is running
  const speechRef = useRef(null)
  const recRef = useRef(null)
  const greetedRef = useRef(false)
  // Shared with the avatar 60×/s without re-rendering React.
  const mouthRef = useRef({ talking: false, lastWord: 0, amp: 0.5 })

  const setPosBoth = (p) => {
    posRef.current = p
    setPos(p)
  }

  // Ask the server for the next beat(s) of section s (it waits up to ~25 s).
  // Shared promise, so a doubt or a pause never starts a second request.
  const loadMore = useCallback(
    (s) => {
      if (!loadingRef.current[s]) {
        loadingRef.current[s] = api
          .beats(cls.id, s, beatCountRef.current[s])
          .then(({ beats, done }) => {
            beatCountRef.current[s] += beats.length
            completeRef.current[s] = done
            if (beats.length) {
              const next = [...sectionsRef.current]
              next[s] = [...next[s], ...beats.flat()]
              sectionsRef.current = next
              setSections(next)
            }
          })
          .finally(() => {
            delete loadingRef.current[s]
          })
      }
      return loadingRef.current[s]
    },
    [cls],
  )

  // Sentence i of section s: a string, null if the section has ended, or
  // undefined if we were stopped while waiting for it.
  const sentenceAt = useCallback(
    async (s, i, token) => {
      for (;;) {
        const sec = sectionsRef.current[s] || []
        if (i < sec.length) return sec[i]
        if (completeRef.current[s]) return null
        if (token !== tokenRef.current) return undefined
        setPhase('preparing')
        await loadMore(s) // throws if the server couldn't write this part
      }
    },
    [loadMore],
  )

  const say = useCallback(async (text) => {
    const m = mouthRef.current
    m.talking = true
    const h = speak(text, {
      voice: voiceRef.current,
      onWord: () => {
        m.lastWord = performance.now()
        m.amp = 0.4 + Math.random() * 0.6
      },
    })
    speechRef.current = h
    const how = await h.done
    m.talking = false
    return how
  }, [])

  // End of section s: the server records it (transcript + progress), then
  // the loop moves on to the next section.
  const finishSection = useCallback(
    async (s) => {
      if (live) api.sectionDone(cls.id, s).catch(() => {})
      setPosBoth({ s: s + 1, i: 0 })
      await sleep(700) // small breath between sections
    },
    [cls, live],
  )

  const stopEverything = useCallback(() => {
    tokenRef.current += 1
    speechRef.current?.cancel()
    recRef.current?.abort()
    recRef.current = null
    mouthRef.current.talking = false
  }, [])

  const run = useCallback(
    async (token) => {
      try {
        if (!greetedRef.current) {
          greetedRef.current = true
          const { s, i } = posRef.current
          const fromNotes = cls.sources.some((src) => src.type !== 'topic') ? ' based on the material you gave me' : ''
          // Same greeting as classroom.plan_lecture.
          const greeting =
            s === 0 && i === 0
              ? `Hey ${STUDENT_NAME}, today we're covering ${cls.title}${fromNotes}. We'll go through it in ${numSections} parts, and just jump in anytime you've got a question.`
              : `Welcome back, ${STUDENT_NAME}. Let's continue with ${cls.outline[Math.min(s, numSections - 1)].toLowerCase()}.`
          if (live && s < numSections) loadMore(s).catch(() => {}) // start writing this section during the greeting
          setPhase('teaching')
          setItems((prev) => [...prev, { kind: 'greeting', text: greeting }])
          await say(greeting)
          if (token !== tokenRef.current) return
          if (live) api.saveProgress(cls.id, s, i).catch(() => {}) // class now counts as started
        }
        while (token === tokenRef.current) {
          const { s, i } = posRef.current
          if (s >= numSections) {
            setPhase('done')
            return
          }
          const text = await sentenceAt(s, i, token)
          if (token !== tokenRef.current || text === undefined) return
          if (text === null) {
            // The section ended with a question → wait for the student's
            // answer (or Skip) before moving on. answerCheck/skipCheck continue.
            const question = findCheckQuestion(sectionsRef.current[s])
            if (question) {
              setCheck({ s, question })
              setPhase('awaiting')
              return
            }
            await finishSection(s)
            continue
          }
          setPhase('teaching')
          const how = await say(text)
          if (token !== tokenRef.current || how === 'cancelled') return
          setItems((prev) => [...prev, { kind: 'sentence', s, i, text }])
          setPosBoth({ s, i: i + 1 })
          if (live) api.saveProgress(cls.id, s, i + 1).catch(() => {}) // resume point survives leaving the page
        }
      } catch (e) {
        if (token !== tokenRef.current) return
        setPhase('paused')
        setNotice(`${e.message} — press play to try again.`)
      }
    },
    [cls, finishSection, live, loadMore, numSections, say, sentenceAt],
  )

  const start = useCallback(() => {
    stopEverything()
    run(tokenRef.current)
  }, [run, stopEverything])

  const pause = useCallback(() => {
    stopEverything()
    setPhase('paused')
  }, [stopEverything])

  const skipCheck = useCallback(async () => {
    const pending = checkRef.current
    if (!pending) return
    stopEverything()
    const token = tokenRef.current
    setCheck(null)
    setItems((prev) => [...prev, { kind: 'teacher', text: SKIP_LINE }])
    setPhase('teaching')
    await say(SKIP_LINE)
    if (token !== tokenRef.current) return
    await finishSection(pending.s)
    if (token !== tokenRef.current) return
    run(token)
  }, [finishSection, run, say, stopEverything])

  // The student's reply to the end-of-section question (anything, even
  // "I don't know"): the teacher judges it, explains the right answer if
  // needed, then the next section starts. Just "skip" / "next" = Skip.
  const answerCheck = useCallback(
    async (reply) => {
      const pending = checkRef.current
      if (!pending || !reply.trim()) return
      if (isSkipReply(reply)) return skipCheck()
      stopEverything()
      const token = tokenRef.current
      setCheck(null)
      setItems((prev) => [...prev, { kind: 'student', text: reply.trim() }])
      setPhase('thinking')
      let answer
      if (live) {
        try {
          const at = sectionsRef.current[pending.s]?.length ?? 0
          answer = (await api.answerCheck(cls.id, pending.s, pending.question, reply.trim(), at)).answer
        } catch (e) {
          answer = `Sorry ${STUDENT_NAME}, I couldn't check that just now (${e.message}). Let's keep going.`
        }
      } else {
        await sleep(1200)
        answer = `Thanks, ${STUDENT_NAME}! In the real class I'd tell you if that's right and explain the answer.`
      }
      if (token !== tokenRef.current) return
      setItems((prev) => [...prev, { kind: 'teacher', text: answer }])
      setPhase('answering')
      await say(answer)
      if (token !== tokenRef.current) return
      await finishSection(pending.s)
      if (token !== tokenRef.current) return
      run(token)
    },
    [cls, finishSection, live, run, say, skipCheck, stopEverything],
  )

  const askDoubt = useCallback(
    async (question) => {
      const q = question.trim()
      if (!q) return
      if (checkRef.current) return answerCheck(q) // typed/spoken while the teacher waits = the answer
      const wasDone = posRef.current.s >= numSections
      stopEverything()
      const token = tokenRef.current
      setItems((prev) => [...prev, { kind: 'student', text: q }])
      setPhase('thinking') // the teacher may search the notes / the web
      let answer
      if (live) {
        try {
          // After the lecture ended, place the doubt after everything else.
          const sentence = wasDone ? 1e6 : posRef.current.i
          answer = (await api.askDoubt(cls.id, Math.min(posRef.current.s, numSections - 1), q, sentence)).answer
        } catch (e) {
          answer = `Sorry ${STUDENT_NAME}, I couldn't reach my notes just now (${e.message}). Let's keep going, and ask me again in a moment.`
        }
      } else {
        await sleep(1400)
        answer = mockDoubtAnswer(q)
      }
      if (token !== tokenRef.current) return
      setItems((prev) => [...prev, { kind: 'teacher', text: answer }])
      setPhase('answering')
      await say(answer)
      if (token !== tokenRef.current) return
      if (wasDone) return setPhase('done')
      await say(RESUME_PHRASES[Math.floor(Math.random() * RESUME_PHRASES.length)])
      if (token !== tokenRef.current) return
      run(token) // restarts the interrupted sentence, like the CLI
    },
    [answerCheck, cls, live, numSections, run, say, stopEverything],
  )

  const listen = useCallback(async () => {
    if (phase === 'listening') {
      recRef.current?.stop() // second click = "I'm done talking"
      return
    }
    if (!recognitionSupported()) {
      setNotice('Voice doubts need Chrome or Edge. Type your doubt instead.')
      return
    }
    stopEverything()
    const token = tokenRef.current
    setPhase('listening')
    setInterim('')
    const rec = listenOnce({ onInterim: setInterim })
    recRef.current = rec
    const text = await rec.done
    if (token !== tokenRef.current) return
    recRef.current = null
    setInterim('')
    if (!text) {
      const why = recognitionErrorMessage(rec.error)
      if (why) setNotice(why)
    }
    if (text) askDoubt(text)
    else if (checkRef.current) setPhase('awaiting') // heard nothing → keep waiting for the answer
    else if (posRef.current.s >= numSections) setPhase('done')
    else run(token) // heard nothing → carry on (same as an empty transcript in the CLI)
  }, [askDoubt, numSections, phase, run, stopEverything])

  const toggleVoice = useCallback(() => {
    voiceRef.current = !voiceRef.current
    setVoiceOn(voiceRef.current)
    // Restart the current sentence in the new mode.
    if (phase === 'teaching') {
      stopEverything()
      run(tokenRef.current)
    }
  }, [phase, run, stopEverything])

  const restart = useCallback(() => {
    stopEverything()
    // Server: progress/transcript reset; the written beats are kept (no new tokens).
    if (live) api.restartClass(cls.id).catch((e) => setNotice(e.message))
    greetedRef.current = false
    setCheck(null)
    setItems([])
    setPosBoth({ s: 0, i: 0 })
    setPhase('ready')
  }, [cls, live, stopEverything])

  // "Planning" screen right after creating a class. A real class was already
  // planned by the server (ClassPage waited for it), so start right away.
  useEffect(() => {
    if (phase !== 'planning') return
    const t = setTimeout(() => start(), live ? 0 : 4200)
    return () => clearTimeout(t)
  }, [live, phase, start])

  // Coming back to a started class: show what was already taught (and the
  // doubts) as captions, instead of an empty page.
  useEffect(() => {
    if (!live || cls.status === 'new') return
    let alive = true
    api
      .history(cls.id)
      .then(({ items: past }) => {
        if (alive && past.length) setItems((prev) => (prev.some((x) => x.kind !== 'greeting') ? prev : [...past, ...prev]))
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [cls, live])

  // Leaving the page stops the voice and the mic.
  useEffect(() => () => stopEverything(), [stopEverything])

  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(''), 6000)
    return () => clearTimeout(t)
  }, [notice])

  const sectionIdx = Math.min(pos.s, numSections - 1)
  const finished = pos.s >= numSections || phase === 'done'
  return {
    phase,
    items,
    sections,
    pos,
    sectionIdx,
    sectionsDone: finished ? numSections : pos.s,
    currentSentence: !finished && pos.s < numSections ? (sections[pos.s]?.[pos.i] ?? null) : null,
    voiceOn,
    interim,
    notice,
    mouthRef,
    check, // { s, question } while waiting for the student's answer
    start,
    pause,
    askDoubt,
    skipCheck,
    listen,
    toggleVoice,
    restart,
    setNotice,
  }
}
