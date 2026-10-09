// Text-to-speech for the PREVIEW. Uses the browser's built-in voices
// (Edge ships Microsoft "Ava" — the same voice the Python teacher uses).
// When the backend exists, sentences will arrive as edge-tts MP3s instead,
// and this module will play those; the rest of the UI stays the same.
//
// speak() reports every spoken word through onWord — the avatar uses those
// "word pulses" to move the mouth. Without a voice, a timer fakes the pulses.

const WORDS_PER_MINUTE = 160

let cachedVoice = null

function pickVoice() {
  if (cachedVoice) return cachedVoice
  if (!('speechSynthesis' in window)) return null
  const voices = window.speechSynthesis.getVoices()
  const prefs = [/Ava.*Natural/i, /Aria.*Natural/i, /Jenny.*Natural/i, /Ava/i, /Zira/i, /Google US English/i, /^en[-_]US/i, /^en/i]
  for (const p of prefs) {
    const v = voices.find((x) => p.test(x.name) || p.test(x.lang))
    if (v) return (cachedVoice = v)
  }
  return null
}

if ('speechSynthesis' in window) {
  // Voices load asynchronously in Chrome/Edge.
  window.speechSynthesis.onvoiceschanged = () => {
    cachedVoice = null
    pickVoice()
  }
}

export function voiceSupported() {
  return 'speechSynthesis' in window
}

// Returns { done: Promise<'ended' | 'cancelled'>, cancel() }.
export function speak(text, { voice = true, onWord = () => {}, rate = 1 } = {}) {
  const wordCount = text.split(/\s+/).filter(Boolean).length
  const estimatedMs = (wordCount / WORDS_PER_MINUTE) * 60000 / rate

  let finished = false
  let resolveDone
  const done = new Promise((r) => (resolveDone = r))
  const timers = []
  const finish = (how) => {
    if (finished) return
    finished = true
    timers.forEach(clearTimeout)
    timers.forEach(clearInterval)
    resolveDone(how)
  }

  // Timer mode: no audio, just pace the text and fake word pulses.
  const runTimer = () => {
    const perWord = estimatedMs / Math.max(wordCount, 1)
    onWord()
    timers.push(setInterval(onWord, perWord))
    timers.push(setTimeout(() => finish('ended'), estimatedMs + 250))
  }

  if (voice && voiceSupported()) {
    const u = new SpeechSynthesisUtterance(text)
    const v = pickVoice()
    if (v) u.voice = v
    u.rate = rate * 0.97
    let gotBoundary = false
    u.onboundary = (e) => {
      if (e.name === 'word' || e.name === undefined) {
        gotBoundary = true
        onWord()
      }
    }
    u.onstart = () => {
      // Some voices never send word boundaries — fall back to timed pulses.
      timers.push(
        setTimeout(() => {
          if (!gotBoundary && !finished) timers.push(setInterval(onWord, estimatedMs / Math.max(wordCount, 1)))
        }, 400),
      )
    }
    u.onend = () => finish('ended')
    u.onerror = (e) => {
      if (e.error === 'interrupted' || e.error === 'canceled') return finish('cancelled')
      // e.g. 'not-allowed' (page not clicked yet) → continue silently.
      runTimer()
    }
    // Safety net: Chrome occasionally never fires onend.
    timers.push(setTimeout(() => finish('ended'), estimatedMs * 2.5 + 4000))
    const synth = window.speechSynthesis
    if (synth.speaking || synth.pending) synth.cancel()
    synth.speak(u)
  } else {
    runTimer()
  }

  return {
    done,
    cancel() {
      if (finished) return
      finish('cancelled')
      if (voiceSupported()) window.speechSynthesis.cancel()
    },
  }
}

// --- Speech-to-text (voice doubts) ---------------------------------------
// Chrome/Edge have a free built-in recogniser (it uses Google's service,
// like the Python side's free Google STT).
export function recognitionSupported() {
  return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window
}

export function listenOnce({ onInterim = () => {}, lang = 'en-IN' } = {}) {
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition
  const rec = new Rec()
  rec.lang = lang
  rec.interimResults = true
  rec.maxAlternatives = 1
  let finalText = ''
  let resolveDone
  const done = new Promise((r) => (resolveDone = r))
  rec.onresult = (e) => {
    let interim = ''
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i]
      if (r.isFinal) finalText += r[0].transcript
      else interim += r[0].transcript
    }
    onInterim((finalText + ' ' + interim).trim())
  }
  // Keep the error code ('network', 'not-allowed', 'no-speech', …) so the
  // page can say WHY nothing was heard instead of silently carrying on.
  let error = null
  rec.onerror = (e) => {
    error = e.error
  }
  rec.onend = () => resolveDone(finalText.trim())
  rec.start()
  return {
    done,
    stop: () => rec.stop(),
    abort: () => rec.abort(),
    get error() {
      return error
    },
  }
}

// A short, plain explanation for a recognition error code (null = say nothing).
export function recognitionErrorMessage(code) {
  switch (code) {
    case 'network':
      // Brave ships the API but blocks the Google speech service behind it.
      return "Voice input isn't available in this browser (Brave blocks the speech service). Use Chrome or Edge, or type your doubt."
    case 'not-allowed':
    case 'service-not-allowed':
      return 'Microphone access is blocked. Allow the mic for this site (icon left of the address bar), or type your doubt.'
    case 'audio-capture':
      return 'No microphone was found. Check your mic, or type your doubt.'
    case 'no-speech':
      return "I didn't hear anything — try again a bit closer to the mic, or type your doubt."
    case 'language-not-supported':
      return "Voice input doesn't support this language here. Type your doubt instead."
    default:
      return null // 'aborted' (we stopped it) or no error
  }
}
