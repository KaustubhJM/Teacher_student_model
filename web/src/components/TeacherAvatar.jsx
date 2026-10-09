import { useEffect, useRef } from 'react'

// Illustrated teacher. The mouth is driven by mouthRef (filled by the speech
// code): every spoken word opens the mouth, which then relaxes. Later, with
// real audio from the backend, mouthRef.current.level (0–1 loudness from the
// Web Audio API) will drive it instead — see `level` below.
//
// It animates by changing SVG attributes directly ~60×/s (requestAnimationFrame)
// instead of through React state, so the page doesn't re-render 60×/s.
//
// mood: 'talk' | 'listen' | 'think' | 'idle'
const SKIN = '#f1c3a3'
const SKIN_SHADE = '#d99a7c'
const HAIR = '#3b2a24'

function mouthPaths(o) {
  // o = how open (0 closed … 1 wide)
  const w = 30 - o * 6
  const lift = 1.5 + o * 3
  const drop = 3 + o * 20
  const mouth = `M ${-w / 2} 0 Q 0 ${-lift} ${w / 2} 0 Q 0 ${drop} ${-w / 2} 0 Z`
  const teeth = `M ${-w / 2 + 4} 0.4 Q 0 ${-lift + 1} ${w / 2 - 4} 0.4 L ${w / 2 - 6} ${2 + o * 3} Q 0 ${1.5 + o * 2} ${-w / 2 + 6} ${2 + o * 3} Z`
  return { mouth, teeth, tongueY: drop * 0.62, tongueOpacity: Math.max(0, (o - 0.3) * 1.6) }
}

export default function TeacherAvatar({ mouthRef, mood = 'idle' }) {
  const mouthEl = useRef(null)
  const teethEl = useRef(null)
  const tongueEl = useRef(null)
  const smileEl = useRef(null)
  const eyesEl = useRef(null)

  useEffect(() => {
    let raf
    let open = 0
    let nextBlink = performance.now() + 2500
    let blinkUntil = 0
    const tick = (t) => {
      const m = mouthRef?.current || {}
      let target = 0
      if (typeof m.level === 'number') {
        target = m.level // real audio loudness (backend phase)
      } else if (m.talking) {
        const since = t - (m.lastWord || 0)
        if (since < 450) {
          // open on each word, close as the word ends, with a little flutter
          const env = since < 80 ? since / 80 : Math.max(0, 1 - (since - 80) / 260)
          target = env * (0.45 + 0.5 * (m.amp ?? 0.6)) + 0.08 * Math.sin(t / 45)
        } else {
          target = 0.05
        }
      }
      open += (Math.max(0, Math.min(1, target)) - open) * 0.4
      const p = mouthPaths(open)
      mouthEl.current?.setAttribute('d', p.mouth)
      teethEl.current?.setAttribute('d', p.teeth)
      teethEl.current?.setAttribute('opacity', open > 0.15 ? 1 : 0)
      tongueEl.current?.setAttribute('cy', p.tongueY)
      tongueEl.current?.setAttribute('opacity', p.tongueOpacity)
      mouthEl.current?.setAttribute('opacity', open > 0.06 ? 1 : 0)
      smileEl.current?.setAttribute('opacity', open > 0.06 ? 0 : 1)

      if (t > nextBlink) {
        blinkUntil = t + 130
        nextBlink = t + 2200 + Math.random() * 3200
      }
      const s = t < blinkUntil ? 0.08 : 1 // squash the eyes around their centre line (y = 138)
      eyesEl.current?.setAttribute('transform', `translate(0 ${138 * (1 - s)}) scale(1 ${s})`)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [mouthRef])

  const browY = mood === 'listen' ? -6 : mood === 'think' ? -3 : 0
  // CSS transforms (not SVG attributes) so the change animates smoothly.
  const look = mood === 'think' ? 'translate(4px, -4px)' : mood === 'listen' ? 'translate(-2px, 0)' : 'none'
  const tilt = mood === 'listen' ? 'rotate(-4deg)' : mood === 'think' ? 'rotate(3deg)' : 'none'

  return (
    <svg viewBox="0 0 300 330" role="img" aria-label={`Teacher, ${mood === 'talk' ? 'speaking' : mood === 'listen' ? 'listening' : mood === 'think' ? 'thinking' : 'waiting'}`}>
      <style>{`
        .breathe { animation: breathe 4s ease-in-out infinite; transform-origin: 150px 330px; }
        @keyframes breathe { 0%,100% { transform: translateY(0) } 50% { transform: translateY(1.5px) } }
        .tilt { transition: transform 0.5s ease; }
        .brows, .pupils { transition: transform 0.35s ease; }
        .dot { animation: dot 1.2s ease-in-out infinite; }
        .dot:nth-child(2) { animation-delay: 0.2s } .dot:nth-child(3) { animation-delay: 0.4s }
        @keyframes dot { 0%,100% { opacity: .25 } 50% { opacity: 1 } }
      `}</style>
      <g className="breathe">
        {/* body */}
        <path d="M22 330 Q30 262 108 244 L192 244 Q270 262 278 330 Z" fill="#33467a" />
        <path d="M126 244 L150 300 L174 244 Z" fill="#eef0f8" />
        <path d="M108 244 L150 312 L128 330 M192 244 L150 312 L172 330" stroke="#25345e" strokeWidth="3" fill="none" />
        <rect x="132" y="196" width="36" height="54" rx="10" fill={SKIN_SHADE} />

        <g className="tilt" style={{ transform: tilt, transformOrigin: '150px 230px' }}>
          {/* hair behind the head */}
          <path d="M76 150 Q66 52 150 46 Q234 52 224 150 L232 238 Q210 256 190 242 L188 160 L112 160 L110 242 Q90 256 68 238 Z" fill={HAIR} />
          {/* ears + head */}
          <ellipse cx="88" cy="146" rx="10" ry="15" fill={SKIN_SHADE} />
          <ellipse cx="212" cy="146" rx="10" ry="15" fill={SKIN_SHADE} />
          <ellipse cx="150" cy="138" rx="62" ry="74" fill={SKIN} />
          {/* fringe */}
          <path d="M86 132 Q82 62 150 58 Q218 62 214 132 Q204 92 160 86 Q118 90 100 112 Q92 122 86 132 Z" fill={HAIR} />
          {/* cheeks */}
          <circle cx="114" cy="166" r="11" fill="#f08a8a" opacity="0.22" />
          <circle cx="186" cy="166" r="11" fill="#f08a8a" opacity="0.22" />
          {/* brows */}
          <g className="brows" style={{ transform: `translateY(${browY}px)` }}>
            <path d="M110 112 Q124 104 139 110" stroke={HAIR} strokeWidth="4.5" strokeLinecap="round" fill="none" />
            <path d="M161 110 Q176 104 190 112" stroke={HAIR} strokeWidth="4.5" strokeLinecap="round" fill="none" />
          </g>
          {/* eyes (blink = squash vertically around y≈138) */}
          <g ref={eyesEl}>
            <g className="pupils" style={{ transform: look }}>
              <circle cx="125" cy="138" r="6.5" fill="#2a2320" />
              <circle cx="175" cy="138" r="6.5" fill="#2a2320" />
              <circle cx="127.5" cy="135.5" r="2" fill="#fff" />
              <circle cx="177.5" cy="135.5" r="2" fill="#fff" />
            </g>
          </g>
          {/* glasses */}
          <g stroke="#1d1d1f" strokeWidth="3" fill="rgba(255,255,255,0.08)">
            <rect x="104" y="122" width="42" height="32" rx="12" />
            <rect x="154" y="122" width="42" height="32" rx="12" />
          </g>
          <path d="M146 136 Q150 132 154 136 M104 134 L90 130 M196 134 L210 130" stroke="#1d1d1f" strokeWidth="3" fill="none" />
          {/* nose */}
          <path d="M151 146 Q146 164 153 168" stroke={SKIN_SHADE} strokeWidth="3" strokeLinecap="round" fill="none" />
          {/* mouth */}
          <g transform="translate(150 186)">
            <path ref={smileEl} d="M-15 -2 Q0 8 15 -2" stroke="#b8545a" strokeWidth="3.5" strokeLinecap="round" fill="none" />
            <path ref={mouthEl} d="" fill="#6e2430" stroke="#c4626a" strokeWidth="2.5" strokeLinejoin="round" opacity="0" />
            <path ref={teethEl} d="" fill="#fbfbfb" opacity="0" />
            <ellipse ref={tongueEl} cx="0" cy="6" rx="7" ry="4" fill="#e07a84" opacity="0" />
          </g>
        </g>

        {/* thinking dots */}
        {mood === 'think' && (
          <g fill="#fdd663">
            <circle className="dot" cx="236" cy="60" r="5" />
            <circle className="dot" cx="254" cy="44" r="6" />
            <circle className="dot" cx="274" cy="26" r="7" />
          </g>
        )}
      </g>
    </svg>
  )
}
