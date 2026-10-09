// MOCK DATA — stands in for the Python backend until the API exists.
// The shapes here are what the real API will return, so the UI code
// won't change much when we switch from mocks to real calls.

export const STUDENT_NAME = 'Kaustubh'
export const TEACHER_NAME = 'Ms. Ava'

export const MOCK_CLASSES = [
  {
    id: 'dbms-u5',
    title: 'DBMS Unit 5 — Concurrency Control',
    emoji: '🗄️',
    createdAt: '2026-10-07',
    sources: [{ type: 'pdf', name: 'DBMS_U_5_ONE_SHOT.pdf', detail: '72/72 pages read' }],
    outline: [
      'Locking fundamentals',
      'Two-phase locking and its variants',
      'Timestamp ordering',
      'Multiversion concurrency control',
      'The phantom problem',
      'Recovery and two-phase commit',
    ],
    sectionsDone: 1,
    status: 'in_progress',
    quizScore: null,
  },
  {
    id: 'tc-u5',
    title: 'TC Unit 5 — Content Creation & Personality',
    emoji: '🧠',
    createdAt: '2026-10-08',
    sources: [{ type: 'pdf', name: 'TC-U-5-ONE-SHOT-Notes.pdf', detail: '58/58 pages read' }],
    outline: [
      'Introduction to content creation',
      'Content formats',
      'Ethics on social media',
      'Personality theories',
      'Strategies for engaging content',
    ],
    sectionsDone: 5,
    status: 'completed',
    quizScore: { got: 8, total: 10 },
  },
  {
    id: 'ai-u1',
    title: 'Foundations of AI and Intelligent Agents',
    emoji: '🤖',
    createdAt: '2026-09-26',
    sources: [{ type: 'pdf', name: 'UNIT 1 (Artificial Intelligence).pdf', detail: '29/29 pages read' }],
    outline: [
      'What is AI?',
      'History of AI',
      'Types of AI',
      'Intelligent agents',
      'Agent environments',
    ],
    sectionsDone: 5,
    status: 'completed',
    quizScore: { got: 9, total: 10 },
  },
  {
    id: 'yt-backprop',
    title: 'Backpropagation, step by step',
    emoji: '📺',
    createdAt: '2026-09-20',
    sources: [{ type: 'youtube', name: 'YouTube video', detail: 'Transcript, 19 min' }],
    outline: [
      'Why we need gradients',
      'The chain rule',
      'Backward pass through a layer',
      'Putting it together',
    ],
    sectionsDone: 0,
    status: 'new',
    quizScore: null,
  },
]

// A short sample lecture. Real beats are ~600 words per section; these are
// trimmed so the preview moves quickly.
const DBMS_SCRIPT = [
  [
    'A lock is a small piece of bookkeeping that says which transaction may touch a data item right now.',
    'There are two basic kinds: a shared lock for reading, and an exclusive lock for writing.',
    'Many transactions can hold a shared lock on the same item, but an exclusive lock must be alone.',
    'So, quick check — can two transactions read the same row at the same time?',
  ],
  [
    'Two-phase locking splits every transaction into two phases.',
    'In the growing phase it may acquire locks but never release one.',
    "Once it releases its first lock, it enters the shrinking phase and can't take any new lock.",
    'This simple rule guarantees conflict serializable schedules.',
    'But be careful: two-phase locking does not prevent deadlocks — two transactions can still wait for each other forever.',
    'Strict two-phase locking goes one step further and holds every exclusive lock until the transaction commits.',
  ],
  [
    'Timestamp ordering takes a different approach: no locks at all.',
    'Every transaction gets a timestamp when it starts, and the database pretends transactions ran in that order.',
    'Each data item remembers the largest read timestamp and the largest write timestamp that touched it.',
    'If an old transaction tries to write something a younger one already read, the old one is rolled back.',
  ],
  [
    'Multiversion concurrency control keeps several versions of each data item.',
    'Readers get the version that was current when they started, so they never wait for writers.',
    'Writers create a new version instead of overwriting the old one.',
    'This is why many real databases let long reports run while updates keep flowing.',
  ],
  [
    'A phantom is a row that appears or disappears between two reads of the same query.',
    'Locking the rows you read is not enough, because the new row did not exist when you locked.',
    'The fix is to lock the condition itself, for example with index or predicate locks.',
  ],
  [
    'Recovery makes sure a crash never leaves the database half updated.',
    'The log records every change before it reaches the disk, so the system can undo or redo work.',
    'For transactions spread over many servers, two-phase commit first asks everyone to prepare, then tells everyone to commit.',
    'A participant that has prepared must wait for the coordinator\'s decision — it cannot decide on its own.',
  ],
]

export function getLectureScript(cls) {
  if (cls.id === 'dbms-u5') return DBMS_SCRIPT
  return cls.outline.map((section) => [
    `Let's talk about ${section.toLowerCase()}.`,
    'In the real class, this part is generated from your notes, three beats per section.',
    'This preview just shows how the lecture will look and feel.',
  ])
}

export const RESUME_PHRASES = [
  'Alright, back to where we were.',
  'Okay, let me pick up from there.',
  'Good. Now, continuing.',
]

export function mockDoubtAnswer(question) {
  return (
    `Good question, ${STUDENT_NAME}! In the real class I'll search your notes for "${question}" ` +
    'and answer from them. This preview only shows how the conversation will flow.'
  )
}

export const MOCK_QUIZ = [
  {
    q: 'Which lock can be held by many transactions on the same item at once?',
    options: ['Exclusive lock', 'Shared lock', 'Predicate lock', 'Intention lock'],
    answer: 1,
    why: 'Shared (read) locks are compatible with each other; an exclusive lock must be alone.',
  },
  {
    q: 'What does two-phase locking guarantee?',
    options: [
      'No deadlocks',
      'Conflict serializable schedules',
      'No cascading rollbacks',
      'No phantoms',
    ],
    answer: 1,
    why: '2PL guarantees conflict serializability. Deadlocks are still possible.',
  },
  {
    q: 'In multiversion concurrency control, what happens to readers?',
    options: [
      'They wait for writers to finish',
      'They lock the whole table',
      'They read the version current when they started',
      'They are rolled back',
    ],
    answer: 2,
    why: 'Readers see a consistent older version, so they never block on writers.',
  },
  {
    q: 'In two-phase commit, a participant that has PREPARED and lost contact with the coordinator should…',
    options: [
      'Commit on its own',
      'Roll back on its own',
      "Wait for the coordinator's decision",
      'Restart the transaction',
    ],
    answer: 2,
    why: 'A prepared participant is "in doubt" and must wait — deciding alone could break atomicity.',
  },
]
