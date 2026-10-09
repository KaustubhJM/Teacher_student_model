import Icon from './Icon.jsx'

// Shown whenever the page is running on mock data (backend not running),
// so a placeholder lecture can't be mistaken for a real one.
export default function PreviewBanner({ compact = false }) {
  return (
    <div className={`preview-banner ${compact ? 'compact' : ''}`} role="status">
      <Icon name="science" />
      <div>
        <strong>Preview — not generated from your source.</strong>
        {!compact && (
          <>
            {' '}The backend isn't running, so lectures, doubts and quizzes here are placeholders. Start it from the
            project folder with <code>uvicorn server.app:app --port 8000</code> and reload this page.
          </>
        )}
      </div>
    </div>
  )
}
