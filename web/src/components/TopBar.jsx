import { Link } from 'react-router-dom'
import Icon from './Icon.jsx'
import { STUDENT_NAME } from '../data/mockData.js'

// Top bar for both pages. Home: brand + search. Class page: logo + class title.
export default function TopBar({ title, search, onSearch, onNewClass }) {
  return (
    <header className="topbar">
      <Link to="/" className="brand" title="All classes">
        <img src="/favicon.svg" alt="" className="brand-logo" />
        {!title && <span>AI Classroom</span>}
      </Link>
      {title && <span className="title">{title}</span>}
      <span className="spacer" />
      {onSearch && (
        <label className="search">
          <Icon name="search" />
          <input
            placeholder="Search classes"
            value={search}
            onChange={(e) => onSearch(e.target.value)}
            aria-label="Search classes"
          />
        </label>
      )}
      {title && onNewClass && (
        <button className="btn small" onClick={onNewClass}>
          <Icon name="add" /> New class
        </button>
      )}
      <button className="icon-btn" aria-label="Settings" title="Settings">
        <Icon name="settings" />
      </button>
      <div className="avatar-chip" title={STUDENT_NAME}>
        {STUDENT_NAME[0]}
      </div>
    </header>
  )
}
