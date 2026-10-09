// <Icon name="add" /> → a Material Symbols icon. Names: fonts.google.com/icons
export default function Icon({ name, filled = false, className = '', style }) {
  return (
    <span className={`icon ${filled ? 'filled' : ''} ${className}`} style={style} aria-hidden="true">
      {name}
    </span>
  )
}
