/** Shared inline error surface: names what failed, offers the retry action. */

interface Props {
  message: string | null
  onRetry?: () => void
}

export function ErrorNote({ message, onRetry }: Props) {
  if (message === null) return null
  return (
    <p className="error" role="alert">
      {message}{' '}
      {onRetry !== undefined && (
        <button type="button" className="link" onClick={onRetry}>
          retry
        </button>
      )}
    </p>
  )
}
