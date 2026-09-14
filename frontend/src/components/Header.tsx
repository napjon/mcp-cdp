import type { AppStatus } from '../types'

type Props = {
  projectName: string
  appStatus: AppStatus | null
  apiReachable: boolean
  bootNotice: string | null
}

export function Header({ projectName, appStatus, apiReachable, bootNotice }: Props) {
  const limitedChat = appStatus != null && !appStatus.llm_configured
  return (
    <header className="header">
      <h1>{projectName}</h1>
      <div className="header-meta">
        {bootNotice ? (
          <p className="notice danger" role="status">
            {bootNotice}
          </p>
        ) : null}
        {!apiReachable && !bootNotice ? (
          <p className="notice danger" role="status">
            Can’t reach the local app.
          </p>
        ) : null}
        {limitedChat ? (
          <p className="notice" role="status">
            Chat answers are limited — training still works.
          </p>
        ) : null}
      </div>
    </header>
  )
}
