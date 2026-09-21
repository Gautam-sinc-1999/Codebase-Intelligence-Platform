import React, { useState } from 'react';

export function Sidebar({
  repositories,
  selectedRepoId,
  onSelectRepo,
  conversations,
  activeConvId,
  onSelectConv,
  onNewChat,
  onDeleteConv,
  onRenameConv,
  onReindexRepo,
  onSyncRepo,
  onDeleteRepo,
  onOpenUploadModal,
  health,
  repoStatus,
  pending = {}
}) {
  const currentRepo = repositories.find(r => r.repository_id === selectedRepoId);
  const [reindexing, setReindexing] = useState(false);
  // The thread currently being renamed, and the text in its input. A title generated from the
  // first question is a fair default and a poor label for something you come back to.
  const [renamingId, setRenamingId] = useState(null);
  const [draftTitle, setDraftTitle] = useState('');

  const startRename = (conv) => {
    setRenamingId(conv.conversation_id);
    setDraftTitle(conv.title || '');
  };

  const commitRename = async (conv) => {
    const next = draftTitle.trim();
    setRenamingId(null);
    // An unchanged or empty title is a cancel, not a rename: the server would reject the empty
    // one anyway, and reporting an error for "pressed Enter having changed nothing" is noise.
    if (!next || next === conv.title) return;
    await onRenameConv(conv.conversation_id, next);
  };
  const degraded = health?.degraded || [];

  // A GitHub repository keeps no local source — the clone is deleted once indexed — so
  // re-indexing it means re-cloning, which is what /sync does. Choosing the wrong one gives the
  // user a 409 telling them to re-upload something they never uploaded.
  const isGithub = currentRepo?.source === 'github' || repoStatus?.source === 'github';

  const refresh = async (event) => {
    if (!selectedRepoId || reindexing) return;
    // Shift-click forces a full rebuild; the default is incremental, which is what you want
    // unless the index itself is suspect rather than the source.
    setReindexing(true);
    try {
      if (isGithub) {
        await onSyncRepo(selectedRepoId, { full: event.shiftKey });
      } else {
        await onReindexRepo(selectedRepoId, { full: event.shiftKey });
      }
    } finally {
      setReindexing(false);
    }
  };

  return (
    <aside className="sidebar glass-panel" style={{ width: '290px', height: '100%', display: 'flex', flexDirection: 'column', padding: '16px', borderRight: '1px solid var(--border-glass)' }}>
      {/* Brand Logo Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
        <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'linear-gradient(135deg, #6366f1, #a855f7)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '1.2rem', color: '#fff' }}>
          C
        </div>
        <div>
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', fontWeight: 700, background: 'linear-gradient(90deg, #fff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Codebase Intelligence
          </h2>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Conversational Code RAG</span>
        </div>
      </div>

      {/* Repository Selector */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
          Active Repository
        </label>
        <select
          value={selectedRepoId || ''}
          onChange={(e) => onSelectRepo(e.target.value)}
          style={{ width: '100%', background: 'rgba(255,255,255,0.06)', color: 'var(--text-primary)', border: '1px solid var(--border-glass)', borderRadius: '8px', padding: '8px', fontSize: '0.85rem', cursor: 'pointer', outline: 'none' }}
        >
          {repositories.map((repo) => (
            <option key={repo.repository_id} value={repo.repository_id} style={{ background: '#111827' }}>
              {repo.name} ({repo.file_count} files)
            </option>
          ))}
        </select>

        <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
          <button
            onClick={onOpenUploadModal}
            className="btn-secondary"
            style={{ flex: 1, justifyContent: 'center', fontSize: '0.8rem' }}
          >
            + Upload Zip
          </button>
          <button
            onClick={refresh}
            disabled={!selectedRepoId || reindexing}
            className="btn-secondary"
            title={isGithub
              ? 'Re-clone from GitHub and re-index what changed. Shift-click for a full rebuild.'
              : 'Re-index from the stored source (incremental). Shift-click for a full rebuild.'}
            style={{ justifyContent: 'center', fontSize: '0.8rem', opacity: reindexing ? 0.6 : 1 }}
          >
            {reindexing ? '⏳' : '⟳'} {isGithub ? 'Sync' : 'Re-index'}
          </button>
          <button
            onClick={() => onDeleteRepo(selectedRepoId)}
            disabled={!selectedRepoId || reindexing}
            className="btn-secondary"
            title="Delete this repository and everything indexed from it"
            style={{ justifyContent: 'center', fontSize: '0.8rem', padding: '0 10px' }}
          >
            🗑
          </button>
        </div>

        {/* Imports that are still running. A GitHub import returns before indexing finishes, so
            without this the repository simply would not appear for half a minute. */}
        {Object.entries(pending).map(([repoId, info]) => (
          <div
            key={repoId}
            style={{
              marginTop: '8px', padding: '8px 10px', borderRadius: '8px', fontSize: '0.75rem',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid',
              borderColor: info.status === 'failed' ? 'var(--accent-rose)' : 'var(--border-glass)',
              color: info.status === 'failed' ? 'var(--accent-rose)' : 'var(--text-secondary)'
            }}
          >
            <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {info.status === 'failed' ? '✕' : '⏳'} {info.name || repoId}
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              {info.status === 'failed' ? info.error : 'cloning & indexing…'}
            </div>
          </div>
        ))}
      </div>

      {/* Selected Repo Stats */}
      {currentRepo && (
        <div className="glass-card" style={{ padding: '10px', marginBottom: '16px', fontSize: '0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)' }}>
            <span>Files: <strong>{currentRepo.file_count}</strong></span>
            <span>Lines: <strong>{currentRepo.total_lines}</strong></span>
          </div>
          <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginTop: '6px' }}>
            {currentRepo.languages.map((lang) => (
              <span key={lang} style={{ background: 'rgba(255,255,255,0.08)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.68rem', color: 'var(--accent-cyan)' }}>
                {lang}
              </span>
            ))}
          </div>
          {/* Whether the index is actually queryable — a repository row exists before indexing
              finishes, so "ready" is not implied by the repository being listed. */}
          {repoStatus && repoStatus.repository_id === selectedRepoId && (
            <div style={{ marginTop: '6px', fontSize: '0.68rem', color: repoStatus.status === 'ready' ? 'var(--accent-emerald)' : 'var(--accent-amber)' }}>
              ● Index: {repoStatus.status}
              {repoStatus.chunk_count ? ` — ${repoStatus.chunk_count} chunks` : ''}
              {repoStatus.upToDate ? ' · up to date' : ''}
            </div>
          )}
          {/* Upstream identity, for a repository that has one. The commit is what any answer
              about this repository was actually based on. */}
          {repoStatus?.source === 'github' && repoStatus.commit_sha && (
            <div style={{ marginTop: '4px', fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              ⑂ {repoStatus.branch} @ {repoStatus.commit_sha.slice(0, 7)}
            </div>
          )}
        </div>
      )}

      {/* New Chat Button */}
      <button
        onClick={onNewChat}
        className="btn-primary"
        style={{ width: '100%', justifyContent: 'center', marginBottom: '16px' }}
      >
        + New Analysis Chat
      </button>

      {/* Saved Conversations List */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
        <h3 style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '10px' }}>
          Saved Discussion Sessions
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {conversations.length === 0 ? (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center', padding: '12px' }}>
              No chats yet. Start a new analysis!
            </div>
          ) : (
            conversations.map((conv) => {
              const isActive = conv.conversation_id === activeConvId;
              return (
                <div
                  key={conv.conversation_id}
                  style={{
                    display: 'flex',
                    alignItems: 'stretch',
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: isActive ? 'var(--accent-indigo)' : 'transparent',
                    background: isActive ? 'rgba(99, 102, 241, 0.2)' : 'rgba(255, 255, 255, 0.03)',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <button
                    onClick={() => onSelectConv(conv.conversation_id)}
                    onDoubleClick={() => startRename(conv)}
                    title="Double-click to rename"
                    style={{
                      flex: 1,
                      minWidth: 0,
                      textAlign: 'left',
                      padding: '10px 4px 10px 12px',
                      background: 'transparent',
                      border: 'none',
                      color: isActive ? '#fff' : 'var(--text-secondary)',
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px'
                    }}
                  >
                    {renamingId === conv.conversation_id ? (
                      <input
                        autoFocus
                        value={draftTitle}
                        maxLength={120}
                        onChange={(e) => setDraftTitle(e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                        onBlur={() => commitRename(conv)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') { e.preventDefault(); commitRename(conv); }
                          if (e.key === 'Escape') { e.preventDefault(); setRenamingId(null); }
                        }}
                        style={{
                          width: '100%', background: 'rgba(0,0,0,0.45)', color: '#fff',
                          border: '1px solid var(--accent-indigo)', borderRadius: '5px',
                          padding: '3px 6px', fontSize: '0.85rem', outline: 'none'
                        }}
                      />
                    ) : (
                      <div style={{ fontWeight: isActive ? 600 : 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        💬 {conv.title || 'Analysis Session'}
                      </div>
                    )}
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', gap: '6px' }}>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>Repo: {conv.repository_id}</span>
                      {/* The list endpoint returns metadata only — message_count — so the sidebar
                          never loads message history it does not display. */}
                      <span style={{ flexShrink: 0 }}>{conv.message_count ?? (conv.messages ? conv.messages.length : 0)} msgs</span>
                    </div>
                  </button>
                  <button
                    onClick={() => startRename(conv)}
                    title="Rename this thread"
                    aria-label={`Rename ${conv.title || 'this thread'}`}
                    style={{
                      flexShrink: 0,
                      padding: '0 6px',
                      background: 'transparent',
                      border: 'none',
                      color: 'var(--text-muted)',
                      cursor: 'pointer',
                      fontSize: '0.78rem'
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent-cyan)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
                  >
                    ✎
                  </button>
                  <button
                    onClick={() => onDeleteConv(conv)}
                    title="Delete this thread"
                    aria-label={`Delete ${conv.title || 'this thread'}`}
                    style={{
                      flexShrink: 0,
                      padding: '0 10px',
                      background: 'transparent',
                      border: 'none',
                      color: 'var(--text-muted)',
                      cursor: 'pointer',
                      fontSize: '0.8rem'
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent-rose)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
                  >
                    ✕
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Every store falls back silently when its server is unreachable, so without this the
          system looks identical whether it is durably persisting data or holding it in memory. */}
      {health && (
        <div
          title={health.backends ? Object.entries(health.backends).map(([k, v]) => `${k}: ${v}`).join('\n') : ''}
          style={{
            marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-glass)',
            fontSize: '0.68rem', color: degraded.length ? 'var(--accent-amber)' : 'var(--text-muted)'
          }}
        >
          {degraded.length
            ? `⚠ Degraded: ${degraded.join(', ')} — using in-process fallbacks`
            : `● All stores connected${health.version ? ` · v${health.version}` : ''}`}
        </div>
      )}
    </aside>
  );
}
