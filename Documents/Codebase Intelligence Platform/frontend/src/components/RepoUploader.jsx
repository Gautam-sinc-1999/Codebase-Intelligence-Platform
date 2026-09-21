import React, { useState } from 'react';
import { uploadRepositoryFile, fetchGithubBranches, importFromGithub } from '../services/api';

/**
 * Adds a repository, by zip upload or by GitHub URL.
 *
 * The two tabs differ in more than appearance. A zip upload is **synchronous**: the response
 * carries the finished index, so the modal can hand a complete repository straight to the caller.
 * A GitHub import is **asynchronous** — cloning and embedding a real repository takes tens of
 * seconds — so it returns a handle, the modal closes immediately, and the sidebar shows the
 * repository indexing while the caller polls.
 */
export function RepoUploader({ isOpen, onClose, onUploadSuccess, onImportStarted }) {
  const [tab, setTab] = useState('zip');

  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [url, setUrl] = useState('');
  const [branches, setBranches] = useState(null);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [loadingBranches, setLoadingBranches] = useState(false);
  const [importing, setImporting] = useState(false);

  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const busy = uploading || loadingBranches || importing;

  const close = () => {
    if (busy) return;
    setError(null);
    setBranches(null);
    setSelectedBranch('');
    onClose();
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      // Returns once the archive is extracted, not once it is indexed. A corrupt or oversized
      // archive still fails here, synchronously, which is what the error box is for.
      const started = await uploadRepositoryFile(file);
      onUploadSuccess(started);
      close();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleFetchBranches = async () => {
    if (!url.trim()) return;
    setLoadingBranches(true);
    setError(null);
    setBranches(null);
    try {
      const result = await fetchGithubBranches(url.trim());
      setBranches(result);
      setSelectedBranch(result.default_branch || result.branches[0]?.name || '');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingBranches(false);
    }
  };

  const handleImport = async () => {
    if (!url.trim() || !selectedBranch) return;
    setImporting(true);
    setError(null);
    try {
      const started = await importFromGithub(url.trim(), selectedBranch);
      // Closes on 202 rather than on completion: indexing continues server-side and the sidebar
      // reports progress, so holding the modal open would only block the user for no reason.
      onImportStarted(started);
      close();
    } catch (err) {
      setError(err.message);
    } finally {
      setImporting(false);
    }
  };

  const tabStyle = (name) => ({
    flex: 1,
    padding: '8px',
    background: tab === name ? 'rgba(99, 102, 241, 0.22)' : 'transparent',
    border: '1px solid',
    borderColor: tab === name ? 'var(--accent-indigo)' : 'var(--border-glass)',
    borderRadius: '8px',
    color: tab === name ? '#fff' : 'var(--text-secondary)',
    fontSize: '0.82rem',
    fontWeight: tab === name ? 600 : 500,
    cursor: 'pointer'
  });

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100, padding: '16px' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '460px', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-glow)' }}>
        <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.2rem', fontWeight: 700, marginBottom: '14px', color: '#fff' }}>
          Add Repository
        </h3>

        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
          <button onClick={() => { setTab('zip'); setError(null); }} style={tabStyle('zip')} disabled={busy}>
            📁 Upload .zip
          </button>
          <button onClick={() => { setTab('github'); setError(null); }} style={tabStyle('github')} disabled={busy}>
            🐙 GitHub URL
          </button>
        </div>

        {tab === 'zip' ? (
          <>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '14px' }}>
              Select a <code>.zip</code> of your codebase. The engine parses AST symbols, builds embeddings, and constructs the call graph — indexing continues in the background once the upload completes.
            </p>

            <div style={{ border: '2px dashed var(--border-glass)', borderRadius: '12px', padding: '24px', textAlign: 'center', background: 'rgba(255,255,255,0.02)', marginBottom: '16px' }}>
              <input
                type="file"
                accept=".zip"
                onChange={(e) => setFile(e.target.files[0])}
                style={{ display: 'none' }}
                id="repo-zip-input"
              />
              <label htmlFor="repo-zip-input" style={{ cursor: 'pointer', display: 'block' }}>
                <div style={{ fontSize: '2rem', marginBottom: '8px' }}>📁</div>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-indigo)' }}>
                  {file ? file.name : 'Click to select repo archive (.zip)'}
                </div>
                {file && <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>{(file.size / 1024 / 1024).toFixed(2)} MB</div>}
              </label>
            </div>
          </>
        ) : (
          <>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '10px' }}>
              Paste a public GitHub repository URL. Only the selected branch is cloned, and the working copy is deleted once indexed.
            </p>

            <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
              <input
                type="text"
                value={url}
                onChange={(e) => { setUrl(e.target.value); setBranches(null); }}
                onKeyDown={(e) => { if (e.key === 'Enter') handleFetchBranches(); }}
                placeholder="https://github.com/owner/repo"
                disabled={busy}
                style={{
                  flex: 1, minWidth: 0, background: 'rgba(255,255,255,0.05)',
                  border: '1px solid var(--border-glass)', borderRadius: '8px',
                  padding: '9px 12px', color: '#fff', fontSize: '0.82rem', outline: 'none'
                }}
              />
              <button onClick={handleFetchBranches} className="btn-secondary" disabled={busy || !url.trim()} style={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                {loadingBranches ? '…' : 'Branches'}
              </button>
            </div>

            {branches && (
              <div style={{ marginBottom: '14px' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Branches in {branches.name}
                </div>
                <div style={{ maxHeight: '190px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {branches.branches.map((b) => (
                    <label
                      key={b.name}
                      style={{
                        display: 'flex', alignItems: 'center', gap: '8px', padding: '7px 10px',
                        borderRadius: '7px', cursor: 'pointer', fontSize: '0.8rem',
                        background: selectedBranch === b.name ? 'rgba(99,102,241,0.18)' : 'rgba(255,255,255,0.03)',
                        border: '1px solid',
                        borderColor: selectedBranch === b.name ? 'var(--accent-indigo)' : 'transparent'
                      }}
                    >
                      <input
                        type="radio"
                        name="branch"
                        checked={selectedBranch === b.name}
                        onChange={() => setSelectedBranch(b.name)}
                        style={{ accentColor: 'var(--accent-indigo)' }}
                      />
                      <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#fff' }}>
                        {b.name}
                      </span>
                      {b.is_default && (
                        <span style={{ fontSize: '0.65rem', color: 'var(--accent-cyan)', flexShrink: 0 }}>default</span>
                      )}
                      {/* Says what is already indexed, rather than making the user remember. */}
                      {b.indexed && (
                        <span style={{ fontSize: '0.65rem', color: 'var(--accent-emerald)', flexShrink: 0 }}>indexed</span>
                      )}
                    </label>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {error && (
          <div style={{ background: 'rgba(244, 63, 94, 0.15)', color: 'var(--accent-rose)', border: '1px solid rgba(244, 63, 94, 0.3)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.8rem', marginBottom: '16px' }}>
            {error}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
          <button onClick={close} className="btn-secondary" disabled={busy}>Cancel</button>
          {tab === 'zip' ? (
            <button onClick={handleUpload} className="btn-primary" disabled={!file || busy}>
              {uploading ? 'Uploading…' : 'Upload & Index'}
            </button>
          ) : (
            <button onClick={handleImport} className="btn-primary" disabled={!selectedBranch || busy}>
              {importing ? 'Starting…' : 'Import & Index'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
