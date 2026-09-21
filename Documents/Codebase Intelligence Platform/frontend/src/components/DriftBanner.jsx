import React, { useState } from 'react';
import { driftBanner } from '../lib/drift';

/**
 * Tells the reader the repository has moved since this thread was written, and offers the choice
 * between staying put and pulling the latest.
 *
 * Whether to show anything at all, and how to phrase it, lives in `lib/drift.js` — a plain module
 * so that decision can be tested directly. This file is only presentation.
 */
export function DriftBanner({ drift, onKeep, onPull, busy }) {
  const [dismissed, setDismissed] = useState(false);

  const banner = driftBanner(drift);
  if (!banner || dismissed) return null;

  const warning = banner.severity === 'warning';
  const accent = warning ? 'var(--accent-rose)' : 'var(--accent-amber)';
  const tint = warning ? 'rgba(244, 63, 94, 0.10)' : 'rgba(245, 158, 11, 0.10)';

  const keep = async () => {
    await onKeep(drift.remote_sha);
    setDismissed(true);
  };

  return (
    <div
      style={{
        display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap',
        padding: '10px 14px', borderRadius: '10px',
        background: tint, border: `1px solid ${accent}`, fontSize: '0.8rem'
      }}
    >
      <div style={{ flex: 1, minWidth: '220px' }}>
        <div style={{ color: '#fff', fontWeight: 600 }}>
          {warning ? '⚠' : '⟳'} {banner.headline}
        </div>
        {banner.detail && (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '3px', fontFamily: 'var(--font-mono)' }}>
            {banner.detail}
          </div>
        )}
      </div>

      {banner.actionable ? (
        <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
          <button onClick={keep} className="btn-secondary" disabled={busy} style={{ fontSize: '0.76rem' }}>
            Keep this version
          </button>
          <button onClick={() => onPull()} className="btn-primary" disabled={busy} style={{ fontSize: '0.76rem' }}>
            {busy ? 'Syncing…' : 'Pull latest & re-index'}
          </button>
        </div>
      ) : (
        <button onClick={() => setDismissed(true)} className="btn-secondary" style={{ fontSize: '0.76rem', flexShrink: 0 }}>
          Dismiss
        </button>
      )}
    </div>
  );
}
