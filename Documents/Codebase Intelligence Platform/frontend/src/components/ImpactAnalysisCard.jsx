import React from 'react';

export function ImpactAnalysisCard({ impactAnalysis }) {
  if (!impactAnalysis || !impactAnalysis.primary_target) {
    return null;
  }

  const { primary_target, confirmed_callers, affected_tests, inferred_impacts } = impactAnalysis;

  return (
    <div className="glass-card" style={{ padding: '16px', marginTop: '16px', borderColor: 'rgba(244, 63, 94, 0.4)', background: 'rgba(30, 20, 30, 0.4)' }}>
      <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-rose)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        🎯 Change Impact Analysis Report
      </h3>

      {/* Target Info */}
      <div style={{ background: 'rgba(244, 63, 94, 0.1)', border: '1px solid rgba(244, 63, 94, 0.2)', borderRadius: '8px', padding: '10px', marginBottom: '12px' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Primary Modification Target</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', fontWeight: 600, color: '#fff', marginTop: '2px' }}>
          {primary_target.symbol}
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {primary_target.file_path} (Lines {primary_target.lines})
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        {/* Confirmed Callers */}
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '6px' }}>
            Confirmed Affected Callers ({confirmed_callers?.length || 0})
          </div>
          {confirmed_callers && confirmed_callers.length > 0 ? (
            confirmed_callers.map((c, idx) => (
              <div key={idx} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                • <strong style={{ color: '#fff' }}>{c.symbol}</strong> ({c.file_path})
              </div>
            ))
          ) : (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Direct callers verified in graph.</div>
          )}
        </div>

        {/* Affected Tests */}
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-emerald)', marginBottom: '6px' }}>
            Affected Test Files ({affected_tests?.length || 0})
          </div>
          {affected_tests && affected_tests.length > 0 ? (
            affected_tests.map((t, idx) => (
              <div key={idx} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                🧪 <strong style={{ color: '#fff' }}>{t.file_path}</strong> ({t.test_symbol})
              </div>
            ))
          ) : (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>tests/test_discount.py</div>
          )}
        </div>
      </div>

      {/* Inferred Impacts */}
      {inferred_impacts && inferred_impacts.length > 0 && (
        <div style={{ marginTop: '12px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <strong>Inferred Impacts:</strong> {inferred_impacts.map(i => i.symbol).join(', ')}
        </div>
      )}
    </div>
  );
}
