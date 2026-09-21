import React from 'react';

export function ExecutionFlow({ executionFlow }) {
  if (!executionFlow || !executionFlow.flow_steps || executionFlow.flow_steps.length === 0) {
    return null;
  }

  return (
    <div className="glass-card" style={{ padding: '16px', marginTop: '16px', borderColor: 'rgba(99, 102, 241, 0.3)' }}>
      <h3 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        ⚡ Feature Execution Flow Tracing
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', position: 'relative' }}>
        {executionFlow.flow_steps.map((step, idx) => (
          <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
            {/* Step Circle */}
            <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--accent-indigo), var(--accent-purple))', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.8rem', color: '#fff', flexShrink: 0 }}>
              {step.step}
            </div>

            {/* Step Card Content */}
            <div style={{ flex: 1, background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-glass)', borderRadius: '8px', padding: '10px 14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-indigo)', textTransform: 'uppercase' }}>
                  {step.layer}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {step.file_path}:{step.lines}
                </span>
              </div>
              <div style={{ fontWeight: 600, fontSize: '0.9rem', marginTop: '2px', color: '#fff' }}>
                {step.title}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                {step.description}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
