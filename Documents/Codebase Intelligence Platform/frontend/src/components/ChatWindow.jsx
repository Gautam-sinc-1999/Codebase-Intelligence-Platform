import React, { useState, useRef, useEffect } from 'react';
import { MessageItem } from './MessageItem';
import { DriftBanner } from './DriftBanner';

export function ChatWindow({
  activeConv,
  messages,
  onSendMessage,
  loading,
  onToggleGraph,
  showGraph,
  drift,
  driftBusy,
  onKeepVersion,
  onPullLatest
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handlePromptClick = (promptText) => {
    onSendMessage(promptText);
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', background: 'var(--bg-primary)' }}>
      {/* Header */}
      <header className="glass-panel" style={{ padding: '12px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-glass)' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1rem', fontWeight: 600, color: '#fff' }}>
            {activeConv ? activeConv.title : 'Conversational Analysis'}
          </h2>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Repo ID: {activeConv ? activeConv.repository_id : 'None Selected'}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={onToggleGraph}
            className="btn-secondary"
            style={{ fontSize: '0.8rem', borderColor: showGraph ? 'var(--accent-purple)' : 'var(--border-glass)' }}
          >
            {showGraph ? '💬 Hide Neo4j Graph' : '🕸️ View Neo4j Graph'}
          </button>
        </div>
      </header>

      {/* Messages Feed */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Above the transcript, not over it: the thread stays readable and usable while this is
            showing, and while a sync it started is running. */}
        <DriftBanner drift={drift} busy={driftBusy} onKeep={onKeepVersion} onPull={onPullLatest} />

        {messages.length === 0 ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <div style={{ width: '56px', height: '56px', borderRadius: '16px', background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2))', border: '1px solid var(--border-glow)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.8rem', marginBottom: '16px' }}>
              🔍
            </div>
            <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.2rem', color: '#fff', marginBottom: '8px' }}>
              Codebase Intelligence & Developer Onboarding
            </h3>
            <p style={{ maxWidth: '480px', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '24px' }}>
              Ask feature flow questions, locate exact line numbers, perform Neo4j call graph analysis, or calculate change impacts.
            </p>

            {/* Quick Sample Prompts */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', maxWidth: '640px', width: '100%' }}>
              <button
                onClick={() => handlePromptClick('How does checkout work?')}
                className="glass-card"
                style={{ padding: '12px', textAlign: 'left', cursor: 'pointer', background: 'rgba(255,255,255,0.03)' }}
              >
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>⚡ Feature Execution Flow</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>"How does checkout work?"</div>
              </button>

              <button
                onClick={() => handlePromptClick('Where is the discount calculation?')}
                className="glass-card"
                style={{ padding: '12px', textAlign: 'left', cursor: 'pointer', background: 'rgba(255,255,255,0.03)' }}
              >
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-indigo)' }}>📍 Code Location</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>"Where is discount calculation?"</div>
              </button>

              <button
                onClick={() => handlePromptClick('What depends on calculate_discount?')}
                className="glass-card"
                style={{ padding: '12px', textAlign: 'left', cursor: 'pointer', background: 'rgba(255,255,255,0.03)' }}
              >
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-purple)' }}>🕸️ Dependency Tracing</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>"What depends on calculate_discount?"</div>
              </button>

              <button
                onClick={() => handlePromptClick('If I change discount calculation, what will be affected?')}
                className="glass-card"
                style={{ padding: '12px', textAlign: 'left', cursor: 'pointer', background: 'rgba(255,255,255,0.03)' }}
              >
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-rose)' }}>🎯 Change Impact Report</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>"If I change discount, what breaks?"</div>
              </button>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <MessageItem
              key={idx}
              message={msg}
              repositoryId={activeConv ? activeConv.repository_id : ''}
              conversationId={activeConv ? activeConv.conversation_id : ''}
            />
          ))
        )}

        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-indigo)', animation: 'pulse 1s infinite' }} />
            Analyzing repository AST, traversing Neo4j graph & constructing context...
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Box */}
      <div className="glass-panel" style={{ padding: '16px 20px', borderTop: '1px solid var(--border-glass)' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything about the codebase (e.g. 'How does checkout work?', 'What depends on DiscountService?')..."
            disabled={loading}
            style={{
              flex: 1,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid var(--border-glass)',
              borderRadius: '10px',
              padding: '12px 16px',
              color: '#fff',
              fontSize: '0.9rem',
              outline: 'none'
            }}
          />
          <button type="submit" className="btn-primary" disabled={loading || !input.trim()}>
            Send Query
          </button>
        </form>
      </div>
    </div>
  );
}
