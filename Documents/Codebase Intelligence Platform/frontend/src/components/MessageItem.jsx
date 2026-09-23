import React, { useState } from 'react';
import { fetchSnippet, sendFeedback } from '../services/api';
import { canRate, nextRating, shouldAskWhy } from '../lib/feedback';
import { ExecutionFlow } from './ExecutionFlow';
import { ImpactAnalysisCard } from './ImpactAnalysisCard';

/**
 * Renders the markdown subset the assistant actually produces: fenced code blocks, inline code,
 * bold, italics, headings and bullets.
 *
 * Hand-written rather than pulling in a markdown library plus a syntax highlighter, and it
 * builds React elements rather than using dangerouslySetInnerHTML — answers quote repository
 * source, which must never be interpreted as markup.
 */
function renderInline(text, keyPrefix) {
  const nodes = [];
  // `code`, **bold**, *italic* — matched in one pass so they cannot nest incorrectly.
  const pattern = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g;
  let lastIndex = 0;
  let match;
  let i = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) nodes.push(text.slice(lastIndex, match.index));

    const token = match[0];
    const key = `${keyPrefix}-i${i++}`;

    if (token.startsWith('`')) {
      nodes.push(
        <code
          key={key}
          style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.85em',
            background: 'rgba(0,0,0,0.45)', border: '1px solid var(--border-glass)',
            borderRadius: '4px', padding: '1px 5px', color: 'var(--accent-cyan)'
          }}
        >
          {token.slice(1, -1)}
        </code>
      );
    } else if (token.startsWith('**')) {
      nodes.push(<strong key={key} style={{ color: '#fff' }}>{token.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) nodes.push(text.slice(lastIndex));
  return nodes;
}

function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false);
  const lines = code.replace(/\n$/, '').split('\n');

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context) — the code is still selectable */
    }
  };

  return (
    <div style={{ margin: '10px 0', border: '1px solid var(--border-glass)', borderRadius: '8px', overflow: 'hidden', background: '#050811' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 10px', borderBottom: '1px solid var(--border-glass)', background: 'rgba(255,255,255,0.03)' }}>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {language || 'code'}
        </span>
        <button
          onClick={copy}
          style={{ background: 'transparent', border: 'none', color: copied ? 'var(--accent-emerald)' : 'var(--text-muted)', cursor: 'pointer', fontSize: '0.68rem' }}
        >
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre style={{ margin: 0, padding: '10px 0', overflowX: 'auto', fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
        {lines.map((line, i) => (
          <div key={i} style={{ display: 'flex', minWidth: 'max-content' }}>
            {/* Line numbers are the point of a code viewer here — answers cite exact ranges. */}
            <span style={{ width: '38px', flexShrink: 0, textAlign: 'right', paddingRight: '10px', color: 'var(--text-muted)', userSelect: 'none' }}>
              {i + 1}
            </span>
            <span style={{ color: 'var(--text-primary)', whiteSpace: 'pre', paddingRight: '12px' }}>{line || ' '}</span>
          </div>
        ))}
      </pre>
    </div>
  );
}

function renderMarkdown(content) {
  if (!content) return null;
  const blocks = [];
  // Split on fenced code blocks first so their contents are never treated as markdown.
  const parts = content.split(/```/);

  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      const newline = part.indexOf('\n');
      const language = newline === -1 ? '' : part.slice(0, newline).trim();
      const code = newline === -1 ? part : part.slice(newline + 1);
      blocks.push(<CodeBlock key={`c${index}`} language={language} code={code} />);
      return;
    }

    part.split('\n').forEach((line, lineIndex) => {
      const key = `t${index}-${lineIndex}`;
      const trimmed = line.trim();

      if (!trimmed) {
        blocks.push(<div key={key} style={{ height: '7px' }} />);
      } else if (/^#{1,4}\s/.test(trimmed)) {
        blocks.push(
          <div key={key} style={{ fontWeight: 700, color: '#fff', marginTop: '8px', fontSize: '0.92rem' }}>
            {renderInline(trimmed.replace(/^#+\s/, ''), key)}
          </div>
        );
      } else if (/^[-*]\s/.test(trimmed)) {
        blocks.push(
          <div key={key} style={{ display: 'flex', gap: '7px', paddingLeft: '4px' }}>
            <span style={{ color: 'var(--accent-indigo)' }}>•</span>
            <span>{renderInline(trimmed.slice(2), key)}</span>
          </div>
        );
      } else if (/^\d+\.\s/.test(trimmed)) {
        const [, num, rest] = trimmed.match(/^(\d+)\.\s(.*)$/);
        blocks.push(
          <div key={key} style={{ display: 'flex', gap: '7px', paddingLeft: '4px' }}>
            <span style={{ color: 'var(--accent-indigo)', fontVariantNumeric: 'tabular-nums' }}>{num}.</span>
            <span>{renderInline(rest, key)}</span>
          </div>
        );
      } else {
        blocks.push(<div key={key}>{renderInline(line, key)}</div>);
      }
    });
  });

  return blocks;
}

function SourceCard({ source, repositoryId, citedSha }) {
  const [open, setOpen] = useState(false);
  // A live response carries the code; a turn reloaded from storage carries only the pointer,
  // because persisting every snippet made threads grow without bound. Fetch on expand so both
  // behave identically to the reader.
  const [code, setCode] = useState(source.code_snippet || '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [stale, setStale] = useState(null);

  const toggle = async () => {
    const next = !open;
    setOpen(next);
    if (!next || code || loading) return;

    setLoading(true);
    setError('');
    try {
      const resolved = await fetchSnippet(repositoryId, source, citedSha);
      setCode(resolved.code || '');
      if (resolved.stale) setStale(resolved);
      if (!resolved.code) setError('No source recorded for this range.');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid var(--border-glass)', borderRadius: '6px', overflow: 'hidden' }}>
      <button
        onClick={toggle}
        title="Show the cited code"
        style={{
          display: 'flex', alignItems: 'center', gap: '6px', width: '100%',
          background: 'transparent', border: 'none', padding: '4px 8px',
          cursor: 'pointer',
          color: 'var(--text-secondary)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)'
        }}
      >
        <span>{loading ? '…' : open ? '▾' : '▸'}</span>
        <span>📄 {source.file_path}:{source.start_line}-{source.end_line}</span>
        {source.symbol && <span style={{ color: 'var(--text-muted)' }}>({source.symbol})</span>}
      </button>

      {open && (error || (!code && !loading)) && (
        <div style={{ padding: '0 8px 8px', fontSize: '0.72rem', color: 'var(--accent-rose)' }}>
          {error || 'No source recorded for this range.'}
        </div>
      )}

      {/* The repository has moved since this answer was written, so the pointer resolves against
          different code. Saying so is the whole point — showing it silently would be worse than
          showing nothing. */}
      {open && stale && (
        <div style={{ margin: '0 8px 6px', padding: '6px 8px', borderRadius: '6px', fontSize: '0.68rem', background: 'rgba(245, 158, 11, 0.12)', border: '1px solid rgba(245, 158, 11, 0.35)', color: 'var(--accent-amber)' }}>
          ⚠ Shown from {stale.indexed_sha ? stale.indexed_sha.slice(0, 7) : 'the current index'} — this answer was written against {stale.cited_sha ? stale.cited_sha.slice(0, 7) : 'an earlier commit'}
          {stale.moved ? '. The cited lines now hold different code.' : '.'}
        </div>
      )}

      {open && code && (
        <div style={{ padding: '0 8px 8px' }}>
          {/* Numbered from the symbol's real start line, so what is shown matches the citation. */}
          <pre style={{ margin: 0, padding: '8px 0', overflowX: 'auto', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', background: '#050811', borderRadius: '6px' }}>
            {code.split('\n').map((line, i) => (
              <div key={i} style={{ display: 'flex', minWidth: 'max-content' }}>
                <span style={{ width: '44px', flexShrink: 0, textAlign: 'right', paddingRight: '10px', color: 'var(--text-muted)', userSelect: 'none' }}>
                  {(source.start_line || 1) + i}
                </span>
                <span style={{ whiteSpace: 'pre', paddingRight: '12px' }}>{line || ' '}</span>
              </div>
            ))}
          </pre>
        </div>
      )}
    </div>
  );
}

/**
 * A reader's verdict on one answer.
 *
 * Every other score this system records is computable from the answer and the index — whether the
 * citations resolve, whether the callers named are real — and all of them can be perfect for an
 * answer that did not help. This is the only signal that measures whether it did.
 *
 * The rating is applied optimistically and rolled back if the request fails, because the button
 * looking pressed when nothing was recorded is worse than a moment's delay.
 */
function FeedbackButtons({ conversationId, seq, initialRating, initialComment }) {
  const [rating, setRating] = useState(initialRating || '');
  const [comment, setComment] = useState(initialComment || '');
  const [draft, setDraft] = useState('');
  const [composing, setComposing] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (nextRating, nextComment) => {
    const previous = { rating, comment };
    setRating(nextRating);
    setComment(nextComment);
    setError('');
    setBusy(true);
    try {
      await sendFeedback(conversationId, seq, nextRating, nextComment);
    } catch (err) {
      // Put it back the way it was — a thumb that looks recorded but was not is a lie.
      setRating(previous.rating);
      setComment(previous.comment);
      setError(err.message || 'Could not record that.');
    } finally {
      setBusy(false);
    }
  };

  const choose = (clicked) => {
    if (busy) return;
    if (shouldAskWhy(rating, clicked)) {
      setDraft('');
      setComposing(true);
    } else {
      setComposing(false);
    }
    const next = nextRating(rating, clicked);
    // A thumbs-up carries no complaint, so switching to it drops the one left on a previous down.
    submit(next, next === 'down' ? comment : '');
  };

  const buttonStyle = (active, tone) => ({
    background: active ? `rgba(${tone}, 0.16)` : 'transparent',
    border: `1px solid ${active ? `rgba(${tone}, 0.55)` : 'var(--border-glass)'}`,
    borderRadius: '6px',
    padding: '2px 8px',
    fontSize: '0.78rem',
    cursor: busy ? 'progress' : 'pointer',
    opacity: busy ? 0.6 : 1,
    lineHeight: 1.6
  });

  return (
    <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Was this useful?</span>
        <button
          type="button"
          onClick={() => choose('up')}
          disabled={busy}
          aria-pressed={rating === 'up'}
          title="This answer helped"
          style={buttonStyle(rating === 'up', '34, 197, 94')}
        >
          👍
        </button>
        <button
          type="button"
          onClick={() => choose('down')}
          disabled={busy}
          aria-pressed={rating === 'down'}
          title="This answer did not help"
          style={buttonStyle(rating === 'down', '244, 63, 94')}
        >
          👎
        </button>
        {rating && !error && (
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Thanks — recorded.</span>
        )}
        {error && (
          <span style={{ fontSize: '0.7rem', color: 'var(--accent-rose)' }}>{error}</span>
        )}
      </div>

      {/* Offered only on a thumbs-down, where "what was wrong" is the part worth knowing. */}
      {composing && rating === 'down' && (
        <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && draft.trim()) {
                submit('down', draft.trim());
                setComposing(false);
              }
            }}
            maxLength={1000}
            placeholder="What was wrong? (optional)"
            style={{
              flex: 1, background: 'rgba(255,255,255,0.05)',
              border: '1px solid var(--border-glass)', borderRadius: '6px',
              padding: '5px 9px', color: '#fff', fontSize: '0.75rem', outline: 'none'
            }}
          />
          <button
            type="button"
            className="btn-secondary"
            style={{ fontSize: '0.72rem', padding: '4px 10px' }}
            onClick={() => { submit('down', draft.trim()); setComposing(false); }}
            disabled={busy}
          >
            Send
          </button>
        </div>
      )}

      {comment && !composing && (
        <div style={{ marginTop: '6px', fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          “{comment}”
        </div>
      )}
    </div>
  );
}

export function MessageItem({ message, repositoryId, conversationId }) {
  const isUser = message.role === 'user';
  const isError = Boolean(message.error);

  // A repository sync records a version boundary in the transcript. It is not a turn — nobody
  // said it and the model never sees it — so it renders as a rule across the thread rather than
  // as another message bubble. Without it, the stale labels that start appearing on citations
  // above the line would look like a malfunction.
  if (message.role === 'system') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%', alignSelf: 'stretch', margin: '4px 0' }}>
        <div style={{ flex: 1, height: '1px', background: 'var(--border-glass)' }} />
        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          ⑂ {message.content}
        </span>
        <div style={{ flex: 1, height: '1px', background: 'var(--border-glass)' }} />
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex', flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        maxWidth: '85%', alignSelf: isUser ? 'flex-end' : 'flex-start'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
          {isUser ? 'You' : 'AI Architect'}
        </span>
        {message.intent && <span className={`badge-intent badge-${message.intent}`}>{message.intent}</span>}
        {message.streaming && <span style={{ fontSize: '0.7rem', color: 'var(--accent-indigo)' }}>generating…</span>}
      </div>

      <div
        className="glass-panel"
        style={{
          padding: '16px',
          borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
          background: isUser
            ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25))'
            : 'var(--bg-glass)',
          border: '1px solid',
          // An error is not an answer; it should not look like one.
          borderColor: isError ? 'var(--accent-rose)' : (isUser ? 'var(--border-glow)' : 'var(--border-glass)'),
          width: '100%'
        }}
      >
        <div style={{ fontSize: '0.9rem', color: isError ? 'var(--accent-rose)' : 'var(--text-primary)', lineHeight: 1.55 }}>
          {isUser || isError
            ? <span style={{ whiteSpace: 'pre-wrap' }}>{message.content}</span>
            : renderMarkdown(message.content)}
          {message.streaming && (
            <span style={{ display: 'inline-block', width: '7px', height: '14px', background: 'var(--accent-indigo)', marginLeft: '2px', verticalAlign: 'text-bottom', animation: 'pulse 1s infinite' }} />
          )}
        </div>

        {message.sources && message.sources.length > 0 && (
          <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px' }}>
              Indexed File References ({message.sources.length}) — click to view the cited code:
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {message.sources.map((src, i) => (
                <SourceCard
                  key={i}
                  source={src}
                  repositoryId={message.repository_id || repositoryId}
                  citedSha={message.commit_sha}
                />
              ))}
            </div>
          </div>
        )}

        {/* The answer came from the rule-based template, not a model — rate-limited, unreachable
            or unconfigured. The citations are real; the prose is not written for this question.
            Saying so is the difference between a degraded answer and an unexplained bad one. */}
        {message.degraded && (
          <div style={{
            marginTop: '12px', padding: '7px 10px', borderRadius: '7px', fontSize: '0.72rem',
            background: 'rgba(245, 158, 11, 0.10)', border: '1px solid var(--accent-amber)',
            color: 'var(--accent-amber)'
          }}>
            ⚠ {message.degraded_reason
              || 'Assembled from the indexed code because the language model was unavailable.'}
          </div>
        )}

        {message.execution_flow && <ExecutionFlow executionFlow={message.execution_flow} />}
        {message.impact_analysis && <ImpactAnalysisCard impactAnalysis={message.impact_analysis} />}

        {/* The rule lives in lib/feedback.js, where it can be tested — see canRate for why each
            of the four disqualifying cases is one. */}
        {canRate(message, conversationId) && (
          <FeedbackButtons
            conversationId={conversationId}
            seq={message.seq}
            initialRating={message.feedback}
            initialComment={message.feedback_comment}
          />
        )}
      </div>
    </div>
  );
}
