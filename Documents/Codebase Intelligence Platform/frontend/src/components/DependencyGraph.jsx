import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { fetchGraphData } from '../services/api';

// Node colours by graph label, drawn from the app's accent tokens.
const NODE_COLORS = {
  File: '#64748b',
  Class: '#a855f7',
  Method: '#6366f1',
  Function: '#06b6d4',
  Component: '#10b981',
  Endpoint: '#f43f5e',
  Table: '#f59e0b',
  Symbol: '#94a3b8'
};

// Edge styling by relationship type. CONTAINS is structural, so it recedes.
const EDGE_STYLES = {
  CALLS: { stroke: '#6366f1', dash: null, width: 1.4 },
  CONTAINS: { stroke: 'rgba(148,163,184,0.28)', dash: '3 3', width: 1 },
  ROUTES_TO: { stroke: '#f43f5e', dash: '5 3', width: 1.4 },
  DEPENDS_ON: { stroke: '#94a3b8', dash: '2 2', width: 1 }
};

const WIDTH = 900;
const HEIGHT = 700;

/**
 * Force-directed layout computed synchronously over a fixed iteration count.
 *
 * Written by hand rather than pulling in d3-force: the repo graph is a few hundred nodes at
 * most, where an O(n^2) repulsion pass is cheap, and it avoids adding a dependency for one view.
 */
function layoutGraph(nodes, edges) {
  const n = nodes.length;
  if (n === 0) return [];

  const deg = new Map(nodes.map((d) => [d.id, 0]));
  edges.forEach((e) => {
    deg.set(e.source, (deg.get(e.source) || 0) + 1);
    deg.set(e.target, (deg.get(e.target) || 0) + 1);
  });

  // Seed on a circle — deterministic, so the layout is stable across re-renders.
  const pts = nodes.map((d, i) => {
    const angle = (i / n) * Math.PI * 2;
    const radius = Math.min(WIDTH, HEIGHT) * 0.32;
    return {
      ...d,
      degree: deg.get(d.id) || 0,
      x: WIDTH / 2 + Math.cos(angle) * radius,
      y: HEIGHT / 2 + Math.sin(angle) * radius,
      vx: 0,
      vy: 0
    };
  });

  const index = new Map(pts.map((p, i) => [p.id, i]));
  const links = edges
    .map((e) => ({ s: index.get(e.source), t: index.get(e.target) }))
    .filter((l) => l.s !== undefined && l.t !== undefined && l.s !== l.t);

  const ITERATIONS = 320;
  const REPULSION = 9000;
  const SPRING = 0.006;
  const IDEAL = 90;
  const CENTER_PULL = 0.004;

  for (let step = 0; step < ITERATIONS; step++) {
    const cooling = 1 - step / ITERATIONS;

    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        let dx = pts[i].x - pts[j].x;
        let dy = pts[i].y - pts[j].y;
        let distSq = dx * dx + dy * dy;
        if (distSq < 1) {
          // Deterministic nudge so coincident nodes separate without Math.random().
          dx = ((i % 7) - 3) * 0.5 || 0.5;
          dy = ((j % 5) - 2) * 0.5 || 0.5;
          distSq = dx * dx + dy * dy;
        }
        const dist = Math.sqrt(distSq);
        const force = REPULSION / distSq;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        pts[i].vx += fx;
        pts[i].vy += fy;
        pts[j].vx -= fx;
        pts[j].vy -= fy;
      }
    }

    links.forEach(({ s, t }) => {
      const dx = pts[t].x - pts[s].x;
      const dy = pts[t].y - pts[s].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = (dist - IDEAL) * SPRING;
      const fx = (dx / dist) * force * dist;
      const fy = (dy / dist) * force * dist;
      pts[s].vx += fx;
      pts[s].vy += fy;
      pts[t].vx -= fx;
      pts[t].vy -= fy;
    });

    pts.forEach((p) => {
      p.vx += (WIDTH / 2 - p.x) * CENTER_PULL;
      p.vy += (HEIGHT / 2 - p.y) * CENTER_PULL;
      p.x += Math.max(-30, Math.min(30, p.vx)) * cooling * 0.1;
      p.y += Math.max(-30, Math.min(30, p.vy)) * cooling * 0.1;
      p.vx *= 0.82;
      p.vy *= 0.82;
      p.x = Math.max(40, Math.min(WIDTH - 40, p.x));
      p.y = Math.max(40, Math.min(HEIGHT - 40, p.y));
    });
  }

  return pts;
}

export function DependencyGraph({ repositoryId }) {
  // `truncated` and the totals come from the server. A force-directed layout cannot usefully
  // render a large repository, so the cap stays — but showing a fifth of a graph without saying
  // so left people studying an arbitrary slice of their own project and unaware of it.
  const [graph, setGraph] = useState({ nodes: [], edges: [], truncated: false });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const [hiddenKinds, setHiddenKinds] = useState(() => new Set());
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragRef = useRef(null);

  useEffect(() => {
    if (!repositoryId) return;
    setLoading(true);
    setError('');
    setSelectedId(null);
    fetchGraphData(repositoryId)
      .then((data) => setGraph({
        nodes: data.nodes || [],
        edges: data.edges || [],
        truncated: Boolean(data.truncated),
        totalNodes: data.total_nodes ?? (data.nodes || []).length,
        totalEdges: data.total_edges ?? (data.edges || []).length,
        shownNodes: data.shown_nodes ?? (data.nodes || []).length,
      }))
      .catch((err) => setError(err.message || 'Failed to load graph'))
      .finally(() => setLoading(false));
  }, [repositoryId]);

  const visible = useMemo(() => {
    const nodes = graph.nodes.filter((nd) => !hiddenKinds.has(nd.label));
    const ids = new Set(nodes.map((nd) => nd.id));
    const edges = graph.edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    return { nodes, edges };
  }, [graph, hiddenKinds]);

  const positioned = useMemo(
    () => layoutGraph(visible.nodes, visible.edges),
    [visible.nodes, visible.edges]
  );

  const posById = useMemo(() => new Map(positioned.map((p) => [p.id, p])), [positioned]);

  // Neighbours of the selection, used to dim everything unrelated.
  const neighbours = useMemo(() => {
    if (!selectedId) return null;
    const set = new Set([selectedId]);
    visible.edges.forEach((e) => {
      if (e.source === selectedId) set.add(e.target);
      if (e.target === selectedId) set.add(e.source);
    });
    return set;
  }, [selectedId, visible.edges]);

  const kinds = useMemo(() => {
    const counts = {};
    graph.nodes.forEach((nd) => {
      counts[nd.label] = (counts[nd.label] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [graph.nodes]);

  const toggleKind = (kind) =>
    setHiddenKinds((prev) => {
      const next = new Set(prev);
      next.has(kind) ? next.delete(kind) : next.add(kind);
      return next;
    });

  const onPointerDown = useCallback((e) => {
    dragRef.current = { x: e.clientX, y: e.clientY, pan: { ...pan } };
  }, [pan]);

  const onPointerMove = useCallback((e) => {
    if (!dragRef.current) return;
    setPan({
      x: dragRef.current.pan.x + (e.clientX - dragRef.current.x),
      y: dragRef.current.pan.y + (e.clientY - dragRef.current.y)
    });
  }, []);

  const onPointerUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  const selected = selectedId ? graph.nodes.find((nd) => nd.id === selectedId) : null;
  const selectedEdges = selectedId
    ? visible.edges.filter((e) => e.source === selectedId || e.target === selectedId)
    : [];

  return (
    <div className="glass-card" style={{ padding: '16px', height: '100%', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', marginBottom: '10px', flexWrap: 'wrap' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-purple)', margin: 0 }}>
          🕸️ Dependency Knowledge Graph
        </h3>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {visible.nodes.length} nodes · {visible.edges.length} relationships
          {graph.truncated ? ` of ${graph.totalNodes}` : ''}
        </span>
      </div>

      {/* The server caps what it returns, so the view must say what it is not showing. Without
          this a developer studies a fifth of their dependency graph believing it is all of it. */}
      {graph.truncated && (
        <div
          title="A force-directed layout cannot usefully render a whole large repository."
          style={{
            display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px',
            padding: '6px 10px', borderRadius: '7px', fontSize: '0.72rem',
            background: 'rgba(245, 158, 11, 0.10)', border: '1px solid var(--accent-amber)',
            color: 'var(--accent-amber)'
          }}
        >
          ⚠ Showing the {graph.shownNodes} most connected of {graph.totalNodes} nodes
          {graph.totalEdges ? ` and ${graph.totalEdges} relationships` : ''}. Ask a question to
          explore the rest.
        </div>
      )}

      {/* Legend doubles as a filter — click a kind to show/hide it. */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '10px' }}>
        {kinds.map(([kind, count]) => {
          const off = hiddenKinds.has(kind);
          return (
            <button
              key={kind}
              onClick={() => toggleKind(kind)}
              title={off ? `Show ${kind}` : `Hide ${kind}`}
              style={{
                display: 'flex', alignItems: 'center', gap: '5px',
                background: off ? 'transparent' : 'rgba(255,255,255,0.05)',
                border: '1px solid var(--border-glass)', borderRadius: '999px',
                padding: '3px 9px', cursor: 'pointer', opacity: off ? 0.4 : 1,
                color: 'var(--text-secondary)', fontSize: '0.7rem', fontFamily: 'var(--font-sans)'
              }}
            >
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: NODE_COLORS[kind] || NODE_COLORS.Symbol }} />
              {kind} <span style={{ color: 'var(--text-muted)' }}>{count}</span>
            </button>
          );
        })}
        {kinds.length > 0 && (
          <button
            onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); setSelectedId(null); }}
            style={{
              marginLeft: 'auto', background: 'transparent', border: '1px solid var(--border-glass)',
              borderRadius: '999px', padding: '3px 9px', cursor: 'pointer',
              color: 'var(--text-muted)', fontSize: '0.7rem'
            }}
          >
            Reset view
          </button>
        )}
      </div>

      <div
        style={{
          flex: 1, background: '#050811', borderRadius: '8px', border: '1px solid var(--border-glass)',
          position: 'relative', overflow: 'hidden', minHeight: 0, cursor: dragRef.current ? 'grabbing' : 'grab'
        }}
        onMouseDown={onPointerDown}
        onMouseMove={onPointerMove}
        onMouseUp={onPointerUp}
        onMouseLeave={onPointerUp}
        onWheel={(e) => setZoom((z) => Math.max(0.3, Math.min(2.5, z - e.deltaY * 0.0012)))}
      >
        {loading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Loading graph topology…
          </div>
        )}

        {!loading && error && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-rose)', fontSize: '0.85rem', padding: '20px', textAlign: 'center' }}>
            {error}
          </div>
        )}

        {!loading && !error && visible.nodes.length === 0 && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px', textAlign: 'center' }}>
            No graph data for this repository yet.
          </div>
        )}

        {!loading && !error && visible.nodes.length > 0 && (
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ width: '100%', height: '100%', display: 'block' }}
          >
            <defs>
              {Object.entries(EDGE_STYLES).map(([type, s]) => (
                <marker
                  key={type}
                  id={`arrow-${type}`}
                  viewBox="0 0 10 10"
                  refX="19"
                  refY="5"
                  markerWidth="5"
                  markerHeight="5"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={s.stroke} />
                </marker>
              ))}
            </defs>

            <g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`} transform-origin="center">
              {visible.edges.map((e, i) => {
                const a = posById.get(e.source);
                const b = posById.get(e.target);
                if (!a || !b) return null;
                const style = EDGE_STYLES[e.type] || EDGE_STYLES.DEPENDS_ON;
                const related = !neighbours || (neighbours.has(e.source) && neighbours.has(e.target));
                return (
                  <line
                    key={`${e.source}->${e.target}-${e.type}-${i}`}
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={style.stroke}
                    strokeWidth={style.width}
                    strokeDasharray={style.dash || undefined}
                    markerEnd={`url(#arrow-${e.type in EDGE_STYLES ? e.type : 'DEPENDS_ON'})`}
                    opacity={related ? 0.9 : 0.08}
                  />
                );
              })}

              {positioned.map((nd) => {
                const color = NODE_COLORS[nd.label] || NODE_COLORS.Symbol;
                const isSelected = nd.id === selectedId;
                const related = !neighbours || neighbours.has(nd.id);
                const r = Math.min(16, 6 + Math.sqrt(nd.degree) * 2.4);
                const name = nd.name || nd.id;
                return (
                  <g
                    key={nd.id}
                    transform={`translate(${nd.x} ${nd.y})`}
                    onClick={(ev) => { ev.stopPropagation(); setSelectedId(isSelected ? null : nd.id); }}
                    style={{ cursor: 'pointer' }}
                    opacity={related ? 1 : 0.15}
                  >
                    <circle
                      r={r}
                      fill={color}
                      fillOpacity={isSelected ? 1 : 0.75}
                      stroke={isSelected ? '#fff' : color}
                      strokeWidth={isSelected ? 2.5 : 1}
                    />
                    <text
                      y={r + 11}
                      textAnchor="middle"
                      fontSize="9.5"
                      fill={isSelected ? '#fff' : 'var(--text-secondary)'}
                      style={{ pointerEvents: 'none', fontFamily: 'var(--font-mono)' }}
                    >
                      {name.length > 22 ? `${name.slice(0, 20)}…` : name}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
        )}

        <div style={{ position: 'absolute', bottom: 8, left: 10, fontSize: '0.65rem', color: 'var(--text-muted)', pointerEvents: 'none' }}>
          drag to pan · scroll to zoom · click a node to isolate
        </div>
      </div>

      {selected && (
        <div style={{ marginTop: '12px', background: 'rgba(0,0,0,0.45)', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--accent-purple)' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#fff', fontFamily: 'var(--font-mono)', wordBreak: 'break-all' }}>
            {selected.name || selected.id}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '3px' }}>
            {selected.label}
            {selected.file_path ? ` · ${selected.file_path}` : ''}
          </div>
          {selectedEdges.length > 0 && (
            <div style={{ marginTop: '7px', maxHeight: '108px', overflowY: 'auto' }}>
              {selectedEdges.map((e, i) => {
                const outgoing = e.source === selectedId;
                const other = outgoing ? e.target : e.source;
                return (
                  <div key={i} style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    <span style={{ color: (EDGE_STYLES[e.type] || EDGE_STYLES.DEPENDS_ON).stroke }}>
                      {outgoing ? '→' : '←'} {e.type}
                    </span>{' '}
                    {other}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
