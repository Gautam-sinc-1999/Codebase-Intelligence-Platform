const API_BASE = '/api';

// Sent only when the backend has auth enabled and VITE_API_KEY is configured. Left unset,
// requests go out unchanged, matching the backend's opt-in default.
const API_KEY = import.meta.env?.VITE_API_KEY || '';

function withAuth(headers = {}) {
  return API_KEY ? { ...headers, 'X-API-Key': API_KEY } : headers;
}

/** Extracts FastAPI's `detail` message, falling back when the body is not JSON. */
async function errorFrom(res, fallback) {
  try {
    const body = await res.json();
    return new Error(body.detail || fallback);
  } catch {
    return new Error(`${fallback} (HTTP ${res.status})`);
  }
}

export async function fetchRepositories() {
  const res = await fetch(`${API_BASE}/repositories`, { headers: withAuth() });
  if (!res.ok) throw await errorFrom(res, 'Failed to fetch repositories');
  return await res.json();
}

export async function uploadRepositoryFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/repositories/upload`, {
    method: 'POST',
    headers: withAuth(),
    body: formData
  });

  if (!res.ok) throw await errorFrom(res, 'Upload failed');

  return await res.json();
}

export async function fetchConversations(repoId) {
  const url = repoId ? `${API_BASE}/conversations?repository_id=${repoId}` : `${API_BASE}/conversations`;
  const res = await fetch(url, { headers: withAuth() });
  if (!res.ok) throw await errorFrom(res, 'Failed to fetch conversations');
  return await res.json();
}

export async function createConversation(repoId, title = 'New Analysis') {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: 'POST',
    headers: withAuth({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ repository_id: repoId, title })
  });
  if (!res.ok) throw await errorFrom(res, 'Failed to create conversation');
  return await res.json();
}

export async function fetchConversationDetails(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, { headers: withAuth() });
  if (!res.ok) throw await errorFrom(res, 'Failed to load conversation history');
  return await res.json();
}

export async function sendMessageToConversation(conversationId, message) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}/messages`, {
    method: 'POST',
    headers: withAuth({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ message })
  });

  if (!res.ok) throw await errorFrom(res, 'Failed to send message');

  return await res.json();
}

/**
 * Streams an answer over server-sent events.
 *
 * `onEvent` receives each parsed event: `meta` (intent + sources, before generation starts),
 * `token` deltas, then `done` with the full payload. Falls back to the buffered endpoint if
 * streaming is unavailable, so the UI works either way.
 */
export async function streamMessageToConversation(conversationId, message, onEvent) {
  let res;
  try {
    res = await fetch(`${API_BASE}/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: withAuth({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ message })
    });
  } catch {
    return sendMessageToConversation(conversationId, message);
  }

  if (!res.ok) throw await errorFrom(res, 'Failed to send message');
  if (!res.body) {
    // No streaming support in this environment — take the buffered answer instead.
    return sendMessageToConversation(conversationId, message);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let final = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; a frame may arrive split across reads.
    const frames = buffer.split('\n\n');
    buffer = frames.pop() || '';

    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data:'));
      if (!line) continue;

      let event;
      try {
        event = JSON.parse(line.slice(5).trim());
      } catch {
        continue;
      }

      if (event.type === 'error') throw new Error(event.detail || 'Streaming failed');
      if (event.type === 'done') final = event;
      onEvent(event);
    }
  }

  if (!final) throw new Error('Stream ended before the answer completed');
  return final;
}

/**
 * Resolves a stored citation to its source.
 *
 * Persisted turns hold pointers rather than inlined code, so the code viewer fetches the body
 * only when a reader expands a citation — which also means a reloaded thread behaves the same
 * as a live one.
 */
export async function fetchSnippet(repoId, { file_path, start_line, end_line }, citedSha = '') {
  const params = new URLSearchParams({
    file_path,
    start_line: String(start_line ?? 1),
    end_line: String(end_line ?? 0)
  });
  // The commit this answer was written against. The server compares it with what is indexed now
  // and flags the result stale when they differ, so an aged citation is labelled rather than
  // silently showing whatever occupies those lines today.
  if (citedSha) params.set('at_sha', citedSha);
  const res = await fetch(`${API_BASE}/repositories/${repoId}/snippet?${params}`, { headers: withAuth() });
  if (!res.ok) throw await errorFrom(res, 'Failed to load the cited code');
  return await res.json();
}

export async function deleteConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: 'DELETE',
    headers: withAuth()
  });
  if (!res.ok) throw await errorFrom(res, 'Failed to delete conversation');
  return await res.json();
}

export async function reindexRepository(repoId, { full = false } = {}) {
  const res = await fetch(`${API_BASE}/repositories/${repoId}/reindex?full=${full}`, {
    method: 'POST',
    headers: withAuth()
  });
  if (!res.ok) throw await errorFrom(res, 'Failed to re-index repository');
  return await res.json();
}

export async function fetchRepositoryStatus(repoId) {
  const res = await fetch(`${API_BASE}/repositories/${repoId}/status`, { headers: withAuth() });
  if (!res.ok) throw await errorFrom(res, 'Failed to fetch repository status');
  return await res.json();
}

/** Liveness plus which backend each store resolved to, so degraded mode is visible in the UI. */
export async function fetchHealth() {
  const res = await fetch('/', { headers: withAuth() });
  if (!res.ok) throw await errorFrom(res, 'Failed to fetch service health');
  return await res.json();
}

export async function fetchGraphData(repoId) {
  const res = await fetch(`${API_BASE}/graph/${repoId}`, { headers: withAuth() });
  if (!res.ok) throw await errorFrom(res, 'Failed to fetch graph data');
  return await res.json();
}

/**
 * Lists a GitHub repository's branches without cloning it.
 *
 * Backed by `git ls-remote` on the server and cached there for a few minutes, since the UI calls
 * this on every paste. Each branch reports whether it is already indexed.
 */
export async function fetchGithubBranches(url) {
  const res = await fetch(`${API_BASE}/repositories/github/branches?url=${encodeURIComponent(url)}`, {
    headers: withAuth()
  });
  if (!res.ok) throw await errorFrom(res, 'Could not read branches from GitHub');
  return await res.json();
}

/**
 * Starts a GitHub import.
 *
 * Returns as soon as the server has accepted the job (202), not when indexing finishes — a real
 * repository takes tens of seconds and the browser would abandon the request. The caller polls
 * `fetchRepositoryStatus` until the status is `ready` or `failed`.
 */
export async function importFromGithub(url, branch) {
  const res = await fetch(`${API_BASE}/repositories/from-github`, {
    method: 'POST',
    headers: withAuth({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ url, branch: branch || null })
  });
  if (!res.ok) throw await errorFrom(res, 'Import failed');
  return await res.json();
}

/** Re-clones a GitHub repository and re-indexes what changed. */
export async function syncRepository(repoId, { full = false } = {}) {
  const res = await fetch(`${API_BASE}/repositories/${repoId}/sync?full=${full}`, {
    method: 'POST',
    headers: withAuth()
  });
  if (!res.ok) throw await errorFrom(res, 'Sync failed');
  return await res.json();
}

/** Removes a repository and everything derived from it. Conversations are kept unless asked. */
export async function deleteRepository(repoId, { deleteConversations = false } = {}) {
  const res = await fetch(
    `${API_BASE}/repositories/${repoId}?delete_conversations=${deleteConversations}`,
    { method: 'DELETE', headers: withAuth() }
  );
  if (!res.ok) throw await errorFrom(res, 'Failed to delete repository');
  return await res.json();
}

/**
 * Asks whether a repository has fallen behind its branch.
 *
 * Called after a thread has rendered, never before: messages come from local disk in
 * milliseconds while this costs a network round trip, and drift is advisory. A thread must open
 * and answer normally even when GitHub is unreachable.
 */
export async function fetchDrift(repoId, conversationId) {
  const params = conversationId ? `?conversation_id=${encodeURIComponent(conversationId)}` : '';
  const res = await fetch(`${API_BASE}/repositories/${repoId}/drift${params}`, { headers: withAuth() });
  if (!res.ok) throw await errorFrom(res, 'Could not check for repository updates');
  return await res.json();
}

/** Records that the user chose to stay on the indexed version, for this commit only. */
export async function acknowledgeDrift(conversationId, sha) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}/ack-drift`, {
    method: 'POST',
    headers: withAuth({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ sha })
  });
  if (!res.ok) throw await errorFrom(res, 'Failed to record your choice');
  return await res.json();
}

/**
 * Renames a thread.
 *
 * Marks the title as deliberately chosen, so the automatic titling from the first question never
 * overwrites it afterwards.
 */
export async function renameConversation(conversationId, title) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: 'PATCH',
    headers: withAuth({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title })
  });
  if (!res.ok) throw await errorFrom(res, 'Failed to rename conversation');
  return await res.json();
}
