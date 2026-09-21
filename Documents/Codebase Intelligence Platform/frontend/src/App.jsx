import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatWindow } from './components/ChatWindow';
import { DependencyGraph } from './components/DependencyGraph';
import { RepoUploader } from './components/RepoUploader';
import {
  fetchRepositories,
  fetchConversations,
  createConversation,
  fetchConversationDetails,
  sendMessageToConversation,
  streamMessageToConversation,
  deleteConversation,
  reindexRepository,
  fetchRepositoryStatus,
  fetchHealth,
  syncRepository,
  deleteRepository,
  fetchDrift,
  acknowledgeDrift,
  renameConversation
} from './services/api';

export function App() {
  const [repositories, setRepositories] = useState([]);
  const [selectedRepoId, setSelectedRepoId] = useState('');
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState('');
  const [activeConv, setActiveConv] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showGraph, setShowGraph] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [health, setHealth] = useState(null);
  const [repoStatus, setRepoStatus] = useState(null);
  // repository_id -> { status, name, error }. A GitHub import returns before indexing finishes,
  // so the sidebar shows these alongside the finished repositories rather than leaving the user
  // watching nothing happen.
  const [pending, setPending] = useState({});
  const [drift, setDrift] = useState(null);
  const [driftBusy, setDriftBusy] = useState(false);

  // 1. Initial Load: Fetch Repositories and All Saved Conversations
  useEffect(() => {
    initPlatform();
  }, []);

  const initPlatform = async () => {
    try {
      // Non-fatal: the UI works without it, it only makes fallback mode visible.
      fetchHealth().then(setHealth).catch(() => setHealth(null));

      const repos = await fetchRepositories();
      setRepositories(repos);

      const convs = await fetchConversations();
      setConversations(convs);

      if (repos.length > 0) {
        const initialRepoId = repos[0].repository_id;
        setSelectedRepoId(initialRepoId);

        const repoConvs = convs.filter(c => c.repository_id === initialRepoId);
        if (repoConvs.length > 0) {
          handleSelectConv(repoConvs[0].conversation_id);
        } else {
          handleNewChat(initialRepoId);
        }
      }
    } catch (err) {
      console.error('Error initializing platform:', err);
    }
  };

  // Index readiness for whichever repository is selected. A repository row exists before
  // indexing finishes, so the sidebar reports the index's own state rather than inferring it.
  useEffect(() => {
    if (!selectedRepoId) {
      setRepoStatus(null);
      return;
    }
    let cancelled = false;
    fetchRepositoryStatus(selectedRepoId)
      .then((s) => { if (!cancelled) setRepoStatus(s); })
      .catch(() => { if (!cancelled) setRepoStatus(null); });
    return () => { cancelled = true; };
  }, [selectedRepoId]);

  // 2. Resume session on sidebar click
  const handleSelectConv = async (convId) => {
    setActiveConvId(convId);
    setDrift(null);
    try {
      const details = await fetchConversationDetails(convId);
      setActiveConv(details);
      setMessages(details.messages || []);
      if (details.repository_id && details.repository_id !== selectedRepoId) {
        setSelectedRepoId(details.repository_id);
      }

      // Deliberately after the messages are on screen and deliberately not awaited: the thread
      // reads from local disk in milliseconds, while this is a network round trip. Drift is
      // advisory, so a failure here must never delay or prevent reading the thread.
      if (details.repository_id) {
        checkDrift(details.repository_id, convId);
      }
    } catch (err) {
      console.error('Error loading session details:', err);
    }
  };

  const checkDrift = async (repositoryId, conversationId) => {
    try {
      setDrift(await fetchDrift(repositoryId, conversationId));
    } catch {
      // Unreachable upstream is not worth interrupting anyone over.
      setDrift(null);
    }
  };

  const handleKeepVersion = async (sha) => {
    if (!activeConvId || !sha) return;
    try {
      await acknowledgeDrift(activeConvId, sha);
      setDrift(null);
    } catch (err) {
      console.error('Error recording drift choice:', err);
    }
  };

  const handlePullLatest = async () => {
    if (!activeConvId || !activeConv?.repository_id) return;
    const repositoryId = activeConv.repository_id;

    setDriftBusy(true);
    try {
      const result = await syncRepository(repositoryId, { full: false });
      if (result && result.status !== 'already current') {
        await trackImport(repositoryId, result.name);
      }
      setDrift(null);
      // The sync appended a version marker to this thread; reload so the reader can see where
      // the boundary falls, which is what makes the stale labels below it make sense.
      const details = await fetchConversationDetails(activeConvId);
      setActiveConv(details);
      setMessages(details.messages || []);
    } catch (err) {
      console.error('Error pulling latest:', err);
      window.alert(err.message);
    } finally {
      setDriftBusy(false);
    }
  };

  // 3. Create New Chat Session
  const handleNewChat = async (repoIdOverride) => {
    const rId = repoIdOverride || selectedRepoId;
    if (!rId) return;

    try {
      const newConv = await createConversation(rId, 'New Analysis');
      setConversations((prev) => [newConv, ...prev.filter(c => c.conversation_id !== newConv.conversation_id)]);
      setActiveConvId(newConv.conversation_id);
      setActiveConv(newConv);
      setMessages([]);
    } catch (err) {
      console.error('Error creating new conversation:', err);
    }
  };

  // 4. Send Message in Current Session
  const handleSendMessage = async (msgText) => {
    if (!activeConvId) return;

    const userMsg = { role: 'user', content: msgText };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    // Placeholder the stream fills in. Sources arrive with the `meta` event, before the first
    // token, so the citations render while the prose is still being generated.
    setMessages((prev) => [...prev, { role: 'assistant', content: '', streaming: true }]);

    const updateLast = (patch) =>
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === 'assistant') next[next.length - 1] = { ...last, ...patch };
        return next;
      });

    try {
      let streamed = '';
      const response = await streamMessageToConversation(activeConvId, msgText, (event) => {
        if (event.type === 'meta') {
          updateLast({ intent: event.intent, sources: event.sources });
        } else if (event.type === 'token') {
          streamed += event.text;
          updateLast({ content: streamed });
        }
      });

      updateLast({
        content: response.answer,
        intent: response.intent,
        sources: response.sources,
        execution_flow: response.execution_flow,
        impact_analysis: response.impact_analysis,
        degraded: response.degraded,
        degraded_reason: response.degraded_reason,
        streaming: false
      });

      // Refresh saved conversations list to reflect updated titles & timestamps
      const updatedConvs = await fetchConversations();
      setConversations(updatedConvs);
    } catch (err) {
      console.error('Error processing query:', err);
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        // Replace the streaming placeholder instead of leaving an empty bubble behind it.
        if (last && last.role === 'assistant' && last.streaming) {
          next[next.length - 1] = { role: 'assistant', content: err.message, error: true };
        } else {
          next.push({ role: 'assistant', content: err.message, error: true });
        }
        return next;
      });
    } finally {
      setLoading(false);
    }
  };

  // 5a. Rename a thread. The list is updated from the server's response rather than from what
  // was typed, so whatever normalisation the server applied is what the sidebar shows.
  const handleRenameConv = async (conversationId, title) => {
    try {
      const result = await renameConversation(conversationId, title);
      setConversations((prev) => prev.map((c) =>
        c.conversation_id === conversationId ? { ...c, title: result.title } : c));
      setActiveConv((prev) =>
        prev && prev.conversation_id === conversationId ? { ...prev, title: result.title } : prev);
    } catch (err) {
      console.error('Error renaming conversation:', err);
      window.alert(err.message);
    }
  };

  // 5. Delete a thread — removes its folder on the server, then picks a neighbouring thread
  // so the pane is never left showing a conversation that no longer exists.
  const handleDeleteConv = async (conv) => {
    const label = conv.title || 'this thread';
    if (!window.confirm(`Delete "${label}"? Its messages cannot be recovered.`)) return;

    try {
      await deleteConversation(conv.conversation_id);
    } catch (err) {
      console.error('Error deleting conversation:', err);
      return;
    }

    const remaining = conversations.filter((c) => c.conversation_id !== conv.conversation_id);
    setConversations(remaining);

    if (conv.conversation_id !== activeConvId) return;

    const sameRepo = remaining.filter((c) => c.repository_id === selectedRepoId);
    if (sameRepo.length > 0) {
      handleSelectConv(sameRepo[0].conversation_id);
    } else {
      setActiveConvId('');
      setActiveConv(null);
      setMessages([]);
    }
  };

  // 6. Re-index the active repository from its stored source.
  const handleReindexRepo = async (repoId, { full = false } = {}) => {
    try {
      const result = await reindexRepository(repoId, { full });
      setRepoStatus({
        repository_id: repoId,
        status: result.status,
        chunk_count: result.chunk_count
      });
      // File counts and languages can change with the source, so take the fresh rows.
      setRepositories(await fetchRepositories());
    } catch (err) {
      console.error('Error re-indexing repository:', err);
      window.alert(err.message);
    }
  };

  /**
   * Follows an asynchronous import to completion.
   *
   * Polls the status endpoint the server already exposes rather than holding a request open.
   * Backs off from 1s to 4s: a small repository is ready almost at once, a large one takes half a
   * minute, and hammering the endpoint for the whole of it buys nothing.
   */
  const trackImport = async (repositoryId, name) => {
    setPending((prev) => ({ ...prev, [repositoryId]: { status: 'indexing', name } }));

    let delay = 1000;
    const deadline = Date.now() + 10 * 60 * 1000;

    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      delay = Math.min(delay * 1.5, 4000);

      let status;
      try {
        status = await fetchRepositoryStatus(repositoryId);
      } catch {
        // A failed import deletes its record, so a 404 here means it is gone for good.
        setPending((prev) => {
          const next = { ...prev };
          delete next[repositoryId];
          return next;
        });
        return;
      }

      if (status.status === 'ready') {
        setPending((prev) => {
          const next = { ...prev };
          delete next[repositoryId];
          return next;
        });
        const repos = await fetchRepositories();
        setRepositories(repos);
        setSelectedRepoId(repositoryId);
        handleNewChat(repositoryId);
        return;
      }

      if (String(status.status).startsWith('failed')) {
        setPending((prev) => ({
          ...prev,
          [repositoryId]: { status: 'failed', name, error: status.error || 'Indexing failed' }
        }));
        return;
      }
    }
  };

  const handleImportStarted = (started) => {
    trackImport(started.repository_id, started.name);
  };

  const handleSyncRepo = async (repoId, { full = false } = {}) => {
    try {
      const result = await syncRepository(repoId, { full });
      if (result.status === 'already current') {
        setRepoStatus((prev) => ({ ...(prev || {}), repository_id: repoId, status: 'ready', upToDate: true }));
        return result;
      }
      trackImport(repoId, result.name);
      return result;
    } catch (err) {
      console.error('Error syncing repository:', err);
      window.alert(err.message);
    }
  };

  const handleDeleteRepo = async (repoId) => {
    const repo = repositories.find((r) => r.repository_id === repoId);
    const label = repo ? repo.name : repoId;
    if (!window.confirm(`Delete "${label}" and everything indexed from it? Conversations are kept.`)) return;

    try {
      await deleteRepository(repoId);
    } catch (err) {
      console.error('Error deleting repository:', err);
      window.alert(err.message);
      return;
    }

    const remaining = repositories.filter((r) => r.repository_id !== repoId);
    setRepositories(remaining);
    setConversations(await fetchConversations());

    if (repoId === selectedRepoId) {
      const next = remaining[0]?.repository_id || '';
      setSelectedRepoId(next);
      setActiveConvId('');
      setActiveConv(null);
      setMessages([]);
      if (next) handleNewChat(next);
    }
  };

  // A zip upload is now asynchronous too: the request returns once the archive is extracted,
  // and indexing continues in the background. Same handling as a GitHub import — the sidebar
  // shows it indexing and polling selects it when it is ready.
  const handleUploadSuccess = (started) => {
    trackImport(started.repository_id, started.name);
  };

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <Sidebar
        repositories={repositories}
        selectedRepoId={selectedRepoId}
        onSelectRepo={(id) => {
          setSelectedRepoId(id);
          const repoConvs = conversations.filter(c => c.repository_id === id);
          if (repoConvs.length > 0) {
            handleSelectConv(repoConvs[0].conversation_id);
          } else {
            handleNewChat(id);
          }
        }}
        conversations={conversations}
        activeConvId={activeConvId}
        onSelectConv={handleSelectConv}
        onNewChat={() => handleNewChat()}
        onDeleteConv={handleDeleteConv}
        onRenameConv={handleRenameConv}
        onReindexRepo={handleReindexRepo}
        onSyncRepo={handleSyncRepo}
        onDeleteRepo={handleDeleteRepo}
        onOpenUploadModal={() => setIsUploadOpen(true)}
        health={health}
        repoStatus={repoStatus}
        pending={pending}
      />

      <main style={{ flex: 1, display: 'flex', height: '100%', overflow: 'hidden' }}>
        <div style={{ flex: showGraph ? 0.55 : 1, transition: 'all 0.3s ease', height: '100%' }}>
          <ChatWindow
            activeConv={activeConv}
            messages={messages}
            onSendMessage={handleSendMessage}
            loading={loading}
            onToggleGraph={() => setShowGraph(!showGraph)}
            showGraph={showGraph}
            drift={drift}
            driftBusy={driftBusy}
            onKeepVersion={handleKeepVersion}
            onPullLatest={handlePullLatest}
          />
        </div>

        {showGraph && (
          <div style={{ flex: 0.45, borderLeft: '1px solid var(--border-glass)', height: '100%' }}>
            <DependencyGraph repositoryId={selectedRepoId} />
          </div>
        )}
      </main>

      <RepoUploader
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={handleUploadSuccess}
        onImportStarted={handleImportStarted}
      />
    </div>
  );
}
export default App;
