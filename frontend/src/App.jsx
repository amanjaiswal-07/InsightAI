// src/App.jsx
// App shell: owns session state, file uploads, the streaming chat loop and tab switching (Chat / Dashboard / Forecast).
import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatView from './components/ChatView';
import DashboardView from './components/DashboardView';
import ForecastView from './components/ForecastView';
import LaunchScreen from './components/LaunchScreen';
import { api } from './api';

export default function App() {
  const [showLaunch, setShowLaunch] = useState(
    () => !localStorage.getItem('analyst_launched')
  );

  const [userId] = useState(() => {
    const saved = localStorage.getItem('analyst_user_id');
    if (saved) return saved;
    const newId = `user_${Math.random().toString(36).substring(2, 10)}`;
    localStorage.setItem('analyst_user_id', newId);
    return newId;
  });

  const [conversations, setConversations] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isQueryLoading, setIsQueryLoading] = useState(false);

  // Always holds the current activeSessionId so async callbacks (stream
  // events, late-arriving fetches) can tell whether they're still relevant
  // by the time they resolve, instead of blindly writing into state that
  // now belongs to a different session.
  const activeSessionRef = useRef(activeSessionId);
  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  // Aborts the in-flight /query/stream request, if any, when the session
  // changes or the component unmounts — prevents a stream started under
  // session A from continuing to append tokens after the user switches to B.
  const streamAbortRef = useRef(null);

  // Load conversations list on mount
  useEffect(() => {
    loadSessions();
  }, []);

  // Load session data when active session changes
  useEffect(() => {
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    setIsQueryLoading(false);

    if (!activeSessionId) {
      setMessages([]);
      setDocuments([]);
      return;
    }
    setMessages([]);
    setDocuments([]);
    loadSessionDetails(activeSessionId);

    return () => {
      streamAbortRef.current?.abort();
      streamAbortRef.current = null;
    };
  }, [activeSessionId]);

  const loadSessions = async () => {
    try {
      const res = await api.getConversations(userId);
      if (res.conversations) {
        setConversations(res.conversations);
        if (!activeSessionId && res.conversations.length > 0) {
          setActiveSessionId(res.conversations[0].conversation_id);
        }
      }
    } catch (e) {
      console.error('Failed to load sessions', e);
    }
  };

  const loadSessionDetails = async (sessionId) => {
    try {
      const res = await api.getConversation(sessionId);
      // Ignore this response if the user has since switched to another
      // session — otherwise a slow fetch for the old session can overwrite
      // the new session's just-loaded (or empty) state.
      if (activeSessionRef.current !== sessionId) return;
      if (res.success && res.conversation) {
        setMessages(res.conversation.messages || []);
      }
      const docsRes = await api.getDocuments(sessionId);
      if (activeSessionRef.current !== sessionId) return;
      if (docsRes.success) {
        setDocuments(docsRes.documents || []);
      }
    } catch (e) {
      console.error('Failed to load session details', e);
    }
  };

  const handleNewSession = async () => {
    try {
      const res = await api.createConversation(userId, 'New Session');
      if (res.success) {
        streamAbortRef.current?.abort();
        streamAbortRef.current = null;
        setMessages([]);
        setDocuments([]);
        setActiveSessionId(res.conversation_id);
        await loadSessions();
      }
    } catch (e) {
      console.error('Failed to create new session', e);
    }
  };

  const handleFileUpload = async (files) => {
    if (!activeSessionId) return;
    setIsUploading(true);
    try {
      const res = await api.uploadFiles(activeSessionId, files);
      if (res.success || res.loaded_files) {
        pollStatus(activeSessionId);
      }
    } catch (e) {
      console.error('Upload failed', e);
      setIsUploading(false);
    }
  };

  const pollStatus = (sessionId) => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > 60) {
        clearInterval(interval);
        setIsUploading(false);
        return;
      }
      try {
        const res = await api.getProcessingStatus(sessionId);
        if (res.status === 'ready') {
          clearInterval(interval);
          setIsUploading(false);
          loadSessionDetails(sessionId);
        } else if (res.status === 'error') {
          clearInterval(interval);
          setIsUploading(false);
        }
      } catch (e) {
        console.error('Status poll error', e);
      }
    }, 1000);
  };

  const handleSendMessage = async (text) => {
    if (!activeSessionId) return;
    const sessionForThisRequest = activeSessionId;

    // Bail out of a state update if the user has switched away from
    // `sessionForThisRequest` since this request started — keeps stream
    // tokens/meta/errors from session A out of session B's message list.
    const isStale = () => activeSessionRef.current !== sessionForThisRequest;

    // Optimistically add user message
    const userMsg = { role: 'user', content: text, timestamp: new Date().toISOString() };
    setMessages((prev) => [...prev, userMsg]);
    setIsQueryLoading(true);

    let assistantMsg = {
      role: 'assistant',
      content: '',
      routing: '',
      metadata: {},
    };

    setMessages((prev) => [...prev, assistantMsg]);

    const controller = new AbortController();
    streamAbortRef.current = controller;

    await api.streamQuery(activeSessionId, text, null, {
      signal: controller.signal,
      onMeta: (meta) => {
        if (isStale()) return;
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.routing = meta.routing;
          last.metadata = { ...last.metadata, plot: meta.plot, sql_query: meta.sql_query };
          updated[updated.length - 1] = last;
          return updated;
        });
      },
      onToken: (chunk) => {
        if (isStale()) return;
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.content = (last.content || '') + chunk;
          updated[updated.length - 1] = last;
          return updated;
        });
      },
      onDone: () => {
        if (streamAbortRef.current === controller) streamAbortRef.current = null;
        if (isStale()) return;
        setIsQueryLoading(false);
        loadSessions();
      },
      onError: (err) => {
        if (streamAbortRef.current === controller) streamAbortRef.current = null;
        if (isStale()) return;
        setIsQueryLoading(false);
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.content = `Error: ${err.message || err.error || 'Failed to get response'}`;
          updated[updated.length - 1] = last;
          return updated;
        });
      },
    });
  };

  const handlePinChart = async (plotSpec) => {
    if (!activeSessionId) return;
    const sessionForThisPin = activeSessionId;
    try {
      const res = await api.getDashboard(sessionForThisPin);
      if (activeSessionRef.current !== sessionForThisPin) return;
      let state = res.state || { dashboards: {}, activeId: null, built: true };
      if (!state.dashboards) state.dashboards = {};

      const dashKeys = Object.keys(state.dashboards);
      let targetId = state.activeId || (dashKeys.length > 0 ? dashKeys[0] : 'dash_main');

      if (!state.dashboards[targetId]) {
        state.dashboards[targetId] = {
          id: targetId,
          title: 'Main Dashboard',
          panels: [],
        };
      }

      state.dashboards[targetId].panels.push({
        title: plotSpec.title || 'Pinned Chart',
        plot: plotSpec,
      });

      state.activeId = targetId;
      state.built = true;

      await api.saveDashboard(sessionForThisPin, state);
      if (activeSessionRef.current === sessionForThisPin) setActiveTab('dashboard');
    } catch (e) {
      console.error('Failed to pin chart', e);
    }
  };

  if (showLaunch) {
    return (
      <LaunchScreen
        onStart={() => {
          localStorage.setItem('analyst_launched', '1');
          setShowLaunch(false);
        }}
      />
    );
  }

  const activeConversation = conversations.find((c) => c.conversation_id === activeSessionId);

  return (
    <div className="flex h-screen w-screen overflow-hidden" style={{ background: 'var(--bg)', color: 'var(--text)' }}>
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewSession={handleNewSession}
        onFileUpload={handleFileUpload}
        isUploading={isUploading}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <Header
          activeTab={activeTab}
          onTabChange={setActiveTab}
          activeSessionId={activeSessionId}
          sessionTitle={activeConversation?.title}
          documentsCount={documents.length}
        />

        <div className="flex-1 overflow-hidden relative">
          {activeTab === 'chat' && (
            <ChatView
              activeSessionId={activeSessionId}
              messages={messages}
              onSendMessage={handleSendMessage}
              onPinChart={handlePinChart}
              isLoading={isQueryLoading}
            />
          )}

          {activeTab === 'dashboard' && <DashboardView activeSessionId={activeSessionId} />}

          {activeTab === 'forecast' && <ForecastView activeSessionId={activeSessionId} />}
        </div>
      </div>
    </div>
  );
}
