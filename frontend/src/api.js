// src/api.js
// Thin client for the FastAPI backend. Every call goes through here so the
// base URL and the optional API key are configured in one place.
import axios from 'axios';

// Empty base URL = same origin (Vite dev proxy, or FastAPI serving the build).
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
// Must match one of the backend's API_KEYS. Leave unset when backend auth is off.
const API_KEY = import.meta.env.VITE_API_KEY || '';
const AUTH_HEADERS = API_KEY ? { 'X-Api-Key': API_KEY } : {};

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
    ...AUTH_HEADERS,
  },
});

export const api = {
  // Session & Conversation
  createConversation: async (userId, title = 'New Session') => {
    const res = await client.post('/conversations', { user_id: userId, title }, {
      headers: { 'X-User-ID': userId },
    });
    return res.data;
  },

  getConversations: async (userId) => {
    const res = await client.get('/conversations', {
      headers: { 'X-User-ID': userId },
    });
    return res.data;
  },

  getConversation: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}`);
    return res.data;
  },

  // Document Management
  uploadFiles: async (conversationId, files) => {
    const formData = new FormData();
    Array.from(files).forEach((f) => formData.append('files', f));
    const res = await axios.post(`${API_BASE}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'X-Conversation-ID': conversationId,
        ...AUTH_HEADERS,
      },
    });
    return res.data;
  },

  getProcessingStatus: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}/processing-status`);
    return res.data;
  },

  getDocuments: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}/documents`);
    return res.data;
  },

  // Numeric columns of the session's uploaded CSVs (for the Forecast tab).
  getNumericColumns: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}/schema`);
    return { success: res.data.success, columns: res.data.numeric_columns || [] };
  },

  // Dashboards (Session Level)
  getDashboard: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}/dashboard`);
    return res.data;
  },

  saveDashboard: async (conversationId, state) => {
    const res = await client.post(`/conversations/${conversationId}/dashboard`, { state });
    return res.data;
  },

  // Query Execution
  runQuery: async (conversationId, question, valueCol = null) => {
    const res = await client.post('/query', {
      question,
      conversation_id: conversationId,
      value_col: valueCol,
    });
    return res.data;
  },

  // Same pipeline as runQuery, but nothing is written to chat history.
  runSilentQuery: async (conversationId, question, valueCol = null) => {
    const res = await client.post('/query/silent', {
      question,
      conversation_id: conversationId,
      value_col: valueCol,
    });
    return res.data;
  },

  // SSE Stream helper. Pass `signal` (from an AbortController) so callers can
  // cancel an in-flight stream, e.g. when the user switches sessions.
  streamQuery: async (conversationId, question, valueCol, { onMeta, onToken, onDone, onError, signal }) => {
    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...AUTH_HEADERS,
        },
        body: JSON.stringify({
          question,
          conversation_id: conversationId,
          value_col: valueCol,
        }),
        signal,
      });

      if (!response.ok) throw new Error(`Stream error: ${response.statusText}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const block of parts) {
          let eventName = 'message';
          let dataStr = '';

          for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) eventName = line.slice(7).trim();
            if (line.startsWith('data: ')) dataStr = line.slice(6).trim();
          }

          if (!dataStr) continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (eventName === 'meta' && onMeta) onMeta(parsed);
            else if (eventName === 'token' && onToken) onToken(parsed);
            else if (eventName === 'done' && onDone) onDone(parsed);
            else if (eventName === 'error' && onError) onError(parsed);
          } catch (e) {
            console.error('SSE JSON parse error', e);
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (onError) onError(err);
    }
  },
};
