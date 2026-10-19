// src/components/ChatView.jsx
// Chat tab: message list with routing badges, charts, sources, and the question input.
import React, { useState, useRef, useEffect } from 'react';
import { Send, Pin, Bot, User, Sparkles } from 'lucide-react';
import { marked } from 'marked';
import ChartCanvas from './ChartCanvas';

marked.setOptions({ breaks: true, gfm: true });

const ROUTING_BADGE = {
  sql: 'badge-accent',
  document: 'badge-blue',
  rag: 'badge-blue',
  hybrid: 'badge-amber',
  forecast: 'badge-purple',
  general: 'badge-neutral',
};

export default function ChatView({
  activeSessionId,
  messages,
  onSendMessage,
  onPinChart,
  isLoading,
}) {
  const [inputText, setInputText] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading || !activeSessionId) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden" style={{ background: 'var(--bg)' }}>
      {/* Messages Feed */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-6 space-y-6">
          {!activeSessionId ? (
            <EmptyState icon={Sparkles} text="Select or create a session to start" />
          ) : messages.length === 0 ? (
            <div className="h-[60vh] flex flex-col items-center justify-center gap-4">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: 'var(--accent-soft)' }}>
                <Bot className="w-5 h-5" style={{ color: 'var(--accent)' }} />
              </div>
              <p className="text-[14px] text-[var(--text-secondary)]">Query your CSV data or documents using natural language</p>
              <div className="flex flex-wrap justify-center gap-2">
                {['Who is the top sales rep?', 'Show revenue by region'].map((q) => (
                  <button
                    key={q}
                    onClick={() => activeSessionId && !isLoading && onSendMessage(q)}
                    className="px-3 py-1.5 rounded-full text-[12.5px] border transition-colors cursor-pointer"
                    style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)', background: 'var(--surface)' }}
                  >
                    "{q}"
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, index) => {
              const isUser = msg.role === 'user';
              const plotData = msg.metadata?.plot || msg.plot;
              const routing = msg.routing;

              return (
                <div key={index} className={`flex flex-col gap-1.5 ${isUser ? 'items-end' : 'items-start'} animate-fade-up`}>
                  <div className="flex items-center gap-1.5 px-1 text-[11.5px] font-medium text-[var(--text-muted)]">
                    {isUser ? (
                      <>
                        <User className="w-3 h-3" />
                        <span>You</span>
                      </>
                    ) : (
                      <>
                        <Bot className="w-3.5 h-3.5" style={{ color: 'var(--accent)' }} />
                        <span style={{ color: 'var(--text)' }}>InsightAI</span>
                        {routing && (
                          <span className={`badge ${ROUTING_BADGE[routing?.toLowerCase()] || 'badge-neutral'}`}>
                            {routing}
                          </span>
                        )}
                      </>
                    )}
                  </div>

                  <div
                    className={`px-4 py-3 rounded-2xl max-w-[90%] text-[14px] ${isUser ? 'rounded-tr-md' : 'rounded-tl-md'}`}
                    style={
                      isUser
                        ? { background: 'var(--accent)', color: 'white' }
                        : { background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }
                    }
                  >
                    {isUser ? (
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    ) : (
                      <div
                        className="prose-answer"
                        dangerouslySetInnerHTML={{ __html: marked.parse(msg.content || '') }}
                      />
                    )}

                    {plotData && (
                      <div className="mt-3 p-3 rounded-lg" style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                        <ChartCanvas plotSpec={plotData} />
                        <div className="mt-2 flex items-center justify-end pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
                          <button
                            onClick={() => onPinChart(plotData)}
                            className="btn btn-ghost !text-[12px] !py-1 !px-2"
                            style={{ color: 'var(--purple)' }}
                          >
                            <Pin className="w-3 h-3" />
                            <span>Pin to dashboard</span>
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
          <div ref={chatEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t shrink-0" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto px-6 py-4 flex gap-2.5">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isLoading || !activeSessionId}
            placeholder={
              !activeSessionId
                ? 'Create a session first…'
                : 'Ask a question (e.g. "Show a bar chart of total revenue by region")'
            }
            className="input flex-1"
          />
          <button
            type="submit"
            disabled={isLoading || !inputText.trim() || !activeSessionId}
            className="btn btn-primary !px-4"
          >
            <span>Send</span>
            <Send className="w-3.5 h-3.5" />
          </button>
        </form>
      </div>
    </div>
  );
}

function EmptyState({ icon: Icon, text }) {
  return (
    <div className="h-[70vh] flex flex-col items-center justify-center gap-3">
      <Icon className="w-8 h-8 text-[var(--text-muted)]" />
      <p className="text-[14px] text-[var(--text-muted)]">{text}</p>
    </div>
  );
}
