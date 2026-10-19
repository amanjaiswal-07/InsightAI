// src/components/Sidebar.jsx
// Left sidebar: session list, new session, file upload and loaded documents.
import React, { useState } from 'react';
import { Plus, Upload, Loader2, MessageSquare, FileStack } from 'lucide-react';

export default function Sidebar({
  conversations,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onFileUpload,
  isUploading,
}) {
  const [isDragging, setIsDragging] = useState(false);

  const formatDate = (d) => {
    if (!d) return '';
    const diff = (new Date() - new Date(d)) / 1000;
    if (diff < 60) return 'now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
    return `${Math.floor(diff / 86400)}d`;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length > 0) onFileUpload(e.dataTransfer.files);
  };

  return (
    <aside
      className="flex flex-col h-full shrink-0 border-r"
      style={{ width: 'var(--sidebar-w)', background: 'var(--bg-subtle)', borderColor: 'var(--border)' }}
    >
      {/* Brand */}
      <div className="h-14 flex items-center gap-2.5 px-4 border-b shrink-0" style={{ borderColor: 'var(--border)' }}>
        <div className="w-7 h-7 rounded-md flex items-center justify-center shrink-0" style={{ background: 'var(--accent)' }}>
          <span className="text-white text-[13px] font-bold">I</span>
        </div>
        <span className="font-semibold text-[14.5px] text-[var(--text)] tracking-tight">InsightAI</span>
      </div>

      {/* New session */}
      <div className="p-3 shrink-0">
        <button onClick={onNewSession} className="btn btn-primary w-full !justify-center !py-2.5">
          <Plus className="w-4 h-4" />
          <span>New session</span>
        </button>
      </div>

      {/* Conversations */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        <div className="px-2.5 py-1.5 text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider">
          Sessions
        </div>

        {conversations.length === 0 ? (
          <div className="px-3 py-6 text-center">
            <MessageSquare className="w-6 h-6 mx-auto text-[var(--text-muted)] mb-2" />
            <p className="text-[12.5px] text-[var(--text-muted)]">No sessions yet</p>
          </div>
        ) : (
          conversations.map((conv) => {
            const isActive = conv.conversation_id === activeSessionId;
            return (
              <button
                key={conv.conversation_id}
                onClick={() => onSelectSession(conv.conversation_id)}
                className="w-full text-left px-3 py-2.5 rounded-lg transition-colors cursor-pointer block"
                style={{
                  background: isActive ? 'var(--accent-soft)' : 'transparent',
                }}
                onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'var(--surface)'; }}
                onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent'; }}
              >
                <div
                  className="text-[13px] font-medium truncate"
                  style={{ color: isActive ? 'var(--accent-text)' : 'var(--text)' }}
                >
                  {conv.title || 'New Session'}
                </div>
                <div className="flex items-center gap-2 mt-1 text-[11px] text-[var(--text-muted)]">
                  <span>{conv.message_count || 0} msgs</span>
                  {conv.documents > 0 && (
                    <span className="flex items-center gap-0.5">
                      <FileStack className="w-2.5 h-2.5" />
                      {conv.documents}
                    </span>
                  )}
                  <span className="ml-auto">{formatDate(conv.updated_at)}</span>
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* Upload zone */}
      <div className="p-3 border-t shrink-0" style={{ borderColor: 'var(--border)' }}>
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className="rounded-lg border border-dashed p-3.5 text-center transition-colors"
          style={{
            borderColor: isDragging ? 'var(--accent)' : 'var(--border-strong)',
            background: isDragging ? 'var(--accent-soft)' : 'var(--surface)',
          }}
        >
          <input
            type="file"
            id="sidebarFileInput"
            multiple
            accept=".csv,.xlsx,.pdf,.docx,.txt"
            className="hidden"
            onChange={(e) => e.target.files?.length && onFileUpload(e.target.files)}
          />
          <label htmlFor="sidebarFileInput" className="cursor-pointer flex flex-col items-center gap-1.5">
            {isUploading ? (
              <Loader2 className="w-4 h-4 text-[var(--accent)] animate-spin" />
            ) : (
              <Upload className="w-4 h-4 text-[var(--text-muted)]" />
            )}
            <span className="text-[12px] font-medium text-[var(--text)]">
              {isUploading ? 'Indexing files…' : 'Attach files to session'}
            </span>
            <span className="text-[10.5px] text-[var(--text-muted)]">CSV, PDF, XLSX, DOCX</span>
          </label>
        </div>
      </div>
    </aside>
  );
}
