// src/components/Header.jsx
// Top bar: active session title and tab navigation.
import React from 'react';
import { MessageSquare, LayoutGrid, LineChart, FileStack } from 'lucide-react';

const TABS = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'dashboard', label: 'Dashboard', icon: LayoutGrid },
  { id: 'forecast', label: 'Forecast', icon: LineChart },
];

export default function Header({
  activeTab,
  onTabChange,
  activeSessionId,
  sessionTitle,
  documentsCount,
}) {
  return (
    <header
      className="h-14 flex items-center px-5 gap-1 border-b shrink-0"
      style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
    >
      {activeSessionId ? (
        <>
          <div className="flex items-center gap-2 pr-4 mr-1 border-r shrink-0 max-w-[220px]" style={{ borderColor: 'var(--border)' }}>
            <span className="text-[13.5px] font-medium text-[var(--text)] truncate">
              {sessionTitle || 'New Session'}
            </span>
            {documentsCount > 0 && (
              <span className="flex items-center gap-1 text-[11px] text-[var(--text-muted)] shrink-0">
                <FileStack className="w-3 h-3" />
                {documentsCount}
              </span>
            )}
          </div>

          <nav className="flex items-center gap-1">
            {TABS.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition-colors cursor-pointer"
                  style={{
                    background: isActive ? 'var(--accent-soft)' : 'transparent',
                    color: isActive ? 'var(--accent-text)' : 'var(--text-secondary)',
                  }}
                  onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'var(--bg-subtle)'; }}
                  onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent'; }}
                >
                  <tab.icon className="w-[15px] h-[15px]" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </>
      ) : (
        <span className="text-[13.5px] text-[var(--text-muted)]">Select or create a session to begin</span>
      )}
    </header>
  );
}
