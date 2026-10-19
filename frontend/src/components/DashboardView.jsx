// src/components/DashboardView.jsx
// Dashboard tab: named dashboards of pinned charts, persisted per session on the backend.
import React, { useState, useEffect, useRef } from 'react';
import { Plus, Sparkles, Trash2, RefreshCw, BarChart2, X } from 'lucide-react';
import ChartCanvas from './ChartCanvas';
import { api } from '../api';

export default function DashboardView({ activeSessionId }) {
  // sessionDashboards structure: { [dashId]: { id, title, prompt, panels: [{ title, plot }] } }
  const [dashboards, setDashboards] = useState({});
  const [activeDashId, setActiveDashId] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newDashTitle, setNewDashTitle] = useState('');
  const [newDashPrompt, setNewDashPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  // Tracks the session the most recently loaded/saved dashboard state
  // belongs to, so a late-arriving response for a session the user has
  // since left can't overwrite the newly active session's dashboard.
  const activeSessionRef = useRef(activeSessionId);
  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  // Load dashboards when session changes
  useEffect(() => {
    // Reset immediately so a brand-new/different session never briefly
    // shows the previous session's dashboards while the fetch is in flight.
    setDashboards({});
    setActiveDashId(null);
    if (!activeSessionId) return;
    loadDashboardState(activeSessionId);
  }, [activeSessionId]);

  const loadDashboardState = async (sessionId) => {
    try {
      const res = await api.getDashboard(sessionId);
      if (activeSessionRef.current !== sessionId) return;
      if (res.success && res.state) {
        if (res.state.dashboards) {
          setDashboards(res.state.dashboards);
          setActiveDashId(res.state.activeId || Object.keys(res.state.dashboards)[0] || null);
        } else if (res.state.panels) {
          const defaultDash = {
            id: 'default',
            title: res.state.title || 'Session Dashboard',
            panels: res.state.panels || [],
          };
          setDashboards({ default: defaultDash });
          setActiveDashId('default');
        }
      } else {
        setDashboards({});
        setActiveDashId(null);
      }
    } catch (e) {
      console.error('Failed to load dashboard state', e);
    }
  };

  const saveDashboardState = async (updatedDict, newActiveId) => {
    setDashboards(updatedDict);
    setActiveDashId(newActiveId);
    if (!activeSessionId) return;
    try {
      await api.saveDashboard(activeSessionId, {
        dashboards: updatedDict,
        activeId: newActiveId,
        built: Object.keys(updatedDict).length > 0,
      });
    } catch (e) {
      console.error('Failed to save dashboard state', e);
    }
  };

  const handleCreateDashboard = async (e) => {
    e.preventDefault();
    if (!newDashTitle.trim() || !activeSessionId) return;

    setIsGenerating(true);
    const newId = `dash_${Date.now()}`;
    let panels = [];

    if (newDashPrompt.trim()) {
      try {
        const queryRes = await api.runSilentQuery(activeSessionId, newDashPrompt.trim());
        if (queryRes.success && queryRes.plot) {
          panels.push({ title: queryRes.plot.title || 'Overview Chart', plot: queryRes.plot });
        }
      } catch (err) {
        console.error('Auto query failed', err);
      }
    }

    const updatedDict = {
      ...dashboards,
      [newId]: { id: newId, title: newDashTitle.trim(), prompt: newDashPrompt.trim(), panels },
    };

    await saveDashboardState(updatedDict, newId);
    setIsGenerating(false);
    setIsModalOpen(false);
    setNewDashTitle('');
    setNewDashPrompt('');
  };

  const handleDeleteDashboard = (dashId) => {
    const nextDict = { ...dashboards };
    delete nextDict[dashId];
    const keys = Object.keys(nextDict);
    saveDashboardState(nextDict, keys.length > 0 ? keys[0] : null);
  };

  const handleRemovePanel = (dashId, panelIndex) => {
    const dash = dashboards[dashId];
    if (!dash) return;
    const updatedPanels = [...dash.panels];
    updatedPanels.splice(panelIndex, 1);
    saveDashboardState({ ...dashboards, [dashId]: { ...dash, panels: updatedPanels } }, dashId);
  };

  const dashList = Object.values(dashboards);
  const activeDash = activeDashId ? dashboards[activeDashId] : null;

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden" style={{ background: 'var(--bg)' }}>
      {/* Dashboard tab strip */}
      <div
        className="flex items-center gap-1 px-5 h-12 border-b shrink-0 overflow-x-auto"
        style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
      >
        {dashList.map((d) => {
          const isActive = d.id === activeDashId;
          return (
            <button
              key={d.id}
              onClick={() => setActiveDashId(d.id)}
              className="group flex items-center gap-2 px-3 py-1.5 rounded-md text-[12.5px] font-medium shrink-0 transition-colors cursor-pointer"
              style={{
                background: isActive ? 'var(--accent-soft)' : 'transparent',
                color: isActive ? 'var(--accent-text)' : 'var(--text-secondary)',
              }}
            >
              <span>{d.title}</span>
              <span className="text-[10.5px] opacity-60">{d.panels?.length || 0}</span>
              <X
                className="w-3 h-3 opacity-0 group-hover:opacity-60 hover:!opacity-100 transition-opacity"
                onClick={(e) => { e.stopPropagation(); handleDeleteDashboard(d.id); }}
              />
            </button>
          );
        })}

        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12.5px] font-medium shrink-0 transition-colors cursor-pointer"
          style={{ color: 'var(--text-muted)' }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-subtle)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
        >
          <Plus className="w-3.5 h-3.5" />
          <span>New dashboard</span>
        </button>
      </div>

      {/* Main Grid View */}
      <div className="flex-1 overflow-y-auto p-6">
        {!activeDash || activeDash.panels.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: 'var(--blue-soft)' }}>
              <BarChart2 className="w-5 h-5" style={{ color: 'var(--blue)' }} />
            </div>
            <div className="text-center">
              <h3 className="text-[14.5px] font-semibold text-[var(--text)]">
                {!activeDash ? 'No dashboard selected' : 'This dashboard is empty'}
              </h3>
              <p className="text-[13px] text-[var(--text-muted)] mt-1.5 max-w-sm">
                {!activeDash
                  ? 'Create a dashboard to start pinning charts.'
                  : 'Pin a chart from any chat answer, or click "New dashboard" above.'}
              </p>
              {!activeDash && (
                <button onClick={() => setIsModalOpen(true)} className="btn btn-primary mt-4">
                  <Plus className="w-4 h-4" />
                  <span>New dashboard</span>
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 max-w-5xl mx-auto">
            {activeDash.panels.map((panel, idx) => (
              <div key={idx} className="card p-4 flex flex-col">
                <div className="flex items-center justify-between pb-3 mb-3 border-b" style={{ borderColor: 'var(--border)' }}>
                  <h4 className="text-[13px] font-semibold text-[var(--text)]">{panel.title}</h4>
                  <button
                    onClick={() => handleRemovePanel(activeDash.id, idx)}
                    className="text-[var(--text-muted)] hover:text-[var(--red)] transition-colors cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                <div className="flex-1 min-h-[260px]">
                  <ChartCanvas plotSpec={panel.plot} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create New Dashboard Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(10,10,14,.5)', backdropFilter: 'blur(2px)' }}>
          <div className="card w-full max-w-md p-6 space-y-4" style={{ boxShadow: 'var(--shadow-lg)' }}>
            <h3 className="text-[14.5px] font-semibold text-[var(--text)] flex items-center gap-2">
              <Sparkles className="w-4 h-4" style={{ color: 'var(--blue)' }} />
              New dashboard
            </h3>

            <form onSubmit={handleCreateDashboard} className="space-y-4">
              <div>
                <label className="block text-[12.5px] font-medium text-[var(--text-secondary)] mb-1.5">Title</label>
                <input
                  type="text"
                  required
                  autoFocus
                  value={newDashTitle}
                  onChange={(e) => setNewDashTitle(e.target.value)}
                  placeholder="e.g. Sales Regional Breakdown"
                  className="input"
                />
              </div>

              <div>
                <label className="block text-[12.5px] font-medium text-[var(--text-secondary)] mb-1.5">Initial chart prompt (optional)</label>
                <input
                  type="text"
                  value={newDashPrompt}
                  onChange={(e) => setNewDashPrompt(e.target.value)}
                  placeholder="e.g. Show a bar chart of total revenue by region"
                  className="input"
                />
              </div>

              <div className="flex justify-end gap-2 pt-1">
                <button type="button" onClick={() => setIsModalOpen(false)} className="btn btn-secondary">
                  Cancel
                </button>
                <button type="submit" disabled={isGenerating || !newDashTitle.trim()} className="btn btn-primary">
                  {isGenerating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
                  <span>Create</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
