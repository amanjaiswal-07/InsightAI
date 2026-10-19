// src/components/ForecastView.jsx
// Forecast tab: pick a numeric column and horizon, run a Prophet forecast, show chart + narrative.
import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Play, Sparkles, Loader2 } from 'lucide-react';
import ChartCanvas from './ChartCanvas';
import { api } from '../api';
import { marked } from 'marked';

export default function ForecastView({ activeSessionId }) {
  const [columns, setColumns] = useState([]);
  const [selectedCol, setSelectedCol] = useState('');
  const [periods, setPeriods] = useState(6);
  const [isLoading, setIsLoading] = useState(false);
  const [forecastResult, setForecastResult] = useState(null);
  const [forecastError, setForecastError] = useState('');

  // Lets async callbacks detect whether the user has switched sessions
  // since the request was fired, so a slow response can't paint a forecast
  // from session A onto session B's view.
  const activeSessionRef = useRef(activeSessionId);
  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    // Clear any forecast from the previous session immediately — otherwise
    // a brand-new/different session keeps showing the old chart/insights
    // until the user manually runs a new forecast.
    setForecastResult(null);
    setForecastError('');
    setColumns([]);
    setSelectedCol('');
    if (!activeSessionId) return;
    fetchNumericColumns(activeSessionId);
  }, [activeSessionId]);

  const fetchNumericColumns = async (sessionId) => {
    try {
      const res = await api.getNumericColumns(sessionId);
      if (activeSessionRef.current !== sessionId) return;
      if (res.success && res.columns) {
        setColumns(res.columns);
        if (res.columns.length > 0) setSelectedCol(res.columns[0]);
      }
    } catch (e) {
      console.error('Failed to fetch numeric columns', e);
    }
  };

  const handleRunForecast = async () => {
    if (!activeSessionId) return;
    const sessionForThisForecast = activeSessionId;
    setIsLoading(true);
    setForecastError('');
    const question = selectedCol
      ? `forecast ${selectedCol} for next ${periods} periods`
      : `forecast the main metric for next ${periods} periods`;

    try {
      const res = await api.runSilentQuery(sessionForThisForecast, question, selectedCol);
      if (activeSessionRef.current !== sessionForThisForecast) return;
      if (res.success) {
        setForecastResult(res);
      } else {
        setForecastResult(null);
        setForecastError(res.error || 'The forecast could not be generated.');
      }
    } catch (e) {
      console.error('Forecast failed', e);
      if (activeSessionRef.current === sessionForThisForecast) {
        setForecastError(e.response?.data?.detail || 'The forecast request failed. Is the backend running?');
      }
    } finally {
      if (activeSessionRef.current === sessionForThisForecast) setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden" style={{ background: 'var(--bg)' }}>
      {/* Top Controls Bar */}
      <div
        className="px-5 py-3.5 border-b flex flex-wrap items-center justify-between gap-4 shrink-0"
        style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
      >
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md flex items-center justify-center" style={{ background: 'var(--purple-soft)' }}>
              <LineChart className="w-3.5 h-3.5" style={{ color: 'var(--purple)' }} />
            </div>
            <span className="text-[13px] font-semibold text-[var(--text)]">Prophet Forecast</span>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-[12px] text-[var(--text-muted)]">Column</label>
            <select
              value={selectedCol}
              onChange={(e) => setSelectedCol(e.target.value)}
              className="input !w-auto !py-1.5 text-[12.5px]"
            >
              {columns.length === 0 ? (
                <option value="">No numeric columns found</option>
              ) : (
                columns.map((c) => <option key={c} value={c}>{c}</option>)
              )}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-[12px] text-[var(--text-muted)]">Horizon</label>
            <div className="flex gap-1 p-0.5 rounded-md" style={{ background: 'var(--bg-subtle)' }}>
              {[3, 6, 12, 24].map((p) => (
                <button
                  key={p}
                  onClick={() => setPeriods(p)}
                  className="px-2.5 py-1 rounded text-[12px] font-medium transition-colors cursor-pointer"
                  style={
                    periods === p
                      ? { background: 'var(--surface)', color: 'var(--text)', boxShadow: 'var(--shadow-sm)' }
                      : { color: 'var(--text-muted)' }
                  }
                >
                  {p}m
                </button>
              ))}
            </div>
          </div>
        </div>

        <button onClick={handleRunForecast} disabled={isLoading || !activeSessionId} className="btn btn-primary" style={{ background: 'var(--purple)' }}>
          {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          <span>Run forecast</span>
        </button>
      </div>

      {/* Main Studio Area */}
      <div className="flex-1 overflow-y-auto p-6">
        {!forecastResult ? (
          <div className="h-full flex flex-col items-center justify-center gap-3">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: 'var(--purple-soft)' }}>
              <LineChart className="w-5 h-5" style={{ color: 'var(--purple)' }} />
            </div>
            <p className="text-[14px] text-[var(--text-muted)]">Select a numeric column above and click Run forecast</p>
            {forecastError && (
              <p role="alert" className="text-[13px] max-w-md text-center" style={{ color: 'var(--red, #dc2626)' }}>
                {forecastError}
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-5 max-w-4xl mx-auto">
            {forecastResult.plot && (
              <div className="card p-5">
                <h3 className="text-[13px] font-semibold flex items-center gap-2 mb-4" style={{ color: 'var(--purple)' }}>
                  <Sparkles className="w-4 h-4" />
                  Historical vs predicted projection
                </h3>
                <ChartCanvas plotSpec={forecastResult.plot} />
              </div>
            )}

            <div className="card p-5">
              <h3 className="text-[11.5px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-3">
                Forecast analysis &amp; insights
              </h3>
              <div
                className="prose-answer"
                dangerouslySetInnerHTML={{ __html: marked.parse(forecastResult.explanation || '') }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
