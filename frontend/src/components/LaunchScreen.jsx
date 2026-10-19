// src/components/LaunchScreen.jsx
// Empty state shown before a session exists: what the app does and how to start.
import React from 'react';
import { Database, FileText, BarChart3, LineChart, ArrowRight, Layers, ShieldCheck } from 'lucide-react';

const FEATURES = [
  {
    icon: Database,
    color: 'var(--accent)',
    soft: 'var(--accent-soft)',
    title: 'SQL Analyst',
    desc: 'Ask questions in plain English and get precise, calculated answers straight from your uploaded data.',
  },
  {
    icon: FileText,
    color: 'var(--blue)',
    soft: 'var(--blue-soft)',
    title: 'Document RAG',
    desc: 'Upload PDFs, Word docs, or reports and query them semantically, in 9 languages.',
  },
  {
    icon: BarChart3,
    color: 'var(--amber)',
    soft: 'var(--amber-soft)',
    title: 'Instant Charts',
    desc: 'Every answer that can be visualized gets charted automatically. Pin any chart to a dashboard.',
  },
  {
    icon: LineChart,
    color: 'var(--purple)',
    soft: 'var(--purple-soft)',
    title: 'Prophet Forecasting',
    desc: 'Project any numeric column forward and get an AI-written narrative of what the trend means.',
  },
];

const POINTS = [
  { icon: Layers, text: 'Every session is fully isolated, with its own files, chat, and dashboards' },
  { icon: ShieldCheck, text: 'Routing picks SQL, documents, or both automatically. No manual mode switching' },
];

export default function LaunchScreen({ onStart }) {
  return (
    <div className="fixed inset-0 bg-[var(--bg)] overflow-y-auto z-[8000]">
      <div className="hero-grid" />
      <div className="hero-glow" />

      <div className="relative z-10 min-h-full flex flex-col items-center px-6 py-20 gap-16">
        {/* Hero */}
        <div className="flex flex-col items-center gap-6 text-center max-w-xl animate-fade-up">
          <div className="badge badge-accent">Multi-agent NL analytics</div>

          <h1 className="text-[2.75rem] leading-[1.1] font-semibold tracking-tight text-[var(--text)]">
            Talk to your data.<br />Get answers, not queries.
          </h1>

          <p className="text-[15px] text-[var(--text-secondary)] leading-relaxed max-w-md">
            Upload a spreadsheet or a document. Ask questions the way you'd ask a colleague.
            Get calculated answers, cited sources, charts, and forecasts, routed automatically
            to the right engine, isolated per session.
          </p>

          <button onClick={onStart} className="btn btn-primary !px-6 !py-3 !text-sm mt-2 shadow-[var(--shadow-lg)]">
            <span>Enter workspace</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <div className="flex flex-col gap-2 mt-2">
            {POINTS.map((p) => (
              <div key={p.text} className="flex items-center gap-2 text-[13px] text-[var(--text-muted)]">
                <p.icon className="w-3.5 h-3.5 shrink-0" />
                <span>{p.text}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full max-w-5xl animate-fade-up" style={{ animationDelay: '.1s' }}>
          {FEATURES.map((f) => (
            <div key={f.title} className="feature-card" style={{ '--fc': f.color, '--fc-soft': f.soft }}>
              <div className="feature-icon">
                <f.icon className="w-[18px] h-[18px]" />
              </div>
              <div className="text-[14px] font-semibold text-[var(--text)] mt-3.5">
                {f.title}
              </div>
              <p className="text-[13px] text-[var(--text-muted)] leading-relaxed mt-1.5">
                {f.desc}
              </p>
            </div>
          ))}
        </div>

        <div className="text-[12px] text-[var(--text-muted)]">
          No sign-up required · Your data stays in your session
        </div>
      </div>
    </div>
  );
}
