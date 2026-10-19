// src/components/ChartCanvas.jsx
// Renders a backend plot spec (bar / line / pie) with Chart.js.
import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  BarController, LineController, PieController, DoughnutController,
  CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement,
  Title, Tooltip, Legend,
} from 'chart.js';

// Tree-shaken Chart.js: every chart type needs its controller registered,
// otherwise `new Chart({ type: 'bar' })` throws and takes the React tree down.
ChartJS.register(
  BarController, LineController, PieController, DoughnutController,
  CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement,
  Title, Tooltip, Legend,
);

const PALETTE = ['#6366f1', '#0ea5e9', '#f59e0b', '#ef4444', '#a855f7', '#10b981', '#fb923c', '#818cf8'];

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export default function ChartCanvas({ plotSpec }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !plotSpec) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext('2d');
    const isPie = plotSpec.type === 'pie' || plotSpec.type === 'doughnut';

    const text = cssVar('--text', '#18181b');
    const muted = cssVar('--text-muted', '#94949c');
    const border = cssVar('--border', '#e8e8ea');
    const accent = cssVar('--accent', '#6366f1');
    const surfaceRaised = cssVar('--surface-raised', '#ffffff');

    // Two spec shapes come from the backend:
    //   simple: { x: [...labels], y: [...values] }               (SQL charts)
    //   series: { data: [{date, actual, forecast}], x_key, series } (forecasts)
    const isMultiSeries = Array.isArray(plotSpec.series) && Array.isArray(plotSpec.data);
    const labels = isMultiSeries
      ? plotSpec.data.map((row) => row[plotSpec.x_key])
      : plotSpec.x || [];
    const datasets = isMultiSeries
      ? plotSpec.series.map((s, i) => ({
          label: s.label || s.key,
          data: plotSpec.data.map((row) => row[s.key] ?? null),
          borderColor: s.color || PALETTE[i % PALETTE.length],
          backgroundColor: `${s.color || PALETTE[i % PALETTE.length]}33`,
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.3,
        }))
      : [
          {
            label: plotSpec.y_label || 'Value',
            data: plotSpec.y || [],
            backgroundColor: isPie ? PALETTE.slice(0, labels.length) : `${accent}33`,
            borderColor: isPie ? PALETTE.slice(0, labels.length) : accent,
            borderWidth: 1.5,
            borderRadius: isPie ? 0 : 4,
            tension: 0.3,
          },
        ];

    chartRef.current = new ChartJS(ctx, {
      type: plotSpec.type || 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: plotSpec.title || 'Data Visualization',
            color: text,
            font: { family: 'Inter, sans-serif', size: 13, weight: '600' },
            padding: { bottom: 12 },
          },
          legend: {
            display: isPie || isMultiSeries,
            labels: { color: muted, font: { family: 'Inter, sans-serif', size: 11 } },
          },
          tooltip: {
            backgroundColor: surfaceRaised,
            borderColor: border,
            borderWidth: 1,
            titleColor: accent,
            bodyColor: text,
            padding: 8,
            cornerRadius: 6,
          },
        },
        scales: isPie
          ? {}
          : {
              x: {
                ticks: { color: muted, font: { family: 'Inter, sans-serif', size: 10.5 } },
                grid: { color: border },
              },
              y: {
                ticks: { color: muted, font: { family: 'Inter, sans-serif', size: 10.5 } },
                grid: { color: border },
                beginAtZero: true,
              },
            },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [plotSpec]);

  return (
    <div className="relative w-full h-64">
      <canvas ref={canvasRef} />
    </div>
  );
}
