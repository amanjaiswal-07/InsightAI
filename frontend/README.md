# InsightAI frontend

React 19 + Vite + Tailwind UI for InsightAI. It has three tabs:

- **Chat:** streaming answers with routing badges, charts and sources.
- **Dashboard:** pinned charts.
- **Forecast:** Prophet forecasts.

See the [root README](../README.md) for the full project.

```bash
npm install
npm run dev
```

`npm run dev` serves on <http://localhost:3000> and proxies API calls to the backend on :8000.

```bash
npm run lint
npm run build
```

`npm run build` writes `dist/`, which FastAPI serves automatically at `/`.

Optional environment variables (`.env`, see `.env.example`):

| Variable | Purpose |
|---|---|
| `VITE_API_BASE_URL` | Backend URL. Leave empty for same origin / the dev proxy. |
| `VITE_API_KEY` | Only if the backend sets `API_KEYS`. Bundled into the browser, so not secret. |
