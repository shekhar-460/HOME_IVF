# Quick Start Guide

Get **AI Engagement Tools (Home IVF)** running in a few steps.

---

## Prerequisites

- **Python** 3.10+
- **PostgreSQL** (default port in config: **5433**)
- **Redis** (optional; for caching)

---

## 1. Setup

```bash
# From project root
cd "HOME IVF"   # or your project path

python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

---

## 2. Database

Create a PostgreSQL database and set the connection URL (default in code uses port **5433**):

```bash
# Example: create DB and set env
createdb patient_bot
export DATABASE_URL="postgresql://postgres:postgres@localhost:5433/patient_bot"
```

Or create a `.env` in the project root:

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/patient_bot
```

Tables are created automatically on first run.

---

## 3. Run the backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **API:** http://localhost:8000  
- **Swagger:** http://localhost:8000/docs  
- **ReDoc:** http://localhost:8000/redoc  

---

## 4. Run the frontend (optional)

In a **second terminal**, from project root:

```bash
cd frontend && python3 -m http.server 3000
```

Open **http://localhost:3000** in your browser. The UI talks to the API at `http://localhost:8000` by default.

---

## Optional: Reduce dependencies for first run

- **No Redis:** App will run; caching is skipped if Redis is unavailable (check logs).
- **No MedGemma:** Set `USE_MEDGEMMA=false` (env or `.env`) to disable AI fallback and engagement insights; chat and engagement tools still work with rule-based logic.
- **Different API port:** e.g. `uvicorn app.main:app --port 8006`. If using the frontend, set `window.API_BASE = "http://localhost:8006"` in `frontend/index.html` before loading `app.js`.

---

## Run tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## Next steps

- Full setup and configuration: [README.md](README.md)  
- API and engagement tools: [README.md#api-reference](README.md#api-reference)  
- Frontend custom API URL: [README.md#frontend-html--css--js](README.md#frontend-html--css--js)
