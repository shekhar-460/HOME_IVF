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
# From project root (e.g. after cloning: cd HOME_IVF)

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

## 3. Run backend + frontend (recommended)

From the project root:

```bash
./start.sh
```

This script checks Python 3.10+, creates/uses `.venv`, installs dependencies if needed, loads `.env` if present, starts the **backend** on **http://0.0.0.0:8000**, then the **frontend** on **http://0.0.0.0:3000**. Press **Ctrl+C** to stop both.

- **Local:** http://localhost:3000 (frontend), http://localhost:8000 (API, docs at /docs)
- **From another device:** http://*your-machine-ip*:3000 (frontend); the UI will call http://*your-machine-ip*:8000 for the API. Use **http://** (not https).

---

## 4. Run manually (two terminals)

**Terminal 1 – backend:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 – frontend:**

```bash
cd frontend && python3 -m http.server 3000 --bind 0.0.0.0
```

Open **http://localhost:3000** (or http://*your-ip*:3000). The frontend uses **http://*current-host*:8000** for the API by default.

---

## Optional: Reduce dependencies / custom API

- **No Redis:** App runs; caching is skipped if Redis is unavailable.
- **No MedGemma:** Set `USE_MEDGEMMA=false` in `.env` to disable AI fallback; chat and engagement tools still work.
- **Different API port:** Run backend on another port (e.g. 8006). In `frontend/index.html`, before `app.js`, add:  
  `<script>window.API_PORT = "8006";</script>`  
  or  
  `<script>window.API_BASE = "http://localhost:8006";</script>`

---

## Run tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## Chat languages

The chat supports **English** and **Hindi**. Language is auto-detected, including **Hinglish** (Roman-script Hindi, e.g. *ivf kya hai?*, *ivf cost kitna hai*). The bot replies in the same language. When the answer is in Hindi, the disclaimer and suggestion buttons (follow-ups, “Get professional help”) are also shown in Hindi so the full response stays in one language.

---

## Key validations (engagement tools)

- **Age:** Female 21–50 (Fertility Readiness, Hormonal Predictor, Treatment Pathway, Home IVF); male 21–55 (Hormonal, Treatment Pathway), male 21–55 (Home IVF).
- **BMI:** Weight/height optional; when provided, calculated BMI must be 15–50 (Fertility Readiness, Visual Health, Treatment Pathway, Home IVF). Frontend shows inline hint and blocks submit if out of range.
- **Treatment Pathway / Home IVF:** Diagnosis and previous treatments can use “Other (not listed)”. Only options from the dropdown list are used for the pathway/eligibility result. If custom text looks nonsensical, a note appears in the result (Reasoning/Reasons and AI Insight). Frontend can block obviously invalid “Other” entries before add.

---

## Next steps

- Full setup and configuration: [README.md](README.md)  
- API and engagement tools: [README.md#api-reference](README.md#api-reference)  
- Engagement tools (age, BMI, custom entries): [README.md#engagement-tools](README.md#engagement-tools)  
- Page translation and CORS: [README.md#page-translation--professional-help](README.md#page-translation--professional-help)
