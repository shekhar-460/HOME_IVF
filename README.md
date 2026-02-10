# AI Engagement Tools (Home IVF)

A **FastAPI** application for multilingual IVF patient education and pre-consultation engagement: conversational chat (FAQ + optional MedGemma), five AI-driven tools (fertility readiness, hormonal predictor, visual wellness, treatment pathway, Home IVF eligibility), and admin/analytics. Built for the **Home IVF** / fertility context.

---

## Table of Contents

- [Quick Start](QUICKSTART.md) – minimal steps to run the app
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Engagement Tools](#engagement-tools)
- [Chat & Knowledge Engine](#chat--knowledge-engine)
- [Testing](#testing)
- [Development & Production](#development--production)
- [License](#license)

---

## Overview

The system has two main pillars:

1. **Conversational patient education (chat)**  
   English and Hindi chat with intent classification, semantic search over FAQs (from `knowledge_base/sample_faqs.json`), and optional **MedGemma-4b-it** fallback for IVF-related questions. All answers are constrained by an **IVF guardrail**. Translation (e.g. MedGemma output to Hindi) uses **googletrans**.

2. **AI engagement tools (REST)**  
   Five standalone POST APIs for pre-consultation use:
   - **Fertility Readiness Calculator** – Risk score (0–100) and next steps from age, medical history, lifestyle, menstrual and pregnancy history.
   - **Hormonal & Ovarian Health Predictor** – When to consider AMH, semen analysis, or specialist (questionnaire + rules).
   - **Visual Health Indicator (exploratory)** – Non-diagnostic wellness from self-reported sleep, stress, BMI (and optional image placeholder).
   - **Treatment Pathway Recommender** – Suggests natural conception, IUI, IVF, or fertility preservation from inputs.
   - **Home IVF Eligibility Checker** – Quick suitability and prompt to book a consultation.

Engagement tools use **rule-based logic** and can optionally add short **MedGemma** insights (`use_ai_insight: true`).

---

## Features

| Area | Description |
|------|-------------|
| **Languages** | English (`en`), Hindi (`hi`) with auto-detection (langdetect) |
| **Chat** | Create conversation, send message, get history, WebSocket; intent classification and escalation |
| **Knowledge** | Semantic search over JSON FAQs; optional MedGemma-4b-it fallback; IVF-only guardrail |
| **Translation** | googletrans (3.1.0a0) for en ↔ hi (e.g. MedGemma responses) |
| **Engagement** | Five POST endpoints under `/api/v1/engagement/` (see [Engagement Tools](#engagement-tools)) |
| **Admin** | Analytics, FAQ/Article CRUD, GPU memory cleanup |
| **Health** | Root, `/health/`, `/health/ready`, `/health/live` for monitoring |

---

## Architecture

```text
                        ┌───────────────────────────────────────┐
                        │          Browser / Frontend           │
                        │   static HTML / CSS / JS (frontend/)  │
                        └──────────────┬────────────────────────┘
                                       │ HTTP (JSON / WebSocket)
                                       ▼
                        ┌───────────────────────────────────────┐
                        │          FastAPI application          │
                        │              app.main                 │
                        └──────────────┬────────────────────────┘
                                       │ includes routers
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
 ┌────────────────┐           ┌─────────────────┐           ┌──────────────────┐
 │  /health/*     │           │ /api/v1/chat    │           │ /api/v1/engagement │
 │  health.py     │           │ chat.py         │           │ engagement.py      │
 └────────────────┘           └─────────────────┘           └──────────────────┘
                                     │                             │
                                     ▼                             ▼
                         ┌────────────────────┐          ┌──────────────────────┐
                         │  ResponseGenerator │          │  EngagementService   │
                         │  (chat pipeline)   │          │  (5 tools, rules)    │
                         └─────────┬──────────┘          └──────────┬───────────┘
                                   │                                │
             ┌─────────────────────┼────────────────────────┐       │
             │                     │                        │       │
             ▼                     ▼                        ▼       ▼
   ┌─────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
   │ IntentClassifier│   │ KnowledgeEngine      │   │ IVFGuardrail         │
   │ (intent, NLU)   │   │ (FAQ search +        │   │ (IVF-only content    │
   └───────┬─────────┘   │  MedGemma fallback)  │   │  and safety filter)  │
           │             └──────────┬───────────┘   └─────────┬────────────┘
           │                        │                         │
           ▼                        ▼                         │
  ┌─────────────────┐     ┌──────────────────────────┐        │
  │Conversation     │     │  MedGemma-4b-it model    │        │
  │Manager          │◄──► │  (local or HF, image+txt)│        │
  │(DB persistence) │     └──────────────────────────┘        │
  └────────┬────────┘                 ▲                       │
           │                          │                       │
           │                          │                       │
           ▼                          │                       ▼
  ┌──────────────────────┐     ┌──────────────────┐   ┌────────────────────┐
  │PostgreSQL            │     │ Redis cache      │   │ EscalationManager  │
  │conversations, msgs,  │◄──► │ FAQ / MedGemma   │   │ + outbound systems │
  │faqs, articles,       │     │ responses        │   │ (future integration│
  │escalations           │     └──────────────────┘   └────────────────────┘
  └──────────────────────┘

Chat flow (high level):
  frontend → /api/v1/chat/message → ResponseGenerator
    → IntentClassifier → KnowledgeEngine (FAQ or MedGemma)
    → IVFGuardrail → store via ConversationManager (PostgreSQL)
    → send BotResponse back to frontend.

Engagement flow (high level):
  frontend or partner app → /api/v1/engagement/*
    → EngagementService (rule-based logic + optional KnowledgeEngine / MedGemma)
    → JSON result (scores, recommendations, ai_insight) returned to caller.
```

- **Chat**: User message → intent → FAQ search (or MedGemma if enabled and no good match) → IVF guardrail → formatted response, follow-ups, suggestions; optional escalation. Conversations and messages are persisted in PostgreSQL, with Redis optionally caching FAQ embeddings and MedGemma responses.
- **Engagement**: JSON body → `EngagementService` (rules; optional `KnowledgeEngine.get_answer_from_medgemma` for `ai_insight`) → JSON response. Inputs/outputs are pure JSON, making the tools easy to call from the bundled frontend or external systems.

---

## Project Structure

```
HOME IVF/
├── app/
│   ├── main.py                    # FastAPI app, CORS, routers, startup/shutdown, GPU cleanup
│   ├── config.py                  # Pydantic Settings (env / .env)
│   ├── api/routes/
│   │   ├── health.py              # GET /health/, /health/ready, /health/live
│   │   ├── chat.py                # POST /api/v1/chat/message, /conversation; GET /conversation/{id}; POST /escalate; WS /ws/{id}
│   │   ├── engagement.py          # POST /api/v1/engagement/* (5 tools)
│   │   └── admin.py               # GET /api/v1/admin/analytics; POST /api/v1/admin/faq, /article; POST /api/v1/admin/cleanup-gpu
│   ├── database/
│   │   ├── connection.py         # Engine, SessionLocal, get_db, init_db
│   │   └── models.py              # Conversation, Message, FAQ, Article, Escalation (PostgreSQL)
│   ├── models/
│   │   ├── schemas.py             # Chat/conversation Pydantic models
│   │   ├── engagement_schemas.py  # Request/response for engagement tools
│   │   └── medgemma-4b-it/        # Optional local MedGemma weights
│   ├── services/
│   │   ├── conversation_manager.py
│   │   ├── intent_classifier.py
│   │   ├── knowledge_engine.py    # FAQ load (knowledge_base/sample_faqs.json), semantic search, MedGemma (lazy), cache
│   │   ├── response_generator.py
│   │   ├── ivf_guardrail.py       # IVF-only content filter
│   │   ├── engagement_service.py # All 5 engagement tools
│   │   ├── escalation_manager.py
│   │   ├── followup_generator.py
│   │   └── proactive_suggestions.py
│   └── utils/
│       ├── language_detector.py   # langdetect
│       └── translator.py         # googletrans (en/hi)
├── knowledge_base/               # Optional: sample_faqs.json (FAQ source if present)
├── tests/
│   ├── conftest.py               # Fixtures, engagement dependency override
│   ├── test_health.py
│   ├── test_engagement_api.py
│   ├── test_engagement_service.py
│   └── test_schemas.py
├── requirements.txt              # App dependencies (googletrans==3.1.0a0)
├── pytest.ini
├── .env                          # Optional
└── README.md
```

---

## Prerequisites

- **Python** 3.10+
- **PostgreSQL** – conversations, messages, FAQs, articles, escalations
- **Redis** – optional; used for FAQ and MedGemma response caching
- **MedGemma** – optional; local path `app/models/medgemma-4b-it` or Hugging Face `google/medgemma-4b-it` (set `USE_MEDGEMMA=false` to disable)

---

## Installation

1. **Clone and enter the project**
   ```bash
   cd "/mnt/NewDisk/SHEKHAR/FERTILITY PREDICTION AI/HOME IVF"
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate   # Windows
   ```

3. **Install app dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   This installs FastAPI, Uvicorn, SQLAlchemy, psycopg2-binary, Redis, sentence-transformers, numpy, langdetect, **googletrans==3.1.0a0**, and (optionally) torch/transformers/accelerate for MedGemma.

4. **Optional: install test dependencies**
   ```bash
   pip install pytest pytest-asyncio httpx
   ```
   Then run tests with `pytest tests/ -v`.

5. **Database**  
   Create a PostgreSQL database and set `DATABASE_URL` (e.g. `postgresql://user:pass@localhost:5433/patient_bot`). On startup, `init_db()` creates tables from `app.database.models`.

6. **Optional: MedGemma**  
   Place MedGemma under `app/models/medgemma-4b-it/` or set `MEDGEMMA_MODEL_PATH`. Set `USE_LOCAL_MEDGEMMA=false` to load from Hugging Face.

7. **FAQ data / knowledge base**  
   A starter JSON knowledge base is provided at `knowledge_base/sample_faqs.json` with bilingual (en/hi) IVF FAQs. Edit or extend this file to customise answers. The `KnowledgeEngine` loads this file as the **primary** source for FAQ answers before falling back to the database or MedGemma.

---

## Configuration

Settings are defined in `app/config.py` and loaded from the environment or `.env`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `False` | Debug logging |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/patient_bot` | PostgreSQL URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL (cache) |
| `USE_MEDGEMMA` | `True` | Use MedGemma for chat fallback and engagement insights |
| `MEDGEMMA_MODEL_PATH` | `app/models/medgemma-4b-it` | Local MedGemma path |
| `USE_LOCAL_MEDGEMMA` | `True` | Use local model; if False, load from Hugging Face |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | FAQ embeddings |
| `SUPPORTED_LANGUAGES` | `["en", "hi"]` | Supported language codes |
| `HOMEIVF_WEBSITE_URL` | `https://homeivf.com/` | HomeIVF link |

---

## Running the Application

From the project root with the virtualenv activated:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **API docs (Swagger):** http://localhost:8000/docs  
- **ReDoc:** http://localhost:8000/redoc  
- **Root:** http://localhost:8000/

### Frontend (HTML / CSS / JS)

A static frontend is provided in the `frontend/` folder. It uses only HTML, CSS, and JavaScript (no build step).

1. **Serve the frontend** from a local server so the browser can call the API without CORS issues (the API allows origins such as `http://localhost:3000`, `http://localhost:5173`). For example:
   ```bash
   cd frontend && python3 -m http.server 3000
   ```
   Then open **http://localhost:3000** in your browser.

2. **Backend** must be running (e.g. `uvicorn app.main:app --port 8000`). The frontend calls **http://localhost:8000** by default.

3. **Custom API URL:** To point the frontend at a different API base URL, set it before loading the app, e.g. in `index.html` add:
   ```html
   <script>window.API_BASE = "https://your-api.example.com";</script>
   <script src="app.js"></script>
   ```

The frontend includes: **Home** (overview and links), **Chat** (send messages, optional image), and the five **engagement** tools (Fertility Readiness, Hormonal Predictor, Visual Health, Treatment Pathway, Home IVF Eligibility) with forms and result display.

---

## API Reference

### Root & Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service name, version, status, supported_languages, docs URL |
| GET | `/health/` | Health check (status, version, timestamp, supported_languages) |
| GET | `/health/ready` | Readiness (e.g. for Kubernetes) |
| GET | `/health/live` | Liveness |
| GET | `/favicon.ico` | 204 No Content |

### Chat (`/api/v1/chat`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/message` | Send message; get bot response, intent, escalation info |
| POST | `/conversation` | Create conversation or send first message |
| GET | `/conversation/{conversation_id}` | Conversation history |
| POST | `/escalate` | Manually escalate to counsellor |
| WebSocket | `/ws/{conversation_id}` | Real-time chat |

### Admin (`/api/v1/admin`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/analytics` | Bot analytics (conversations, messages, resolution/escalation rates, intents, language dist.) |
| POST | `/faq` | Create FAQ |
| PUT | `/faq/{faq_id}` | Update FAQ |
| POST | `/article` | Create article |
| POST | `/cleanup-gpu` | Trigger MedGemma GPU memory cleanup |

### Engagement (`/api/v1/engagement`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/fertility-readiness` | Fertility readiness calculator |
| POST | `/hormonal-predictor` | When to test AMH / semen / specialist |
| POST | `/visual-health` | Exploratory wellness (non-diagnostic) |
| POST | `/treatment-pathway` | Treatment pathway recommender |
| POST | `/home-ivf-eligibility` | Home IVF eligibility checker |

---

## Engagement Tools

All engagement endpoints accept JSON and return JSON. Each supports `language` (`en`/`hi`) and `use_ai_insight` (optional MedGemma snippet).

- **Fertility Readiness** – Inputs: age, medical_history, lifestyle_*, bmi, menstrual_pattern, cycle_length_days, previous_pregnancies, live_births, miscarriages, years_trying. Outputs: risk_score (0–100), risk_level (low/moderate/high), next_steps, guidance_text, ai_insight.
- **Hormonal Predictor** – Inputs: age, sex (female/male/couple), irregular_cycles, symptoms_*, years_trying, previous_tests_*. Outputs: suggest_amh, suggest_semen_analysis, suggest_specialist, when_to_test, reasoning, ai_insight.
- **Visual Health** – Inputs: optional image_base64, self_reported_sleep_hours, stress_level, bmi, skin_concerns. Outputs: disclaimer, wellness_indicators, recommendations, ai_insight (non-diagnostic).
- **Treatment Pathway** – Inputs: age, sex, years_trying, known_diagnosis, previous_treatments, preserving_fertility. Outputs: suggested_pathways, primary_recommendation, reasoning, ai_insight.
- **Home IVF Eligibility** – Inputs: female_age, male_age, medical_contraindications, has_consulted_specialist, ovarian_reserve_known, semen_analysis_known, stable_relationship_or_single_with_donor. Outputs: eligible, reasons, missing_criteria, prompt_consultation, booking_message, ai_insight.

---

## Chat & Knowledge Engine

- **Intent classification** and **semantic search** over FAQs (from `knowledge_base/sample_faqs.json` when present; see `knowledge_engine.py`).
- Low confidence or no good match → **MedGemma** (if `USE_MEDGEMMA=True`) for an IVF-focused answer.
- **IVFGuardrail** restricts queries and MedGemma output to IVF-related content.
- MedGemma is **lazy-loaded**; GPU memory is released after each use. Startup/shutdown and admin cleanup also release GPU.

### Multimodal (image + text)

MedGemma is a **vision–language model**. The app supports **image + text** input in two places:

1. **Chat** – `POST /api/v1/chat/message` accepts optional **`image_base64`**. When provided with a text message, MedGemma receives both the image and the question (e.g. for scans or photos in an IVF context). IVF guardrail and formatting still apply.
2. **Visual Health** – `POST /api/v1/engagement/visual-health` accepts optional **`image_base64`**. When `use_ai_insight=true` and an image is sent, MedGemma returns a short **non-diagnostic** wellness awareness snippet based on the image. Responses are for general awareness only.

Images are decoded with Pillow (PIL) and passed to the processor as PIL Images; the rest of the pipeline is unchanged. Multimodal responses are not cached.

---

## Testing

Tests live under `tests/` and use pytest. Engagement tests override the engagement service dependency so **no database or MedGemma** is required for the engagement API or service unit tests.

**Run all tests:**
```bash
pip install -r requirements.txt
pytest tests/ -v
```
(If you need test-only deps: `pytest`, `pytest-asyncio`, `httpx`; see `tests/conftest.py`.)

**Coverage:** Health (root, /health/, /health/ready, /health/live, favicon); all five engagement POSTs (status and response shape); engagement service unit tests; Pydantic schema validation. Chat and admin routes are not covered (they require a DB session).

---

## Development & Production

- **Linting:** Use ruff, flake8, or similar on `app/`.
- **Migrations:** Use Alembic (or similar) if you change `app/database/models.py`.
- **Production:** Set `DEBUG=False`, use a strong `DATABASE_URL` and `REDIS_URL`, consider multiple Uvicorn workers and a reverse proxy (e.g. Nginx). Ensure `JWT_SECRET_KEY` and other secrets are set via environment.

---

## License

MIT (see `license_info` in `app/main.py`).
