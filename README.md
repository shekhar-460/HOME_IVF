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
- [Page Translation & Professional Help](#page-translation--professional-help)
- [Testing](#testing)
- [Development & Production](#development--production)
- [License](#license)

---

## Overview

The system has two main pillars:

1. **Conversational patient education (chat)**  
   English and Hindi chat with intent classification, semantic search over FAQs (from `knowledge_base/sample_faqs.json`), and optional **MedGemma-4b-it** fallback for IVF-related questions. **Language is auto-detected** (including **Hinglish**—Roman-script Hindi, e.g. *ivf kya hai?*, *ivf cost kitna hai*), and the bot **replies in the same language**. The **response language** (disclaimer, follow-up and suggestion buttons, “Get professional help” link) is aligned with the answer content so the full reply is monolingual when the answer is in Hindi. All answers are constrained by an **IVF guardrail**. Translation (e.g. MedGemma output to Hindi) uses **googletrans**.

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
| **Languages** | English (`en`), Hindi (`hi`), and Hinglish (Roman-script Hindi); auto-detection; reply, disclaimer, and suggestion labels follow answer language |
| **Chat** | Create conversation, send message, get history, WebSocket; intent classification and escalation |
| **Knowledge** | Semantic search over JSON FAQs; optional MedGemma-4b-it fallback; IVF-only guardrail (incl. Hinglish “ivf kya hai?”) |
| **Translation** | googletrans (3.1.0a0): chat en ↔ hi; **page translation** (header dropdown) for 14+ languages |
| **Professional help** | [HomeIVF](https://homeivf.com/) link and phone in chat suggestions, escalation message, and footer |
| **Engagement** | Five POST endpoints under `/api/v1/engagement/` (see [Engagement Tools](#engagement-tools)); age (female 21–50, male 21–55), BMI 15–50, custom diagnosis/treatment validated and explained in result |
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
 ┌────────────────┐           ┌─────────────────┐           ┌────────────────────┐
 │  /health/*     │           │ /api/v1/chat    │           │ /api/v1/engagement │
 │  health.py     │           │ chat.py         │           │ engagement.py      │
 └────────────────┘           └─────────────────┘           └────────────────────┘
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
  ┌──────────────────────┐     ┌──────────────────┐   ┌─────────────────────┐
  │PostgreSQL            │     │ Redis cache      │   │ EscalationManager   │
  │conversations, msgs,  │◄──► │ FAQ / MedGemma   │   │ + outbound systems  │
  │faqs, articles,       │     │ responses        │   │ (future integration)│
  │escalations           │     └──────────────────┘   └─────────────────────┘
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
│   ├── main.py                    # FastAPI app, CORS (allow *), ensure_cors_headers, routers, startup/shutdown, GPU cleanup
│   ├── config.py                  # Pydantic Settings (env / .env)
│   ├── api/routes/
│   │   ├── health.py              # GET /health/, /health/ready, /health/live
│   │   ├── chat.py                # POST /api/v1/chat/message, /conversation; GET /conversation/{id}; POST /escalate; WS /ws/{id}
│   │   ├── engagement.py          # POST /api/v1/engagement/* (5 tools)
│   │   ├── translate.py           # GET /api/v1/translate/languages; POST /api/v1/translate (page translation)
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
│       ├── language_detector.py   # langdetect + Hinglish (Roman-script Hindi) detection
│       └── translator.py         # googletrans (en/hi + page languages)
├── frontend/                     # Static UI: index.html, app.js, styles.css (chat, tools, translate dropdown)
├── knowledge_base/               # Optional: sample_faqs.json (FAQ source if present)
├── tests/
│   ├── README.md                 # Test overview and how to run
│   ├── conftest.py               # Fixtures, engagement dependency override
│   ├── test_health.py
│   ├── test_engagement_api.py
│   ├── test_engagement_service.py
│   └── test_schemas.py
├── requirements.txt              # App dependencies (googletrans==3.1.0a0)
├── start.sh                      # Check env and run backend (0.0.0.0:8000) + frontend (0.0.0.0:3000); Ctrl+C stops both
├── pytest.ini
├── QUICKSTART.md                 # Minimal steps to run the app
├── README.md                     # Full documentation
├── .env                          # Optional (HOST, PORT, DATABASE_URL, etc.)
└── .gitignore                    # Python, venv, .env, tests, IDE, OS, logs, model cache
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
   git clone <repository-url>
   cd HOME_IVF
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
| `SUPPORTED_LANGUAGES` | `["en", "hi"]` | Chat language codes |
| `HOMEIVF_WEBSITE_URL` | `https://homeivf.com/` | HomeIVF professional help link |
| `HOMEIVF_PHONE` | `+91-9958885250` | HomeIVF contact number (escalation message, etc.) |

---

## Running the Application

**Option 1 – One-command script (recommended)**  
From the project root, run `start.sh`. It checks Python 3.10+, creates/uses `.venv`, installs or updates dependencies from `requirements.txt`, optionally loads `.env`, checks PostgreSQL/Redis (optional), starts the **backend** (Uvicorn on `0.0.0.0:8000`), then starts the **frontend** (Python http.server on `0.0.0.0:3000`). You can run it from any directory; it changes into the project root automatically. Press Ctrl+C to stop both.

```bash
./start.sh
```

- **Backend:** http://localhost:8000 (and http://*your-ip*:8000 from other devices)  
- **API docs (Swagger):** http://localhost:8000/docs  
- **ReDoc:** http://localhost:8000/redoc  
- **Frontend:** http://localhost:3000 (and http://*your-ip*:3000 from other devices)

**Option 2 – Manual**  
From the project root with the virtualenv activated:

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal, serve the frontend: `cd frontend && python3 -m http.server 3000 --bind 0.0.0.0`

### Frontend (HTML / CSS / JS)

A static frontend is provided in the `frontend/` folder. It uses only HTML, CSS, and JavaScript (no build step).

1. **API URL:** The frontend uses **http://*current-host*:8000** by default (e.g. when you open http://localhost:3000 it calls http://localhost:8000; when you open http://192.168.15.18:3000 from another device it calls http://192.168.15.18:8000). No CORS issues when backend and frontend are on the same host. Override with `window.API_BASE` or `window.API_PORT` in `index.html` if the API runs on a different port.

2. **Custom API URL:** In `index.html`, before `<script src="app.js">`, add for example:
   ```html
   <script>window.API_BASE = "http://your-server:8000";</script>
   <!-- or only change port: -->
   <script>window.API_PORT = "8000";</script>
   <script src="app.js"></script>
   ```

3. **CORS:** The backend allows any origin (`Access-Control-Allow-Origin: *`) with `allow_credentials=False`, so the frontend works when opened from any device or port (e.g. 3000, 4200).

The frontend includes: **Home** (overview and links), **Chat** (send messages, optional image, suggested actions with “Get professional help” → HomeIVF), a **Translate** dropdown (default **English**; Indian languages listed first; 14+ languages via googletrans; loader while translating; cache for instant re-select), and the five **engagement** tools with forms and result display. The footer includes a link to **HomeIVF** (https://homeivf.com/) and the contact number. Use **http://** (not https) when opening from the network; the server is HTTP-only.

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

### Translate (`/api/v1`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/translate/languages` | List language codes and names for page translation |
| POST | `/translate` | Batch translate strings (body: `texts`, `dest`, optional `src`) for UI translation |

---

## Engagement Tools

All engagement endpoints accept JSON and return JSON. Each supports `language` (`en`/`hi`) and `use_ai_insight` (optional MedGemma snippet).

### Age and BMI rules

- **Age:** Female 21–50 (Fertility Readiness; Hormonal Predictor and Treatment Pathway when sex is female). Male 21–55 (Hormonal Predictor, Treatment Pathway). Home IVF: female 21–50, male 21–55 (optional).
- **BMI:** When weight and height are provided, calculated BMI must be in range **15–50** for Fertility Readiness, Visual Health, Treatment Pathway, and Home IVF. Frontend validates before submit and shows an inline hint; API returns a clear error if out of range.

### Tool summaries

- **Fertility Readiness** – Inputs: age (female 21–50), medical_history, lifestyle_*, bmi (optional, 15–50), menstrual_pattern, cycle_length_days, previous_pregnancies, live_births, miscarriages, years_trying. Outputs: risk_score (0–100), risk_level (low/moderate/high), next_steps, guidance_text, ai_insight.
- **Hormonal Predictor** – Inputs: age (female 21–50, male 21–55), sex (female/male), irregular_cycles, symptoms_*, years_trying, previous_tests_*. Outputs: suggest_amh, suggest_semen_analysis, suggest_specialist, when_to_test, reasoning, ai_insight. When there are no symptoms and AI insight is off, a default suggestions message is still returned.
- **Visual Health** – Inputs: optional image_base64, self_reported_sleep_hours, stress_level, self_reported_bmi (15–50 when provided), skin_concerns. Outputs: disclaimer, wellness_indicators, recommendations, ai_insight (non-diagnostic).
- **Treatment Pathway** – Inputs: age (female 21–50, male 21–55), sex, years_trying, known_diagnosis, previous_treatments, preserving_fertility, optional weight_kg/height_cm (BMI 15–50). Diagnosis and previous treatments accept dropdown options or custom “Other” text. **Only listed (allowlist) options are used for the pathway logic.** Custom entries that look nonsensical (e.g. random alphanumeric) trigger a note in **Reasoning** and **AI Insight** asking the user to use the dropdown for accurate guidance. Outputs: suggested_pathways, primary_recommendation, reasoning, ai_insight.
- **Home IVF Eligibility** – Inputs: female_age (21–50), male_age (21–55, optional), known_diagnosis, previous_treatments (same allowlist/custom behaviour as Treatment Pathway), medical_contraindications, has_consulted_specialist, ovarian_reserve_known, semen_analysis_known, stable_relationship_or_single_with_donor, optional weight_kg/height_cm (BMI 15–50). Outputs: eligible, reasons, missing_criteria, prompt_consultation, booking_message, ai_insight.

### Custom diagnosis / treatment (Treatment Pathway & Home IVF)

- Users can choose from the dropdown or add “Other (not listed)” with free text.
- **Pathway/eligibility logic** uses only values that match the built-in list (e.g. tubal factor, PCOS, IUI, IVF). Custom “Other” text is not used to compute the result.
- **Semantic check:** If custom text looks nonsensical (e.g. high digit ratio, tokens with no vowel), the backend adds an explanation in the **final result** (Reasoning or Reasons) and in **AI Insight**, and the frontend can block obviously invalid “Other” entries when adding. No 422 rejection; the result is still returned with the explanatory message.

---

## Chat & Knowledge Engine

- **Language:** Chat supports English, Hindi (Devanagari), and **Hinglish** (Roman-script Hindi, e.g. *ivf kya hai?*, *batao*, *ivf cost kitna hai*). Language is auto-detected; the bot replies in the same language (templates, FAQ content, and MedGemma output translated to Hindi when needed). **Response language alignment:** When the answer is in Hindi (e.g. Devanagari content), the AI disclaimer, follow-up question labels, proactive suggestion buttons, and “Get professional help” link are also shown in Hindi so the full response is monolingual.
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

## Page Translation & Professional Help

### Page translation (googletrans)

The frontend header includes a **Translate** dropdown (default **English**). Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam) are listed first; then Spanish, French, German, Arabic, Chinese. Selecting a language sends the visible static text to the backend; the app uses **googletrans (3.1.0a0)** and returns translated strings. The UI shows a loader while translating and caches results for instant re-select. Choosing **English** restores the original copy. API: **GET /api/v1/translate/languages** (list) and **POST /api/v1/translate** (body: `{ "texts": ["..."], "dest": "hi", "src": "en" }`).

### Professional help (HomeIVF)

The [HomeIVF](https://homeivf.com/) site and contact number are wired in for professional fertility care:

- **Chat:** Suggested actions include “Get professional help (HomeIVF)” (or “पेशेवर सहायता – HomeIVF” when the answer is in Hindi) linking to `HOMEIVF_WEBSITE_URL`. Follow-up and topic suggestion labels follow the answer language.
- **Escalation:** When a conversation is escalated to a counsellor, the message includes the HomeIVF URL and `HOMEIVF_PHONE` (default +91-9958885250).
- **Frontend:** Footer shows “For professional help: HomeIVF · +91-9958885250” with links. Chat response links are rendered as buttons when the API returns `suggested_actions` with type `link`.

Configure via `HOMEIVF_WEBSITE_URL` and `HOMEIVF_PHONE` in `.env` or environment.

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
