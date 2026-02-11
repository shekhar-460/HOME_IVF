#!/usr/bin/env bash
# start.sh — Check environment and run AI Engagement Tools (Home IVF)

set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
APP_NAME="AI Engagement Tools (Home IVF)"

echo "=============================================="
echo "  $APP_NAME"
echo "=============================================="

# --- Python version check (3.10+) ---
need_python() {
  echo "ERROR: Python 3.10 or higher is required."
  echo "  Install from https://www.python.org/ or your package manager."
  exit 1
}

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found."
  need_python
fi

PY_VER=$(python3 -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")' 2>/dev/null || true)
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null)
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null)

if [[ -z "$PY_MAJOR" || -z "$PY_MINOR" ]]; then
  echo "ERROR: Could not detect Python version."
  need_python
fi

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
  echo "ERROR: Found Python $PY_VER; 3.10+ required."
  need_python
fi

echo "[OK] Python $PY_VER"

# --- Virtual environment ---
VENV_DIR="$PROJECT_ROOT/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv "$VENV_DIR"
fi

echo "[OK] Virtual environment: .venv"
source "$VENV_DIR/bin/activate"

# --- Dependencies ---
if [[ ! -f "$VENV_DIR/installed.flag" ]] || [[ requirements.txt -nt "$VENV_DIR/installed.flag" ]]; then
  echo "Installing/updating dependencies from requirements.txt ..."
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  touch "$VENV_DIR/installed.flag"
  echo "[OK] Dependencies installed."
else
  echo "[OK] Dependencies up to date."
fi

# --- Optional: load .env ---
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "[OK] Loading .env"
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

# --- Optional service checks (warn only) ---
check_port() {
  local host=$1
  local port=$2
  local name=$3
  if command -v nc &>/dev/null; then
    if nc -z "$host" "$port" 2>/dev/null; then
      echo "[OK] $name (reachable at $host:$port)"
      return 0
    fi
  fi
  echo "[--] $name not detected at $host:$port (optional; app may still run)"
  return 1
}

check_port "localhost" "5433" "PostgreSQL"
check_port "localhost" "6379" "Redis" || true

# --- Port from env or default ---
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "=============================================="
echo "  Starting server at http://${HOST}:${PORT}"
echo "  API docs: http://localhost:${PORT}/docs"
echo "  Press Ctrl+C to stop"
echo "=============================================="

exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
