# iSpeak Global — Production Setup Guide

## Files Changed and Why

| File | What Changed |
|---|---|
| `config.py` | All secrets read from env vars; `MAX_PIPELINE_QUEUE` added |
| `database/connection.py` | `get_connection()` replaced with `ThreadedConnectionPool` |
| `database/queries.py` | Uses `release_connection()` instead of `conn.close()`; adds rollback on error |
| `logger.py` | New file — centralized rotating file logger replacing `print()` |
| `report_writer.py` | New file — thread-safe background queue writer for `feedback_reports.jsonl` |
| `auth.py` | New file — constant-time token validation for WebSocket endpoint |
| `backend/main.py` | All 8 fixes integrated; all `print()` replaced with `logger.*` |

---

## Step 1 — Set Environment Variables

Create a `.env` file in the project root (never commit this to git):

```bash
# .env
POSTGRES_HOST=localhost
POSTGRES_DB=ispeak_global
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_real_password_here

DEFAULT_DEPARTMENT_ID=b6f8468a-477c-4045-a696-c402afae99a5

# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
SESSION_TOKEN=paste_your_generated_token_here

# Path to your PyNaCl public key file
SERVER_PUBLIC_KEY_PATH=/absolute/path/to/server_public.key

# Optional: set to DEBUG to see all VAD/STT/translation detail in logs
LOG_LEVEL=INFO
```

Add `.env` to `.gitignore`:
```
echo ".env" >> .gitignore
```

Load the `.env` file automatically by adding this to the very top of `main.py`
(before any other import):
```python
from dotenv import load_dotenv
load_dotenv()
```

Install python-dotenv:
```bash
pip install python-dotenv
```

---

## Step 2 — Generate the Session Token

```bash
python -c "import secrets; print(secrets.token_hex(32))"
# → e.g. a3f9c2b1d8e4f7a0b5c2d1e6f3a4b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3
```

Paste this into `SESSION_TOKEN` in your `.env` file.

---

## Step 3 — Update the Frontend

In `frontend/index.html`, change the WebSocket URL to include the token:

```javascript
// Before:
const WS_URL = `${protocol}//${host}/ws/translate`;

// After:
const SESSION_TOKEN = "paste_your_token_here";  // same value as .env
const WS_URL = `${protocol}//${host}/ws/translate?token=${SESSION_TOKEN}`;
```

For a real multi-user system you would fetch the token from a `/login` endpoint
instead of hardcoding it. But for a single-team deployment, a shared token is fine.

---

## Step 4 — Generate the Encryption Key Pair

```bash
python generate_keys.py
```

Create `generate_keys.py` in the project root:

```python
# generate_keys.py — run once, then delete
import nacl.public
import nacl.encoding

priv = nacl.public.PrivateKey.generate()
pub  = priv.public_key

with open("server_public.key",  "wb") as f:
    f.write(bytes(pub))

with open("server_private.key", "wb") as f:
    f.write(bytes(priv))

print("Keys written: server_public.key, server_private.key")
print("IMPORTANT: Move server_private.key OFF this machine immediately.")
print("The public key stays here. The private key goes on your secure review machine.")
```

**After running:**
- `server_public.key` stays on the server (path set in `SERVER_PUBLIC_KEY_PATH`)
- `server_private.key` must be moved to a separate, air-gapped machine
- Delete `generate_keys.py` after use

---

## Step 5 — Verify DB Schema

```sql
-- Run in psql to confirm column types match what the code writes:
\d utterances

-- Confirm total_latency_ms is INTEGER:
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'utterances';
```

---

## Step 6 — Start the Server

```bash
# Development (auto-reload):
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Production (4 workers, no reload):
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note:** Use `--workers 1` for now. The RouterService holds GPU models in memory.
Multiple workers would each load their own copy of Whisper + IndicTrans2,
exhausting VRAM immediately on the RTX 3050.

---

## Step 7 — Verify Everything Works

**Check the health endpoint:**
```
GET http://localhost:8000/health
```
Expected response:
```json
{
  "status": "ready",
  "models": {"whisper": "loaded", "translation": "loaded", "tts": "disabled"},
  "active_connections": 0,
  "gpu": "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
}
```

**Check logs are writing:**
```bash
tail -f logs/ispeak.log
```

**Test auth rejection:**
```bash
# This should be rejected with code 4003:
wscat -c "ws://localhost:8000/ws/translate"

# This should connect:
wscat -c "ws://localhost:8000/ws/translate?token=your_token_here"
```

---

## What Is NOT Changed (Intentionally)

- **`RouterService` / `STTService` / `TranslationService`** — pipeline logic is unchanged
- **`ConnectionManager`** — already production-quality
- **`ChunkingService` / `VADService`** — unchanged
- **`frontend/index.html`** — only the WS URL construction needs updating (Step 3)

---

## Production Readiness Checklist

- [ ] `.env` file created and not committed to git
- [ ] `SESSION_TOKEN` generated and set
- [ ] `server_public.key` generated; private key moved off server
- [ ] `SERVER_PUBLIC_KEY_PATH` set in `.env`
- [ ] `POSTGRES_PASSWORD` set to real password in `.env`
- [ ] `DEFAULT_DEPARTMENT_ID` UUID exists in `departments` table
- [ ] `total_latency_ms` column is `INTEGER` in `utterances` table
- [ ] `logs/` directory writable by the process user
- [ ] Frontend `WS_URL` updated with `?token=...`
- [ ] `python-dotenv` installed if using `.env` file
