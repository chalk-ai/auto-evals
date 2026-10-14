"""HTTP server for the auto-evals demo: the support assistant, its traffic log, and a webhook receiver.

Public:
  GET  /                    the demo page (index.html; ?embed=1 drops its own title)
  GET  /api/state           live traffic, evaluation runs, received webhooks, prompt version
  POST /api/chat            {"message"} -> the assistant's reply (rate limited)
  POST /api/prompt          {"version": "v1"|"v2"} ships a system prompt; v2 reverts by itself

Called by the scheduled `juniper-auto-eval` Chalk function, with the internal token:
  POST /internal/traffic/claim   unscored turns for the next evaluation window
  POST /internal/runs            progress and results of an evaluation run

Webhook receiver (HMAC-SHA256 signed with the same token):
  POST /hooks/eval-alerts        threshold alerts fired by the evaluator
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import agent
import traffic

PORT = int(os.environ.get("PORT", "8080"))
HERE = Path(__file__).resolve().parent
API_SERVER = os.environ.get("CHALK_API_SERVER", "https://api.chalk.ai").rstrip("/")
MODEL = os.environ.get("AGENT_MODEL", "anthropic/claude-haiku-5-5")
INTERNAL_TOKEN = os.environ["AUTO_EVALS_TOKEN"]
SYNTHETIC_EVERY_S = float(os.environ.get("SYNTHETIC_EVERY_S", "30"))
# A shipped v2 prompt reverts after two evaluation windows, so the page never stays broken.
V2_TTL_S = 12 * 60
MAX_MESSAGE_CHARS = 400
CHATS_PER_HOUR = 20
GLOBAL_CHATS_PER_HOUR = 400
PROMPT_SWITCHES_PER_HOUR = 6
MAX_CLAIM = 40

log = logging.getLogger("auto-evals")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Limiter:
    """Sliding one-hour window of events per key."""

    def __init__(self, per_hour: int) -> None:
        super().__init__()
        self.per_hour = per_hour
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.events[key]
            while q and q[0] < now - 3600:
                q.popleft()
            if len(q) >= self.per_hour:
                return False
            q.append(now)
            return True


class State:
    """Everything the page shows. One replica, so plain memory under one lock is enough."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.turns: deque[dict[str, Any]] = deque(maxlen=400)
        self.runs: dict[str, dict[str, Any]] = {}
        self.webhooks: deque[dict[str, Any]] = deque(maxlen=40)
        self.prompt_version = "v1"
        self.revert_at: float | None = None
        self.prompt_log: deque[dict[str, Any]] = deque(maxlen=20)

    def current_version(self) -> str:
        with self.lock:
            if self.revert_at is not None and time.time() >= self.revert_at:
                self.prompt_version, self.revert_at = "v1", None
                self.prompt_log.append(
                    {"at": time.time(), "version": "v1", "by": "auto-revert"}
                )
            return self.prompt_version


STATE = State()
CHAT_LIMIT = Limiter(CHATS_PER_HOUR)
GLOBAL_LIMIT = Limiter(GLOBAL_CHATS_PER_HOUR)
PROMPT_LIMIT = Limiter(PROMPT_SWITCHES_PER_HOUR)


def model_config() -> agent.ModelConfig:
    key = os.environ["ROUTER_API_KEY"]
    # A router API key ("ck-...") goes in x-api-key; a Chalk access token is a bearer token.
    auth = (
        {"api_key": key}
        if key.startswith("ck-")
        else {"auth_token": key, "api_key": None}
    )
    client = anthropic.Anthropic(
        base_url=f"{API_SERVER}/v1/router",
        default_headers={"X-Chalk-Env-Id": os.environ["CHALK_ENVIRONMENT_ID"]},
        timeout=60.0,
        max_retries=2,
        **auth,
    )
    return agent.ModelConfig(client, MODEL)


CFG = model_config()


def take_turn(question: str, source: str) -> dict[str, Any]:
    version = STATE.current_version()
    started = time.time()
    answer = agent.reply(CFG, version, question)
    turn = {
        "id": uuid.uuid4().hex[:12],
        "at": time.time(),
        "source": source,
        "prompt_version": version,
        "question": question,
        "answer": answer,
        "latency_ms": int((time.time() - started) * 1000),
        "run_id": None,
    }
    with STATE.lock:
        STATE.turns.append(turn)
    return turn


def synthetic_loop() -> None:
    # Steady background traffic, so every evaluation window has something to score.
    while True:
        time.sleep(SYNTHETIC_EVERY_S * random.uniform(0.7, 1.3))
        try:
            take_turn(random.choice(traffic.QUESTIONS), "synthetic")
        except Exception:
            log.exception("synthetic turn failed")


def next_scheduled_run(now: float) -> float:
    """The evaluator runs on a */5 cron in UTC."""
    return (int(now) // 300 + 1) * 300


def require_token(request: Request) -> None:
    if not hmac.compare_digest(
        request.headers.get("x-internal-token", ""), INTERNAL_TOKEN
    ):
        raise HTTPException(status_code=401, detail="bad token")


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )


app = FastAPI()


@app.on_event("startup")
def start_background() -> None:
    threading.Thread(target=synthetic_loop, daemon=True).start()


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((HERE / "index.html").read_text())


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/state")
def state() -> dict[str, Any]:
    now = time.time()
    version = STATE.current_version()
    with STATE.lock:
        turns = list(STATE.turns)[-30:]
        runs = sorted(STATE.runs.values(), key=lambda r: r["started_at"], reverse=True)[
            :12
        ]
        hooks = list(STATE.webhooks)[::-1][:12]
        unscored = sum(1 for t in STATE.turns if t["run_id"] is None)
        prompt_log = list(STATE.prompt_log)[::-1]
        revert_at = STATE.revert_at
    return {
        "now": now,
        "company": agent.COMPANY,
        "assistant": agent.ASSISTANT,
        "model": MODEL,
        "prompt_version": version,
        "revert_at": revert_at,
        "next_run_at": next_scheduled_run(now),
        "unscored_turns": unscored,
        "turns": turns[::-1],
        "runs": runs,
        "webhooks": hooks,
        "prompt_log": prompt_log,
    }


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


@app.post("/api/chat")
def chat(body: ChatIn, request: Request) -> dict[str, Any]:
    if not CHAT_LIMIT.allow(client_key(request)) or not GLOBAL_LIMIT.allow("all"):
        raise HTTPException(
            status_code=429, detail="Too many messages for now. Try again later."
        )
    return take_turn(body.message.strip(), "visitor")


class PromptIn(BaseModel):
    version: str = Field(pattern="^v[12]$")


@app.post("/api/prompt")
def ship_prompt(body: PromptIn, request: Request) -> dict[str, Any]:
    if not PROMPT_LIMIT.allow(client_key(request)):
        raise HTTPException(
            status_code=429, detail="That's enough deploys for one hour."
        )
    with STATE.lock:
        STATE.prompt_version = body.version
        STATE.revert_at = time.time() + V2_TTL_S if body.version == "v2" else None
        STATE.prompt_log.append(
            {"at": time.time(), "version": body.version, "by": "visitor"}
        )
    return {"prompt_version": body.version, "revert_at": STATE.revert_at}


@app.post("/internal/traffic/claim")
def claim(request: Request) -> dict[str, Any]:
    require_token(request)
    run_id = request.query_params.get("run_id") or uuid.uuid4().hex
    min_rows = int(request.query_params.get("min_rows", "1"))
    with STATE.lock:
        batch = [t for t in STATE.turns if t["run_id"] is None][-MAX_CLAIM:]
        # A window too small to score stays unclaimed, so its turns roll into the next one.
        if len(batch) < min_rows:
            batch = []
        for t in batch:
            t["run_id"] = run_id
        # Alerts fire on transitions, so the evaluator needs the last finished window's state.
        finished = [r for r in STATE.runs.values() if "breached" in r]
        breached = (
            max(finished, key=lambda r: r["started_at"])["breached"] if finished else []
        )
    return {"run_id": run_id, "turns": batch, "breached": breached}


@app.post("/internal/runs")
async def record_run(request: Request) -> dict[str, bool]:
    require_token(request)
    run = await request.json()
    with STATE.lock:
        merged = {**STATE.runs.get(run["id"], {}), **run}
        STATE.runs[run["id"]] = merged
        # Keep the page's run list bounded.
        for old in sorted(STATE.runs.values(), key=lambda r: r["started_at"])[:-50]:
            STATE.runs.pop(old["id"], None)
    return {"ok": True}


@app.post("/hooks/eval-alerts")
async def receive_alert(request: Request) -> dict[str, bool]:
    raw = await request.body()
    expected = hmac.new(INTERNAL_TOKEN.encode(), raw, hashlib.sha256).hexdigest()
    signature = request.headers.get("x-signature-sha256", "")
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="bad signature")
    payload = json.loads(raw)
    with STATE.lock:
        STATE.webhooks.append(
            {
                "received_at": time.time(),
                "received_at_iso": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "event": request.headers.get("x-event", payload.get("event", "")),
                "signature_ok": True,
                "payload": payload,
            }
        )
    log.info("webhook %s %s", payload.get("event"), payload.get("metric"))
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
