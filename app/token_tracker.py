# app/token_tracker.py
import os, json
from datetime import date
from pathlib import Path

TRACK_FILE = Path(".token_usage.json")
DAILY_LIMIT = int(os.getenv("DAILY_TOKEN_LIMIT", "500000"))  # default 500k

def _load_state():
    if TRACK_FILE.exists():
        with open(TRACK_FILE, "r") as f:
            return json.load(f)
    return {"day": str(date.today()), "used": 0}

def _save_state(state):
    with open(TRACK_FILE, "w") as f:
        json.dump(state, f)

def add_tokens(count: int):
    state = _load_state()
    today = str(date.today())
    if state["day"] != today:
        state = {"day": today, "used": 0}
    state["used"] += count
    _save_state(state)
    return DAILY_LIMIT - state["used"], state["used"], DAILY_LIMIT
