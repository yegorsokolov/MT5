from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from subprocess import Popen
from typing import Dict, Any
from pydantic import BaseModel
import os
from utils import update_config
from log_utils import LOG_FILE


class ConfigUpdate(BaseModel):
    key: str
    value: Any
    reason: str


app = FastAPI(title="MT5 Bot Controller")

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def authorize(key: str = Security(api_key_header)) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

bots: Dict[str, Popen] = {}

@app.get("/bots")
async def list_bots(_: None = Depends(authorize)):
    """Return running bot IDs and their status."""
    return {bid: proc.poll() is None for bid, proc in bots.items()}

@app.post("/bots/{bot_id}/start")
async def start_bot(bot_id: str, _: None = Depends(authorize)):
    """Launch a realtime training instance for the given bot."""
    if bot_id in bots and bots[bot_id].poll() is None:
        raise HTTPException(status_code=400, detail="Bot already running")
    proc = Popen(["python", "realtime_train.py"])
    bots[bot_id] = proc
    return {"bot": bot_id, "status": "started"}

@app.post("/bots/{bot_id}/stop")
async def stop_bot(bot_id: str, _: None = Depends(authorize)):
    """Terminate a running bot."""
    proc = bots.get(bot_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Bot not found")
    proc.terminate()
    bots.pop(bot_id)
    return {"bot": bot_id, "status": "stopped"}


@app.get("/logs")
async def get_logs(lines: int = 50, _: None = Depends(authorize)):
    """Return the last N lines from the bot log file."""
    if not LOG_FILE.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    with open(LOG_FILE, "r") as f:
        data = f.readlines()[-lines:]
    return {"logs": "".join(data)}


@app.post("/config")
async def update_configuration(change: ConfigUpdate, _: None = Depends(authorize)):
    """Update a configuration variable."""
    update_config(change.key, change.value, change.reason)
    return {"status": "updated", change.key: change.value}
