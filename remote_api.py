from fastapi import FastAPI, HTTPException
from subprocess import Popen
from typing import Dict

app = FastAPI(title="MT5 Bot Controller")

bots: Dict[str, Popen] = {}

@app.get("/bots")
async def list_bots():
    """Return running bot IDs and their status."""
    return {bid: proc.poll() is None for bid, proc in bots.items()}

@app.post("/bots/{bot_id}/start")
async def start_bot(bot_id: str):
    """Launch a realtime training instance for the given bot."""
    if bot_id in bots and bots[bot_id].poll() is None:
        raise HTTPException(status_code=400, detail="Bot already running")
    proc = Popen(["python", "realtime_train.py"])
    bots[bot_id] = proc
    return {"bot": bot_id, "status": "started"}

@app.post("/bots/{bot_id}/stop")
async def stop_bot(bot_id: str):
    """Terminate a running bot."""
    proc = bots.get(bot_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Bot not found")
    proc.terminate()
    bots.pop(bot_id)
    return {"bot": bot_id, "status": "stopped"}
