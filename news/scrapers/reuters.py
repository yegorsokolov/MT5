import aiohttp
import json
from datetime import datetime, timezone
from typing import List, Dict

URL = "https://www.reuters.com/markets/api/topNews"


def parse(text: str) -> List[Dict]:
    data = json.loads(text)
    items: List[Dict] = []
    for item in data.get("items", []):
        ts = datetime.fromisoformat(item.get("time", "").replace("Z", "+00:00"))
        items.append({
            "timestamp": ts,
            "title": item.get("headline"),
            "url": item.get("url"),
            "symbols": item.get("symbols", []),
        })
    return items


async def fetch() -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(URL, timeout=10) as resp:
            text = await resp.text()
    return parse(text)
