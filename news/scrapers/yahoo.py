import aiohttp
import aiohttp
import json
from datetime import datetime, timezone
from typing import List, Dict

URL = "https://finance.yahoo.com/news"


def parse(text: str) -> List[Dict]:
    data = json.loads(text)
    items: List[Dict] = []
    for item in data.get("items", []):
        ts = datetime.fromisoformat(item.get("publishedAt", "").replace("Z", "+00:00"))
        items.append({
            "timestamp": ts,
            "title": item.get("title"),
            "url": item.get("link"),
            "symbols": item.get("symbols", []),
        })
    return items


async def fetch() -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(URL, timeout=10) as resp:
            text = await resp.text()
    return parse(text)
