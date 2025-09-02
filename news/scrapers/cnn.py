import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List, Dict

URL = "https://www.cnn.com/business"


def parse(text: str) -> List[Dict]:
    soup = BeautifulSoup(text, "html.parser")
    items: List[Dict] = []
    for art in soup.select("article"):
        link = art.find("a")
        if not link:
            continue
        ts_str = art.get("data-time") or (art.find("time").get("datetime") if art.find("time") else None)
        if not ts_str:
            continue
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        symbols = [s.strip() for s in (art.get("data-symbols") or art.get("data-ticker") or "").split(",") if s.strip()]
        items.append({
            "timestamp": ts,
            "title": link.get_text(strip=True),
            "url": link.get("href"),
            "symbols": symbols,
        })
    return items


async def fetch() -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(URL, timeout=10) as resp:
            text = await resp.text()
    return parse(text)
