import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List, Dict

URL = "https://www.forexfactory.com/news"


def parse(text: str) -> List[Dict]:
    soup = BeautifulSoup(text, "html.parser")
    items: List[Dict] = []
    for div in soup.select("div.news-article"):
        ts_tag = div.find(class_="timestamp")
        link = div.find("a", class_="title")
        if not ts_tag or not link:
            continue
        ts = datetime.fromisoformat(ts_tag.get_text(strip=True).replace("Z", "+00:00"))
        symbols = [s.strip() for s in (div.get("data-symbols") or "").split(",") if s.strip()]
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
