import os
from pathlib import Path
import httpx

# Base directory for caching raw responses
CACHE_DIR = Path(os.environ.get("SCRAPER_CACHE_DIR", Path(__file__).resolve().parents[1] / "external_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

async def fetch_with_cache(url: str, cache_name: str, client: httpx.AsyncClient | None = None, force_refresh: bool = False) -> str:
    """Fetch a URL asynchronously and cache the raw response.

    Parameters
    ----------
    url: str
        Target URL to fetch.
    cache_name: str
        File name for caching the raw response.
    client: httpx.AsyncClient | None
        Optional existing HTTP client. Created if not provided.
    force_refresh: bool
        If True, ignore any cached file and refetch.
    Returns
    -------
    str
        The response text.
    """
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not force_refresh:
        return cache_path.read_text()

    created = False
    if client is None:
        client = httpx.AsyncClient()
        created = True
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        text = resp.text
        cache_path.write_text(text)
        return text
    finally:
        if created:
            await client.aclose()
