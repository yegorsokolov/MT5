import asyncio
from pathlib import Path
import types
import sys

import httpx
import pytest

# Stub the top-level data package to import scrapers without heavy deps
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
stub = types.ModuleType("data")
stub.__path__ = [str(DATA_ROOT)]
sys.modules.setdefault("data", stub)

import data.scrapers as scrapers
from data.scrapers import yahoo_financials as yf

DATA_DIR = Path(__file__).parent / "data" / "scrapers"


def read(name: str) -> str:
    return (DATA_DIR / name).read_text()


def test_parse():
    raw = read("yahoo_financials.json")
    df = yf.parse("AAPL", raw)
    assert df.loc[0, "eps"] == 3.5
    assert df.loc[0, "revenue"] == 1000000
    assert df.loc[0, "market_cap"] == 500000000


def test_fetch_uses_cache(tmp_path):
    scrapers.CACHE_DIR = tmp_path
    sample = read("yahoo_financials.json")

    def handler(request):
        return httpx.Response(200, text=sample)

    async def run_first():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await yf.fetch("AAPL", client=client, force_refresh=True)

    df = asyncio.run(run_first())
    assert (tmp_path / "yahoo_AAPL.json").exists()
    assert df.loc[0, "ebitda"] == 200000

    def fail_handler(request):
        return httpx.Response(500)

    async def run_second():
        transport_fail = httpx.MockTransport(fail_handler)
        async with httpx.AsyncClient(transport=transport_fail) as client:
            return await yf.fetch("AAPL", client=client, force_refresh=False)

    df = asyncio.run(run_second())
    assert df.loc[0, "eps"] == 3.5
