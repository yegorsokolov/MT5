import asyncio
from pathlib import Path
import types
import sys

import httpx
import pytest

# The top-level `data` package has heavy import side effects. To avoid pulling
# in unnecessary dependencies during tests, we create a light-weight package
# stub pointing to the repository's data directory before importing the
# scrapers modules.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
stub = types.ModuleType("data")
stub.__path__ = [str(DATA_ROOT)]
sys.modules.setdefault("data", stub)

import data.scrapers as scrapers
from data.scrapers import tradingeconomics_calendar as te
from data.scrapers import bloomberg_bdi as bloomberg
from data.scrapers import cnbc_badi as cnbc
from data.scrapers import census_retail as census

DATA_DIR = Path(__file__).parent / "data" / "scrapers"


def read(name: str) -> str:
    return (DATA_DIR / name).read_text()


def test_tradingeconomics_parse():
    raw = read("tradingeconomics_calendar.json")
    items = te.parse(raw)
    assert items[0]["symbol"] == "GDP"
    assert items[0]["value"] == 2.5
    assert items[0]["source"] == "TradingEconomics"


def test_bloomberg_parse():
    raw = read("bloomberg_bdi.html")
    items = bloomberg.parse(raw)
    assert items[0]["symbol"] == "BDIY:IND"
    assert items[0]["value"] == 1234.56


def test_cnbc_parse():
    raw = read("cnbc_badi.html")
    items = cnbc.parse(raw)
    assert items[0]["symbol"] == ".BADI"
    assert items[0]["value"] == 1150.0


def test_census_parse():
    raw = read("census_retail.csv")
    items = census.parse(raw)
    assert items[0]["symbol"] == "US_RETAIL_SALES"
    assert items[0]["value"] == 680.3


def test_fetch_uses_cache(tmp_path):
    """Ensure fetch uses cached response when available."""
    scrapers.CACHE_DIR = tmp_path
    sample = read("bloomberg_bdi.html")

    def handler(request):
        return httpx.Response(200, text=sample)

    async def run_first():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await bloomberg.fetch(client=client, force_refresh=True)

    data = asyncio.run(run_first())
    assert (tmp_path / "bloomberg_bdi.html").exists()
    assert data[0]["value"] == 1234.56

    # Second call with failing transport should still use cache
    def fail_handler(request):
        return httpx.Response(500)

    async def run_second():
        transport_fail = httpx.MockTransport(fail_handler)
        async with httpx.AsyncClient(transport=transport_fail) as client:
            return await bloomberg.fetch(client=client, force_refresh=False)

    data = asyncio.run(run_second())
    assert data[0]["value"] == 1234.56
