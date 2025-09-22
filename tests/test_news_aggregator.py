import sys
from pathlib import Path

import pytest

pytest.importorskip("bs4")
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news.aggregator import NewsAggregator

FAIRECONOMY_XML = """
<weeklyevents>
  <event id="1">
    <title>Non-Farm Employment Change</title>
    <currency>USD</currency>
    <date>2023-09-01</date>
    <time>08:30</time>
    <impact>High</impact>
    <forecast>180K</forecast>
    <actual>200K</actual>
  </event>
</weeklyevents>
"""

FOREXFACTORY_HTML = """
<table>
  <tr data-event-id="1" data-timestamp="1693557000" data-impact="High">
    <td class="calendar__currency">USD</td>
    <td class="calendar__event">Non-Farm Employment Change</td>
    <td class="calendar__actual">205K</td>
    <td class="calendar__forecast">190K</td>
  </tr>
</table>
"""

def test_parsers(tmp_path):
    agg = NewsAggregator(cache_dir=tmp_path)
    xml_events = agg.parse_faireconomy_xml(FAIRECONOMY_XML, source="xml")
    html_events = agg.parse_forexfactory_html(FOREXFACTORY_HTML, source="html")

    assert len(xml_events) == 1
    assert len(html_events) == 1

    expected_ts = datetime(2023, 9, 1, 8, 30, tzinfo=timezone.utc)
    assert xml_events[0]["timestamp"] == expected_ts
    assert html_events[0]["timestamp"] == expected_ts
    assert xml_events[0]["event"] == "Non-Farm Employment Change"
    assert html_events[0]["actual"] == "205K"


def test_dedup_and_cache(tmp_path):
    agg = NewsAggregator(cache_dir=tmp_path)
    xml_events = agg.parse_faireconomy_xml(FAIRECONOMY_XML, source="xml")
    html_events = agg.parse_forexfactory_html(FOREXFACTORY_HTML, source="html")
    combined = agg._dedupe(xml_events + html_events)

    assert len(combined) == 1
    ev = combined[0]
    assert set(ev["sources"]) == {"xml", "html"}
    assert ev["actual"] == "200K"  # first event wins

    # Save and reload via get_news
    agg._save_cache(combined)
    start = datetime(2023, 9, 1, 8, 0, tzinfo=timezone.utc)
    end = datetime(2023, 9, 1, 9, 0, tzinfo=timezone.utc)
    result = agg.get_news(start, end)
    assert len(result) == 1
    assert result[0]["event"] == "Non-Farm Employment Change"
