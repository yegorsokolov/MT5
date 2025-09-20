from news import ai_enrichment as ai


def test_ai_helpers_provide_reasonable_defaults() -> None:
    text = "Stocks rally as the Fed signals a possible rate cut, boosting optimism."
    summary = ai.summarize_text(text, max_length=80)
    assert summary
    keywords = ai.extract_keywords(text)
    assert "stocks" in keywords
    topics = ai.classify_topics(text)
    assert "markets" in topics or "policy" in topics
    sentiment = ai.predict_sentiment(text)
    assert sentiment is not None and sentiment > 0
