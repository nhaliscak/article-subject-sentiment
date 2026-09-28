from app.pipeline import analyze

MIXED_SENTIMENT_ARTICLE = """
Apple shares surged after the company posted record iPhone revenue and raised
its full-year guidance, with analysts praising strong demand and raising
their price targets. Meanwhile, rival Nexlon Corp tumbled after missing
earnings expectations and warning of weak demand, sending its stock to a
52-week low. Nexlon's management blamed persistent supply chain issues for
the disappointing quarter.
"""

MIXED_RELEVANCE_ARTICLE = """
Apple unveiled a new charity partnership with the Red Cross to support
disaster relief efforts in flood-affected regions. Separately, Apple's stock
rose three percent after analysts raised their price target following
strong quarterly earnings and upbeat guidance. The Red Cross said it plans
to expand its regional volunteer program next year.
"""


def _find(results, name):
    for r in results:
        if r["subject"] == name:
            return r
    raise AssertionError(f"{name!r} not found among subjects: {[r['subject'] for r in results]}")


def test_extracts_known_subjects():
    results = analyze(MIXED_SENTIMENT_ARTICLE)
    names = {r["subject"] for r in results}
    assert "Apple" in names
    assert any("Nexlon" in n for n in names)


def test_sentiment_distinguishes_positive_and_negative_subjects():
    results = analyze(MIXED_SENTIMENT_ARTICLE)
    apple = _find(results, "Apple")
    nexlon = _find(results, "Nexlon Corp")

    assert apple["sentiment"]["compound"] > 0.05
    assert apple["sentiment"]["label"] == "positive"

    assert nexlon["sentiment"]["compound"] < -0.05
    assert nexlon["sentiment"]["label"] == "negative"


def test_summary_is_extractive_not_generated():
    results = analyze(MIXED_SENTIMENT_ARTICLE)
    apple = _find(results, "Apple")
    # The summary must be verbatim material lifted from the source text,
    # not a paraphrase - i.e. it should appear as a substring of the article.
    normalized_article = " ".join(MIXED_SENTIMENT_ARTICLE.split())
    normalized_summary = " ".join(apple["summary"].split())
    assert normalized_summary in normalized_article
    assert "Apple" in apple["summary"]


def test_mentions_are_counted():
    results = analyze(MIXED_SENTIMENT_ARTICLE)
    nexlon = _find(results, "Nexlon Corp")
    # "Nexlon Corp" and "Nexlon's" should merge into one subject with 2 mentions.
    assert nexlon["mentions"] == 2


def test_financial_mode_filters_out_irrelevant_subjects():
    all_results = analyze(MIXED_RELEVANCE_ARTICLE, financial_mode=False)
    names = {r["subject"] for r in all_results}
    assert "Apple" in names
    assert "the Red Cross" in names

    apple = _find(all_results, "Apple")
    red_cross = _find(all_results, "the Red Cross")
    assert apple["financial_relevant"] is True
    assert red_cross["financial_relevant"] is False

    filtered = analyze(MIXED_RELEVANCE_ARTICLE, financial_mode=True)
    filtered_names = {r["subject"] for r in filtered}
    assert "Apple" in filtered_names
    assert "the Red Cross" not in filtered_names


def test_ticker_symbol_is_recognized_as_financial():
    from app.pipeline import _is_financially_relevant

    assert _is_financially_relevant("Widgetco", "Widgetco ($WDGT) rallied today.")
    assert _is_financially_relevant("Widgetco", "Widgetco (NASDAQ: WDGT) rallied today.")


def test_no_subjects_in_empty_text():
    assert analyze("") == []
    assert analyze("   ") == []


def test_headline_is_included_in_analysis_context():
    results = analyze(
        "The company reported strong results and beat expectations on every metric.",
        headline="Apple posts blowout earnings",
    )
    names = {r["subject"] for r in results}
    assert "Apple" in names
