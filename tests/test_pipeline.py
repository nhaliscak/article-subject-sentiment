from app.pipeline import analyze, _merge_cross_label_surname_matches

# Frozen verbatim from a real Benzinga article pulled live via Alpaca's News API
# (2026-09-27, symbols=META). Real headline-style capitalization reproduces both the
# entity-boundary noise and the cross-mention PERSON/ORG label split documented in
# README.md's findings section - not a hand-written worst case.
REAL_HEADLINE_ZUCKERBERG = (
    "Mark Zuckerberg Loses Nearly $9 Billion in One Day as Meta Stock Slides 4% "
    "After Goldman Questions AI Spending Payoff"
)
REAL_BODY_ZUCKERBERG = (
    "Meta stock fell 4%, wiping nearly $9 billion from Zuckerberg’s wealth after "
    "Goldman Sachs warned AI spending faces a massive revenue hurdle."
)

# Frozen verbatim real headlines (Alpaca News API, 2026-09-27) reproducing the same
# headline-boundary noise on different subjects/verbs.
REAL_HEADLINE_APPLE = (
    "Apple Hit With $5.7 Billion Patent Verdict Over iPhone, Apple Watch Haptics"
)
REAL_HEADLINE_SPACEX = (
    "Elon Musk's Path to Trillionaire Stalls Ahead of Key Tesla and SpaceX Events"
)

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


def test_ner_tags_a_real_company_name_as_org_not_norp():
    # Regression coverage: en_core_web_sm (the original model) mistagged "Rivian" as NORP
    # (nationality/political-group) instead of ORG - reproduced directly comparing sm vs md
    # on this exact sentence, 2026-09-28. Subject extraction is this project's whole job, so
    # this is a correctness bug on a real company name, not a hypothetical - en_core_web_md
    # (the current model, see app/pipeline.py's _nlp()) gets it right.
    results = analyze("Rivian shares surged 8% after the company reported record deliveries. "
                       "Rivian said production would ramp through the fourth quarter.")
    rivian = _find(results, "Rivian")
    assert rivian["label"] == "ORG"


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


def test_headline_boundary_noise_is_trimmed_goldman_and_meta():
    # Regression for the real spot-check finding: "Goldman Questions AI Spending
    # Payoff" captured as one ORG span, and "Meta Stock" swallowing a common noun -
    # both from the same real headline (see REAL_HEADLINE_ZUCKERBERG above).
    results = analyze(REAL_BODY_ZUCKERBERG, headline=REAL_HEADLINE_ZUCKERBERG)
    names = {r["subject"] for r in results}
    assert "Meta" in names
    assert "Meta Stock" not in names
    assert not any("Questions" in n or "Payoff" in n for n in names)


def test_headline_boundary_noise_is_trimmed_apple_hit():
    # Regression for the real spot-check finding: "Apple Hit" (a verb swallowed
    # into the entity span).
    results = analyze("The verdict followed a lengthy trial.", headline=REAL_HEADLINE_APPLE)
    names = {r["subject"] for r in results}
    assert "Apple" in names
    assert "Apple Hit" not in names


def test_headline_boundary_noise_is_trimmed_spacex_events():
    # Regression for the real spot-check finding: "SpaceX Events" (a following
    # common noun swallowed into the entity span).
    results = analyze("The mission proceeded on schedule.", headline=REAL_HEADLINE_SPACEX)
    names = {r["subject"] for r in results}
    assert "SpaceX" in names
    assert "SpaceX Events" not in names


def test_headline_style_detection_does_not_affect_normal_prose():
    # The boundary trim must stay scoped to headline-style text - normal-prose
    # entities like "Nexlon Corp" must keep their full canonical name untouched.
    results = analyze(MIXED_SENTIMENT_ARTICLE)
    nexlon = _find(results, "Nexlon Corp")
    assert nexlon["mentions"] == 2


def test_cross_mention_label_split_is_merged_zuckerberg():
    # Regression for the real spot-check finding: spaCy tags "Mark Zuckerberg" as
    # PERSON in the headline and the later standalone "Zuckerberg" as ORG in the
    # body - two mentions of the same person, split across labels.
    results = analyze(REAL_BODY_ZUCKERBERG, headline=REAL_HEADLINE_ZUCKERBERG)
    zuck = _find(results, "Mark Zuckerberg")
    assert zuck["label"] == "PERSON"
    assert zuck["mentions"] == 2


def test_cross_label_surname_merge_is_narrow():
    # Direct unit test of the merge guard itself (bypassing spaCy, whose real-world
    # tagging of ambiguous names like "Washington" varies): a single-token group
    # must only merge into a multi-token cross-label group when it matches the
    # LAST word, not any word - otherwise a person entity that happens to share the
    # FIRST word of an org's name (the "Washington" / "Washington Post" case the
    # existing label-must-match rule was built to avoid) would wrongly merge.
    person_washington = {
        "canonical": "Washington", "label": "PERSON", "norm": "washington",
        "aliases_norm": {"washington"}, "spans": ["span_a"],
    }
    org_washington_post = {
        "canonical": "Washington Post", "label": "ORG", "norm": "washington post",
        "aliases_norm": {"washington post"}, "spans": ["span_b"],
    }
    result = _merge_cross_label_surname_matches([person_washington, org_washington_post])
    assert len(result) == 2

    person_mark_zuckerberg = {
        "canonical": "Mark Zuckerberg", "label": "PERSON", "norm": "mark zuckerberg",
        "aliases_norm": {"mark zuckerberg"}, "spans": ["span_a"],
    }
    org_zuckerberg = {
        "canonical": "Zuckerberg", "label": "ORG", "norm": "zuckerberg",
        "aliases_norm": {"zuckerberg"}, "spans": ["span_b"],
    }
    result = _merge_cross_label_surname_matches([person_mark_zuckerberg, org_zuckerberg])
    assert len(result) == 1
    assert result[0]["canonical"] == "Mark Zuckerberg"
    assert result[0]["spans"] == ["span_a", "span_b"]


def test_two_subjects_in_one_sentence_are_scored_per_clause():
    results = analyze("Apple shares surged on record sales, but Nexlon Corp tumbled after missing earnings.")
    assert _find(results, "Apple")["sentiment"]["compound"] > 0.05
    assert _find(results, "Nexlon Corp")["sentiment"]["compound"] < -0.05


def test_pronoun_sentence_is_attributed_to_previous_subject():
    results = analyze("Nexlon Corp missed earnings expectations. It also warned of weak demand and layoffs.")
    ev = _find(results, "Nexlon Corp")["evidence"]
    assert [e["inferred"] for e in ev] == [False, True]
    assert "warned of weak demand" in _find(results, "Nexlon Corp")["summary"]


def test_mixed_flag_when_subject_gets_opposing_sentences():
    results = analyze("Apple shares surged on record sales. Later, Apple was sued and faced a terrible, "
                      "damaging verdict.")
    assert _find(results, "Apple")["sentiment"]["mixed"] is True


GEO_ARTICLE = (
    "Tesla shares fell after China imposed new tariffs on electric vehicles. "
    "Analysts at Goldman Sachs cut their price target, citing weak demand in China."
)


def test_detailed_mode_adds_geopolitical_entities():
    std = {r["subject"] for r in analyze(GEO_ARTICLE)}
    det = {r["subject"]: r for r in analyze(GEO_ARTICLE, mode="detailed")}
    assert "China" not in std
    assert det["China"]["label"] == "GPE"
    assert std <= set(det)


def test_financial_lite_is_compact_and_finance_only():
    res = analyze(MIXED_RELEVANCE_ARTICLE, mode="financial_lite")
    assert res and all(set(r) == {"subject", "label", "mentions", "sentiment"} for r in res)
    assert "Red Cross" not in {r["subject"] for r in res}
    assert all(r["sentiment"]["label"] in ("positive", "negative", "neutral") for r in res)


def test_unknown_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        analyze("Apple rose.", mode="bogus")
