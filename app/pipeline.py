"""Subject-level sentiment analysis for a single article.

Pipeline: spaCy NER finds candidate subjects (companies, people, products) ->
group raw entity mentions into subjects by simple alias overlap -> for each
subject, pull the sentences that actually mention it (via spaCy's own
sentence attribution of each entity span, not string matching) -> rank those
sentences with a hand-rolled TF-IDF score and keep the top few as an
extractive summary -> score sentiment on that subject-specific text with
VADER -> flag financial relevance via a ticker regex + keyword list.

No LLM, no training, no generative summarization anywhere in here.
"""

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Optional

import spacy
from spacy.tokens import Span
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Entity types we treat as "subjects" per the brief: companies, people, products,
# organizations. GPE/EVENT/etc. are deliberately excluded to keep results to
# genuinely discussable subjects rather than every named thing in the text.
SUBJECT_LABELS = {"ORG", "PERSON", "PRODUCT"}

FINANCIAL_KEYWORDS = {
    "earnings", "revenue", "guidance", "shares", "share", "stock", "stocks",
    "analyst", "analysts", "upgrade", "downgrade", "quarter", "quarterly",
    "ipo", "market cap", "dividend", "eps", "profit", "profits", "loss",
    "losses", "sec filing", "nasdaq", "nyse", "valuation", "buyback",
    "price target", "outlook", "forecast", "investors", "trading",
    "shareholders", "acquisition", "merger", "ceo", "cfo", "bankruptcy",
}

# $TICK style, or "(NASDAQ: TICK)" / "(NYSE: TICK)" style.
TICKER_RE = re.compile(
    r"\$[A-Z]{1,5}\b|\((?:NASDAQ|NYSE|AMEX)\s*:\s*[A-Z.]{1,6}\)",
    re.IGNORECASE,
)

_MIN_ALIAS_LEN = 3  # ponytail: guards against short-name false substring merges (e.g. "Bo"/"Bob")
_SUMMARY_MAX_SENTENCES = 3

# Headline title-case capitalizes every real word, not just proper nouns, which is what
# breaks spaCy's boundary signal (see _trim_headline_entity). Detection looks only at
# words this long so short function words ("of", "as", "AI") that are legitimately
# lowercase-or-not either way don't skew the ratio.
_HEADLINE_MIN_WORDS = 3
_HEADLINE_MIN_WORD_LEN = 4
_HEADLINE_CAP_RATIO = 0.7

_vader = SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _nlp():
    # en_core_web_md, not _sm: verified 2026-09-28 that _sm mistags real company names as
    # NORP (nationality/political-group) instead of ORG - e.g. "Rivian shares surged 8%..."
    # tags "Rivian" as NORP under _sm, ORG under _md, reproduced directly with both models
    # loaded side by side. Subject extraction is this project's whole job, so that's a
    # correctness bug for exactly the kind of company name a real financial article uses,
    # not a hypothetical edge case - worth _md's larger download for the accuracy.
    return spacy.load("en_core_web_md")


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def _merge_entities(ents):
    """Group raw entity mentions into subjects.

    ponytail: heuristic alias merging (substring overlap on normalized text,
    same label), not full coreference resolution. Handles the common case
    ("Apple" / "Apple Inc.", "Elon Musk" / "Musk") but will under- or
    over-merge on trickier aliasing. Upgrade path: spacy coref/experimental
    if this proves to be a real problem on real articles.
    """
    groups = []
    for ent in sorted(ents, key=lambda e: -len(e.text)):
        norm = _normalize(ent.text)
        if not norm:
            continue
        match = None
        for g in groups:
            if g["label"] != ent.label_:
                continue
            if norm == g["norm"] or (
                len(norm) >= _MIN_ALIAS_LEN
                and any(
                    len(a) >= _MIN_ALIAS_LEN and (a in norm or norm in a)
                    for a in g["aliases_norm"]
                )
            ):
                match = g
                break
        if match:
            match["aliases_norm"].add(norm)
            match["spans"].append(ent)
        else:
            groups.append(
                {
                    "canonical": ent.text,
                    "label": ent.label_,
                    "norm": norm,
                    "aliases_norm": {norm},
                    "spans": [ent],
                }
            )
    return _merge_cross_label_surname_matches(groups)


def _merge_cross_label_surname_matches(groups):
    """Reconcile spaCy's own per-mention label inconsistency for one person
    (e.g. "Mark Zuckerberg" tagged PERSON, a later standalone "Zuckerberg"
    tagged ORG - reproduced directly on a real article). This isn't a
    merge-logic bug (the label-must-match rule above is doing its job); it's
    spaCy tagging the same surname differently mention to mention.

    Deliberately narrow: only merges a single-token group into a multi-token
    group when the token equals the multi-token group's LAST word, and one
    side is PERSON. Matching on the trailing word only (not any substring)
    is what keeps a person named "Washington" from merging into the org
    "Washington Post" (last word "post" != "washington") while still
    catching the surname case (last word "zuckerberg" == "zuckerberg").
    """
    consumed = set()
    n = len(groups)
    for i in range(n):
        for j in range(i + 1, n):
            g, other = groups[i], groups[j]
            if id(g) in consumed or id(other) in consumed:
                continue
            if g["label"] == other["label"] or "PERSON" not in (g["label"], other["label"]):
                continue
            long_g, short_g = (
                (g, other) if len(g["norm"].split()) >= len(other["norm"].split()) else (other, g)
            )
            short_tokens = short_g["norm"].split()
            long_tokens = long_g["norm"].split()
            if (
                len(short_tokens) == 1
                and len(long_tokens) >= 2
                and len(short_tokens[0]) >= _MIN_ALIAS_LEN
                and short_tokens[0] == long_tokens[-1]
            ):
                long_g["aliases_norm"] |= short_g["aliases_norm"]
                long_g["spans"].extend(short_g["spans"])
                consumed.add(id(short_g))
    return [g for g in groups if id(g) not in consumed]


def _is_headline_style(sent) -> bool:
    """True if `sent` capitalizes essentially every real word (headline
    title case) rather than just proper nouns/sentence starts (normal
    prose). The first token is skipped since sentence-initial capitalization
    happens in normal prose too and isn't evidence either way.
    """
    words = [
        t for t in sent[1:] if t.is_alpha and len(t.text) >= _HEADLINE_MIN_WORD_LEN
    ]
    if len(words) < _HEADLINE_MIN_WORDS:
        return False
    capitalized = sum(1 for t in words if t.text[0].isupper())
    return capitalized / len(words) >= _HEADLINE_CAP_RATIO


def _trim_headline_entity(ent, lower_sent_doc):
    """spaCy's NER (and its POS tagger - checked directly, it has the same
    blind spot) over-extends entity spans on headline-style text into a
    following verb/noun clause, because title case capitalizes every real
    word rather than just proper nouns (e.g. "Apple Hit", "Goldman Questions
    AI Spending Payoff" as one ORG span - both reproduced on real articles).

    Fix: re-tag the same sentence fully lowercased. That strips the
    headline-casing bias, so a trailing common noun/verb goes back to
    tagging as non-PROPN even though it also erases the FIRST token's own
    PROPN signal - irrelevant here, since the original cased NER already
    anchored the entity's start there. Extend from that anchor only while
    the lowercased re-tag still calls each next token PROPN; cut at the
    first one it doesn't.
    """
    sent = ent.sent
    offset = ent.start - sent.start
    end_offset = ent.end - sent.start
    new_end_offset = offset + 1
    while new_end_offset < end_offset and lower_sent_doc[new_end_offset].pos_ == "PROPN":
        new_end_offset += 1
    if new_end_offset == end_offset:
        return ent
    return Span(ent.doc, ent.start, sent.start + new_end_offset, label=ent.label_)


def _sentences_for_group(group):
    """Sentences containing a mention of this subject, in document order.

    Uses each entity span's own `.sent` (spaCy's sentence segmentation),
    which sidesteps string-matching pitfalls entirely (possessives,
    punctuation, partial words).
    """
    seen_starts = set()
    sents = []
    for span in group["spans"]:
        sent = span.sent
        if sent.start not in seen_starts:
            seen_starts.add(sent.start)
            sents.append(sent)
    return sents


def _tfidf_sentence_scores(doc):
    """Score every sentence in the doc by summed TF-IDF weight of its terms.

    ponytail: hand-rolled TF-IDF over lemmatized, stopword-filtered tokens
    instead of pulling in scikit-learn for one small computation. This is
    the "TF-IDF-weighted sentence selection" extractive approach from the
    brief, just implemented with stdlib math instead of a new dependency.
    """
    sents = list(doc.sents)
    tokenized = [
        [t.lemma_.lower() for t in s if t.is_alpha and not t.is_stop]
        for s in sents
    ]
    doc_freq = Counter()
    for toks in tokenized:
        doc_freq.update(set(toks))
    n_sents = len(sents)

    scores = {}
    for sent, toks in zip(sents, tokenized):
        if not toks:
            scores[sent.start] = 0.0
            continue
        term_freq = Counter(toks)
        score = sum(
            (count / len(toks)) * math.log((n_sents + 1) / (doc_freq[term] + 1) + 1)
            for term, count in term_freq.items()
        )
        scores[sent.start] = score
    return scores


def _summarize(subject_sents, sentence_scores):
    """Extractive summary: top-scoring sentences about this subject, kept in
    original reading order. No generation, just selection."""
    ranked = sorted(
        subject_sents, key=lambda s: sentence_scores.get(s.start, 0.0), reverse=True
    )
    top = ranked[:_SUMMARY_MAX_SENTENCES]
    top_in_order = sorted(top, key=lambda s: s.start)
    return " ".join(s.text.strip() for s in top_in_order)


def _sentiment_for_text(text: str) -> dict:
    compound = _vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"compound": compound, "label": label}


def _is_financially_relevant(subject_text: str, sentence_text: str) -> bool:
    combined = f"{subject_text} {sentence_text}".lower()
    if TICKER_RE.search(combined):
        return True
    return any(kw in combined for kw in FINANCIAL_KEYWORDS)


def analyze(text: str, headline: Optional[str] = None, financial_mode: bool = False) -> list:
    """Run the full pipeline and return a list of subject results, each:

        {
          "subject": str,
          "label": "ORG" | "PERSON" | "PRODUCT",
          "mentions": int,
          "summary": str,          # extractive, from the article's own sentences
          "sentiment": {"compound": float, "label": str},
          "financial_relevant": bool,
        }

    Sorted by mention count, descending. If financial_mode is True, subjects
    not flagged as financially relevant are dropped entirely rather than
    just tagged, since the point of the mode is to act as a filter.
    """
    if not text or not text.strip():
        return []

    full_text = f"{headline.strip()}. {text}" if headline and headline.strip() else text
    doc = _nlp()(full_text)

    ents = [e for e in doc.ents if e.label_ in SUBJECT_LABELS]
    headline_lower_docs = {}
    trimmed_ents = []
    for e in ents:
        if _is_headline_style(e.sent):
            if e.sent.start not in headline_lower_docs:
                headline_lower_docs[e.sent.start] = _nlp()(e.sent.text.lower())
            e = _trim_headline_entity(e, headline_lower_docs[e.sent.start])
        trimmed_ents.append(e)
    groups = _merge_entities(trimmed_ents)
    sentence_scores = _tfidf_sentence_scores(doc)

    results = []
    for group in groups:
        sents = _sentences_for_group(group)
        if not sents:
            continue
        sentence_text = " ".join(s.text for s in sents)
        results.append(
            {
                "subject": group["canonical"],
                "label": group["label"],
                "mentions": len(group["spans"]),
                "summary": _summarize(sents, sentence_scores),
                "sentiment": _sentiment_for_text(sentence_text),
                "financial_relevant": _is_financially_relevant(
                    group["canonical"], sentence_text
                ),
            }
        )

    results.sort(key=lambda r: r["mentions"], reverse=True)

    if financial_mode:
        results = [r for r in results if r["financial_relevant"]]

    return results
