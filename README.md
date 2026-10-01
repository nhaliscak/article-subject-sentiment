# article-subject-sentiment

Subject-level sentiment analysis for a single article: given article text,
find the distinct subjects it discusses (companies, people, products), pull
an extractive summary of what's said about each one, and score sentiment
*per subject* rather than one score for the whole article. Includes a
`financial_mode` filter that keeps only the subjects that are actually
relevant to trading.

This is a companion to `market-sentiment-service` (which scores whole
articles with VADER, tagged to whatever tickers Alpaca's News API already
associates with them). This project does the finer-grained thing: an article
can be positive about one company and negative about another, and this
distinguishes them. It takes raw article text as input — it does not ingest
news itself.

**NLP only.** No LLM calls, no generative summarization, no model training.
Entity extraction is spaCy NER (`en_core_web_md`). Sentiment is VADER
(`vaderSentiment`), the same lexicon-based tool `market-sentiment-service`
already uses. Extractive summaries are a hand-rolled TF-IDF sentence ranking
(see `app/pipeline.py`) — no scikit-learn dependency, just stdlib math.

## API

### `POST /analyze`

```json
{"headline": "optional headline", "text": "article body text...", "financial_mode": false}
```

`financial_mode` can also be passed as a query param (`?financial_mode=true`),
which takes precedence over the body field if both are given.

Response:

```json
{
  "headline": "...",
  "financial_mode": false,
  "subjects": [
    {
      "subject": "Apple",
      "label": "ORG",
      "mentions": 3,
      "summary": "extractive sentences pulled verbatim from the article",
      "sentiment": {"compound": 0.72, "label": "positive"},
      "financial_relevant": true
    }
  ]
}
```

When `financial_mode` is true, subjects not flagged `financial_relevant` are
dropped from the list entirely (it's a filter, not just a label).

### `POST /analyze/url`

Convenience endpoint: `{"url": "https://...", "financial_mode": false}`.
Fetches the page, pulls text out of `<p>` tags, and runs it through the same
pipeline. This is a plain scrape, not a boilerplate/readability extractor —
it works on simple article pages and will pick up nav/ad text on
heavily-templated sites. Not the focus of this project; use at your own risk
for real-world URLs.

### `GET /health`

Liveness/readiness check.

## How subject/sentiment/summary/financial-relevance are computed

1. spaCy NER extracts entities labeled `ORG`, `PERSON`, or `PRODUCT`. Entities
   found inside a headline-style (title-case) sentence are first trimmed back
   to the leading run of tokens a lowercase re-tag still calls a proper noun,
   since headline capitalization otherwise fools spaCy into swallowing a
   trailing verb/common noun into the span (`_is_headline_style`,
   `_trim_headline_entity` in `app/pipeline.py`).
2. Mentions are grouped into subjects by normalized substring overlap
   (`"Apple"` / `"Apple Inc."`, `"Elon Musk"` / `"Musk"`). This is a
   heuristic, not real coreference resolution — see the `ponytail:` comment
   in `_merge_entities` in `app/pipeline.py` for the known ceiling. A narrow
   follow-up pass (`_merge_cross_label_surname_matches`) also reconciles the
   case where spaCy tags the same person's surname with two different labels
   across mentions (e.g. `PERSON` then `ORG`).
3. For each subject, the sentences that mention it are found via spaCy's own
   sentence attribution of each entity span (not string re-matching, so
   possessives/punctuation don't break it).
4. Those sentences are ranked by a TF-IDF score computed over the whole
   article and the top 3 are kept, in original order, as the extractive
   summary.
5. VADER runs on the subject's own sentence text (not the whole article) to
   get a subject-localized sentiment score.
6. `financial_relevant` is true if the subject's sentences contain a ticker
   pattern (`$AAPL`, `(NASDAQ: AAPL)`) or common financial-market vocabulary
   (earnings, guidance, shares, analyst, price target, etc.).

## What's tested vs. what's assumed

**Tested** (see `tests/`, 21 tests, all passing against the real spaCy model
and real VADER — no mocking of the NLP):

- Entity extraction against known sample articles with known expected subjects.
- Sentiment correctly distinguishes a positive-about-X / negative-about-Y
  article (Apple up, "Nexlon Corp" down, in the same article).
- Mention merging (`"Nexlon Corp"` + `"Nexlon's"` count as one subject).
- Extractive summaries are verbatim substrings of the source article (never
  generated text).
- `financial_mode` filtering against a mixed article (Apple/earnings vs. a
  Red Cross charity partnership in the same piece).
- Ticker-pattern recognition (`$WDGT`, `(NASDAQ: WDGT)`).
- The API layer end-to-end (FastAPI `TestClient`), including the
  query-param-overrides-body-field behavior for `financial_mode`, empty-text
  rejection, and the URL endpoint's error handling for an unreachable host.
- **Headline entity-boundary trimming and cross-mention label reconciliation**
  (both added 2026-09-28, see below) against real headline/body text pulled
  live from Alpaca's News API and frozen verbatim as test fixtures
  (`REAL_HEADLINE_ZUCKERBERG`, `REAL_HEADLINE_APPLE`, `REAL_HEADLINE_SPACEX`
  in `tests/test_pipeline.py`), plus a spaCy-independent unit test of the
  merge guard's safety case.

**Real-world spot-check performed** (2026-09-28, against 5 live articles
pulled from Alpaca's News API - real headlines, not hand-written samples).
Result: subject-level sentiment/summary/financial-relevance all worked as
designed, but entity extraction on real headline-style text had two real,
reproduced rough edges the hand-written test articles didn't surface. Both
are now fixed:

- **Switched to `en_core_web_md`** (was `en_core_web_sm`) after confirming
  the smaller model mistags real company names - e.g. "Rivian shares surged
  8%..." tags "Rivian" as `NORP` under `_sm`, `ORG` under `_md`, reproduced
  directly comparing both models on the same sentence. Regression test:
  `test_ner_tags_a_real_company_name_as_org_not_norp`.
- **Fixed: entity span boundaries were noisy on headline-style text.** Real
  examples from the spot-check, pulled fresh again on 2026-09-28 to build
  regression fixtures: `"Apple Hit"` (verb swallowed into the entity),
  `"Meta Stock"` / `"SpaceX Events"` (a following common noun swallowed in),
  and worst, `"Goldman Questions AI Spending Payoff"` - an entire headline
  clause captured as one `ORG` entity. Root cause: headline title case
  capitalizes every real word, not just proper nouns, and spaCy's NER *and*
  its POS tagger both use capitalization as their dominant boundary signal -
  confirmed directly by re-running the tagger on these exact headlines
  lowercased, which mistagged the same trailing common nouns/verbs as
  `PROPN` when cased, `NOUN`/`VERB` when not. Fix (`_is_headline_style`,
  `_trim_headline_entity` in `app/pipeline.py`): detect headline-style
  sentences by capitalization ratio, then for entities inside them, re-tag
  that one sentence lowercased and trim the entity down to the leading run
  of tokens the lowercased re-tag still calls `PROPN` - anchored at the
  original (correctly-placed) entity start, so the trim can't eat the whole
  span. Deliberately scoped to headline-style sentences only; normal prose
  (where capitalization *is* a reliable signal) is untouched. Regression
  tests: `test_headline_boundary_noise_is_trimmed_*`,
  `test_headline_style_detection_does_not_affect_normal_prose`.
- **Fixed: cross-mention label inconsistency was defeating the alias merge.**
  Reproduced directly: in one real article, spaCy tags `"Mark Zuckerberg"` as
  `PERSON` in the headline and the later standalone `"Zuckerberg"` as `ORG`
  in the body - two mentions of the same real person, two different NER
  labels within the same document. `_merge_entities`' label-must-match check
  (deliberate - it's what stops merging, say, a person named Washington with
  the org "Washington Post") was doing exactly what it's designed to do;
  the actual root cause was spaCy's own per-mention label inconsistency, not
  a merge-logic bug. Fix (`_merge_cross_label_surname_matches` in
  `app/pipeline.py`): a narrow post-pass that merges a single-token group
  into a cross-label multi-token group only when the token matches the
  multi-token group's *last* word and one side is `PERSON` - matching on the
  trailing word only (not any substring) is what keeps this from merging a
  person named Washington into "Washington Post" (last word "post", not
  "washington"). Known remaining edge case, documented in the function's own
  docstring: a person who happens to share an org's *last* word (e.g. a
  person literally named "Post" near "The Washington Post") would still
  wrongly merge - accepted as a narrow, safe-by-construction heuristic, not
  full coreference resolution. Regression tests:
  `test_cross_mention_label_split_is_merged_zuckerberg`,
  `test_cross_label_surname_merge_is_narrow`.
- **The alias-merging heuristic** (step 2 above), aside from the label-
  mismatch case just described, handles the common cases in the tests; it's
  a substring-overlap heuristic, not coreference resolution, so unusual
  nicknames/abbreviations/subsidiary-vs-parent naming can still under- or
  over-merge.
- **The financial-relevance keyword/ticker list** is a reasonable starting
  set, not validated against a large labeled corpus of financial vs.
  non-financial subjects. It will miss financial language it doesn't
  recognize and won't infer that a company is publicly traded from its name
  alone without a ticker or market-language sentence nearby.
- **ARM64 / Turing Pi RK1 behavior was unverified for a while after this was
  first believed fixed.** See "ARM64 status" below for what actually
  happened and what's confirmed now.
- **The `/analyze/url` scraper** is a plain `<p>`-tag pull, not tested
  against real news sites (paywalls, JS-rendered content, and heavily
  templated pages will all degrade or break it). It's the convenience
  endpoint the brief allowed for, not a hardened ingestion path.

## ARM64 status (flagging clearly, not guessing)

- **Update (2026-09-28): the `linux/arm64` build itself is confirmed to
  complete.** `.github/workflows/build-and-deploy.yml` ran for real on push
  (`gh run watch`) and its `linux/arm64` build - via QEMU emulation on
  GitHub's amd64 runners - succeeded, meaning `spacy`/`numpy`/`blis`/`thinc`/
  `vaderSentiment` all either found prebuilt `manylinux_aarch64` wheels or
  successfully built from source under `build-essential`. The image *builds*
  for arm64 - QEMU emulation proves the build step, not runtime behavior.
- **2026-09-30: the headline-trim/cross-label-merge fix had never actually
  been deployed, on any architecture.** `app/pipeline.py` and
  `tests/test_pipeline.py` were written, tested locally, and described in
  this README as shipped in `bec1109` - but those two files were never
  `git add`ed, so every commit since (`bec1109` through `ba34015`) actually
  deployed the pre-fix pipeline. This was caught while deploying the service
  to the real cluster for the first time (tradebot's own evaluation):
  `kubectl exec`-ing a real test case into the live pod showed `"Apple Hit"`
  un-trimmed, which the local test suite said should be impossible. The
  first theory - an ARM64-specific floating-point divergence in spaCy's
  tagger - turned out to be wrong; checking `git show HEAD:app/pipeline.py`
  showed the deployed source never had the trim logic at all, on any commit.
  Fixed by actually committing the working-tree changes. The incorrect
  ARM64-divergence theory (briefly written into this file and into a
  docstring/test comment) has been removed now that the real cause is known.
- Real ARM64 runtime behavior of the actual trim/merge logic, now that it's
  genuinely deployed, is pending re-verification against the live pod -
  update this section once that's done rather than assume either way.
- `en_core_web_md` itself is a pure data package (no native code, just
  larger than `_sm` - includes word vectors), so the model download step was
  never really a risk here.
- **Recommendation:** when a deployed service doesn't match local test
  results, check what's actually committed and built before reaching for a
  more exotic explanation (architecture, floating-point, etc.) - the mundane
  explanation (uncommitted files) turned out to be the real one here.

## Running locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m spacy download en_core_web_md
.venv/bin/python -m pytest tests/ -v
.venv/bin/uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t article-subject-sentiment .
docker run -p 8000:8000 article-subject-sentiment
```

## Kubernetes

`k8s/deployment.yaml`, `k8s/service.yaml`, and `k8s/argocd-application.yaml`
match the real conventions used by every other app on this cluster
(confirmed 2026-09-28 via `kubectl` against the live, healthy `tradebot-hub`
Application - see that file's own comment). `.github/workflows/
build-and-deploy.yml` builds and pushes this multi-arch (amd64+arm64, via
QEMU on GitHub's runners) on every push to `main`, mirroring `tradebot_hub`'s
own proven workflow.

```bash
kubectl apply -f k8s/argocd-application.yaml   # one-time; Argo CD takes it from here
```
