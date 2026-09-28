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

1. spaCy NER extracts entities labeled `ORG`, `PERSON`, or `PRODUCT`.
2. Mentions are grouped into subjects by normalized substring overlap
   (`"Apple"` / `"Apple Inc."`, `"Elon Musk"` / `"Musk"`). This is a
   heuristic, not real coreference resolution — see the `ponytail:` comment
   in `_merge_entities` in `app/pipeline.py` for the known ceiling.
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

**Tested** (see `tests/`, 14 tests, all passing against the real spaCy model
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

**Real-world spot-check performed** (2026-09-28, against 5 live articles
pulled from Alpaca's News API - real headlines, not hand-written samples).
Result: subject-level sentiment/summary/financial-relevance all worked as
designed, but entity extraction on real headline-style text has real,
reproduced rough edges the hand-written test articles didn't surface:

- **Switched to `en_core_web_md`** (was `en_core_web_sm`) after confirming
  the smaller model mistags real company names - e.g. "Rivian shares surged
  8%..." tags "Rivian" as `NORP` under `_sm`, `ORG` under `_md`, reproduced
  directly comparing both models on the same sentence. Regression test:
  `test_ner_tags_a_real_company_name_as_org_not_norp`.
- **Entity span boundaries are noisy on headline-style text.** Real examples
  from the spot-check: `"Apple Hit"` (verb swallowed into the entity),
  `"Meta Stock"` / `"SpaceX Events"` (a following common noun swallowed in),
  and worst, `"Goldman Questions AI Spending Payoff"` - an entire headline
  clause captured as one `ORG` entity. spaCy's NER expects normal prose
  capitalization; headline title-case (every word capitalized) removes the
  signal it normally uses to find entity boundaries. Not fixed here - would
  need headline-specific preprocessing or entity-boundary post-filtering,
  which is a real design task, not a quick patch.
- **Cross-mention label inconsistency defeats the alias merge.** Reproduced
  directly: in one real article, spaCy tags `"Mark Zuckerberg"` as `PERSON`
  in the first sentence and the later standalone `"Zuckerberg"` as `ORG` in
  the second - two mentions of the same real person, two different NER
  labels within the same document. `_merge_entities`' label-must-match check
  (deliberate - it's what stops merging, say, a person named Washington with
  the org "Washington Post") is doing exactly what it's designed to do here;
  the actual root cause is spaCy's own per-mention label inconsistency, not
  a merge-logic bug. Fixing this needs cross-mention label reconciliation or
  spaCy's experimental coreference resolution (already noted as the upgrade
  path in `_merge_entities`'s own docstring) - flagged as a follow-up task,
  not fixed in this pass.
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
- **ARM64 / Turing Pi RK1 behavior is completely unverified.** This was
  built and tested only on x86_64. See "ARM64 status" below.
- **The `/analyze/url` scraper** is a plain `<p>`-tag pull, not tested
  against real news sites (paywalls, JS-rendered content, and heavily
  templated pages will all degrade or break it). It's the convenience
  endpoint the brief allowed for, not a hardened ingestion path.

## ARM64 status (flagging clearly, not guessing)

**Untested on real ARM64 hardware** — this was built in an x86_64 sandbox
with no access to a Turing Pi RK1 or any other ARM64 machine, so none of the
following has been verified, only reasoned about:

- `spacy`, its `numpy`/`blis`/`thinc` dependencies, and `vaderSentiment` all
  install cleanly via prebuilt wheels on x86_64 (verified: this is exactly
  what happened building this repo). Whether PyPI has prebuilt
  `manylinux_aarch64` wheels for the pinned dependency versions is not
  something this environment could check.
- If aarch64 wheels are missing for some dependency, pip will fall back to
  building from source, which is why the Dockerfile installs
  `build-essential` defensively — this trades a larger image for avoiding a
  silent build failure. If a wheel-only aarch64 build turns out to be
  needed later (e.g. for a smaller image), that's a follow-up, not done
  here.
- `en_core_web_md` itself is a pure data package (no native code, just
  larger than `_sm` - includes word vectors), so the
  model download step should be architecture-independent regardless of the
  above.
- **Recommendation:** build and run the Docker image on the actual RK1
  hardware (or an aarch64 emulation layer like `docker buildx --platform
  linux/arm64`) before trusting this in the cluster. Don't assume it "just
  works" from this README.

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
