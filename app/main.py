from typing import Optional

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .pipeline import analyze

app = FastAPI(
    title="article-subject-sentiment",
    description="Subject-level sentiment analysis for article text (NLP only, no LLM).",
)


class AnalyzeRequest(BaseModel):
    headline: Optional[str] = None
    text: str
    financial_mode: bool = False


class AnalyzeUrlRequest(BaseModel):
    url: str
    financial_mode: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


def _resolve_financial_mode(query_value: Optional[bool], body_value: bool) -> bool:
    return query_value if query_value is not None else body_value


@app.post("/analyze")
def analyze_endpoint(
    req: AnalyzeRequest, financial_mode: Optional[bool] = Query(None)
):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    fm = _resolve_financial_mode(financial_mode, req.financial_mode)
    subjects = analyze(req.text, headline=req.headline, financial_mode=fm)
    return {"headline": req.headline, "financial_mode": fm, "subjects": subjects}


@app.post("/analyze/url")
def analyze_url_endpoint(
    req: AnalyzeUrlRequest, financial_mode: Optional[bool] = Query(None)
):
    # ponytail: simple <p>-tag scrape, not a real readability/boilerplate
    # extractor. Fine for plain article pages; will pull nav/ad text on
    # heavily-templated sites. Upgrade to trafilatura/readability-lxml if
    # that turns out to matter in practice.
    try:
        resp = requests.get(
            req.url, timeout=10, headers={"User-Agent": "article-subject-sentiment/1.0"}
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"failed to fetch url: {exc}")

    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)
    if not text.strip():
        raise HTTPException(
            status_code=422, detail="could not extract article text from url"
        )

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    fm = _resolve_financial_mode(financial_mode, req.financial_mode)
    subjects = analyze(text, headline=title, financial_mode=fm)
    return {"url": req.url, "headline": title, "financial_mode": fm, "subjects": subjects}
