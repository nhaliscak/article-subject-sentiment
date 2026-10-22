# article-subject-sentiment — Project Log

Dated entries, newest last. Each: goal, what was tried, symptoms, debugging steps, root cause, fix, result (include failures and dead ends).

## Log

## 2026-10-21 — Thoroughness pass, detailed and financial_lite modes

Goal: make per-subject analysis more thorough. Replaced concatenated-text VADER with per-sentence
scores averaged (concatenation saturates), clause-level scoring for sentences naming several
subjects, previous-sentence pronoun/"the company" attribution (`inferred` evidence), and a
`sentiment.mixed` flag. Added `mode` (standard/detailed/financial_lite): detailed adds GPE, NORP,
EVENT, LAW, FAC, WORK_OF_ART; financial_lite = financial filter + compact output. 29 tests pass.
Not validated on real articles yet; the pronoun and clause heuristics are marked `ponytail:`.
