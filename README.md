# GlucoCare AI Service

Internal FastAPI service for the GlucoCare Node.js backend: CGM preprocessing, descriptive
analytics and (upcoming) pattern/anomaly detection, risk scoring and a safe RAG chatbot.

> Outputs are informational analytics - **not a medical diagnosis**.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GLUCOCARE_ENV=dev                       # dev: auth optional. Anything else fails closed.
# production: export GLUCOCARE_INTERNAL_API_KEY='<long random secret>'
uvicorn app.main:app --reload --port 8000
pytest -q
```
Swagger UI: `http://localhost:8000/docs` - OpenAPI JSON: `/openapi.json`

## Endpoints (v1) - all require `X-Internal-API-Key`

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Liveness (no auth) |
| POST | `/api/v1/preprocess/clean` | Clean + resample series, returns data-quality report |
| POST | `/api/v1/preprocess/features` | Flat user-level feature vector |
| POST | `/api/v1/analytics/summary` | TIR (5 bands), mean/SD/CV%, GMI, data-sufficiency warnings |
| POST | `/api/v1/analytics/rolling` | Trailing rolling mean/SD (multiple windows) |
| POST | `/api/v1/analytics/daily` | Per local-day statistics |
| POST | `/api/v1/analytics/agp` | Hour-of-day percentile profile |
| POST | `/api/v1/patterns/detect` | Rule-based recurring patterns + K-Means day types |
| POST | `/api/v1/anomalies/detect` | Dangerous-event detection: safety rules + Isolation Forest / robust z |

Errors: request validation -> FastAPI default 422; domain errors (e.g. no valid data) -> 422
`{"error": "insufficient_data", "detail": "...", "request_id": "..."}`; bad key -> 401.

```bash
curl -X POST localhost:8000/api/v1/analytics/summary \
  -H 'Content-Type: application/json' -H 'X-Internal-API-Key: $KEY' \
  -d '{"user_id":"u_8f3a2c","options":{"timezone":"Africa/Cairo"},
       "readings":[{"timestamp":"2026-03-01T00:00:00Z","value_mg_dl":112}]}'
```

## Layout

```
app/
  core/       constants (clinical thresholds), config, security, errors
  schemas/    Pydantic request/response models  (the OpenAPI contract)
  services/   preprocessing.py  analytics.py  features.py   <- pure pandas/numpy, no FastAPI
  api/v1/     thin routers
tests/        service tests + API smoke tests
```

## Key design decisions

- **Stateless**: Node sends the series in each request; nothing is stored or logged here.
- **Privacy**: `user_id` must be an opaque pseudonym (regex-enforced); bodies are never logged.
- **Gap handling**: only gaps <= 30 min (configurable) are interpolated; longer gaps stay missing
  so we never invent hours of data.
- **Stats use observed readings only**; imputed points never affect TIR/GMI/coverage.
- **Clipping vs invalid**: values <= 0/NaN/inf are treated as *missing*; positive out-of-range
  values are *clipped* to 38-400 and counted in the report.
- **Time-based features use the patient's IANA timezone** (`options.timezone`); the Node backend
  should always pass it.
- **Thresholds** live only in `core/constants.py` (International TIR Consensus 2019;
  GMI = 3.31 + 0.02392 x mean).
- Sync (`def`) endpoints so CPU-bound pandas work runs in FastAPI's threadpool.

## Pattern detection (`services/patterns.py`, `services/clustering.py`)

Rules evaluate each local day, then require recurrence (>=50% of evaluable days; hypoglycaemia
>=2 days and >=15%). Pattern IDs are stable: `hypo_{overnight|morning|afternoon|evening}`,
`dawn_rise`, `postmeal_{breakfast|lunch|dinner}`, `hyper_{block}`, `high_variability_days`.
Severity: `priority` (level-2 hypo <54, or block mean >250), `attention`, `info`.
K-Means: days x 24 hourly means, K by silhouette (2..5), `no_distinct_structure` if silhouette
< 0.25, cluster 0 = most common day type, deterministic (`random_state=42`).
Thresholds: `PatternConfig` (rules) and `cluster_daily_profiles()` arguments.

## Anomaly detection (`services/anomalies.py`)

Layer 1 (rules, never delegated to a model): sustained (>=15 min) <54 `urgent_low`, <70 `low`,
>250 `urgent_high`; `rapid_fall` >=2 and `rapid_rise` >=3 mg/dL/min over 15 min (>=2 consecutive
points). Layer 2 (statistical): features are deviations from the person's own median at the same
half-hour of day (recurring meal spikes are not anomalies). Isolation Forest *proposes*
(contamination 1%), a robust z-score (|z|>=3) plus a minimum practical size (40 mg/dL level /
1 mg/dL/min rate) *confirms*; <3 days of data -> robust z-score alone (|z|>=3.5).
Single-point spikes are dropped (counted in `n_isolated_points_ignored`).
Headline type: levels outrank rates. Severity: `priority` = urgent_low or rapid_fall+low.
Retrospective only - **not a real-time alarm**.

## Roadmap (next deliverables)

1. ~~Pattern detection~~ done
2. ~~Anomaly detection~~ done
3. `services/risk.py` - Logistic regression/RF + coefficient/permutation explanations
4. `services/chatbot/` - safety guard (pre/post filters), retrieval over vetted corpus, disclaimer
