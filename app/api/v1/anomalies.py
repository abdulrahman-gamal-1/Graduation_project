from dataclasses import asdict

from fastapi import APIRouter

from app.api.deps import clean_from_request
from app.schemas.anomalies import AnomalyRequest, AnomalyResponse
from app.schemas.glucose import DataQualityReport
from app.services.anomalies import AnomalyConfig, detect_anomalies

router = APIRouter(
    prefix="/anomalies", tags=["Anomaly detection"],
    responses={
        401: {"description": "Missing or invalid X-Internal-API-Key (`{\"detail\": ...}`)."},
        422: {"description": "Request validation error, or domain error such as "
              "`insufficient_data` (`ErrorResponse`: error/detail/request_id)."},
    },
)


@router.post(
    "/detect", response_model=AnomalyResponse, summary="Detect dangerous glucose events",
    description=(
        "**Layer 1 - deterministic rules** (never delegated to a model): sustained <54 and <70 "
        "mg/dL, sustained >250 mg/dL (>= 15 min), rapid falls (>= 2 mg/dL/min) and rises "
        "(>= 3 mg/dL/min).\n\n"
        "**Layer 2 - statistical**: deviations from the person's own pattern at the same time "
        "of day. Isolation Forest proposes, a robust z-score confirms; falls back to robust "
        "z-score alone when < 3 days of data. Single-point spikes are dropped as likely sensor "
        "artifacts.\n\n"
        "Retrospective only - not a real-time alarm. Never a diagnosis."
    ),
)
def detect(req: AnomalyRequest) -> AnomalyResponse:
    clean = clean_from_request(req)
    o = req.anomaly_options
    cfg = AnomalyConfig(
        method=o.method, include_statistical=o.include_statistical,
        contamination=o.contamination, merge_gap_minutes=o.merge_gap_minutes,
    )
    report = detect_anomalies(clean, cfg, max_events=o.max_events)
    return AnomalyResponse(
        user_id=req.user_id, quality=DataQualityReport(**asdict(clean.report)), **asdict(report)
    )
