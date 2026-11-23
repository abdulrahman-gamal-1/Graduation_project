from dataclasses import asdict

from fastapi import APIRouter

from app.api.deps import clean_from_request
from app.schemas.glucose import DataQualityReport
from app.schemas.patterns import PatternRequest, PatternResponse
from app.services.patterns import PatternConfig, detect_all

router = APIRouter(
    prefix="/patterns", tags=["Pattern detection"],
    responses={
        401: {"description": "Missing or invalid X-Internal-API-Key (`{\"detail\": ...}`)."},
        422: {"description": "Request validation error, or domain error such as "
              "`insufficient_data` (`ErrorResponse`: error/detail/request_id)."},
    },
)


@router.post(
    "/detect", response_model=PatternResponse, summary="Detect recurring glucose patterns",
    description=(
        "**Rules**: recurrent hypoglycaemia by time block, early-morning rise, post-meal-window "
        "spikes, sustained hyperglycaemia, high-variability days - each with evidence "
        "(days affected / evaluated, example dates, metrics).\n\n"
        "**Clustering**: K-Means on days x 24 hourly profiles (local time), K chosen by "
        "silhouette. Returns `no_distinct_structure` rather than forcing clusters.\n\n"
        "Needs >= 14 days for reliable results; fewer days simply yield fewer/no patterns. "
        "Descriptive only - never a diagnosis or treatment advice."
    ),
)
def detect_patterns(req: PatternRequest) -> PatternResponse:
    clean = clean_from_request(req)
    o = req.pattern_options
    report = detect_all(
        clean, include_rules=o.include_rules, include_clusters=o.include_clusters,
        max_clusters=o.max_clusters, cfg=PatternConfig(min_days_evaluated=o.min_days_evaluated),
    )
    return PatternResponse(
        user_id=req.user_id, quality=DataQualityReport(**asdict(clean.report)), **asdict(report)
    )
