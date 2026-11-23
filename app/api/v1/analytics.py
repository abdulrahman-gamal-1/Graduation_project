from dataclasses import asdict

from fastapi import APIRouter

from app.api.deps import clean_from_request
from app.schemas.analytics import (
    AGPResponse, DailyResponse, RollingRequest, RollingResponse, SummaryResponse,
)
from app.schemas.glucose import DataQualityReport, GlucoseSeriesRequest
from app.services import analytics as svc

router = APIRouter(
    prefix="/analytics", tags=["Descriptive analytics"],
    responses={
        401: {"description": "Missing or invalid X-Internal-API-Key (`{\"detail\": ...}`)."},
        422: {"description": "Request validation error, or domain error such as "
              "`insufficient_data` (`ErrorResponse`: error/detail/request_id)."},
    },
)


@router.post(
    "/summary", response_model=SummaryResponse, summary="TIR, SD, CV and GMI",
    description="Time-in-Range (5 consensus bands), mean/median/SD, CV%, GMI. Computed on "
    "observed readings only; `warnings` explains data-sufficiency caveats.",
)
def summary(req: GlucoseSeriesRequest) -> SummaryResponse:
    clean = clean_from_request(req)
    return SummaryResponse(
        user_id=req.user_id, quality=DataQualityReport(**asdict(clean.report)),
        **svc.summarize(clean),
    )


@router.post("/rolling", response_model=RollingResponse, summary="Rolling mean and SD")
def rolling(req: RollingRequest) -> RollingResponse:
    clean = clean_from_request(req)
    return RollingResponse(
        user_id=req.user_id,
        series=svc.rolling_statistics(clean, req.windows_minutes, req.output_step_minutes),
    )


@router.post("/daily", response_model=DailyResponse, summary="Per-day statistics (local time)")
def daily(req: GlucoseSeriesRequest) -> DailyResponse:
    return DailyResponse(user_id=req.user_id, days=svc.daily_summaries(clean_from_request(req)))


@router.post(
    "/agp", response_model=AGPResponse, summary="Ambulatory Glucose Profile",
    description="Hour-of-day 5/25/50/75/95th percentile curves across all days.",
)
def agp(req: GlucoseSeriesRequest) -> AGPResponse:
    return AGPResponse(
        user_id=req.user_id, profile=svc.ambulatory_glucose_profile(clean_from_request(req))
    )
