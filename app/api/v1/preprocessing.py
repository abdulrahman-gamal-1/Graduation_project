from dataclasses import asdict

import numpy as np
from fastapi import APIRouter

from app.api.deps import clean_from_request
from app.core.constants import FEATURE_SCHEMA_VERSION
from app.schemas.glucose import (
    CleanedPoint, CleanResponse, DataQualityReport, FeaturesResponse, GlucoseSeriesRequest,
)
from app.services.features import extract_user_features

router = APIRouter(
    prefix="/preprocess", tags=["Preprocessing"],
    responses={
        401: {"description": "Missing or invalid X-Internal-API-Key (`{\"detail\": ...}`)."},
        422: {"description": "Request validation error, or domain error such as "
              "`insufficient_data` (`ErrorResponse`: error/detail/request_id)."},
    },
)


# NOTE: endpoints are plain `def` (not `async def`) on purpose: pandas work is CPU-bound and
# FastAPI runs sync endpoints in a threadpool, keeping the event loop responsive.
@router.post(
    "/clean", response_model=CleanResponse, summary="Clean and resample a CGM series",
    description="Validates, de-duplicates, clips to the physical range, resamples to a regular "
    "grid and imputes short gaps. Returns the cleaned series plus a data-quality report.",
)
def clean_series(req: GlucoseSeriesRequest) -> CleanResponse:
    clean = clean_from_request(req)
    f = clean.frame
    series = [
        CleanedPoint(
            timestamp=ts.to_pydatetime(),
            glucose_mg_dl=None if np.isnan(g) else round(float(g), 2),
            is_imputed=bool(imp),
        )
        for ts, g, imp in zip(f.index, f["glucose"].to_numpy(), f["is_imputed"].to_numpy())
    ]
    return CleanResponse(
        user_id=req.user_id, quality=DataQualityReport(**asdict(clean.report)), series=series
    )


@router.post(
    "/features", response_model=FeaturesResponse, summary="Extract user-level features",
    description="Flat, stable-named feature vector (glycaemic metrics, excursion events, "
    "time-of-day blocks, variability, trend). Same features feed the risk classifier.",
)
def user_features(req: GlucoseSeriesRequest) -> FeaturesResponse:
    clean = clean_from_request(req)
    return FeaturesResponse(
        user_id=req.user_id,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        quality=DataQualityReport(**asdict(clean.report)),
        features=extract_user_features(clean),
    )
