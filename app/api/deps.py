from app.schemas.glucose import GlucoseSeriesRequest
from app.services.preprocessing import (
    CleanedSeries, PreprocessConfig, clean_cgm_series, readings_to_frame,
)


def clean_from_request(req: GlucoseSeriesRequest) -> CleanedSeries:
    """Shared by every endpoint so they all see identically cleaned data."""
    raw = readings_to_frame([(r.timestamp, r.value_mg_dl) for r in req.readings])
    return clean_cgm_series(raw, PreprocessConfig(**req.options.model_dump()))
