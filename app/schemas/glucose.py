"""Request/response models shared by preprocessing and analytics endpoints."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import settings
from app.core.constants import PHYSICAL_MAX_MG_DL, PHYSICAL_MIN_MG_DL


class GlucoseReading(BaseModel):
    timestamp: datetime = Field(
        ..., description="ISO-8601. Timezone-aware preferred; naive values are treated as UTC."
    )
    value_mg_dl: float | None = Field(
        None, description="Glucose in mg/dL. null = missing. Out-of-range values are clipped."
    )


class PreprocessingOptions(BaseModel):
    interval_minutes: int = Field(5, ge=1, le=60, description="Target CGM grid interval.")
    clip_low: float = Field(PHYSICAL_MIN_MG_DL, ge=0, description="Lower physical bound, mg/dL.")
    clip_high: float = Field(PHYSICAL_MAX_MG_DL, le=1000, description="Upper physical bound, mg/dL.")
    max_interpolation_gap_minutes: int = Field(
        30, ge=0, le=180,
        description="Gaps up to this length are linearly interpolated; longer gaps stay missing.",
    )
    timezone: str = Field(
        "UTC", description="IANA timezone of the patient (e.g. 'Africa/Cairo'). Used for "
        "time-of-day blocks, daily summaries and AGP."
    )

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"Unknown IANA timezone: {v!r}") from e
        return v

    @model_validator(mode="after")
    def _bounds(self):
        if self.clip_low >= self.clip_high:
            raise ValueError("clip_low must be smaller than clip_high")
        return self


class GlucoseSeriesRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$",
        description="Opaque pseudonymous ID. Never send names, emails or phone numbers.",
    )
    readings: list[GlucoseReading] = Field(
        ..., min_length=1, max_length=settings.max_readings_per_request
    )
    options: PreprocessingOptions = Field(default_factory=PreprocessingOptions)

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "u_8f3a2c",
                "readings": [
                    {"timestamp": "2026-03-01T00:00:00Z", "value_mg_dl": 112.0},
                    {"timestamp": "2026-03-01T00:05:00Z", "value_mg_dl": 115.0},
                    {"timestamp": "2026-03-01T00:10:00Z", "value_mg_dl": None},
                ],
                "options": {"timezone": "Africa/Cairo"},
            }
        }
    }


class DataQualityReport(BaseModel):
    n_raw: int
    n_invalid_timestamp: int
    n_invalid_value: int
    n_duplicate_timestamps: int
    n_clipped_low: int
    n_clipped_high: int
    n_grid_points: int
    n_observed: int
    n_imputed: int
    n_missing_remaining: int
    coverage_pct: float = Field(..., description="Observed readings / expected grid points (%).")
    n_gaps_remaining: int
    longest_gap_minutes: int
    start: datetime
    end: datetime
    span_days: float


class CleanedPoint(BaseModel):
    timestamp: datetime
    glucose_mg_dl: float | None
    is_imputed: bool


class CleanResponse(BaseModel):
    user_id: str
    quality: DataQualityReport
    series: list[CleanedPoint]


class FeaturesResponse(BaseModel):
    user_id: str
    feature_schema_version: str
    quality: DataQualityReport
    features: dict[str, float | int | None] = Field(
        ..., description="Flat feature dict. Names are stable; null = not enough data."
    )
