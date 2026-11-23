from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from typing import Annotated

from pydantic import Field

from app.schemas.glucose import DataQualityReport, GlucoseSeriesRequest
from pydantic import BaseModel


class TimeInRange(BaseModel):
    very_low_pct: float = Field(..., description="< 54 mg/dL")
    low_pct: float = Field(..., description="54-69 mg/dL")
    in_range_pct: float = Field(..., description="70-180 mg/dL (inclusive)")
    high_pct: float = Field(..., description="181-250 mg/dL")
    very_high_pct: float = Field(..., description="> 250 mg/dL")
    tbr_pct: float = Field(..., description="Time below range (<70) = very_low + low")
    tar_pct: float = Field(..., description="Time above range (>180) = high + very_high")


class GlucoseStats(BaseModel):
    n_readings: int
    mean_mg_dl: float
    median_mg_dl: float
    sd_mg_dl: float
    cv_pct: float = Field(..., description="Coefficient of variation = SD / mean x 100")
    cv_is_stable: bool = Field(..., description="CV <= 36%")
    min_mg_dl: float
    max_mg_dl: float
    gmi_pct: float = Field(..., description="GMI = 3.31 + 0.02392 x mean glucose (mg/dL)")
    time_in_range: TimeInRange


class SummaryResponse(GlucoseStats):
    user_id: str
    quality: DataQualityReport
    meets_consensus_criteria: bool = Field(
        ..., description=">= 14 days of data and >= 70% sensor coverage"
    )
    warnings: list[str] = []


class RollingRequest(GlucoseSeriesRequest):
    windows_minutes: list[Annotated[int, Field(ge=10, le=10080)]] = Field(
        default_factory=lambda: [60, 360, 1440], min_length=1, max_length=5,
        description="Trailing window sizes in minutes (10 min - 7 days).",
    )
    output_step_minutes: int = Field(60, ge=5, le=1440, description="Down-sampling step.")


class RollingPoint(BaseModel):
    timestamp: datetime
    mean_mg_dl: float | None
    sd_mg_dl: float | None


class RollingSeries(BaseModel):
    window_minutes: int
    points: list[RollingPoint]


class RollingResponse(BaseModel):
    user_id: str
    series: list[RollingSeries]


class DailySummary(BaseModel):
    date: date_type = Field(..., description="Local calendar date in the requested timezone")
    coverage_pct: float
    n_readings: int
    mean_mg_dl: float
    sd_mg_dl: float
    cv_pct: float
    gmi_pct: float
    min_mg_dl: float
    max_mg_dl: float
    time_in_range: TimeInRange


class DailyResponse(BaseModel):
    user_id: str
    days: list[DailySummary]


class AGPPoint(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


class AGPResponse(BaseModel):
    user_id: str
    profile: list[AGPPoint]
