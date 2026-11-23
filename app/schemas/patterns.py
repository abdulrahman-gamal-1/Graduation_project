from __future__ import annotations

from datetime import date as date_type
from typing import Literal

from pydantic import BaseModel, Field

from app.core.constants import DISCLAIMER
from app.schemas.glucose import DataQualityReport, GlucoseSeriesRequest


class PatternOptions(BaseModel):
    include_rules: bool = Field(True, description="Run the rule-based detectors.")
    include_clusters: bool = Field(True, description="Run K-Means on daily profiles.")
    max_clusters: int = Field(5, ge=2, le=6, description="Upper bound for K (chosen by silhouette).")
    min_days_evaluated: int = Field(
        5, ge=3, le=30, description="A rule needs at least this many evaluable days to report."
    )


class PatternRequest(GlucoseSeriesRequest):
    pattern_options: PatternOptions = Field(default_factory=PatternOptions)


class PatternModel(BaseModel):
    id: str = Field(..., examples=["hypo_overnight"], description="Stable machine-readable ID.")
    category: Literal["hypoglycemia", "hyperglycemia", "dawn_rise", "postmeal_spike", "variability"]
    label: str
    severity: Literal["priority", "attention", "info"]
    window: str = Field(..., examples=["00:00-06:00"], description="Local time window.")
    description: str
    n_days_affected: int
    n_days_evaluated: int
    fraction: float = Field(..., ge=0, le=1)
    example_dates: list[date_type] = Field(..., description="Up to 5 most recent affected days.")
    metrics: dict[str, float]


class ClusterProfileModel(BaseModel):
    cluster_id: int = Field(..., description="0 = most common day type.")
    label: str
    n_days: int
    fraction: float
    centroid_mg_dl: list[float] = Field(..., min_length=24, max_length=24,
                                        description="Hourly mean glucose, local hours 0-23.")
    mean_mg_dl: float
    peak_hour: int
    peak_mg_dl: float
    nadir_hour: int
    nadir_mg_dl: float
    weekend_fraction: float
    dates: list[date_type]


class DayAssignment(BaseModel):
    day: date_type
    cluster_id: int


class ClusteringModel(BaseModel):
    status: Literal["ok", "insufficient_days", "no_distinct_structure"]
    message: str
    n_days: int
    k: int | None
    silhouette: float | None
    candidate_silhouettes: dict[int, float]
    clusters: list[ClusterProfileModel]
    assignments: list[DayAssignment]


class PatternResponse(BaseModel):
    user_id: str
    quality: DataQualityReport
    n_days_profiled: int = Field(..., description="Complete days used for clustering.")
    patterns: list[PatternModel] | None = Field(
        None, description="Sorted by severity then recurrence. null if include_rules=false."
    )
    clustering: ClusteringModel | None = Field(None, description="null if include_clusters=false.")
    disclaimer: str = DISCLAIMER
