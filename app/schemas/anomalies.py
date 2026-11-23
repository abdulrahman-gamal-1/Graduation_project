from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.core.constants import DISCLAIMER
from app.schemas.glucose import DataQualityReport, GlucoseSeriesRequest

EventType = Literal["urgent_low", "low", "urgent_high", "rapid_fall", "rapid_rise",
                    "statistical_outlier"]


class AnomalyOptions(BaseModel):
    method: Literal["auto", "isolation_forest", "robust_z"] = Field(
        "auto", description="auto = Isolation Forest when >= 3 days of data, else robust z-score."
    )
    include_statistical: bool = Field(
        True, description="false = deterministic safety rules only."
    )
    contamination: float = Field(
        0.01, ge=0.001, le=0.05,
        description="Share of points Isolation Forest proposes as candidates. Candidates are "
        "only reported if a robust z-score also confirms them.",
    )
    merge_gap_minutes: int = Field(15, ge=5, le=60, description="Merge flagged points this close.")
    max_events: int = Field(100, ge=1, le=500)


class AnomalyRequest(GlucoseSeriesRequest):
    anomaly_options: AnomalyOptions = Field(default_factory=AnomalyOptions)


class MethodModel(BaseModel):
    mode: Literal["isolation_forest", "robust_z", "rules_only"]
    reason: str | None = Field(None, description="Why a fallback/mode was used, if applicable.")
    n_points_scored: int
    n_candidates: int
    n_confirmed: int


class AnomalyEventModel(BaseModel):
    id: str = Field(..., examples=["ev_20260105T0300"])
    primary_type: EventType = Field(..., description="Headline type (levels outrank rates).")
    types: list[EventType]
    severity: Literal["priority", "attention", "info"]
    methods: list[Literal["rule", "isolation_forest", "robust_z"]]
    start: datetime = Field(..., description="UTC. For rate types: when the 15-min rate threshold was met.")
    end: datetime
    local_start: datetime = Field(..., description="Start in the patient's timezone.")
    duration_minutes: float
    glucose_min: float = Field(..., description="Over the event +/- 15 min of context.")
    glucose_max: float
    peak_rate_mg_dl_per_min: float | None = Field(None, description="Signed; negative = falling.")
    max_abs_robust_z: float | None
    isolation_score: float | None = Field(None, description="Higher = more isolated (IF mode only).")
    driver: Literal["resid_level", "resid_roc"] | None = Field(
        None, description="Which deviation drove a statistical flag."
    )
    description: str


class AnomalyResponse(BaseModel):
    user_id: str
    quality: DataQualityReport
    method: MethodModel
    n_events_total: int
    counts_by_type: dict[str, int]
    events_by_local_hour: list[int] = Field(..., min_length=24, max_length=24)
    n_isolated_points_ignored: int = Field(
        ..., description="Points that tripped a detector but were dropped as single-point artifacts."
    )
    events: list[AnomalyEventModel] = Field(..., description="Priority first, then most recent.")
    note: str = ("Retrospective analysis of the submitted data. This is not a real-time alarm "
                 "system and must not be used for urgent decisions.")
    disclaimer: str = DISCLAIMER
