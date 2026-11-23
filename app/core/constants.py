"""Clinical constants. Single source of truth - never hard-code thresholds elsewhere.

References: International Consensus on Time in Range (Battelino et al., Diabetes Care 2019)
and GMI formula (Bergenstal et al., Diabetes Care 2018).
"""

# Physical plausibility range of CGM sensors (mg/dL). Readings outside are clipped.
PHYSICAL_MIN_MG_DL = 38.0
PHYSICAL_MAX_MG_DL = 400.0

# Time-in-Range band boundaries (mg/dL)
VERY_LOW_BELOW = 54.0      # Level 2 hypoglycaemia:  < 54
TARGET_LOW = 70.0          # Level 1 hypoglycaemia:  54 - 69
TARGET_HIGH = 180.0        # In range:               70 - 180
VERY_HIGH_ABOVE = 250.0    # Level 1 hyper: 181-250, Level 2 hyper: > 250

# CGM data sufficiency (consensus: >=14 days, >=70% of expected readings)
MIN_DAYS_FOR_CONSENSUS = 14
MIN_COVERAGE_PCT_FOR_CONSENSUS = 70.0

# Glycaemic variability: CV <= 36% is considered "stable"
CV_STABLE_THRESHOLD_PCT = 36.0

# Minimum duration for a glycaemic excursion to count as an "event"
EVENT_MIN_DURATION_MINUTES = 15

# Time-of-day blocks (local hours, [start, end))
TIME_BLOCKS = {
    "overnight": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}

FEATURE_SCHEMA_VERSION = "1.0"

DISCLAIMER = "For informational purposes only, not a medical diagnosis."
