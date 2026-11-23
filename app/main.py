import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1
from app.core.config import settings
from app.core.errors import GlucoCareError, InsufficientDataError
from app.schemas.common import HealthResponse

VERSION = "0.1.0"
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# Privacy: we never log request bodies (they contain health data).

TAGS = [
    {"name": "Health", "description": "Liveness/readiness probes (no auth)."},
    {"name": "Preprocessing", "description": "Cleaning, resampling, feature extraction."},
    {"name": "Descriptive analytics", "description": "TIR, SD, CV, GMI, rolling stats, AGP."},
    {"name": "Pattern detection", "description": "Rule-based patterns + K-Means day types."},
    {"name": "Anomaly detection", "description": "Safety rules + Isolation Forest / robust z-score."},
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="GlucoCare AI Service",
        version=VERSION,
        summary="Internal analytics/ML service consumed by the GlucoCare Node.js backend.",
        description=(
            "**Internal only** - not exposed to end users. All `/api/v1` routes require the "
            "`X-Internal-API-Key` header.\n\n"
            "All glucose values are **mg/dL**. Timestamps are ISO-8601 (UTC preferred). "
            "Outputs are informational analytics, **not a medical diagnosis**."
        ),
        openapi_tags=TAGS,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(GlucoCareError)
    async def domain_error_handler(request: Request, exc: GlucoCareError):
        code = "insufficient_data" if isinstance(exc, InsufficientDataError) else "domain_error"
        return JSONResponse(
            status_code=422,
            content={"error": code, "detail": str(exc),
                     "request_id": getattr(request.state, "request_id", None)},
        )

    @app.get("/health", response_model=HealthResponse, tags=["Health"], summary="Liveness probe")
    def health() -> HealthResponse:
        return HealthResponse(version=VERSION)

    app.include_router(api_v1)
    return app


app = create_app()



