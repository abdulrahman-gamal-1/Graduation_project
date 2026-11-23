from fastapi import APIRouter, Depends

from app.api.v1 import analytics, anomalies, patterns, preprocessing
from app.core.security import require_internal_key

# Every v1 route requires the internal API key (adds the "Authorize" button in Swagger).
api_v1 = APIRouter(prefix="/api/v1", dependencies=[Depends(require_internal_key)])
api_v1.include_router(preprocessing.router)
api_v1.include_router(analytics.router)
api_v1.include_router(patterns.router)
api_v1.include_router(anomalies.router)
