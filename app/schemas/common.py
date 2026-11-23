from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(..., examples=["insufficient_data"])
    detail: str = Field(..., examples=["No valid glucose readings after cleaning."])
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str = Field("ok", examples=["ok"])
    service: str = "glucocare-ai"
    version: str
