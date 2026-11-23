"""Runtime settings, read from environment variables (prefix GLUCOCARE_)."""
import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(f"GLUCOCARE_{name}", default)


@dataclass(frozen=True)
class Settings:
    # "dev" disables auth when no key is configured. Anything else fails closed.
    env: str = field(default_factory=lambda: _env("ENV", "prod") or "prod")
    internal_api_key: str | None = field(default_factory=lambda: _env("INTERNAL_API_KEY","123456"))
    max_readings_per_request: int = field(
        default_factory=lambda: int(_env("MAX_READINGS", "50000") or 50000)
    )
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO") or "INFO")


settings = Settings()
