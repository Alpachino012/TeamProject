"""Typed application settings loaded from environment variables and ``.env``."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings shared by every application layer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.0-flash"
    web_search_provider: str = "tavily"

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    serper_api_key: SecretStr | None = None

    log_level: str = "INFO"
    cache_dir: Path = Path(".cache")
    cache_ttl_seconds: int = Field(default=86_400, ge=1)
    database_path: Path = Path("research.db")
    per_source_timeout_seconds: float = Field(default=20.0, gt=0)
    synthesis_timeout_seconds: float = Field(default=30.0, gt=0)
    external_call_timeout_seconds: float = Field(default=6.0, gt=0)
    max_sources_per_query: int = Field(default=3, ge=1, le=10)
    max_parallel: int = Field(default=3, ge=1, le=20)
    max_question_chars: int = Field(default=1_000, ge=20, le=2_000)
    retry_attempts: int = Field(default=3, ge=1, le=8)
    retry_initial_delay_seconds: float = Field(default=0.5, ge=0, le=30)
    retry_max_delay_seconds: float = Field(default=4.0, ge=0, le=60)
    rate_limit_capacity: int = Field(default=6, ge=1, le=100)
    rate_limit_refill_per_second: float = Field(default=2.0, gt=0, le=100)

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"anthropic", "openai", "google", "gemini"}:
            raise ValueError("LLM_PROVIDER must be anthropic, openai, or gemini")
        return normalized

    @field_validator("web_search_provider")
    @classmethod
    def validate_web_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"tavily", "serper", "duckduckgo", "ddg"}:
            raise ValueError("WEB_SEARCH_PROVIDER must be tavily, serper, or duckduckgo")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return normalized

    @field_validator("retry_max_delay_seconds")
    @classmethod
    def validate_retry_delays(cls, value: float, info: object) -> float:
        data = getattr(info, "data", {})
        initial = data.get("retry_initial_delay_seconds", 0)
        if value < initial:
            raise ValueError("RETRY_MAX_DELAY_SECONDS cannot be below the initial delay")
        return value


def configure_logging(level: str) -> None:
    """Configure application logging once at the selected level."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

