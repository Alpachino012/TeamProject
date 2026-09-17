"""Pydantic models passed between the application's own modules."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai import Citation, Source


class SourceName(StrEnum):
    """Source identifiers accepted by the CLI and orchestration layer."""

    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    WEB = "web"


class ResearchRequest(BaseModel):
    """A validated request to research one question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1, max_length=2_000)
    sources: tuple[SourceName, ...] = (
        SourceName.WIKIPEDIA,
        SourceName.ARXIV,
        SourceName.WEB,
    )
    use_cache: bool = True

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        if any(unicodedata.category(char) == "Cc" and not char.isspace() for char in value):
            raise ValueError("question contains unsupported control characters")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized

    @field_validator("sources")
    @classmethod
    def require_sources(cls, value: tuple[SourceName, ...]) -> tuple[SourceName, ...]:
        if not value:
            raise ValueError("at least one source must be selected")
        return tuple(dict.fromkeys(value))


class SourceFailure(BaseModel):
    """A source that could not contribute to the answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SourceName
    reason: str


class SourceTiming(BaseModel):
    """Timing and cache metadata for one source call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SourceName
    elapsed_ms: float = Field(ge=0)
    cache_hit: bool
    item_count: int = Field(ge=0)


class SourceCollection(BaseModel):
    """Combined result of a multi-source fetch."""

    model_config = ConfigDict(extra="forbid")

    sources: list[Source] = Field(default_factory=list)
    failures: list[SourceFailure] = Field(default_factory=list)
    timings: list[SourceTiming] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """Persisted result returned by the business layer."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    failures: list[SourceFailure] = Field(default_factory=list)
    timings: list[SourceTiming] = Field(default_factory=list)
    total_duration_ms: float = Field(ge=0)
    created_at: datetime
    persisted: bool = True

    @property
    def sources_retrieved(self) -> int:
        """Return the number of excerpts collected across source calls."""

        return sum(timing.item_count for timing in self.timings)


_SOURCE_ALIASES = {
    "wiki": SourceName.WIKIPEDIA,
    "wikipedia": SourceName.WIKIPEDIA,
    "arxiv": SourceName.ARXIV,
    "web": SourceName.WEB,
}


def parse_source_names(value: str) -> tuple[SourceName, ...]:
    """Parse a comma-separated source list and reject unknown names."""

    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--sources must contain wiki, arxiv, or web")
    unknown = [part for part in parts if part not in _SOURCE_ALIASES]
    if unknown:
        raise ValueError(f"unknown source: {unknown[0]}; choose wiki, arxiv, or web")
    return tuple(dict.fromkeys(_SOURCE_ALIASES[part] for part in parts))


def canonicalize_query(value: str) -> str:
    """Create a stable cache key for queries differing only by case or punctuation."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.rstrip(" ?.!;:")
