"""Tests for rate limiting, retries and response validation."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai import AnswerWithCitations, Source
from ai.providers.base import ProviderError
from src.config import Settings
from src.exceptions import ExternalServiceError, InvalidAIResponseError
from src.models import SourceName
from src.services.ai_service import AIFunctions, AIService
from src.services.offline import build_offline_functions
from src.services.rate_limiter import AsyncTokenBucket


def settings(**changes: Any) -> Settings:
    values = {
        "retry_attempts": 3,
        "retry_initial_delay_seconds": 0,
        "retry_max_delay_seconds": 0,
        "external_call_timeout_seconds": 0.2,
        "synthesis_timeout_seconds": 0.2,
        "rate_limit_capacity": 20,
        "rate_limit_refill_per_second": 20,
    }
    values.update(changes)
    return Settings(**values)


def valid_source(origin: str) -> Source:
    return Source(
        title="Title",
        url=f"https://example.com/{origin}",
        snippet="Excerpt",
        origin=origin,
    )


@pytest.mark.asyncio
async def test_fetch_source_retries_transient_provider_error():
    calls = 0

    async def flaky(query: str, **kwargs: Any) -> list[Source]:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderError("temporary")
        return [valid_source("wikipedia")]

    functions = build_offline_functions(0)
    service = AIService(
        settings(),
        functions=AIFunctions(flaky, functions.arxiv, functions.web, functions.synthesizer),
    )
    async with service.make_client() as client:
        result = await service.fetch_source(SourceName.WIKIPEDIA, "question", client=client)
    assert calls == 3
    assert result[0].origin == "wikipedia"


@pytest.mark.asyncio
async def test_fetch_source_converts_exhausted_error():
    async def broken(query: str, **kwargs: Any) -> list[Source]:
        raise ProviderError("down")

    functions = build_offline_functions(0)
    service = AIService(
        settings(retry_attempts=1),
        functions=AIFunctions(broken, functions.arxiv, functions.web, functions.synthesizer),
    )
    async with service.make_client() as client:
        with pytest.raises(ExternalServiceError, match="failed after retries"):
            await service.fetch_source(SourceName.WIKIPEDIA, "question", client=client)


@pytest.mark.asyncio
async def test_fetch_source_times_out():
    async def slow(query: str, **kwargs: Any) -> list[Source]:
        await asyncio.sleep(0.1)
        return []

    functions = build_offline_functions(0)
    service = AIService(
        settings(retry_attempts=1, external_call_timeout_seconds=0.01),
        functions=AIFunctions(slow, functions.arxiv, functions.web, functions.synthesizer),
    )
    async with service.make_client() as client:
        with pytest.raises(ExternalServiceError):
            await service.fetch_source(SourceName.WIKIPEDIA, "question", client=client)


@pytest.mark.asyncio
async def test_fetch_source_drops_semantically_invalid_items():
    async def mixed(query: str, **kwargs: Any) -> list[Source]:
        return [
            valid_source("wikipedia"),
            Source(title="Wrong", url="not-a-url", snippet="x", origin="wikipedia"),
            Source(title="Wrong origin", url="https://example.com", snippet="x", origin="web"),
        ]

    functions = build_offline_functions(0)
    service = AIService(
        settings(),
        functions=AIFunctions(mixed, functions.arxiv, functions.web, functions.synthesizer),
    )
    async with service.make_client() as client:
        result = await service.fetch_source(SourceName.WIKIPEDIA, "question", client=client)
    assert result == [valid_source("wikipedia")]


@pytest.mark.asyncio
async def test_fetch_source_rejects_non_list_response():
    async def malformed(query: str, **kwargs: Any) -> list[Source]:
        return "wrong"  # type: ignore[return-value]

    functions = build_offline_functions(0)
    service = AIService(
        settings(),
        functions=AIFunctions(malformed, functions.arxiv, functions.web, functions.synthesizer),
    )
    async with service.make_client() as client:
        with pytest.raises(InvalidAIResponseError):
            await service.fetch_source(SourceName.WIKIPEDIA, "question", client=client)


@pytest.mark.asyncio
async def test_synthesize_validates_response_type():
    functions = build_offline_functions(0)

    def malformed(question: str, sources: list[Source]) -> AnswerWithCitations:
        return "wrong"  # type: ignore[return-value]

    service = AIService(
        settings(),
        functions=AIFunctions(functions.wikipedia, functions.arxiv, functions.web, malformed),
    )
    with pytest.raises(InvalidAIResponseError):
        await service.synthesize("question", [valid_source("web")])


@pytest.mark.asyncio
async def test_token_bucket_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        AsyncTokenBucket(0, 1)
    with pytest.raises(ValueError):
        AsyncTokenBucket(1, 0)

