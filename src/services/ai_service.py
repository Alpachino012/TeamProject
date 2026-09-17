"""Retrying, rate-limited and validated wrapper around the provided ``ai`` package."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import urlparse

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai import AnswerWithCitations, Source, fetch_arxiv, fetch_web, fetch_wikipedia, synthesize
from ai.providers.base import ProviderError
from src.config import Settings
from src.exceptions import ExternalServiceError, InvalidAIResponseError
from src.models import SourceName
from src.services.rate_limiter import AsyncTokenBucket

logger = logging.getLogger(__name__)
T = TypeVar("T")

SourceFetcher = Callable[..., Awaitable[list[Source]]]
Synthesizer = Callable[[str, list[Source]], AnswerWithCitations]


@dataclass(frozen=True)
class AIFunctions:
    """Injectable boundary around the four provided AI functions."""

    wikipedia: SourceFetcher = fetch_wikipedia
    arxiv: SourceFetcher = fetch_arxiv
    web: SourceFetcher = fetch_web
    synthesizer: Synthesizer = synthesize


class AIService:
    """Apply cross-cutting reliability controls to every ``ai.*`` call."""

    def __init__(
        self,
        settings: Settings,
        *,
        functions: AIFunctions | None = None,
        rate_limiter: AsyncTokenBucket | None = None,
    ) -> None:
        self._settings = settings
        self._functions = functions or AIFunctions()
        self._rate_limiter = rate_limiter or AsyncTokenBucket(
            settings.rate_limit_capacity,
            settings.rate_limit_refill_per_second,
        )

    def make_client(self) -> httpx.AsyncClient:
        """Create the shared connection pool used by all source fetchers in one request."""

        return httpx.AsyncClient(
            timeout=self._settings.external_call_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "AIENG-Research-Assistant/1.0"},
        )

    async def fetch_source(
        self,
        source: SourceName,
        question: str,
        *,
        client: httpx.AsyncClient,
    ) -> list[Source]:
        """Fetch one source with rate limiting, retry, timeout, validation and logging."""

        fetcher = self._fetcher_for(source)

        async def call() -> list[Source]:
            await self._rate_limiter.acquire()
            return await asyncio.wait_for(
                fetcher(
                    question,
                    max_results=self._settings.max_sources_per_query,
                    client=client,
                ),
                timeout=self._settings.external_call_timeout_seconds,
            )

        started = time.perf_counter()
        logger.info("source_fetch_started source=%s query_chars=%d", source.value, len(question))
        logger.debug("source_fetch_input source=%s question=%r", source.value, question)
        try:
            raw_sources = await self._run_with_retry(call, operation_name=f"fetch_{source.value}")
        except (ProviderError, httpx.HTTPError, TimeoutError) as exc:
            raise ExternalServiceError(f"{source.value} failed after retries: {exc}") from exc

        sources = self._validate_sources(source, raw_sources)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        logger.info(
            "source_fetch_finished source=%s items=%d duration_ms=%.1f",
            source.value,
            len(sources),
            elapsed_ms,
        )
        logger.debug("source_fetch_output source=%s payload=%r", source.value, sources)
        return sources

    async def synthesize(
        self,
        question: str,
        sources: list[Source],
    ) -> AnswerWithCitations:
        """Synthesize and validate a cited answer without blocking the event loop."""

        async def call() -> AnswerWithCitations:
            await self._rate_limiter.acquire()
            return await asyncio.wait_for(
                asyncio.to_thread(self._functions.synthesizer, question, sources),
                timeout=self._settings.synthesis_timeout_seconds,
            )

        started = time.perf_counter()
        logger.info("synthesis_started sources=%d question_chars=%d", len(sources), len(question))
        try:
            result = await self._run_with_retry(call, operation_name="synthesize")
        except (ProviderError, httpx.HTTPError, TimeoutError) as exc:
            raise ExternalServiceError(f"answer synthesis failed after retries: {exc}") from exc
        self._validate_answer(result, question, sources)
        logger.info("synthesis_finished duration_ms=%.1f", (time.perf_counter() - started) * 1_000)
        logger.debug("synthesis_output payload=%r", result)
        return result

    async def _run_with_retry(
        self,
        call: Callable[[], Awaitable[T]],
        *,
        operation_name: str,
    ) -> T:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._settings.retry_attempts),
            wait=wait_exponential(
                multiplier=self._settings.retry_initial_delay_seconds,
                max=self._settings.retry_max_delay_seconds,
            ),
            retry=retry_if_exception_type((ProviderError, httpx.HTTPError, TimeoutError)),
            before_sleep=lambda state: self._log_retry(state, operation_name),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await call()
        raise RuntimeError("retry loop ended without a result")

    @staticmethod
    def _log_retry(state: RetryCallState, operation: str) -> None:
        failure = state.outcome.exception() if state.outcome else None
        logger.warning(
            "external_call_retry operation=%s attempt=%d error=%s",
            operation,
            state.attempt_number,
            failure,
        )

    def _fetcher_for(self, source: SourceName) -> SourceFetcher:
        if source is SourceName.WIKIPEDIA:
            return self._functions.wikipedia
        if source is SourceName.ARXIV:
            return self._functions.arxiv
        return self._functions.web

    @staticmethod
    def _validate_sources(source: SourceName, items: object) -> list[Source]:
        if not isinstance(items, list):
            raise InvalidAIResponseError(f"{source.value} returned a non-list response")
        valid: list[Source] = []
        for item in items:
            if not isinstance(item, Source):
                logger.warning("source_item_dropped source=%s reason=wrong_type", source.value)
                continue
            parsed = urlparse(item.url)
            if item.origin != source.value or parsed.scheme not in {"http", "https"}:
                logger.warning(
                    "source_item_dropped source=%s reason=invalid_metadata",
                    source.value,
                )
                continue
            if not item.snippet.strip():
                logger.warning("source_item_dropped source=%s reason=empty_snippet", source.value)
                continue
            valid.append(item)
        return valid

    @staticmethod
    def _validate_answer(result: object, question: str, sources: list[Source]) -> None:
        if not isinstance(result, AnswerWithCitations):
            raise InvalidAIResponseError("synthesizer returned an unexpected response type")
        if result.question != question.strip() or not result.answer.strip():
            raise InvalidAIResponseError("synthesizer returned an invalid question or empty answer")
        indices = [citation.index for citation in result.citations]
        invalid_index = any(index < 1 or index > len(sources) for index in indices)
        if len(indices) != len(set(indices)) or invalid_index:
            raise InvalidAIResponseError("synthesizer returned invalid citation indices")
        if any(citation.source != sources[citation.index - 1] for citation in result.citations):
            raise InvalidAIResponseError("synthesizer returned a citation/source mismatch")
