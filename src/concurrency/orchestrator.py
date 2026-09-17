"""Concurrent source collection with caching and graceful degradation."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from ai import Source
from src.config import Settings
from src.exceptions import CacheError, ExternalServiceError, InvalidAIResponseError
from src.models import SourceCollection, SourceFailure, SourceName, SourceTiming
from src.services.ai_service import AIService
from src.storage.cache_store import SourceCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SourceOutcome:
    source: SourceName
    items: list[Source]
    failure: SourceFailure | None
    timing: SourceTiming


class SourceOrchestrator:
    """Query selected sources concurrently while isolating each failure."""

    def __init__(self, settings: Settings, ai_service: AIService, cache: SourceCache) -> None:
        self._settings = settings
        self._ai_service = ai_service
        self._cache = cache
        self._semaphore = asyncio.Semaphore(settings.max_parallel)

    async def collect(
        self,
        question: str,
        sources: tuple[SourceName, ...],
        *,
        use_cache: bool,
        concurrent: bool = True,
    ) -> SourceCollection:
        """Collect, deduplicate and report source results."""

        async with self._ai_service.make_client() as client:
            if concurrent:
                gathered = await asyncio.gather(
                    *(self._collect_one(source, question, use_cache, client) for source in sources),
                    return_exceptions=True,
                )
            else:
                gathered = []
                for source in sources:
                    try:
                        gathered.append(
                            await self._collect_one(source, question, use_cache, client)
                        )
                    except (ExternalServiceError, InvalidAIResponseError, TimeoutError) as exc:
                        gathered.append(exc)

        outcomes: list[_SourceOutcome] = []
        for source, result in zip(sources, gathered, strict=True):
            if isinstance(result, BaseException):
                logger.error("source_task_failed source=%s error=%s", source.value, result)
                outcomes.append(
                    _SourceOutcome(
                        source=source,
                        items=[],
                        failure=SourceFailure(source=source, reason=str(result)),
                        timing=SourceTiming(
                            source=source,
                            elapsed_ms=0,
                            cache_hit=False,
                            item_count=0,
                        ),
                    )
                )
            else:
                outcomes.append(result)

        seen_urls: set[str] = set()
        combined: list[Source] = []
        for outcome in outcomes:
            for item in outcome.items:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    combined.append(item)
        return SourceCollection(
            sources=combined,
            failures=[outcome.failure for outcome in outcomes if outcome.failure is not None],
            timings=[outcome.timing for outcome in outcomes],
        )

    async def _collect_one(
        self,
        source: SourceName,
        question: str,
        use_cache: bool,
        client: httpx.AsyncClient,
    ) -> _SourceOutcome:
        started = time.perf_counter()
        if use_cache:
            try:
                cached = await self._cache.get(source, question)
            except CacheError as exc:
                logger.warning("cache_read_failed source=%s error=%s", source.value, exc)
                cached = None
            if cached is not None:
                return self._outcome(source, cached, started, cache_hit=True)

        try:
            async with asyncio.timeout(self._settings.per_source_timeout_seconds):
                async with self._semaphore:
                    items = await self._ai_service.fetch_source(source, question, client=client)
        except (ExternalServiceError, InvalidAIResponseError, TimeoutError) as exc:
            logger.warning("source_unavailable source=%s error=%s", source.value, exc)
            return self._outcome(source, [], started, failure=str(exc))

        if use_cache:
            try:
                await self._cache.set(source, question, items)
            except CacheError as exc:
                logger.warning("cache_write_failed source=%s error=%s", source.value, exc)
        return self._outcome(source, items, started)

    @staticmethod
    def _outcome(
        source: SourceName,
        items: list[Source],
        started: float,
        *,
        cache_hit: bool = False,
        failure: str | None = None,
    ) -> _SourceOutcome:
        return _SourceOutcome(
            source=source,
            items=items,
            failure=SourceFailure(source=source, reason=failure) if failure else None,
            timing=SourceTiming(
                source=source,
                elapsed_ms=(time.perf_counter() - started) * 1_000,
                cache_hit=cache_hit,
                item_count=len(items),
            ),
        )
