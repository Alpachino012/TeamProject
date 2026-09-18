"""TTL-aware persistent source cache."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from ai import Source
from src.exceptions import CacheError
from src.models import SourceName, canonicalize_query

logger = logging.getLogger(__name__)


class CacheRecord(BaseModel):
    """Serialized source results and their insertion timestamp."""

    model_config = ConfigDict(extra="forbid")

    stored_at: float
    sources: list[Source]


class SourceCache(ABC):
    """Contract used by the orchestration layer for source-result caching."""

    @abstractmethod
    async def get(self, source: SourceName, query: str) -> list[Source] | None:
        """Return an unexpired entry, or ``None`` on a miss."""

    @abstractmethod
    async def set(self, source: SourceName, query: str, sources: list[Source]) -> None:
        """Persist a source result under its canonical key."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove every cached entry."""


class JsonFileSourceCache(SourceCache):
    """An atomic JSON cache intended for one CLI process at a time."""

    def __init__(
        self,
        path: Path,
        ttl_seconds: int,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = asyncio.Lock()

    async def get(self, source: SourceName, query: str) -> list[Source] | None:
        async with self._lock:
            records = await asyncio.to_thread(self._read_records)
            raw = records.get(self._key(source, query))
            if raw is None:
                return None
            try:
                record = CacheRecord.model_validate(raw)
            except ValidationError:
                logger.warning("cache_record_invalid source=%s", source.value)
                return None
            if self._clock() - record.stored_at >= self._ttl_seconds:
                return None
            return record.sources

    async def set(self, source: SourceName, query: str, sources: list[Source]) -> None:
        async with self._lock:
            records = await asyncio.to_thread(self._read_records)
            records[self._key(source, query)] = CacheRecord(
                stored_at=self._clock(), sources=sources
            ).model_dump(mode="json")
            await asyncio.to_thread(self._write_records, records)

    async def clear(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write_records, {})

    def _read_records(self) -> dict[str, object]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("cache_file_corrupt path=%s", self._path)
            return {}
        except OSError as exc:
            raise CacheError(f"cannot read cache file {self._path}: {exc}") from exc
        if not isinstance(raw, dict):
            logger.warning("cache_file_invalid path=%s", self._path)
            return {}
        return raw

    def _write_records(self, records: dict[str, object]) -> None:
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self._path)
        except OSError as exc:
            raise CacheError(f"cannot write cache file {self._path}: {exc}") from exc

    @staticmethod
    def _key(source: SourceName, query: str) -> str:
        return f"{source.value}:{canonicalize_query(query)}"
