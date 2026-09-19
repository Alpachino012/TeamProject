"""Offline tests for JSON caching and SQLite persistence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai import Citation, Source
from src.exceptions import StorageError
from src.models import ResearchResult, SourceFailure, SourceName, SourceTiming
from src.storage.cache_store import JsonFileSourceCache
from src.storage.session_store import SQLiteResearchRepository


def source(origin: str = "web") -> Source:
    return Source(
        title="Reference",
        url=f"https://example.com/{origin}",
        snippet="A useful excerpt.",
        origin=origin,
    )


@pytest.mark.asyncio
async def test_cache_uses_canonical_source_query_key(tmp_path):
    cache = JsonFileSourceCache(tmp_path / "cache.json", 60)
    await cache.set(SourceName.WEB, "What is AI?", [source()])
    result = await cache.get(SourceName.WEB, "  WHAT is ai ")
    assert result == [source()]


@pytest.mark.asyncio
async def test_cache_expires_and_clears(tmp_path):
    now = [100.0]
    cache = JsonFileSourceCache(tmp_path / "cache.json", 10, clock=lambda: now[0])
    await cache.set(SourceName.WEB, "question", [source()])
    now[0] = 110.0
    assert await cache.get(SourceName.WEB, "question") is None
    now[0] = 100.0
    await cache.set(SourceName.WEB, "question", [source()])
    await cache.clear()
    assert await cache.get(SourceName.WEB, "question") is None


@pytest.mark.asyncio
async def test_cache_ignores_corrupt_file(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("not-json", encoding="utf-8")
    cache = JsonFileSourceCache(path, 60)
    assert await cache.get(SourceName.WEB, "question") is None


@pytest.mark.asyncio
async def test_cache_ignores_wrong_top_level_and_invalid_record(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("[]", encoding="utf-8")
    cache = JsonFileSourceCache(path, 60)
    assert await cache.get(SourceName.WEB, "q") is None
    path.write_text('{"web:q": {"stored_at": "wrong", "sources": []}}', encoding="utf-8")
    assert await cache.get(SourceName.WEB, "q") is None


def make_result() -> ResearchResult:
    cited_source = source()
    return ResearchResult(
        session_id="session-1",
        question="What is AI?",
        answer="A cited answer [1].",
        citations=[Citation(index=1, source=cited_source)],
        failures=[SourceFailure(source=SourceName.ARXIV, reason="timed out")],
        timings=[
            SourceTiming(source=SourceName.WEB, elapsed_ms=12.5, cache_hit=False, item_count=1),
            SourceTiming(source=SourceName.ARXIV, elapsed_ms=20, cache_hit=False, item_count=0),
        ],
        total_duration_ms=35,
        created_at=datetime(2026, 5, 20, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_sqlite_repository_round_trip(tmp_path):
    repository = SQLiteResearchRepository(tmp_path / "research.db")
    await repository.save(make_result())
    rows = await repository.recent()
    assert len(rows) == 1
    assert rows[0].question == "What is AI?"
    assert rows[0].citations[0].source.url == "https://example.com/web"
    assert rows[0].failures[0].source is SourceName.ARXIV
    assert rows[0].sources_retrieved == 1


@pytest.mark.asyncio
async def test_sqlite_repository_rejects_duplicate_session(tmp_path):
    repository = SQLiteResearchRepository(tmp_path / "research.db")
    await repository.save(make_result())
    with pytest.raises(StorageError):
        await repository.save(make_result())


@pytest.mark.asyncio
async def test_sqlite_repository_validates_history_limit(tmp_path):
    repository = SQLiteResearchRepository(tmp_path / "research.db")
    with pytest.raises(ValueError):
        await repository.recent(0)

