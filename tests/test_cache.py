"""Tests for the JSON source cache."""

from __future__ import annotations

import pytest

from ai import Source
from src.models import SourceName
from src.storage.cache_store import JsonFileSourceCache


def source() -> Source:
    return Source(
        title="Reference",
        url="https://example.com/web",
        snippet="A useful excerpt.",
        origin="web",
    )


@pytest.mark.asyncio
async def test_cache_uses_canonical_source_query_key(tmp_path):
    cache = JsonFileSourceCache(tmp_path / "cache.json", 60)
    await cache.set(SourceName.WEB, "What is AI?", [source()])

    assert await cache.get(SourceName.WEB, "  WHAT is ai ") == [source()]


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
async def test_cache_ignores_corrupt_or_invalid_file(tmp_path):
    path = tmp_path / "cache.json"
    cache = JsonFileSourceCache(path, 60)

    path.write_text("not-json", encoding="utf-8")
    assert await cache.get(SourceName.WEB, "question") is None

    path.write_text('{"web:q": {"stored_at": "wrong", "sources": []}}', encoding="utf-8")
    assert await cache.get(SourceName.WEB, "q") is None
