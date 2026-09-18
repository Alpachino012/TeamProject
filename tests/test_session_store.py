"""Tests for persistent research history."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai import Citation, Source
from src.exceptions import StorageError
from src.models import ResearchResult, SourceFailure, SourceName, SourceTiming
from src.storage.session_store import SQLiteResearchRepository


def make_result() -> ResearchResult:
    cited_source = Source(
        title="Reference",
        url="https://example.com/web",
        snippet="A useful excerpt.",
        origin="web",
    )
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

    with pytest.raises(ValueError, match="history limit"):
        await repository.recent(0)
