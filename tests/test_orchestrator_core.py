"""Concurrency and core-use-case tests."""

from __future__ import annotations

import time
from typing import Any

import pytest

from ai import AnswerWithCitations, Citation, Source
from ai.providers.base import ProviderError
from src.concurrency.orchestrator import SourceOrchestrator
from src.config import Settings
from src.core.researcher import Researcher, _sanitize_answer
from src.exceptions import InvalidQuestionError, NoSourcesError, StorageError
from src.models import ResearchRequest, ResearchResult, SourceName
from src.services.ai_service import AIFunctions, AIService
from src.services.offline import build_offline_functions
from src.storage.cache_store import JsonFileSourceCache
from src.storage.session_store import ResearchRepository, SQLiteResearchRepository


def settings(tmp_path, **changes: Any) -> Settings:
    values = {
        "cache_dir": tmp_path,
        "database_path": tmp_path / "research.db",
        "retry_attempts": 1,
        "retry_initial_delay_seconds": 0,
        "retry_max_delay_seconds": 0,
        "external_call_timeout_seconds": 1,
        "per_source_timeout_seconds": 1,
        "rate_limit_capacity": 100,
        "rate_limit_refill_per_second": 100,
    }
    values.update(changes)
    return Settings(**values)


def make_orchestrator(tmp_path, functions: AIFunctions, **changes: Any) -> SourceOrchestrator:
    config = settings(tmp_path, **changes)
    service = AIService(config, functions=functions)
    cache = JsonFileSourceCache(tmp_path / "cache.json", config.cache_ttl_seconds)
    return SourceOrchestrator(config, service, cache)


@pytest.mark.asyncio
async def test_orchestrator_degrades_when_one_source_fails(tmp_path):
    offline = build_offline_functions(0)

    async def broken(query: str, **kwargs: Any) -> list[Source]:
        raise ProviderError("arXiv unavailable")

    functions = AIFunctions(
        offline.wikipedia,
        broken,
        offline.web,
        offline.synthesizer,
    )
    result = await make_orchestrator(tmp_path, functions).collect(
        "question",
        (SourceName.WIKIPEDIA, SourceName.ARXIV, SourceName.WEB),
        use_cache=False,
    )
    assert len(result.sources) == 2
    assert result.failures[0].source is SourceName.ARXIV
    assert len(result.timings) == 3


@pytest.mark.asyncio
async def test_orchestrator_cache_hit_avoids_second_call(tmp_path):
    calls = 0

    async def counted(query: str, **kwargs: Any) -> list[Source]:
        nonlocal calls
        calls += 1
        return [
            Source(
                title="Reference",
                url="https://example.com/wiki",
                snippet="Excerpt",
                origin="wikipedia",
            )
        ]

    offline = build_offline_functions(0)
    orchestrator = make_orchestrator(
        tmp_path,
        AIFunctions(counted, offline.arxiv, offline.web, offline.synthesizer),
    )
    await orchestrator.collect("What is AI?", (SourceName.WIKIPEDIA,), use_cache=True)
    result = await orchestrator.collect(" WHAT is ai ", (SourceName.WIKIPEDIA,), use_cache=True)
    assert calls == 1
    assert result.timings[0].cache_hit is True


@pytest.mark.asyncio
async def test_concurrent_collection_is_faster_than_sequential(tmp_path):
    functions = build_offline_functions(0.05)
    orchestrator = make_orchestrator(tmp_path, functions)
    source_names = (SourceName.WIKIPEDIA, SourceName.ARXIV, SourceName.WEB)
    started = time.perf_counter()
    await orchestrator.collect("q1", source_names, use_cache=False, concurrent=False)
    sequential = time.perf_counter() - started
    started = time.perf_counter()
    await orchestrator.collect("q2", source_names, use_cache=False, concurrent=True)
    concurrent = time.perf_counter() - started
    assert concurrent < sequential * 0.7


@pytest.mark.asyncio
async def test_orchestrator_reports_unexpected_task_exception(tmp_path):
    offline = build_offline_functions(0)

    async def unexpected(query: str, **kwargs: Any) -> list[Source]:
        raise RuntimeError("bad adapter")

    functions = AIFunctions(
        unexpected,
        offline.arxiv,
        offline.web,
        offline.synthesizer,
    )
    result = await make_orchestrator(tmp_path, functions).collect(
        "question",
        (SourceName.WIKIPEDIA, SourceName.WEB),
        use_cache=False,
    )
    assert len(result.sources) == 1
    assert "bad adapter" in result.failures[0].reason


@pytest.mark.asyncio
async def test_researcher_happy_path_persists_result(tmp_path):
    config = settings(tmp_path)
    functions = build_offline_functions(0)
    service = AIService(config, functions=functions)
    orchestrator = SourceOrchestrator(
        config,
        service,
        JsonFileSourceCache(tmp_path / "cache.json", 60),
    )
    repository = SQLiteResearchRepository(tmp_path / "research.db")
    researcher = Researcher(orchestrator, service, repository, max_question_chars=100)
    result = await researcher.ask(
        ResearchRequest(question="What is AI?", sources=(SourceName.WEB,), use_cache=False)
    )
    assert "[1]" in result.answer
    assert result.persisted is True
    assert (await researcher.history())[0].session_id == result.session_id


@pytest.mark.asyncio
async def test_researcher_rejects_configured_oversize_question(tmp_path):
    config = settings(tmp_path)
    functions = build_offline_functions(0)
    service = AIService(config, functions=functions)
    researcher = Researcher(
        make_orchestrator(tmp_path, functions),
        service,
        SQLiteResearchRepository(tmp_path / "research.db"),
        max_question_chars=5,
    )
    with pytest.raises(InvalidQuestionError):
        await researcher.ask(ResearchRequest(question="a valid but long question"))


@pytest.mark.asyncio
async def test_researcher_fails_cleanly_when_all_sources_empty(tmp_path):
    offline = build_offline_functions(0)

    async def empty(query: str, **kwargs: Any) -> list[Source]:
        return []

    functions = AIFunctions(empty, empty, empty, offline.synthesizer)
    config = settings(tmp_path)
    service = AIService(config, functions=functions)
    researcher = Researcher(
        make_orchestrator(tmp_path, functions),
        service,
        SQLiteResearchRepository(tmp_path / "research.db"),
        max_question_chars=100,
    )
    with pytest.raises(NoSourcesError):
        await researcher.ask(ResearchRequest(question="question"))


class BrokenRepository(ResearchRepository):
    async def save(self, result: ResearchResult) -> None:
        raise StorageError("disk full")

    async def recent(self, limit: int = 10) -> list[ResearchResult]:
        return []


@pytest.mark.asyncio
async def test_researcher_returns_answer_when_history_save_fails(tmp_path):
    config = settings(tmp_path)
    functions = build_offline_functions(0)
    service = AIService(config, functions=functions)
    researcher = Researcher(
        make_orchestrator(tmp_path, functions),
        service,
        BrokenRepository(),
        max_question_chars=100,
    )
    result = await researcher.ask(
        ResearchRequest(question="question", sources=(SourceName.WEB,), use_cache=False)
    )
    assert result.persisted is False


def test_sanitize_answer_removes_controls_and_dangling_citations():
    source = Source(title="T", url="https://example.com", snippet="S", origin="web")
    answer = AnswerWithCitations(
        question="q",
        answer="Claim\x00 [1,99] and bad [88].",
        citations=[Citation(index=1, source=source)],
    )
    result = _sanitize_answer(answer, 1)
    assert result.answer == "Claim [1] and bad ."
