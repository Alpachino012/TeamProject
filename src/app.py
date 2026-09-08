from __future__ import annotations
from src.concurrency.orchestrator import SourceOrchestrator
from src.config import Settings
from src.core.researcher import Researcher
from src.services.ai_service import AIService
from src.services.offline import build_offline_functions
from src.storage.cache_store import JsonFileSourceCache
from src.storage.session_store import SQLiteResearchRepository


def build_researcher(settings: Settings, *, offline: bool = False) -> Researcher:
    functions = build_offline_functions() if offline else None
    ai_service = AIService(settings, functions=functions)
    cache = JsonFileSourceCache(
        settings.cache_dir / "sources.json",
        settings.cache_ttl_seconds,
    )
    repository = SQLiteResearchRepository(settings.database_path)
    orchestrator = SourceOrchestrator(settings, ai_service, cache)
    return Researcher(
        orchestrator,
        ai_service,
        repository,
        max_question_chars=settings.max_question_chars,
    )

