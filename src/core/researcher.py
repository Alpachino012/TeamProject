"""End-to-end research use case."""

from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import UTC, datetime

from ai import AnswerWithCitations
from src.concurrency.orchestrator import SourceOrchestrator
from src.exceptions import InvalidQuestionError, NoSourcesError, StorageError
from src.models import ResearchRequest, ResearchResult
from src.services.ai_service import AIService
from src.storage.session_store import ResearchRepository

logger = logging.getLogger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class Researcher:
    """Coordinate collection, synthesis, sanitization and persistence."""

    def __init__(
        self,
        orchestrator: SourceOrchestrator,
        ai_service: AIService,
        repository: ResearchRepository,
        *,
        max_question_chars: int,
    ) -> None:
        self._orchestrator = orchestrator
        self._ai_service = ai_service
        self._repository = repository
        self._max_question_chars = max_question_chars

    async def ask(self, request: ResearchRequest) -> ResearchResult:
        """Run one complete research request and persist the result."""

        if len(request.question) > self._max_question_chars:
            raise InvalidQuestionError(
                f"question is too long; maximum is {self._max_question_chars} characters"
            )
        started = time.perf_counter()
        collection = await self._orchestrator.collect(
            request.question,
            request.sources,
            use_cache=request.use_cache,
        )
        if not collection.sources:
            details = "; ".join(
                f"{failure.source.value}: {failure.reason}" for failure in collection.failures
            )
            suffix = f" ({details})" if details else ""
            raise NoSourcesError(f"no usable sources were retrieved{suffix}")

        answer = await self._ai_service.synthesize(request.question, collection.sources)
        sanitized = _sanitize_answer(answer, len(collection.sources))
        result = ResearchResult(
            session_id=uuid.uuid4().hex,
            question=sanitized.question,
            answer=sanitized.answer,
            citations=sanitized.citations,
            failures=collection.failures,
            timings=collection.timings,
            total_duration_ms=(time.perf_counter() - started) * 1_000,
            created_at=datetime.now(UTC),
        )
        try:
            await self._repository.save(result)
        except StorageError as exc:
            logger.error("history_save_failed session_id=%s error=%s", result.session_id, exc)
            return result.model_copy(update={"persisted": False})
        return result

    async def history(self, limit: int = 10) -> list[ResearchResult]:
        """Return recent persisted research sessions."""

        return await self._repository.recent(limit)


def _sanitize_answer(answer: AnswerWithCitations, source_count: int) -> AnswerWithCitations:
    """Remove terminal controls and dangling citation numbers from model prose."""

    clean = _CONTROL_CHARS.sub("", answer.answer).strip()

    def valid_group(match: re.Match[str]) -> str:
        indices = [int(piece.strip()) for piece in match.group(1).split(",")]
        valid = [index for index in indices if 1 <= index <= source_count]
        return f"[{','.join(str(index) for index in valid)}]" if valid else ""

    clean = _CITATION_GROUP.sub(valid_group, clean)
    return answer.model_copy(update={"answer": clean})

