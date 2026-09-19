"""SQLite repository for completed research sessions."""

from __future__ import annotations

import asyncio
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from pathlib import Path

from ai import Citation, Source
from src.exceptions import StorageError
from src.models import ResearchResult, SourceFailure, SourceName, SourceTiming


class ResearchRepository(ABC):
    """Contract for storing and retrieving completed research sessions."""

    @abstractmethod
    async def save(self, result: ResearchResult) -> None:
        """Persist one completed research result."""

    @abstractmethod
    async def recent(self, limit: int = 10) -> list[ResearchResult]:
        """Return completed sessions from newest to oldest."""


class SQLiteResearchRepository(ResearchRepository):
    """Normalized SQLite storage with foreign-key cleanup."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._initialize()

    async def save(self, result: ResearchResult) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save_sync, result)

    async def recent(self, limit: int = 10) -> list[ResearchResult]:
        if not 1 <= limit <= 100:
            raise ValueError("history limit must be between 1 and 100")
        async with self._lock:
            return await asyncio.to_thread(self._recent_sync, limit)

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self._path, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open research database {self._path}: {exc}") from exc

    def _initialize(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection, connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS research_sessions (
                        session_id TEXT PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        total_duration_ms REAL NOT NULL CHECK (total_duration_ms >= 0),
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_sessions_created_at
                        ON research_sessions(created_at DESC);
                    CREATE TABLE IF NOT EXISTS citations (
                        session_id TEXT NOT NULL,
                        citation_index INTEGER NOT NULL CHECK (citation_index > 0),
                        origin TEXT NOT NULL CHECK (origin IN ('wikipedia', 'arxiv', 'web')),
                        title TEXT NOT NULL,
                        url TEXT NOT NULL,
                        snippet TEXT NOT NULL,
                        PRIMARY KEY (session_id, citation_index),
                        FOREIGN KEY (session_id) REFERENCES research_sessions(session_id)
                            ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS source_events (
                        session_id TEXT NOT NULL,
                        source TEXT NOT NULL CHECK (source IN ('wikipedia', 'arxiv', 'web')),
                        status TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
                        elapsed_ms REAL NOT NULL CHECK (elapsed_ms >= 0),
                        cache_hit INTEGER NOT NULL CHECK (cache_hit IN (0, 1)),
                        item_count INTEGER NOT NULL CHECK (item_count >= 0),
                        error TEXT,
                        PRIMARY KEY (session_id, source),
                        FOREIGN KEY (session_id) REFERENCES research_sessions(session_id)
                            ON DELETE CASCADE
                    );
                    """
                )
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"cannot initialize research database {self._path}: {exc}") from exc

    def _save_sync(self, result: ResearchResult) -> None:
        failure_reasons = {failure.source: failure.reason for failure in result.failures}
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO research_sessions VALUES (?, ?, ?, ?, ?)",
                    (
                        result.session_id,
                        result.question,
                        result.answer,
                        result.total_duration_ms,
                        result.created_at.isoformat(),
                    ),
                )
                connection.executemany(
                    "INSERT INTO citations VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            result.session_id,
                            citation.index,
                            citation.source.origin,
                            citation.source.title,
                            citation.source.url,
                            citation.source.snippet,
                        )
                        for citation in result.citations
                    ],
                )
                connection.executemany(
                    "INSERT INTO source_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            result.session_id,
                            timing.source.value,
                            "failed" if timing.source in failure_reasons else "ok",
                            timing.elapsed_ms,
                            int(timing.cache_hit),
                            timing.item_count,
                            failure_reasons.get(timing.source),
                        )
                        for timing in result.timings
                    ],
                )
        except sqlite3.Error as exc:
            raise StorageError(f"cannot save research session: {exc}") from exc

    def _recent_sync(self, limit: int) -> list[ResearchResult]:
        try:
            with closing(self._connect()) as connection:
                sessions = connection.execute(
                    "SELECT * FROM research_sessions ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
                return [self._result_from_rows(connection, session) for session in sessions]
        except sqlite3.Error as exc:
            raise StorageError(f"cannot read research history: {exc}") from exc

    @staticmethod
    def _result_from_rows(connection: sqlite3.Connection, session: sqlite3.Row) -> ResearchResult:
        citation_rows = connection.execute(
            "SELECT * FROM citations WHERE session_id = ? ORDER BY citation_index",
            (session["session_id"],),
        ).fetchall()
        event_rows = connection.execute(
            "SELECT * FROM source_events WHERE session_id = ? ORDER BY source",
            (session["session_id"],),
        ).fetchall()
        return ResearchResult(
            session_id=session["session_id"],
            question=session["question"],
            answer=session["answer"],
            citations=[
                Citation(
                    index=row["citation_index"],
                    source=Source(
                        title=row["title"],
                        url=row["url"],
                        snippet=row["snippet"],
                        origin=row["origin"],
                    ),
                )
                for row in citation_rows
            ],
            failures=[
                SourceFailure(source=SourceName(row["source"]), reason=row["error"])
                for row in event_rows
                if row["status"] == "failed"
            ],
            timings=[
                SourceTiming(
                    source=SourceName(row["source"]),
                    elapsed_ms=row["elapsed_ms"],
                    cache_hit=bool(row["cache_hit"]),
                    item_count=row["item_count"],
                )
                for row in event_rows
            ],
            total_duration_ms=session["total_duration_ms"],
            created_at=session["created_at"],
        )
