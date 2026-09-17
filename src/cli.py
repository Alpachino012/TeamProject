"""Command-line interface for research, demos and local history."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.app import build_researcher
from src.config import Settings, configure_logging
from src.core.researcher import Researcher
from src.exceptions import ResearchAssistantError
from src.models import ResearchRequest, ResearchResult, SourceName, parse_source_names

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI grammar."""

    parser = argparse.ArgumentParser(prog="researcher", description="Async cited research CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="research one question")
    ask.add_argument("question")
    ask.add_argument(
        "--sources",
        default="wiki,arxiv,web",
        help="comma-separated: wiki,arxiv,web",
    )
    ask.add_argument("--no-cache", action="store_true", help="bypass source cache")
    ask.add_argument("--offline", action="store_true", help="use deterministic local providers")

    demo = commands.add_parser("demo", help="run all bundled example questions")
    demo.add_argument(
        "--sources",
        default="wiki,arxiv,web",
        help="comma-separated: wiki,arxiv,web",
    )
    demo.add_argument("--no-cache", action="store_true", help="bypass source cache")
    demo.add_argument("--offline", action="store_true", help="use deterministic local providers")

    history = commands.add_parser("history", help="show recent saved sessions")
    history.add_argument("--limit", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        if args.command == "history":
            return asyncio.run(_run_history(settings, args.limit))
        selected = parse_source_names(args.sources)
        app = build_researcher(settings, offline=args.offline)
        if args.command == "ask":
            request = ResearchRequest(
                question=args.question,
                sources=selected,
                use_cache=not args.no_cache,
            )
            result = asyncio.run(app.ask(request))
            print(render_result(result))
            return 0
        return asyncio.run(_run_demo(app, selected, not args.no_cache))
    except (ResearchAssistantError, ValidationError, ValueError) as exc:
        logger.error("command_failed error=%s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 2


async def _run_demo(
    app: Researcher,
    sources: tuple[SourceName, ...],
    use_cache: bool,
) -> int:
    questions_path = Path(__file__).resolve().parents[1] / "data" / "research_questions.json"
    try:
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchAssistantError(f"cannot load demo questions: {exc}") from exc
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ResearchAssistantError("demo question file has an invalid format")
    for number, question in enumerate(questions, start=1):
        if not isinstance(question, dict) or not isinstance(question.get("text"), str):
            raise ResearchAssistantError("demo question file contains an invalid item")
        request = ResearchRequest(
            question=question["text"],
            sources=sources,
            use_cache=use_cache,
        )
        result = await app.ask(request)
        print(f"\n--- Question {number} of {len(questions)} ---")
        print(render_result(result))
    return 0


async def _run_history(settings: Settings, limit: int) -> int:
    app = build_researcher(settings)
    rows = await app.history(limit)
    if not rows:
        print("No saved research sessions.")
        return 0
    for row in rows:
        timestamp = row.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        print(f"{timestamp}  {row.session_id[:8]}  {row.question}")
    return 0


def render_result(result: ResearchResult) -> str:
    

    lines = [f"Q: {result.question}", "", f"A: {result.answer}"]
    if result.citations:
        lines.extend(["", "References:"])
        for citation in result.citations:
            lines.append(
                f"  [{citation.index}] ({citation.source.origin}) {citation.source.title}"
            )
            lines.append(f"      {citation.source.url}")
    if result.failures:
        lines.extend(["", "Unavailable sources:"])
        for failure in result.failures:
            lines.append(f"  - {failure.source.value}: {failure.reason}")
    hits = sum(1 for timing in result.timings if timing.cache_hit)
    lines.extend(
        [
            "",
            f"Retrieved {result.sources_retrieved} excerpts in {result.total_duration_ms:.1f} ms "
            f"({hits}/{len(result.timings)} cache hits).",
        ]
    )
    if not result.persisted:
        lines.append("History note: this result could not be saved.")
    return "\n".join(lines)
