"""Reproducible offline sequential-vs-concurrent source benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.concurrency.orchestrator import SourceOrchestrator
from src.config import Settings
from src.models import SourceName
from src.services.ai_service import AIService
from src.services.offline import build_offline_functions
from src.services.rate_limiter import AsyncTokenBucket
from src.storage.cache_store import JsonFileSourceCache


def load_questions(iterations: int) -> list[str]:
    """Load the supplied questions, cycling only when a larger N is requested."""

    path = PROJECT_ROOT / "data" / "research_questions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    available = [item["text"] for item in payload["questions"]]
    return [available[index % len(available)] for index in range(iterations)]


async def measure(*, questions: list[str], delay: float, concurrent: bool) -> float:
    """Measure total wall time for identical no-cache source work."""

    samples: list[float] = []
    with TemporaryDirectory() as temporary:
        settings = Settings(
            cache_dir=Path(temporary),
            max_parallel=3,
            retry_initial_delay_seconds=0,
            retry_max_delay_seconds=0,
            rate_limit_capacity=100,
            rate_limit_refill_per_second=100,
        )
        service = AIService(
            settings,
            functions=build_offline_functions(delay_seconds=delay),
            rate_limiter=AsyncTokenBucket(100, 100),
        )
        cache = JsonFileSourceCache(Path(temporary) / "bench.json", 60)
        orchestrator = SourceOrchestrator(settings, service, cache)
        for question in questions:
            started = time.perf_counter()
            await orchestrator.collect(
                question,
                (SourceName.WIKIPEDIA, SourceName.ARXIV, SourceName.WEB),
                use_cache=False,
                concurrent=concurrent,
            )
            samples.append(time.perf_counter() - started)
    return sum(samples)


async def run(iterations: int, delay: float) -> None:
    """Run both modes and print a compact comparison table."""

    questions = load_questions(iterations)
    sequential = await measure(questions=questions, delay=delay, concurrent=False)
    parallel = await measure(questions=questions, delay=delay, concurrent=True)
    speedup = sequential / parallel
    print("Mode        N   Wall seconds")
    print(f"Sequential {iterations:>3}   {sequential:.3f}")
    print(f"Parallel   {iterations:>3}   {parallel:.3f}")
    print(f"Speedup          {speedup:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.10)
    args = parser.parse_args()
    if args.iterations < 1 or args.delay < 0:
        parser.error("iterations must be positive and delay cannot be negative")
    asyncio.run(run(args.iterations, args.delay))


if __name__ == "__main__":
    main()
