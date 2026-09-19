"""Diagnose the live pipeline one stage at a time, without running a full query.

``researcher ask`` always ends in synthesis, so a missing LLM credit hides whether retrieval
is healthy. This script probes each stage independently and reports which ones work, so a
failure can be attributed to configuration, a single source, or the LLM account.

    python scripts/check_live.py              # probe sources only (free)
    python scripts/check_live.py --with-llm   # also send one tiny billable LLM request
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings  # noqa: E402
from src.models import SourceName  # noqa: E402
from src.services.ai_service import AIService  # noqa: E402
from src.services.live_sources import build_live_functions  # noqa: E402

OK = "PASS"
BAD = "FAIL"


def report_configuration(settings: Settings) -> None:
    """Print the resolved settings, showing only whether each key is present."""

    keys = {
        "openai": settings.openai_api_key,
        "gemini": settings.google_api_key,
        "anthropic": settings.anthropic_api_key,
    }
    selected = keys.get("gemini" if settings.llm_provider == "google" else settings.llm_provider)
    print("Configuration")
    print(f"  cwd            : {Path.cwd()}")
    print(f"  .env found     : {(Path.cwd() / '.env').is_file()}")
    print(f"  llm            : {settings.llm_provider} / {settings.llm_model}")
    print(f"  llm key present: {selected is not None and bool(selected.get_secret_value())}")
    print(f"  web provider   : {settings.web_search_provider}")
    print()


async def probe_sources(settings: Settings) -> dict[SourceName, str]:
    """Fetch from each source live and summarise the outcome per source."""

    service = AIService(settings, functions=build_live_functions(settings))
    question = "What is photosynthesis?"
    results: dict[SourceName, str] = {}

    print(f"Source retrieval (query: {question!r})")
    async with service.make_client() as client:
        for source in SourceName:
            try:
                found = await service.fetch_source(source, question, client=client)
            except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
                results[source] = f"{BAD} {type(exc).__name__}: {exc}"
            else:
                verdict = OK if found else BAD
                detail = found[0].url if found else "returned zero sources"
                results[source] = f"{verdict} {len(found)} source(s) - {detail}"
            print(f"  {source.value:10s}: {results[source]}")
    print()
    return results


def probe_llm(settings: Settings) -> str:
    """Send the smallest useful completion to confirm the account can actually be billed."""

    from src.services.live_sources import _build_llm

    print("LLM synthesis")
    try:
        llm = _build_llm(settings)
        reply = llm.complete("Reply with the single word: OK", max_tokens=5)
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        outcome = f"{BAD} {type(exc).__name__}: {exc}"
    else:
        outcome = f"{OK} model replied {reply.strip()[:40]!r}"
    print(f"  {settings.llm_provider:10s}: {outcome}")
    print()
    return outcome


def main(argv: list[str] | None = None) -> int:
    """Run the probes and return 0 only when every attempted stage passed."""

    parser = argparse.ArgumentParser(description="Check the live research pipeline")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="also send one small billable request to the configured LLM",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    report_configuration(settings)
    outcomes = list(asyncio.run(probe_sources(settings)).values())
    if args.with_llm:
        outcomes.append(probe_llm(settings))

    failures = [line for line in outcomes if line.startswith(BAD)]
    if not failures:
        print("All probed stages passed.")
        return 0
    print(f"{len(failures)} of {len(outcomes)} stages failed (see above).")
    print("Sources can fail individually and still produce an answer by graceful degradation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
