"""Live provider adapters that wire ``Settings`` into the unmodified ``ai`` package.

The provided ``ai`` package resolves its providers from process environment variables.
This application resolves configuration into a typed :class:`Settings` object instead, and
``pydantic-settings`` deliberately does not export ``.env`` values into ``os.environ``. The
adapters below close that gap through the injection points the package already exposes -
``fetch_web(provider=...)`` and ``synthesize(llm=...)`` - so credentials travel through the
typed settings object and ``ai`` itself stays untouched.

Two upstream behaviours are also compensated for here:

``fetch_wikipedia``
    Queries the ``opensearch`` endpoint, which is a title-prefix autocomplete rather than a
    full-text search, so natural-language questions match nothing. The adapter resolves the
    question to article titles first, then hands each title to the provided fetcher.

``DuckDuckGoProvider``
    Imports ``duckduckgo_search``, a package since renamed to ``ddgs``. The old name now
    installs a shim that returns no results. The adapter subclasses the provided
    ``WebSearchProvider`` contract against the current package.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

from ai import Source, fetch_arxiv, fetch_web, fetch_wikipedia, synthesize
from ai.providers.base import LLMProvider, ProviderError
from ai.schemas import AnswerWithCitations
from ai.sources import WebSearchProvider
from src.config import Settings
from src.services.ai_service import AIFunctions

logger = logging.getLogger(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


async def _resolve_wikipedia_titles(
    query: str,
    *,
    max_results: int,
    client: Any,
) -> list[str]:
    """Resolve a natural-language question to Wikipedia article titles.

    Uses the full-text search endpoint (``list=search``) rather than the title-prefix
    autocomplete the provided fetcher relies on. Returns an empty list when the search
    endpoint is unavailable or answers in an unexpected shape, letting the caller fall back.
    """

    try:
        response = await client.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - upstream fetcher is the fallback path
        logger.warning("wikipedia_title_resolution_failed error=%s", exc)
        return []

    hits = payload.get("query", {}).get("search", [])
    if not isinstance(hits, list):
        logger.warning("wikipedia_title_resolution_failed error=unexpected_payload_shape")
        return []
    return [hit["title"] for hit in hits if isinstance(hit, dict) and hit.get("title")]


async def fetch_wikipedia_full_text(
    query: str,
    *,
    max_results: int = 3,
    client: Any = None,
) -> list[Source]:
    """Search Wikipedia by content, then retrieve each article via the provided fetcher."""

    if not query.strip():
        return []

    titles = await _resolve_wikipedia_titles(query, max_results=max_results, client=client)
    if not titles:
        logger.info("wikipedia_title_resolution_empty falling_back=direct_fetch")
        direct: list[Source] = await fetch_wikipedia(
            query, max_results=max_results, client=client
        )
        return direct

    logger.debug("wikipedia_titles_resolved titles=%r", titles)
    batches = await asyncio.gather(
        *(fetch_wikipedia(title, max_results=1, client=client) for title in titles),
        return_exceptions=True,
    )

    sources: list[Source] = []
    seen: set[str] = set()
    for title, batch in zip(titles, batches, strict=True):
        if isinstance(batch, BaseException):
            logger.warning("wikipedia_article_skipped title=%r error=%s", title, batch)
            continue
        for source in batch:
            if source.url in seen:
                continue
            seen.add(source.url)
            sources.append(source)
    return sources[:max_results]


class DdgsWebSearchProvider(WebSearchProvider):  # type: ignore[misc]
    """DuckDuckGo search through the ``ddgs`` package, the successor to ``duckduckgo-search``.

    Implements the provided :class:`WebSearchProvider` contract. The underlying library is
    synchronous, so each search runs in a worker thread to keep the event loop free.
    """

    def __init__(self) -> None:
        try:
            import ddgs  # noqa: F401
        except ImportError as exc:
            raise ProviderError(
                "The `ddgs` package is required for DuckDuckGo search. "
                "Install with `pip install ddgs`."
            ) from exc

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        """Return web results; ``client`` is unused because the library owns its transport."""

        def run() -> list[Source]:
            from ddgs import DDGS

            results: list[Source] = []
            with DDGS() as ddgs_client:
                for item in ddgs_client.text(query, max_results=max_results):
                    href = item.get("href")
                    if not href:
                        continue
                    results.append(
                        Source(
                            title=item.get("title", "(untitled)"),
                            url=href,
                            snippet=item.get("body", ""),
                            origin="web",
                        )
                    )
            return results

        try:
            return await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001 - normalised to the package's error type
            raise ProviderError(f"DuckDuckGo search failed: {exc}") from exc


def _build_web_provider(settings: Settings) -> WebSearchProvider | None:
    """Select a web-search provider, or ``None`` to let the package resolve it itself."""

    if settings.web_search_provider in {"duckduckgo", "ddg"}:
        return DdgsWebSearchProvider()
    return None


def _build_llm(settings: Settings) -> LLMProvider:
    """Construct the configured LLM with explicit credentials from typed settings."""

    provider = settings.llm_provider
    if provider == "openai":
        from ai.providers.openai import OpenAILLM

        return OpenAILLM(settings.llm_model, api_key=_secret(settings.openai_api_key, "OPENAI"))
    if provider in {"google", "gemini"}:
        from ai.providers.google import GeminiLLM

        return GeminiLLM(settings.llm_model, api_key=_secret(settings.google_api_key, "GOOGLE"))
    from ai.providers.anthropic import AnthropicLLM

    key = _secret(settings.anthropic_api_key, "ANTHROPIC")
    return AnthropicLLM(settings.llm_model, api_key=key)


def _secret(value: Any, name: str) -> str:
    """Unwrap a configured API key, failing with the package's own error type."""

    if value is None:
        raise ProviderError(f"{name}_API_KEY is not set in the environment or .env file.")
    secret = str(value.get_secret_value())
    if not secret.strip():
        raise ProviderError(f"{name}_API_KEY is empty in the environment or .env file.")
    return secret


def build_live_functions(settings: Settings) -> AIFunctions:
    """Bind the provided AI functions to this application's typed configuration.

    The LLM is constructed on first synthesis rather than eagerly, so commands that never
    synthesise - ``researcher history``, for example - run without any API key configured.
    """

    web_provider = _build_web_provider(settings)
    llm_cache: list[LLMProvider] = []

    async def web(
        query: str,
        *,
        max_results: int = 3,
        client: Any = None,
    ) -> list[Source]:
        results: list[Source] = await fetch_web(
            query,
            max_results=max_results,
            provider=web_provider,
            client=client,
        )
        return results

    def synthesizer(question: str, sources: Iterable[Source]) -> AnswerWithCitations:
        if not llm_cache:
            llm_cache.append(_build_llm(settings))
        return synthesize(question, list(sources), llm=llm_cache[0])

    return AIFunctions(
        wikipedia=fetch_wikipedia_full_text,
        arxiv=fetch_arxiv,
        web=web,
        synthesizer=synthesizer,
    )
