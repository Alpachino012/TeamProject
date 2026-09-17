"""Retrying, rate-limited and validated wrapper around the provided ``ai`` package."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import urlparse

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai import AnswerWithCitations, Source, fetch_arxiv, fetch_web, fetch_wikipedia, synthesize
from ai.providers.base import ProviderError
from src.config import Settings
from src.exceptions import ExternalServiceError, InvalidAIResponseError
from src.models import SourceName
from src.services.rate_limiter import AsyncTokenBucket

logger = logging.getLogger(__name__)
T = TypeVar("T")

SourceFetcher = Callable[..., Awaitable[list[Source]]]
Synthesizer = Callable[[str, list[Source]], AnswerWithCitations]


@dataclass(frozen=True)
class AIFunctions:
    """Injectable boundary around the four provided AI functions."""

    wikipedia: SourceFetcher = fetch_wikipedia
    arxiv: SourceFetcher = fetch_arxiv
    web: SourceFetcher = fetch_web
    synthesizer: Synthesizer = synthesize
