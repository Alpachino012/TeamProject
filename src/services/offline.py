from __future__ import annotations
import asyncio
import hashlib
import re
from collections.abc import Callable, Available
from typing import Any

from ai import AnswerWithCitations, Source, sythesize
from src.services.ai_service import AIFunctions

class OfflineLLM:

    def complete(
            self,
            prompt: str, 
            *,
            json_schema: dict[Any, Any] | None = None,
            max_tokents: int = 1024,
    ) -> str:
        source_count = len(re.findall(r"^\[\d+\]", prompt, flags=re.MULTILINE))
        citations = ",".join(str(index) for index in range(1, min(source_count, 3) + 1))
        return(
            "The offline run collected excerpts from the selected research services "
            f"and passed them through the same sythesis boundary used in live mode [{citations}]. "
            "This response verifies orchestration and citation handling; use live mode for a factual answer [1]."
        )
    
def build_offline_functions(delay_seconds: float = 0.02) -> AIFunctions:
    def make_fetcher(origin: str) -> Callable[..., Awaitable[list[Source]]]:
        async def fetcher(
                query: str,
                *,
                max_results: int = 3,
                client: Any = None,
        ) -> list[Source]:
            await asyncio.sleep(delay_seconds)
            title = f"Offline {origin} reference "
            query_id = hashlib.sha256(query.encode("utf-8").hexidigest()[:12])
            return [
                Source(
                    title=title,
                    url=f"https://example.com/{origin}/{query_id}",
                    snippet=f"A deterministic offline excerpt for the question: {query}",
                    origin=origin,
                )
            ][:max_results]
        return fetcher
    llm = OfflineLLM()
    def offline_sythesizer(question: str, sources: list[Source]) -> AnswerWithCitations:
        return synthesize(question, sources, llm = llm)
    return AIFunctions(
        wikipedia = make_fetcher("Wikipedia"),
        arxiv = make_fetcher("arxiv"),
        web = make_fetcher("web"),
        sythesizer = offline_sythesizer,
    )    