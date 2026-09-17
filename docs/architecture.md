# Architecture

The command line creates the application in `src/app.py`, which is the only place that chooses
concrete storage and provider adapters.

```text
CLI (`python -m researcher`)
             |
             v
       Researcher core --------------> SQLite history
             |
             v
     SourceOrchestrator --------------> JSON TTL cache
             |
       bounded gather
       /      |      \
 Wikipedia  arXiv   web search
       \      |      /
        AIService wrappers
   (retry, timeout, rate limit, logs)
             |
             v
      provided `ai/` package
```

`ResearchRequest`, `SourceCollection`, and `ResearchResult` are typed models at module
boundaries. The provided `Source`, `Citation`, and `AnswerWithCitations` models are reused where
their contract already fits. `ResearchRepository` and `SourceCache` are abstract base classes;
their SQLite and JSON implementations are composed into the core.

All three source functions receive one shared `httpx.AsyncClient` per research request. A
configurable semaphore bounds source calls, and `asyncio.gather(return_exceptions=True)` keeps an
arXiv outage, for example, from discarding useful Wikipedia and web results.

