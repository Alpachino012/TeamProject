# Async Research Assistant

A command-line research tool that collects Wikipedia, arXiv, and web-search excerpts in parallel,
then produces a concise answer with numbered references.

**Topic:** 4 — Async Research Assistant  
**Course:** AI-ENG-110 Software Engineering, AI Academy

## Quick start

Python 3.12 is the supported runtime.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Add a Gemini key and a Tavily key to `.env`, then run:

```bash
python -m researcher ask "How does CRISPR-Cas9 work?"
python -m researcher ask "How do transformers handle long contexts?" --sources wiki,arxiv
python -m researcher ask "What is fusion energy?" --no-cache
python -m researcher history --limit 5
```

For a deterministic run without keys or network:

```bash
python -m researcher demo --offline --no-cache
```

Live mode uses the provided `ai.fetch_wikipedia`, `ai.fetch_arxiv`, `ai.fetch_web`, and
`ai.synthesize` functions. The `ai/` package has not been modified.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `LLM_PROVIDER` | `gemini` | `anthropic`, `openai`, or `gemini` |
| `LLM_MODEL` | `gemini-2.0-flash` | Provider model identifier |
| `GOOGLE_API_KEY` | empty | Required by the default LLM |
| `WEB_SEARCH_PROVIDER` | `tavily` | `tavily`, `serper`, or `duckduckgo` |
| `TAVILY_API_KEY` | empty | Required by the default web provider |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `CACHE_DIR` | `./.cache` | JSON source-cache directory |
| `CACHE_TTL_SECONDS` | `86400` | Source-cache lifetime |
| `DATABASE_PATH` | `./research.db` | SQLite research history |
| `PER_SOURCE_TIMEOUT_SECONDS` | `20` | Total budget for one source task |
| `SYNTHESIS_TIMEOUT_SECONDS` | `30` | Timeout for each synthesis attempt |
| `EXTERNAL_CALL_TIMEOUT_SECONDS` | `6` | Timeout for each fetch attempt |
| `MAX_SOURCES_PER_QUERY` | `3` | Excerpts requested from each source |
| `MAX_PARALLEL` | `3` | Semaphore bound for outbound source calls |
| `MAX_QUESTION_CHARS` | `1000` | Input size limit |
| `RETRY_ATTEMPTS` | `3` | Maximum attempts for transient failures |
| `RETRY_INITIAL_DELAY_SECONDS` | `0.5` | First exponential-backoff delay |
| `RETRY_MAX_DELAY_SECONDS` | `4` | Backoff ceiling |
| `RATE_LIMIT_CAPACITY` | `6` | Maximum request burst |
| `RATE_LIMIT_REFILL_PER_SECOND` | `2` | Request tokens restored per second |

The other supported provider keys are listed in `.env.example`. Secrets are read by the provided
provider package and never logged.

## How failures are handled

Each provided AI call sits behind the service wrapper. Transient provider, HTTP, and timeout
failures are retried with exponential backoff. Source tasks have separate time budgets and run
through a token bucket plus a configurable semaphore. If one source remains unavailable, the
other sources are still synthesized and the CLI prints an `Unavailable sources` note. If every
source fails, the CLI returns a short error rather than a traceback.

The cache is keyed by `(source, canonical question)`, so differences in case, whitespace, or final
punctuation do not cause duplicate entries. `--no-cache` bypasses both reads and writes. Completed
answers are stored in normalized SQLite tables with foreign keys for citations and source events.

## Sequential vs concurrent benchmark

The benchmark uses deterministic 100 ms offline source adapters, three sources per question, a
disabled cache, and the same workload in both modes.

```bash
python scripts/bench.py --iterations 5 --delay 0.10
```

| Workload | N | Sequential | Concurrent (`MAX_PARALLEL=3`) | Speedup |
|---|---:|---:|---:|---:|
| Three source fetches for each supplied question | 5 | 1.740 s | 0.547 s | 3.18x |

The result is close to the 3x expected from three independent I/O tasks; small timing variance can
put a short synthetic run just above or below that ratio. In live mode, network latency and provider
rate limits become the new bottlenecks.

## Tests and quality checks

All tests are offline. The application tests inject deterministic source and LLM functions; the
provided AI smoke tests use their supplied fakes.

```bash
pytest
pytest --cov=src --cov=researcher --cov-report=term-missing --cov-fail-under=60
python -m mypy src researcher.py scripts
python -m ruff check src tests researcher.py scripts
python demo_ai.py --offline
```

The second test command enforces the required 60% coverage gate. Keeping it separate means the
provided smoke-only command still reports its own contract result cleanly.

Verified locally: **63 tests passed** (including all 16 provided smoke tests) with **92% coverage**
of the application and CLI layers. Strict mypy and Ruff checks also pass.

## Docker

The multi-stage image runs as a non-root user and defaults to the complete five-question offline
demo.

```bash
docker build -t research-assistant .
docker run --rm research-assistant
docker run --rm --env-file .env research-assistant \
  python -m researcher ask "What is retrieval-augmented generation?"
```

## Project layout

```text
ai/                 provided package; unchanged
src/config.py       typed environment settings
src/services/       retries, rate limiting, AI validation
src/concurrency/    bounded gather and graceful degradation
src/core/           research use case
src/storage/        JSON TTL cache and SQLite history
tests/              supplied smoke tests plus offline application tests
scripts/bench.py    reproducible concurrency benchmark
data/               five supplied questions
artefacts/          one recorded offline run
docs/architecture.md
```

## Limitations

- The JSON cache and SQLite history suit a single process; a multi-instance deployment would need
  PostgreSQL and a shared cache.
- The in-process rate limiter does not coordinate budgets across containers.
- Citation indices are checked, but factual entailment still depends on the configured LLM.
- DuckDuckGo needs the optional `duckduckgo-search` package, which is not in the default image.

AI-assisted implementation is disclosed in the final contribution statement and report. The
project is academic coursework and is not published as a reusable library.
