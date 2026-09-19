"""Unit tests for typed settings and boundary models."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from src.config import Settings, configure_logging
from src.models import ResearchRequest, SourceName, canonicalize_query, parse_source_names


def test_settings_normalize_provider_and_log_level(tmp_path):
    settings = Settings(
        llm_provider=" OpenAI ",
        web_search_provider="DDG",
        log_level="warning",
        database_path=tmp_path / "history.db",
    )
    assert settings.llm_provider == "openai"
    assert settings.web_search_provider == "ddg"
    assert settings.log_level == "WARNING"


@pytest.mark.parametrize(
    ("field", "value"),
    [("llm_provider", "other"), ("web_search_provider", "google"), ("log_level", "LOUD")],
)
def test_settings_reject_invalid_enums(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_settings_reject_max_delay_below_initial():
    with pytest.raises(ValidationError):
        Settings(retry_initial_delay_seconds=2, retry_max_delay_seconds=1)


def test_configure_logging_sets_root_level():
    configure_logging("ERROR")
    assert logging.getLogger().level == logging.ERROR


def test_research_request_normalizes_and_deduplicates_sources():
    request = ResearchRequest(
        question="  How   does this work?  ",
        sources=(SourceName.WEB, SourceName.WEB, SourceName.ARXIV),
    )
    assert request.question == "How does this work?"
    assert request.sources == (SourceName.WEB, SourceName.ARXIV)


def test_research_request_rejects_empty_sources():
    with pytest.raises(ValidationError):
        ResearchRequest(question="A valid question?", sources=())


def test_research_request_rejects_control_character():
    with pytest.raises(ValidationError):
        ResearchRequest(question="bad\x00question")


def test_parse_source_names_supports_alias_and_order():
    assert parse_source_names("web, wiki,arxiv,web") == (
        SourceName.WEB,
        SourceName.WIKIPEDIA,
        SourceName.ARXIV,
    )


@pytest.mark.parametrize("value", ["", ", ,", "reddit", "wiki,news"])
def test_parse_source_names_rejects_bad_values(value):
    with pytest.raises(ValueError):
        parse_source_names(value)


def test_canonicalize_query_normalizes_case_space_and_punctuation():
    assert canonicalize_query("  WHAT  is Photosynthesis?! ") == "what is photosynthesis"

