from __future__ import annotations

import researcher
from src.cli import main


def configure_test_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "research.db"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("RATE_LIMIT_CAPACITY", "100")
    monkeypatch.setenv("RATE_LIMIT_REFILL_PER_SECOND", "100")
    monkeypatch.setenv("RETRY_INITIAL_DELAY_SECONDS", "0")
    monkeypatch.setenv("RETRY_MAX_DELAY_SECONDS", "0")


def test_module_entry_point_exports_cli_main():
    assert researcher.main is main


def test_cli_ask_offline_end_to_end(monkeypatch, tmp_path, capsys):
    configure_test_environment(monkeypatch, tmp_path)
    code = main(
        [
            "ask",
            "What is photosynthesis?",
            "--sources",
            "wiki,arxiv",
            "--offline",
            "--no-cache",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "A: The offline run" in captured.out
    assert "[1] (wikipedia)" in captured.out
    assert "Retrieved 2 excerpts" in captured.out
    assert captured.err == ""


def test_cli_history_lists_saved_session(monkeypatch, tmp_path, capsys):
    configure_test_environment(monkeypatch, tmp_path)
    assert main(["ask", "Saved question?", "--offline", "--no-cache"]) == 0
    capsys.readouterr()
    assert main(["history", "--limit", "1"]) == 0
    captured = capsys.readouterr()
    assert "Saved question?" in captured.out


def test_cli_empty_history(monkeypatch, tmp_path, capsys):
    configure_test_environment(monkeypatch, tmp_path)
    assert main(["history"]) == 0
    assert "No saved research sessions." in capsys.readouterr().out


def test_cli_rejects_unknown_source(monkeypatch, tmp_path, capsys):
    configure_test_environment(monkeypatch, tmp_path)
    code = main(["ask", "question", "--sources", "news", "--offline"])
    assert code == 2
    assert "unknown source" in capsys.readouterr().err


def test_cli_rejects_oversize_question(monkeypatch, tmp_path, capsys):
    configure_test_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("MAX_QUESTION_CHARS", "20")
    code = main(["ask", "x" * 21, "--offline"])
    assert code == 2
    assert "question is too long" in capsys.readouterr().err
