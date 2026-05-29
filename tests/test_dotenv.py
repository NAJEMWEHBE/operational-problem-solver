"""CLI .env loader: allow-list + empty-var override. Offline."""

import os

from ops_solver.cli import _load_dotenv


def test_overrides_empty_env_var(tmp_path, monkeypatch):
    # The bug: ANTHROPIC_API_KEY present-but-empty shadowed the .env value.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY= sk-test-123\n", encoding="utf-8")
    _load_dotenv(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-123"  # leading space stripped


def test_does_not_clobber_real_existing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "real-key")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-other\n", encoding="utf-8")
    _load_dotenv(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "real-key"  # top guard wins


def test_allowlist_blocks_arbitrary_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EVIL_VAR", raising=False)
    (tmp_path / ".env").write_text("EVIL_VAR=x\nANTHROPIC_API_KEY=sk-ok\n", encoding="utf-8")
    _load_dotenv(tmp_path)
    assert "EVIL_VAR" not in os.environ
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ok"


def test_strips_surrounding_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text('ANTHROPIC_API_KEY="sk-quoted"\n', encoding="utf-8")
    _load_dotenv(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-quoted"
