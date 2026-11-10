"""CLI .env loader: allow-list + empty-var override. Offline."""

import os

import pytest

from ops_solver.cli import _dotenv_value, _load_dotenv


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


def test_inline_comment_is_not_part_of_the_value(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-test-abc   # my work key\n", encoding="utf-8")
    _load_dotenv(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-abc"


def test_hash_inside_quotes_or_without_space_is_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        'ANTHROPIC_API_KEY="sk-a #b"  # note\nANTHROPIC_BASE_URL=http://h/x#frag\n', encoding="utf-8"
    )
    _load_dotenv(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-a #b"
    assert _dotenv_value("http://h/x#frag") == "http://h/x#frag"


def test_dotenv_never_chooses_the_api_endpoint(tmp_path, monkeypatch):
    # an operator signed in another way (auth token / profile): the .env loads,
    # but the endpoint that receives that credential stays the operator's choice
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-dummy-token")
    (tmp_path / ".env").write_text("ANTHROPIC_BASE_URL=https://example.invalid\n", encoding="utf-8")
    _load_dotenv(tmp_path)
    assert "ANTHROPIC_BASE_URL" not in os.environ


def test_dotenv_endpoint_is_ignored_next_to_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-test\nANTHROPIC_BASE_URL=https://example.invalid\n", encoding="utf-8"
    )
    _load_dotenv(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test"
    assert "ANTHROPIC_BASE_URL" not in os.environ


def test_endpoint_from_the_real_environment_is_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example")
    (tmp_path / ".env").write_text("ANTHROPIC_BASE_URL=https://example.invalid\n", encoding="utf-8")
    _load_dotenv(tmp_path)
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://proxy.example"


def test_a_dotenv_that_is_not_a_regular_file_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").mkdir()  # a folder named .env
    _load_dotenv(tmp_path)
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_a_linked_dotenv_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    target = tmp_path / "elsewhere.env"
    target.write_text("ANTHROPIC_API_KEY=sk-linked\n", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    try:
        (proj / ".env").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("this OS/account can't create symlinks")
    _load_dotenv(proj)
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_dotenv_read_is_bounded(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    padding = "# " + "x" * (70 * 1024) + "\n"  # past the 64 KiB read limit
    (tmp_path / ".env").write_text(padding + "ANTHROPIC_API_KEY=sk-late\n", encoding="utf-8")
    _load_dotenv(tmp_path)
    assert "ANTHROPIC_API_KEY" not in os.environ
