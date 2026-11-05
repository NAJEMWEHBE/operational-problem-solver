"""Environment probe: language detection + safe directory walk. Offline."""

from ops_solver.env_probe import _detect_language, probe


def test_marker_file_wins(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert _detect_language(tmp_path, []) == "python"


def test_package_json_marker(tmp_path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert _detect_language(tmp_path, []) == "typescript"


def test_extension_frequency_vote(tmp_path):
    assert _detect_language(tmp_path, ["a.go", "b.go", "c.go", "d.py"]) == "go"


def test_empty_is_auto(tmp_path):
    assert _detect_language(tmp_path, []) == "auto"


def test_probe_runs_on_empty_dir(tmp_path):
    out = probe(tmp_path)
    assert out["language"] == "auto"
    assert out["file_tree"] == []
    assert isinstance(out["linters_available"], list)


def test_probe_collects_files(tmp_path):
    (tmp_path / "main.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("rich==13.0\n# comment\ntyper\n", encoding="utf-8")
    out = probe(tmp_path)
    assert out["language"] == "python"
    assert "main.py" in out["file_tree"]
    assert "rich" in out["dependencies"] and "typer" in out["dependencies"]


def test_parent_folder_named_like_a_skip_dir_does_not_hide_the_project(tmp_path):
    for parent in ("build", "runs", "dist", "venv"):
        proj = tmp_path / parent / "myproj"
        proj.mkdir(parents=True)
        (proj / "main.py").write_text("print(1)\n", encoding="utf-8")
        out = probe(proj)
        assert out["file_tree"] == ["main.py"], parent
        assert out["language"] == "python", parent


def test_skip_dirs_inside_the_project_are_not_listed(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    for skip in ("node_modules/pkg", ".git/objects", "build/lib"):
        d = tmp_path / skip
        d.mkdir(parents=True)
        (d / "junk.js").write_text("//\n", encoding="utf-8")
    assert probe(tmp_path)["file_tree"] == ["app.py"]
