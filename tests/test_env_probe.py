"""Environment probe: language detection + safe directory walk. Offline."""

import json

import pytest

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


# --- reads stay inside the working directory and are bounded ----------------


def _link(link, target):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("this OS/account can't create symlinks")


@pytest.mark.parametrize("name", ["README.md", "requirements.txt", "package.json"])
def test_probe_does_not_follow_links_out_of_the_working_directory(tmp_path, name):
    secret = "OUTSIDE_CONTENT_MARKER"
    outside = tmp_path / "outside.txt"
    body = json.dumps({"dependencies": {secret: "1"}}) if name == "package.json" else secret
    outside.write_text(body + "\n", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    # control: the same content as a regular file is read
    (proj / name).write_text(body + "\n", encoding="utf-8")
    seen = probe(proj)
    assert secret in seen["readme_excerpt"] or secret in seen["dependencies"]
    (proj / name).unlink()
    _link(proj / name, outside)
    out = probe(proj)
    assert secret not in out["readme_excerpt"]
    assert not any(secret in d for d in out["dependencies"])
    assert out["language"] == "auto"  # a linked marker file is not a marker


def test_a_linked_readme_falls_through_to_a_regular_one(tmp_path):
    (tmp_path / "outside.md").write_text("outside", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    _link(proj / "README.md", tmp_path / "outside.md")
    (proj / "README.rst").write_text("inside", encoding="utf-8")
    assert probe(proj)["readme_excerpt"] == "inside"


def test_dependency_entries_are_bounded_in_length_and_count(tmp_path):
    pkg = {"dependencies": {f"{i}" + "x" * 3000: "1" for i in range(60)}}  # ~180 KB
    (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
    deps = probe(tmp_path)["dependencies"]
    assert len(deps) == 50
    assert all(len(d) <= 100 for d in deps)


def test_an_oversized_package_json_is_not_read_whole(tmp_path):
    pkg = {"dependencies": {"x" * 1_000_000: "1"}}
    (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
    assert probe(tmp_path)["dependencies"] == []  # cut at the read limit: not valid JSON


def test_requirements_line_without_a_version_is_bounded(tmp_path):
    (tmp_path / "requirements.txt").write_text("y" * 100_000 + "\nrich==13\n", encoding="utf-8")
    deps = probe(tmp_path)["dependencies"]
    assert deps == ["y" * 100, "rich"]


def test_manifest_reads_are_bounded(tmp_path):
    lines = "".join(f"pkg{i}\n" for i in range(200_000))  # ~1.5 MB
    (tmp_path / "requirements.txt").write_text(lines, encoding="utf-8")
    assert len(probe(tmp_path)["dependencies"]) == 50


def test_odd_package_json_shapes_are_ignored(tmp_path):
    for body in ('["a", "b"]', '{"dependencies": ["a"]}', '{"dependencies": "a"}', "not json"):
        (tmp_path / "package.json").write_text(body, encoding="utf-8")
        assert probe(tmp_path)["dependencies"] == [], body


def test_package_json_that_cannot_be_parsed_is_ignored(tmp_path):
    too_deep = "[" * 100_000
    too_long_a_number = '{"dependencies": {"a": 1}, "n": ' + "1" * 5000 + "}"
    for body in (too_deep, too_long_a_number):
        (tmp_path / "package.json").write_text(body, encoding="utf-8")
        assert probe(tmp_path)["dependencies"] == [], body[:20]


def test_readme_excerpt_is_bounded(tmp_path):
    (tmp_path / "README.md").write_text("é" * 5000, encoding="utf-8")
    assert probe(tmp_path)["readme_excerpt"] == "é" * 1200


def test_facts_block_has_a_fixed_ceiling():
    from ops_solver.intelligence import _facts_block

    probe_out = {
        "language": "python",
        "dependencies": ["d" * 100] * 50,
        "file_tree": ["f" * 250] * 40,
        "readme_excerpt": "r" * 1200,
    }
    assert len(_facts_block(probe_out)) < 6000
