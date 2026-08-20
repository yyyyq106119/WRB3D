from pathlib import Path

from wrb3d.training.checkpoint import _project_path
from wrb3d.training.manifests import resolve_manifest_dir
from wrb3d.utils.config import project_root, resolve_project_path


def _make_project(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "package"
    (root / "src" / "wrb3d").mkdir(parents=True)
    (root / "configs" / "nested").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='test'\nversion='0'\n")
    config_path = root / "configs" / "nested" / "formal.yaml"
    config_path.write_text("{}\n")
    config = {
        "_config_path": str(config_path),
        "data": {"manifest_dir": "splits"},
    }
    return root, config


def test_project_relative_paths_use_package_root_for_nested_config(tmp_path: Path) -> None:
    root, config = _make_project(tmp_path)
    assert project_root(config) == root.resolve()
    assert resolve_project_path(config, "artifacts/statistics.json") == (
        root / "artifacts" / "statistics.json"
    ).resolve()
    assert resolve_manifest_dir(config) == (root / "splits").resolve()
    assert _project_path(config, "artifacts/statistics.json") == (
        root / "artifacts" / "statistics.json"
    ).resolve()


def test_absolute_paths_are_unchanged(tmp_path: Path) -> None:
    _, config = _make_project(tmp_path)
    absolute = (tmp_path / "external" / "statistics.json").resolve()
    assert resolve_project_path(config, absolute) == absolute
