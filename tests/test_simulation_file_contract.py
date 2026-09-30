"""Migration contract for issue #2 simulation files."""

import yaml

import peslite
from conftest import EXAMPLES, ROOT
from peslite.solver.simulation import _config_path

EXPECTED = {
    "gfl-example.pes",
    "gfm-droop-example.pes",
    "gfm-dvoc-example.pes",
    "gfm-matching-example.pes",
    "gfm-psc-example.pes",
    "gfm-vsg-example.pes",
    "two-converters-example.pes",
}


def test_bundled_examples_are_named_pes_files_containing_yaml():
    paths = {path.name: path for path in EXAMPLES.iterdir()}
    assert set(paths) == EXPECTED
    for path in paths.values():
        assert path.is_file()
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)


def test_examples_have_one_root_location_and_no_python_files():
    assert EXAMPLES == ROOT / "examples"
    assert not (ROOT / "src" / "peslite" / "examples").exists()
    assert not (ROOT / "src" / "peslite" / "configs").exists()
    assert not list(EXAMPLES.rglob("*.py"))


def test_command_line_finds_bundled_pes_files():
    found = _config_path("gfl-example")
    assert found == (EXAMPLES / "gfl-example.pes").resolve()
    assert _config_path("gfl-example.pes") == found
    assert _config_path(None) == found


def test_issue_two_does_not_change_the_release_version():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '\nversion = "0.1.1"\n' in pyproject
    assert peslite.__version__ == "0.1.1"


def test_wheel_data_comes_from_the_root_examples_directory():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"share/peslite/examples" = ["examples/*.pes"]' in pyproject
