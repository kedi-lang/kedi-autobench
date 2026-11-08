from __future__ import annotations

import tomllib
from pathlib import Path

import kedi
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from kedi_autobench import Kedi

ROOT = Path(__file__).resolve().parents[1]


def test_supported_kedi_range_is_the_package_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependency = next(
        requirement
        for requirement in map(Requirement, project["dependencies"])
        if requirement.name == "kedi"
    )
    supported = Kedi().info.supported_versions

    assert supported is not None
    assert SpecifierSet(supported) == dependency.specifier
    assert Version(kedi.__version__) in dependency.specifier
