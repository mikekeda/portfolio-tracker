"""Contracts between modules: everything imports, commits are self-contained, writers fill their tables."""

import ast
import importlib
import subprocess
from pathlib import Path

import pytest

from models import FeaturesDaily
from scripts.update_features import _row_from_holding

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("backend", "scripts", "utils", "celery_tasks")
PROJECT_TOP_LEVEL = {*PACKAGES, "config", "models", "data"}
MODULES = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
    for package in PACKAGES
    for path in (ROOT / package).rglob("*.py")
)


def _tracked_python_files() -> set[str]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout")
    return set(listing.split())


def _project_imports(source: str, package: tuple[str, ...]) -> set[str]:
    """Dotted names of this project's modules imported by `source`, a module inside `package`."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: max(0, len(package) - node.level + 1)] if node.level else ()
            module = ".".join((*base, *([node.module] if node.module else [])))
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return {name for name in names if name.split(".")[0] in PROJECT_TOP_LEVEL}


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_committed_code_imports_only_committed_files():
    # An uncommitted dependency imports fine here but is missing from every other checkout.
    tracked = _tracked_python_files()
    untracked = set()
    for importer in sorted(tracked):
        if not (ROOT / importer).exists():
            continue  # deleted locally, deletion not yet staged
        for name in _project_imports((ROOT / importer).read_text(encoding="utf-8"), Path(importer).parent.parts):
            stem = name.replace(".", "/")
            for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
                if (ROOT / candidate).exists() and candidate not in tracked:
                    untracked.add(f"{importer} -> {candidate}")
    assert not untracked, f"uncommitted dependencies: {sorted(untracked)}"


def test_commit_guard_resolves_relative_imports():
    source = "from .fx import latest_rates_to_gbp\nfrom .. import views\nimport config\nimport numpy\n"
    assert _project_imports(source, ("backend", "utils")) == {
        "backend.utils.fx", "backend.utils.fx.latest_rates_to_gbp", "backend", "backend.views", "config",
    }


def test_features_writer_fills_every_column():
    # The nightly upsert sets every column from the row, so an omitted one is nulled on each run.
    columns = {column.name for column in FeaturesDaily.__table__.columns} - {"id"}
    assert set(_row_from_holding({}, instrument_id=1)) == columns
