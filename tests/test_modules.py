"""The pipeline modules all import.

Nothing calls the Sprint 1–5 stubs yet. Importing them here means a typo
or a syntax error in one of those files fails the tests immediately,
instead of waiting until the sprint that first uses the module.
"""

import importlib

import pytest

# importlib is used so the module named filter does not clash with
# Python's built-in filter function in this file.
PIPELINE_MODULES = [
    "kilonova_hunter.fetch",
    "kilonova_hunter.clean",
    "kilonova_hunter.features",
    "kilonova_hunter.filter",
    "kilonova_hunter.inject",
    "kilonova_hunter.tns",
    "kilonova_hunter.report",
    "kilonova_hunter.cli",
]


@pytest.mark.parametrize("module_name", PIPELINE_MODULES)
def test_pipeline_module_imports(module_name: str) -> None:
    """The module loads, so its functions and docstrings are valid Python."""
    importlib.import_module(module_name)
