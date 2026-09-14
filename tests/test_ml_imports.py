"""Fail if app.ml imports web or LLM stacks."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "app.api",
    "app.mcp",
    "app.services.chat",
    "openai",
    "anthropic",
)

ML_ROOT = Path(__file__).resolve().parents[1] / "app" / "ml"


def _imported_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _is_forbidden(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)


def test_ml_source_does_not_import_web_or_llm():
    offenders: list[str] = []
    for path in sorted(ML_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_names(tree):
            if _is_forbidden(name):
                offenders.append(f"{path.relative_to(ML_ROOT.parent.parent)} imports {name}")
    assert offenders == []


def test_importing_app_ml_does_not_load_web_or_llm():
    before = set(sys.modules)
    import app.ml
    import app.ml.runner  # noqa: F401

    loaded = set(sys.modules) - before
    bad = [name for name in loaded if _is_forbidden(name)]
    assert bad == []
