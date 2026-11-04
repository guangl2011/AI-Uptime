"""Governance guardrail: no paths that could write to facility systems.

The MVP posture (MVP spec art_SSJ84EWf, novelty D) is zero autonomous
control: integrations are structurally read-only. Connectors arrive in later
phases; this test pins the invariant now so the first connector lands inside
it, not after a retrofit: no industrial-control or remote-execution client
may enter the backend source.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Libraries whose surface is (or includes) writing to/controlling external
# equipment or hosts. httpx/requests are allowed: read-only pulls of alarm
# exports and work orders are in scope; anything else is review-gated.
FORBIDDEN_MODULES = {
    "pymodbus",
    "snap7",
    "bacpypes",
    "pycomm3",
    "paramiko",
    "fabric",
    "asyncssh",
}

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_control_write_clients_in_backend_source() -> None:
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        modules = _imported_modules(path)
        forbidden = modules & FORBIDDEN_MODULES
        if forbidden:
            offenders.append(f"{path}: {sorted(forbidden)}")
    assert not offenders, f"control-capable imports found — read-only posture violated: {offenders}"
