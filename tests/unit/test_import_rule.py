"""The import rule (D32), checked on the source: every import, also inside functions.

- bm4tc.core imports neither bm4tc.analysis nor bm4tc.pipeline;
- bm4tc.analysis imports bm4tc.core only (of bm4tc);
- nothing in bm4tc imports the repository's other packages (analysis/, baselines/,
  tests/), which are scripts and notebooks on top of it.
"""
import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "bm4tc"
FORBIDDEN = {
    "core": ("bm4tc.analysis", "bm4tc.pipeline"),
    "analysis": ("bm4tc.pipeline",),
    "pipeline": (),
}
OUTSIDE = ("analysis", "baselines", "tests", "experiments", "src")


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: resolve against the module's package
                base = path.relative_to(PACKAGE.parent).with_suffix("").parts[:-node.level]
                module = ".".join(base + ((node.module,) if node.module else ()))
            else:
                module = node.module
            yield node.lineno, module


def _violations(area: str):
    for path in sorted((PACKAGE / area).rglob("*.py")):
        for line, module in _imports(path):
            top = module.split(".")[0]
            if top in OUTSIDE or any(module == f or module.startswith(f + ".")
                                     for f in FORBIDDEN[area]):
                yield f"{path.relative_to(PACKAGE.parent)}:{line} imports {module}"


@pytest.mark.parametrize("area", sorted(FORBIDDEN))
def test_import_rule(area):
    assert list(_violations(area)) == []


def test_the_rule_sees_violations(tmp_path, monkeypatch):
    fake = tmp_path / "bm4tc"
    (fake / "core").mkdir(parents=True)
    (fake / "core" / "bad.py").write_text(
        "def f():\n    from ..pipeline import runs\n    import analysis.utils\n")
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", fake)
    assert len(list(_violations("core"))) == 2
