"""
Check that every BaseActivation subclass has a :math: formula in its docstring.
Usage: python scripts/check_docstrings.py [paths...]
Exits 1 if any violations found.
"""

import ast
import sys
from pathlib import Path


def check_file(path: Path) -> list[str]:
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
        if "BaseActivation" not in bases:
            continue
        docstring = ast.get_docstring(node)
        if not docstring or (":math:" not in docstring and ".. math::" not in docstring):
            violations.append(f"{path}:{node.lineno}: {node.name} missing :math: in docstring")
    return violations


def main(paths: list[str]) -> int:
    roots = [Path(p) for p in paths] if paths else [Path("torch_activation")]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))

    violations = []
    for f in sorted(files):
        violations.extend(check_file(f))

    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
