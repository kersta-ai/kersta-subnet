"""What miner code may import and call.

The real security boundary is the container: no network, read only package,
read only root filesystem, no capabilities, an unprivileged user. These
checks are a second layer. They refuse a package before it ever runs, give a
miner a clear message, and stop code that only works because it reaches
outside the model (pickles, eval, the filesystem, introspection tricks).

Two checks share one allow list:
  * ``check_source`` reads each .py file's syntax tree. The validator and
    ``kersta miner check`` both run it.
  * ``install_import_guard`` refuses, at run time inside the sandbox, any
    import made by package code that is not on the list.
"""

from __future__ import annotations

import ast
import builtins
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ALLOWED_MODULES: Final[frozenset[str]] = frozenset(
    {
        # standard library, pure computation only
        "__future__",
        "abc",
        "cmath",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "enum",
        "functools",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "time",
        "typing",
        "typing_extensions",
        "warnings",
        # numerical and model libraries installed in the sandbox image
        "einops",
        "numpy",
        "PIL",
        "safetensors",
        "scipy",
        "timm",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
    }
)

FORBIDDEN_CALLS: Final[frozenset[str]] = frozenset(
    {"eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint",
     "input", "open", "memoryview"}
)

FORBIDDEN_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        "__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__",
        "__mro__", "__bases__", "__base__", "__getattribute__", "__reduce__",
        "__reduce_ex__", "__import__", "__loader__", "__spec__", "__dict__",
        "f_globals", "f_locals", "f_back", "f_builtins", "gi_frame", "cr_frame", "tb_frame",
    }
)

# (object name, attribute): calls that load pickles or reach the network.
FORBIDDEN_MEMBERS: Final[frozenset[tuple[str, str]]] = frozenset(
    {("torch", "load"), ("torch", "hub"), ("jit", "load"), ("numpy", "load"), ("np", "load"),
     ("pickle", "loads")}
)

FORBIDDEN_KEYWORDS: Final[dict[str, object]] = {
    "allow_pickle": True,
    "trust_remote_code": True,
    "weights_only": False,
}


@dataclass(frozen=True, slots=True)
class Violation:
    file: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.message}"


def _allowed_import(name: str, siblings: frozenset[str], fromlist: Iterable[str] = ()) -> bool:
    top = name.split(".")[0]
    if top in ALLOWED_MODULES or top in siblings:
        return True
    if name == "os.path" or name.startswith("os.path."):
        return True
    return name == "os" and set(fromlist) == {"path"}


class _Checker(ast.NodeVisitor):
    def __init__(self, file: str, siblings: frozenset[str]) -> None:
        self.file = file
        self.siblings = siblings
        self.found: list[Violation] = []

    def flag(self, node: ast.AST, message: str) -> None:
        self.found.append(Violation(self.file, getattr(node, "lineno", 0), message))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _allowed_import(alias.name, self.siblings):
                self.flag(node, f"import of {alias.name!r} is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = [alias.name for alias in node.names]
        if node.level:
            # relative import: only modules that live in this package
            target = node.module.split(".")[0] if node.module else None
            pulled = [target] if target else names
            for name in pulled:
                if name not in self.siblings:
                    self.flag(node, f"relative import of {name!r} names no file in the package")
        elif not _allowed_import(node.module or "", self.siblings, names):
            self.flag(node, f"import from {node.module!r} is not allowed")
        if "*" in names:
            self.flag(node, "star imports are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.flag(node, f"call to {node.func.id}() is not allowed")
        if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "setattr", "delattr"):
            for arg in node.args[1:2]:
                if isinstance(arg, ast.Constant) and arg.value in FORBIDDEN_ATTRIBUTES:
                    self.flag(node, f"{node.func.id}() of {arg.value!r} is not allowed")
        for keyword in node.keywords:
            if (
                keyword.arg in FORBIDDEN_KEYWORDS
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == FORBIDDEN_KEYWORDS[keyword.arg]
            ):
                self.flag(node, f"{keyword.arg}={keyword.value.value!r} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.flag(node, f"access to .{node.attr} is not allowed")
        owner = node.value
        owner_name = owner.id if isinstance(owner, ast.Name) else (
            owner.attr if isinstance(owner, ast.Attribute) else None
        )
        if owner_name and (owner_name, node.attr) in FORBIDDEN_MEMBERS:
            self.flag(node, f"{owner_name}.{node.attr} is not allowed (pickle or network)")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "__builtins__":
            self.flag(node, "access to __builtins__ is not allowed")
        self.generic_visit(node)


def siblings_of(package_dir: Path) -> frozenset[str]:
    return frozenset(path.stem for path in package_dir.glob("*.py"))


def check_source(source: str, file: str, siblings: frozenset[str]) -> list[Violation]:
    try:
        tree = ast.parse(source, filename=file)
    except SyntaxError as exc:
        return [Violation(file, exc.lineno or 0, f"syntax error: {exc.msg}")]
    checker = _Checker(file, siblings)
    checker.visit(tree)
    return checker.found


def check_package_code(package_dir: Path) -> list[Violation]:
    """Check every .py file of one lane package."""
    siblings = siblings_of(package_dir)
    found: list[Violation] = []
    for path in sorted(package_dir.rglob("*.py")):
        rel = path.relative_to(package_dir).as_posix()
        found.extend(check_source(path.read_text(encoding="utf-8"), rel, siblings))
    return found


def install_import_guard(package_dir: Path) -> None:
    """Refuse, at run time, imports that package code makes off the list.

    Only imports issued *from a file inside the package* are checked, so the
    libraries themselves (torch imports os, ctypes and much more) still work.
    """
    root = str(package_dir.resolve())
    siblings = siblings_of(package_dir)
    original = builtins.__import__

    def guarded(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (),
                level: int = 0) -> Any:
        importer = (globals_ or {}).get("__file__") or ""
        inside = bool(importer) and str(Path(importer).resolve()).startswith(root)
        if inside and level == 0 and not _allowed_import(name, siblings, fromlist or ()):
            raise ImportError(f"kersta sandbox: import of {name!r} is not allowed")
        return original(name, globals_, locals_, fromlist, level)

    builtins.__import__ = guarded
    sys.dont_write_bytecode = True
