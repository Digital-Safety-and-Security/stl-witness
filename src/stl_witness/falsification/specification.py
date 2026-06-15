"""IA-STL file parsing and RTAMT formula normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

IoType = Literal["input", "output"]

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_COMMENT_RE = re.compile(r"(#|//).*$")
_DECL_PREFIX_RE = re.compile(rf"^(input|output)\s*:?\s+(.+)$", re.IGNORECASE)
_DECL_SUFFIX_RE = re.compile(rf"^({_IDENTIFIER})\s*:\s*(input|output)\b", re.IGNORECASE)
_TOKEN_RE = re.compile(_IDENTIFIER)
_ASSIGNMENT_RE = re.compile(rf"^\s*({_IDENTIFIER})\s*=")

_KEYWORDS = {
    "G",
    "F",
    "H",
    "O",
    "X",
    "U",
    "R",
    "and",
    "or",
    "not",
    "implies",
    "always",
    "eventually",
    "until",
    "since",
    "once",
    "historically",
    "prev",
    "next",
    "rise",
    "fall",
    "abs",
    "exp",
    "pow",
    "true",
    "false",
    "inf",
    "out",
}


@dataclass(frozen=True)
class VariableDeclaration:
    """Input/output declaration for an IA-STL variable."""

    name: str
    io_type: IoType


@dataclass(frozen=True)
class IaStlSpecification:
    """Parsed IA-STL specification with input/output variable declarations."""

    formula: str
    declarations: tuple[VariableDeclaration, ...]
    output_name: str = "out"

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(decl.name for decl in self.declarations)

    def rtamt_formula(self) -> str:
        """Return an RTAMT-friendly formula with an output assignment."""

        formula = normalize_formula_for_rtamt(self.formula)
        match = _ASSIGNMENT_RE.match(formula)
        if match:
            return _ensure_semicolon(formula)
        return _ensure_semicolon(f"{self.output_name} = {formula}")


def read_ia_stl_file(path: str | Path) -> IaStlSpecification:
    """Read an `.stl` file containing declarations plus an IA-STL formula.

    Supported declaration forms include:

    - ``input request``
    - ``output response``
    - ``input: request, another_request``
    - ``response: output``
    """

    declarations: list[VariableDeclaration] = []
    formula_lines: list[str] = []

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = _COMMENT_RE.sub("", raw_line).strip()
        if not line:
            continue

        declaration = _parse_declaration(line)
        if declaration:
            declarations.extend(declaration)
            continue

        formula_lines.append(line)

    if not declarations:
        raise ValueError("STL file must declare at least one input or output variable")
    if not formula_lines:
        raise ValueError("STL file must contain a formula")

    spec = IaStlSpecification(
        formula=" ".join(formula_lines).strip(),
        declarations=tuple(declarations),
    )
    _validate_formula_variables(spec)
    return spec


def normalize_formula_for_rtamt(formula: str) -> str:
    """Translate common IA-STL shorthand into RTAMT syntax."""

    normalized = formula.strip().rstrip(";")
    normalized = re.sub(
        r"\bG_\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]",
        r"always[\1:\2]",
        normalized,
    )
    normalized = re.sub(
        r"\bF_\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]",
        r"eventually[\1:\2]",
        normalized,
    )
    normalized = re.sub(
        r"\bG\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]",
        r"always[\1:\2]",
        normalized,
    )
    normalized = re.sub(
        r"\bF\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]",
        r"eventually[\1:\2]",
        normalized,
    )
    normalized = re.sub(r"\bG\s*\(", "always(", normalized)
    normalized = re.sub(r"\bF\s*\(", "eventually(", normalized)
    normalized = normalized.replace("->", "implies")
    return normalized


def extract_formula_variables(formula: str) -> set[str]:
    """Return identifier-like tokens that look like STL variables."""

    normalized = normalize_formula_for_rtamt(formula)
    match = _ASSIGNMENT_RE.match(normalized)
    assigned_name = match.group(1) if match else None
    variables = set()

    for token in _TOKEN_RE.findall(normalized):
        if token == assigned_name:
            continue
        if token in _KEYWORDS or token.lower() in _KEYWORDS:
            continue
        variables.add(token)
    return variables


def _parse_declaration(line: str) -> list[VariableDeclaration]:
    prefix_match = _DECL_PREFIX_RE.match(line)
    if prefix_match:
        io_type = prefix_match.group(1).lower()
        return [
            VariableDeclaration(name=name, io_type=io_type) for name in _parse_names(prefix_match.group(2))
        ]

    suffix_match = _DECL_SUFFIX_RE.match(line)
    if suffix_match:
        return [
            VariableDeclaration(
                name=suffix_match.group(1),
                io_type=suffix_match.group(2).lower(),
            )
        ]

    return []


def _parse_names(value: str) -> list[str]:
    names: list[str] = []
    for part in re.split(r"[,\s]+", value):
        candidate = part.strip().strip(";")
        if not candidate:
            continue
        name = candidate.split(":", maxsplit=1)[0]
        if re.fullmatch(_IDENTIFIER, name):
            names.append(name)
    return names


def _validate_formula_variables(spec: IaStlSpecification) -> None:
    declared = set(spec.variables)
    used = extract_formula_variables(spec.formula)
    missing = sorted(used - declared)
    if missing:
        raise ValueError(
            "formula references undeclared variable(s): " + ", ".join(missing)
        )


def _ensure_semicolon(formula: str) -> str:
    return formula if formula.strip().endswith(";") else f"{formula};"
