"""Configuration objects for witness-guided IA-STL search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

Target = Literal["satisfy", "falsify"]
DifferentialTarget = Literal["both", "phi1_only", "phi2_only", "neither"]
DIFFERENTIAL_TARGETS: tuple[DifferentialTarget, ...] = (
    "both",
    "phi1_only",
    "phi2_only",
    "neither",
)


@dataclass(frozen=True)
class VariableConfig:
    """Configuration for one IA-STL variable."""

    signal: str
    io: Literal["input", "output"] | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FalsificationConfig:
    """Top-level configuration for witness search."""

    specification: Path
    variables: Mapping[str, VariableConfig]
    targets: tuple[Target, ...] = ("falsify",)
    horizon: float = 10.0
    dt: float | None = None
    sample_rate: float | None = 100.0
    runs: int = 1
    iterations: int = 100
    witnesses: int = 1
    seed: int = 1
    optimizer: Literal["dual_annealing", "uniform_random"] = "dual_annealing"
    backend: Literal["psytaliro", "random", "auto"] = "psytaliro"
    trace_diagnostics: bool = True
    output_csv: Path | None = None
    output_png: Path | None = None

    @property
    def max_iterations_per_signal_generation(self) -> int:
        """Maximum falsification iterations used for one witness-generation attempt."""

        return self.iterations


@dataclass(frozen=True)
class DifferentialConfig:
    """Top-level configuration for two-spec differential witness search."""

    phi1: Path
    phi2: Path
    variables: Mapping[str, VariableConfig]
    targets: tuple[DifferentialTarget, ...] = DIFFERENTIAL_TARGETS
    horizon: float = 10.0
    dt: float | None = None
    sample_rate: float | None = 100.0
    runs: int = 1
    iterations: int = 100
    witnesses: int = 1
    seed: int = 1
    optimizer: Literal["dual_annealing", "uniform_random"] = "dual_annealing"
    backend: Literal["psytaliro", "random", "auto"] = "psytaliro"
    trace_diagnostics: bool = True
    output_csv: Path | None = None
    output_png: Path | None = None

    @property
    def max_iterations_per_signal_generation(self) -> int:
        """Maximum falsification iterations used for one witness-generation attempt."""

        return self.iterations


def load_config(path: str | Path) -> FalsificationConfig | DifferentialConfig:
    """Load a JSON falsification or differential-analysis configuration."""

    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))

    if _is_differential_config(data):
        return _load_differential_config(config_path, data)
    return _load_falsification_config(config_path, data)


def _load_falsification_config(
    config_path: Path, data: Mapping[str, Any]
) -> FalsificationConfig:
    spec_path = _resolve_path(config_path, data["specification"])
    variables = {
        name: _load_variable_config(value) for name, value in data.get("variables", {}).items()
    }

    targets = _load_targets(data.get("target", data.get("targets", "falsify")))
    output_csv = _optional_path(config_path, data.get("output_csv"))
    output_png = _optional_path(config_path, data.get("output_png"))

    return FalsificationConfig(
        specification=spec_path,
        variables=variables,
        targets=targets,
        horizon=float(data.get("horizon", 10.0)),
        dt=_optional_float(data.get("dt")),
        sample_rate=_time_base_sample_rate(data),
        runs=int(data.get("runs", 1)),
        iterations=_iterations(data),
        witnesses=_positive_int(data.get("witnesses", 1), "witnesses"),
        seed=int(data.get("seed", 1)),
        optimizer=data.get("optimizer", "dual_annealing"),
        backend=data.get("backend", "psytaliro"),
        trace_diagnostics=_bool(data.get("trace_diagnostics", True), "trace_diagnostics"),
        output_csv=output_csv,
        output_png=output_png,
    )


def _load_differential_config(
    config_path: Path, data: Mapping[str, Any]
) -> DifferentialConfig:
    specs = data.get("specifications", {})
    phi1 = data.get("phi1", specs.get("phi1"))
    phi2 = data.get("phi2", specs.get("phi2"))
    if phi1 is None or phi2 is None:
        raise ValueError("differential mode requires phi1 and phi2 specification paths")

    variables = {
        name: _load_variable_config(value) for name, value in data.get("variables", {}).items()
    }
    targets = _load_differential_targets(
        data.get("differential_targets", data.get("targets", "all"))
    )

    return DifferentialConfig(
        phi1=_resolve_path(config_path, phi1),
        phi2=_resolve_path(config_path, phi2),
        variables=variables,
        targets=targets,
        horizon=float(data.get("horizon", 10.0)),
        dt=_optional_float(data.get("dt")),
        sample_rate=_time_base_sample_rate(data),
        runs=int(data.get("runs", 1)),
        iterations=_iterations(data),
        witnesses=_positive_int(data.get("witnesses", 1), "witnesses"),
        seed=int(data.get("seed", 1)),
        optimizer=data.get("optimizer", "dual_annealing"),
        backend=data.get("backend", "psytaliro"),
        trace_diagnostics=_bool(data.get("trace_diagnostics", True), "trace_diagnostics"),
        output_csv=_optional_path(config_path, data.get("output_csv")),
        output_png=_optional_path(config_path, data.get("output_png")),
    )


def _load_variable_config(value: Mapping[str, Any]) -> VariableConfig:
    return VariableConfig(
        signal=str(value["signal"]),
        io=value.get("io"),
        parameters=dict(value.get("parameters", {})),
    )


def _load_targets(value: Any) -> tuple[Target, ...]:
    if isinstance(value, str):
        if value == "both":
            return ("satisfy", "falsify")
        values = [value]
    else:
        values = list(value)

    targets: list[Target] = []
    for target in values:
        if target not in {"satisfy", "falsify"}:
            raise ValueError("targets must contain 'satisfy', 'falsify', or use 'both'")
        targets.append(target)
    return tuple(targets)


def _load_differential_targets(value: Any) -> tuple[DifferentialTarget, ...]:
    if isinstance(value, str):
        if value in {"all", "four", "both_modes"}:
            return DIFFERENTIAL_TARGETS
        values = [value]
    else:
        values = list(value)

    targets: list[DifferentialTarget] = []
    valid = set(DIFFERENTIAL_TARGETS)
    for target in values:
        if target not in valid:
            raise ValueError(
                "differential targets must be 'both', 'phi1_only', "
                "'phi2_only', 'neither', or 'all'"
            )
        targets.append(target)
    return tuple(targets)


def _is_differential_config(data: Mapping[str, Any]) -> bool:
    if data.get("mode") == "differential":
        return True
    return "phi1" in data or "phi2" in data or "specifications" in data


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _optional_path(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    return _resolve_path(config_path, value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _iterations(data: Mapping[str, Any]) -> int:
    value = data.get(
        "max_iterations_per_signal_generation",
        data.get("iterations", 100),
    )
    return _positive_int(value, "max_iterations_per_signal_generation")


def _positive_int(value: Any, name: str) -> int:
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be at least 1")
    return integer


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _time_base_sample_rate(data: Mapping[str, Any]) -> float | None:
    if "sample_rate" in data:
        return _optional_float(data.get("sample_rate"))
    if "dt" in data:
        return None
    return 100.0
