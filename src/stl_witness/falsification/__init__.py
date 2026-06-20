"""Psy-TaLiRo/RTAMT integration for signal-witness generation."""

from .config import (
    DifferentialConfig,
    DifferentialTarget,
    FalsificationConfig,
    Target,
    VariableConfig,
    load_config,
)
from .io import plot_trace, save_trace_csv
from .parameters import SignalParameterSpace, available_signal_classes
from .runner import (
    DifferentialResult,
    DifferentialTargetResult,
    FalsificationResult,
    TargetResult,
    run_differential,
    run_falsification,
)
from .specification import IaStlSpecification, read_ia_stl_file

__all__ = [
    "DifferentialConfig",
    "DifferentialResult",
    "DifferentialTarget",
    "DifferentialTargetResult",
    "FalsificationConfig",
    "FalsificationResult",
    "IaStlSpecification",
    "SignalParameterSpace",
    "Target",
    "TargetResult",
    "VariableConfig",
    "available_signal_classes",
    "load_config",
    "plot_trace",
    "read_ia_stl_file",
    "run_differential",
    "run_falsification",
    "save_trace_csv",
]
