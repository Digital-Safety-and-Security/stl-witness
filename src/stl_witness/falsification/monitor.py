"""RTAMT monitoring adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from stl_witness.signals import Signal

from .specification import IaStlSpecification


class RtamtMonitor:
    """Dense-time RTAMT monitor for IA-STL specifications."""

    def __init__(self, specification: IaStlSpecification):
        self.specification = specification
        self._rtamt = _import_rtamt()
        self._monitor = self._build_monitor()

    def evaluate(self, trace: Mapping[str, Signal]) -> float:
        """Evaluate robustness at time zero for a generated trace."""

        args = []
        for declaration in self.specification.declarations:
            signal = trace[declaration.name]
            samples = [[float(t), float(y)] for t, y in zip(signal.t, signal.y)]
            args.append([declaration.name, samples])

        robustness = self._monitor.evaluate(*args)
        return _extract_robustness(robustness)

    def _build_monitor(self) -> Any:
        monitor = self._rtamt.StlDenseTimeSpecification()
        monitor.name = "stl-witness IA-STL dense-time monitor"
        monitor.declare_var(self.specification.output_name, "float")

        for declaration in self.specification.declarations:
            monitor.declare_var(declaration.name, "float")
            monitor.set_var_io_type(declaration.name, declaration.io_type)

        monitor.spec = self.specification.rtamt_formula()
        monitor.parse()
        return monitor


def _import_rtamt() -> Any:
    try:
        import rtamt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RTAMT is required for robustness evaluation. Install project "
            "dependencies so `rtamt @ git+https://github.com/nickovic/rtamt.git@master` "
            "is available."
        ) from exc
    return rtamt


def _extract_robustness(value: Any) -> float:
    if isinstance(value, (int, float)):
        return _validate_robustness(float(value))

    if isinstance(value, tuple) and len(value) >= 2:
        return _extract_robustness(value[1])

    if isinstance(value, list):
        if not value:
            raise ValueError("RTAMT returned an empty robustness result")
        first = value[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return _extract_robustness(first[1])
        return _extract_robustness(first)

    try:
        robustness = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"unsupported RTAMT robustness result: {value!r}") from exc

    return _validate_robustness(robustness)


def _validate_robustness(robustness: float) -> float:
    if math.isnan(robustness):
        raise ValueError("RTAMT returned NaN robustness")
    return robustness
