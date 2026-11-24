"""RTAMT monitoring adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from stl_witness.signals import Signal

from .specification import IaStlSpecification

ExplanationPolarity = Literal["satisfaction", "violation"]


@dataclass(frozen=True)
class TraceExplanation:
    """Time-domain explanation overlays for one trace variable."""

    polarity: ExplanationPolarity
    intervals: tuple[tuple[float, float], ...] = tuple()
    points: tuple[float, ...] = tuple()


class RtamtMonitor:
    """Discrete-time RTAMT monitor for STL witness evaluation."""

    def __init__(self, specification: IaStlSpecification):
        self.specification = specification
        self._rtamt = _import_rtamt()
        self._monitor = self._build_monitor()

    def evaluate(self, trace: Mapping[str, Signal]) -> float:
        """Evaluate robustness at time zero for a generated trace."""

        robustness = self._evaluate_raw(trace)
        return _extract_robustness(robustness)

    def explain(
        self,
        trace: Mapping[str, Signal],
        polarity: ExplanationPolarity,
    ) -> dict[str, tuple[TraceExplanation, ...]]:
        """Compute RTAMT explanation overlays for trace variables."""

        self._evaluate_raw(trace)
        explainer = getattr(self._monitor, "explainer", None)
        if explainer is None:
            return {}

        try:
            explainer.explanations = {}
            if not self._run_explainer(explainer, polarity):
                return {}
        except Exception:
            return {}

        trace_variables = tuple(
            declaration.name for declaration in self.specification.declarations
        )
        return _normalize_explanations(
            getattr(explainer, "explanations", {}),
            trace,
            trace_variables,
            polarity,
        )

    def _evaluate_raw(self, trace: Mapping[str, Signal]) -> Any:
        return self._monitor.evaluate(_dataset(trace, self.specification))

    def _run_explainer(
        self,
        explainer: Any,
        polarity: ExplanationPolarity,
    ) -> bool:
        specs = getattr(self._monitor, "specs", None)
        if specs is not None and hasattr(explainer, "visit"):
            explainer.spec = self._monitor
            satisfied = polarity == "satisfaction"
            for spec in specs:
                explainer.visit(spec, [[[0, 0]], satisfied])
            return True

        if hasattr(self._monitor, "explain"):
            self._monitor.explain()
            return True

        return False

    def _build_monitor(self) -> Any:
        try:
            monitor = self._rtamt.StlDiscreteTimeOfflineSpecification()
        except AttributeError:
            monitor = self._rtamt.StlDiscreteTimeSpecification()

        monitor.name = "stl-witness STL discrete-time monitor"
        monitor.declare_var(self.specification.output_name, "float")

        for declaration in self.specification.declarations:
            monitor.declare_var(declaration.name, "float")
            if hasattr(monitor, "set_var_io_type"):
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


def _dataset(
    trace: Mapping[str, Signal],
    specification: IaStlSpecification,
) -> dict[str, list[float]]:
    dataset = {"time": []}
    for declaration in specification.declarations:
        signal = trace[declaration.name]
        if not dataset["time"]:
            dataset["time"] = [float(value) for value in signal.t]
        dataset[declaration.name] = [float(value) for value in signal.y]
    return dataset


def _normalize_explanations(
    raw: Mapping[Any, Any],
    trace: Mapping[str, Signal],
    variables: tuple[str, ...],
    polarity: ExplanationPolarity,
) -> dict[str, tuple[TraceExplanation, ...]]:
    explanations: dict[str, tuple[TraceExplanation, ...]] = {}

    for variable in variables:
        if variable not in raw or variable not in trace:
            continue

        intervals, points = _normalize_explanation_items(raw[variable], trace[variable])
        if intervals or points:
            explanations[variable] = (
                TraceExplanation(
                    polarity=polarity,
                    intervals=intervals,
                    points=points,
                ),
            )

    return explanations


def _normalize_explanation_items(
    raw: Any,
    signal: Signal,
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    interval_items, point_items = _split_explanation_items(raw)
    time_values = tuple(float(value) for value in signal.t)
    intervals: set[tuple[float, float]] = set()
    points: set[float] = set()

    for item in interval_items:
        interval = _normalize_interval(item, time_values)
        if interval is None:
            continue
        start, end = interval
        if start == end:
            points.add(start)
        else:
            intervals.add((start, end))

    for item in point_items:
        point = _normalize_point(item, time_values)
        if point is not None:
            points.add(point)

    return tuple(sorted(intervals)), tuple(sorted(points))


def _split_explanation_items(raw: Any) -> tuple[Any, Any]:
    if isinstance(raw, Mapping):
        intervals = raw.get("intervals", raw.get("ranges", tuple()))
        points = raw.get("points", raw.get("singular_points", tuple()))
        return intervals, points
    return raw or tuple(), tuple()


def _normalize_interval(
    item: Any,
    time_values: tuple[float, ...],
) -> tuple[float, float] | None:
    if hasattr(item, "start") and hasattr(item, "end"):
        raw_start = item.start
        raw_end = item.end
    else:
        if isinstance(item, (int, float)):
            point = _normalize_point(item, time_values)
            return None if point is None else (point, point)
        try:
            values = list(item)
        except TypeError:
            return None
        if len(values) != 2:
            return None
        raw_start, raw_end = values

    start = _normalize_point(raw_start, time_values)
    end = _normalize_point(raw_end, time_values)
    if start is None or end is None:
        return None
    if start <= end:
        return start, end
    return end, start


def _normalize_point(value: Any, time_values: tuple[float, ...]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        index = int(number)
        if 0 <= index < len(time_values):
            return time_values[index]
    return number


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
