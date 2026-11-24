"""Robustness objectives used by search backends."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from stl_witness.signals import Signal, time_vector

from .config import DifferentialTarget, Target
from .monitor import ExplanationPolarity, RtamtMonitor, TraceExplanation
from .parameters import SignalParameterSpace

_VACUOUS_OR_INVALID_COST = 1.0e12


@dataclass(frozen=True)
class CandidateTrace:
    """One evaluated trace and its robustness metadata."""

    target: Target
    robustness: float
    cost: float
    accepted: bool
    parameters: Mapping[str, float]
    trace: Mapping[str, Signal]
    explanations: Mapping[str, tuple[TraceExplanation, ...]] = field(default_factory=dict)


class RobustnessObjective:
    """Generate signals, evaluate RTAMT robustness, and expose a scalar cost."""

    def __init__(
        self,
        *,
        target: Target,
        parameter_space: SignalParameterSpace,
        monitor: RtamtMonitor,
        horizon: float,
        dt: float | None,
        sample_rate: float | None,
        trace_diagnostics: bool = True,
    ):
        self.target = target
        self.parameter_space = parameter_space
        self.monitor = monitor
        self.trace_diagnostics = trace_diagnostics
        self.t = time_vector(horizon, dt=dt, sample_rate=sample_rate)

    def evaluate(self, sample: Mapping[str, float]) -> CandidateTrace:
        parameters = {name: float(value) for name, value in sample.items()}
        trace = self.parameter_space.generate_trace(self.t, parameters)
        robustness = self.monitor.evaluate(trace)
        polarity = explanation_polarity(robustness)
        cost = robustness_to_cost(robustness, self.target)
        return CandidateTrace(
            target=self.target,
            robustness=robustness,
            cost=cost,
            accepted=is_target_hit(robustness, self.target),
            parameters=parameters,
            trace=trace,
            explanations=(
                self.monitor.explain(trace, polarity) if self.trace_diagnostics else {}
            ),
        )

    def evaluate_vector(self, values: NDArray[np.float64] | list[float]) -> CandidateTrace:
        sample = {
            parameter.name: float(value)
            for parameter, value in zip(self.parameter_space.search_parameters, values)
        }
        return self.evaluate(sample)


@dataclass(frozen=True)
class DifferentialCandidateTrace:
    """One evaluated trace against two specifications."""

    target: DifferentialTarget
    robustness: Mapping[str, float]
    margins: Mapping[str, float]
    cost: float
    accepted: bool
    parameters: Mapping[str, float]
    trace: Mapping[str, Signal]
    explanations: Mapping[str, tuple[TraceExplanation, ...]] = field(default_factory=dict)


class DifferentialRobustnessObjective:
    """Evaluate one generated trace against phi1 and phi2."""

    def __init__(
        self,
        *,
        target: DifferentialTarget,
        parameter_space: SignalParameterSpace,
        monitors: Mapping[str, RtamtMonitor],
        horizon: float,
        dt: float | None,
        sample_rate: float | None,
        trace_diagnostics: bool = True,
    ):
        self.target = target
        self.parameter_space = parameter_space
        self.monitors = dict(monitors)
        self.trace_diagnostics = trace_diagnostics
        self.t = time_vector(horizon, dt=dt, sample_rate=sample_rate)

    def evaluate(self, sample: Mapping[str, float]) -> DifferentialCandidateTrace:
        parameters = {name: float(value) for name, value in sample.items()}
        trace = self.parameter_space.generate_trace(self.t, parameters)
        robustness = {
            name: monitor.evaluate(trace) for name, monitor in self.monitors.items()
        }
        explanations = (
            _merge_explanations(
                monitor.explain(trace, explanation_polarity(robustness[name]))
                for name, monitor in self.monitors.items()
            )
            if self.trace_diagnostics
            else {}
        )
        margins = differential_margins(robustness, self.target)
        cost = differential_robustness_to_cost(robustness, self.target)
        return DifferentialCandidateTrace(
            target=self.target,
            robustness=robustness,
            margins=margins,
            cost=cost,
            accepted=is_differential_target_hit(robustness, self.target),
            parameters=parameters,
            trace=trace,
            explanations=explanations,
        )

    def evaluate_vector(
        self, values: NDArray[np.float64] | list[float]
    ) -> DifferentialCandidateTrace:
        sample = {
            parameter.name: float(value)
            for parameter, value in zip(self.parameter_space.search_parameters, values)
        }
        return self.evaluate(sample)


def robustness_to_cost(robustness: float, target: Target) -> float:
    """Convert robustness to a minimization cost."""

    if target == "satisfy":
        if not math.isfinite(robustness):
            return _VACUOUS_OR_INVALID_COST
        return -robustness

    if math.isnan(robustness):
        return _VACUOUS_OR_INVALID_COST
    return robustness


def explanation_polarity(robustness: float) -> ExplanationPolarity:
    """Return the qualitative trace polarity used for explanation overlays."""

    return "satisfaction" if robustness >= 0.0 else "violation"


def differential_robustness_to_cost(
    robustness: Mapping[str, float], target: DifferentialTarget
) -> float:
    """Convert phi1/phi2 robustness values to a minimization cost."""

    margins = differential_margins(robustness, target)
    if any(math.isnan(margin) for margin in margins.values()):
        return _VACUOUS_OR_INVALID_COST
    return -min(margins.values())


def is_target_hit(robustness: float, target: Target) -> bool:
    if target == "satisfy":
        return math.isfinite(robustness) and robustness > 0.0
    return robustness < 0.0


def is_differential_target_hit(
    robustness: Mapping[str, float], target: DifferentialTarget
) -> bool:
    return all(margin > 0.0 for margin in differential_margins(robustness, target).values())


def differential_margins(
    robustness: Mapping[str, float], target: DifferentialTarget
) -> dict[str, float]:
    desired = _differential_truth_table(target)
    margins: dict[str, float] = {}
    for name, should_satisfy in desired.items():
        value = float(robustness[name])
        if math.isnan(value):
            margins[name] = float("nan")
        elif should_satisfy:
            # Infinite positive robustness is treated as vacuous satisfaction.
            margins[name] = value if math.isfinite(value) else -_VACUOUS_OR_INVALID_COST
        else:
            margins[name] = -value
    return margins


def best_candidate(candidates: list[CandidateTrace], target: Target) -> CandidateTrace | None:
    if not candidates:
        return None
    accepted = [candidate for candidate in candidates if candidate.accepted]
    pool = accepted or candidates
    if target == "satisfy":
        finite = [candidate for candidate in pool if math.isfinite(candidate.robustness)]
        pool = finite or pool
        return max(pool, key=lambda candidate: candidate.robustness)
    return min(pool, key=lambda candidate: candidate.robustness)


def best_differential_candidate(
    candidates: list[DifferentialCandidateTrace], target: DifferentialTarget
) -> DifferentialCandidateTrace | None:
    if not candidates:
        return None
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if accepted:
        return max(accepted, key=lambda candidate: min(candidate.margins.values()))
    return min(candidates, key=lambda candidate: candidate.cost)


def _merge_explanations(
    explanations_by_spec,
) -> dict[str, tuple[TraceExplanation, ...]]:
    merged: dict[str, list[TraceExplanation]] = {}
    for explanations in explanations_by_spec:
        for variable, variable_explanations in explanations.items():
            merged.setdefault(variable, []).extend(variable_explanations)
    return {
        variable: tuple(explanations)
        for variable, explanations in merged.items()
    }


def _differential_truth_table(target: DifferentialTarget) -> dict[str, bool]:
    if target == "both":
        return {"phi1": True, "phi2": True}
    if target == "phi1_only":
        return {"phi1": True, "phi2": False}
    if target == "phi2_only":
        return {"phi1": False, "phi2": True}
    if target == "neither":
        return {"phi1": False, "phi2": False}
    raise ValueError(f"unknown differential target: {target}")
