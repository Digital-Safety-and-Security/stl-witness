"""End-to-end witness search orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import numpy as np

from .config import DifferentialConfig, DifferentialTarget, FalsificationConfig, Target
from .monitor import RtamtMonitor
from .objective import (
    CandidateTrace,
    DifferentialCandidateTrace,
    DifferentialRobustnessObjective,
    RobustnessObjective,
    best_candidate,
    best_differential_candidate,
)
from .parameters import SignalParameterSpace
from .specification import read_ia_stl_file

SearchCandidate = CandidateTrace | DifferentialCandidateTrace
WitnessCandidate = TypeVar("WitnessCandidate", CandidateTrace, DifferentialCandidateTrace)


@dataclass(frozen=True)
class TargetResult:
    """Search result for one target polarity."""

    target: Target
    found: bool
    best: CandidateTrace | None
    witnesses: tuple[CandidateTrace, ...]
    evaluations: tuple[CandidateTrace, ...]


@dataclass(frozen=True)
class FalsificationResult:
    """Search result for all requested targets."""

    results: Mapping[Target, TargetResult]

    def __getitem__(self, target: Target) -> TargetResult:
        return self.results[target]


@dataclass(frozen=True)
class DifferentialTargetResult:
    """Search result for one differential truth quadrant."""

    target: DifferentialTarget
    found: bool
    best: DifferentialCandidateTrace | None
    witnesses: tuple[DifferentialCandidateTrace, ...]
    evaluations: tuple[DifferentialCandidateTrace, ...]


@dataclass(frozen=True)
class DifferentialResult:
    """Search result for all requested differential targets."""

    results: Mapping[DifferentialTarget, DifferentialTargetResult]

    def __getitem__(self, target: DifferentialTarget) -> DifferentialTargetResult:
        return self.results[target]


def run_falsification(config: FalsificationConfig) -> FalsificationResult:
    """Run witness search for all requested targets."""

    specification = read_ia_stl_file(config.specification)
    parameter_space = SignalParameterSpace.from_spec(specification, config.variables)
    monitor = RtamtMonitor(specification)
    results: dict[Target, TargetResult] = {}

    for target in config.targets:
        objective = RobustnessObjective(
            target=target,
            parameter_space=parameter_space,
            monitor=monitor,
            horizon=config.horizon,
            dt=config.dt,
            sample_rate=config.sample_rate,
            trace_diagnostics=config.trace_diagnostics,
        )
        results[target] = _run_target(config, target, parameter_space, objective)

    return FalsificationResult(results=results)


def run_differential(config: DifferentialConfig) -> DifferentialResult:
    """Run two-spec differential witness search."""

    phi1 = read_ia_stl_file(config.phi1)
    phi2 = read_ia_stl_file(config.phi2)
    parameter_space = SignalParameterSpace.from_specs((phi1, phi2), config.variables)
    monitors = {"phi1": RtamtMonitor(phi1), "phi2": RtamtMonitor(phi2)}
    results: dict[DifferentialTarget, DifferentialTargetResult] = {}

    for target in config.targets:
        objective = DifferentialRobustnessObjective(
            target=target,
            parameter_space=parameter_space,
            monitors=monitors,
            horizon=config.horizon,
            dt=config.dt,
            sample_rate=config.sample_rate,
            trace_diagnostics=config.trace_diagnostics,
        )
        results[target] = _run_differential_target(
            config, target, parameter_space, objective
        )

    return DifferentialResult(results=results)


def _run_target(
    config: FalsificationConfig,
    target: Target,
    parameter_space: SignalParameterSpace,
    objective: RobustnessObjective,
) -> TargetResult:
    evaluations = _run_witness_search(config, parameter_space, objective)
    witnesses = _select_target_witnesses(
        evaluations,
        target,
        parameter_space,
        config.witnesses,
    )
    best = witnesses[0] if witnesses else best_candidate(evaluations, target)
    return TargetResult(
        target=target,
        found=bool(witnesses),
        best=best,
        witnesses=witnesses,
        evaluations=tuple(evaluations),
    )


def _run_differential_target(
    config: DifferentialConfig,
    target: DifferentialTarget,
    parameter_space: SignalParameterSpace,
    objective: DifferentialRobustnessObjective,
) -> DifferentialTargetResult:
    evaluations = _run_witness_search(config, parameter_space, objective)
    witnesses = _select_differential_witnesses(
        evaluations,
        parameter_space,
        config.witnesses,
    )
    best = witnesses[0] if witnesses else best_differential_candidate(evaluations, target)
    return DifferentialTargetResult(
        target=target,
        found=bool(witnesses),
        best=best,
        witnesses=witnesses,
        evaluations=tuple(evaluations),
    )


def _run_witness_search(
    config: FalsificationConfig | DifferentialConfig,
    parameter_space: SignalParameterSpace,
    objective: RobustnessObjective | DifferentialRobustnessObjective,
) -> list[Any]:
    evaluations: list[Any] = []

    for generation in range(config.witnesses):
        generation_config = _search_config_for_generation(config, generation)
        evaluations.extend(_run_search(generation_config, parameter_space, objective))
        if _accepted_count(evaluations) >= config.witnesses:
            break

    return evaluations


def _search_config_for_generation(
    config: FalsificationConfig | DifferentialConfig, generation: int
) -> FalsificationConfig | DifferentialConfig:
    if generation == 0:
        return config
    return replace(config, seed=config.seed + generation)


def _run_search(
    config: FalsificationConfig | DifferentialConfig,
    parameter_space: SignalParameterSpace,
    objective: RobustnessObjective | DifferentialRobustnessObjective,
) -> list[Any]:
    if config.backend == "psytaliro":
        return _run_psytaliro(config, parameter_space, objective)
    if config.backend == "random":
        return _run_random(config, parameter_space, objective)

    try:
        return _run_psytaliro(config, parameter_space, objective)
    except RuntimeError:
        return _run_random(config, parameter_space, objective)


def _run_psytaliro(
    config: FalsificationConfig | DifferentialConfig,
    parameter_space: SignalParameterSpace,
    objective: RobustnessObjective | DifferentialRobustnessObjective,
) -> list[Any]:
    try:
        import staliro
        from staliro import optimizers
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Psy-TaLiRo is required for the psytaliro backend. Install `psy-taliro`."
        ) from exc

    @staliro.costfunc
    def cost_function(sample: Any) -> Any:
        candidate = objective.evaluate(sample.static)
        return staliro.Result(candidate.cost, candidate)

    options = staliro.TestOptions(
        runs=config.runs,
        iterations=config.max_iterations_per_signal_generation,
        seed=config.seed,
        static_inputs=parameter_space.bounds,
    )
    optimizer = _make_optimizer(config, optimizers)
    runs = staliro.staliro(cost_function, optimizer, options)

    evaluations: list[Any] = []
    for run in runs:
        evaluations.extend(evaluation.extra for evaluation in run.evaluations)
    return evaluations


def _accepted_count(candidates: list[Any]) -> int:
    return sum(1 for candidate in candidates if candidate.accepted)


def _select_target_witnesses(
    candidates: list[CandidateTrace],
    target: Target,
    parameter_space: SignalParameterSpace,
    count: int,
) -> tuple[CandidateTrace, ...]:
    def quality(candidate: CandidateTrace) -> float:
        if target == "satisfy":
            return candidate.robustness
        return -candidate.robustness

    return _select_diverse_witnesses(candidates, parameter_space, count, quality)


def _select_differential_witnesses(
    candidates: list[DifferentialCandidateTrace],
    parameter_space: SignalParameterSpace,
    count: int,
) -> tuple[DifferentialCandidateTrace, ...]:
    return _select_diverse_witnesses(
        candidates,
        parameter_space,
        count,
        lambda candidate: min(candidate.margins.values()),
    )


def _select_diverse_witnesses(
    candidates: list[WitnessCandidate],
    parameter_space: SignalParameterSpace,
    count: int,
    quality: Callable[[WitnessCandidate], float],
) -> tuple[WitnessCandidate, ...]:
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if not accepted:
        return tuple()

    selected = [max(accepted, key=quality)]
    remaining = [candidate for candidate in accepted if candidate is not selected[0]]

    while len(selected) < count and remaining:
        next_candidate = max(
            remaining,
            key=lambda candidate: (
                _min_normalized_parameter_distance(
                    candidate,
                    selected,
                    parameter_space,
                ),
                quality(candidate),
            ),
        )
        selected.append(next_candidate)
        remaining.remove(next_candidate)

    return tuple(selected)


def _min_normalized_parameter_distance(
    candidate: SearchCandidate,
    selected: list[SearchCandidate],
    parameter_space: SignalParameterSpace,
) -> float:
    return min(
        _normalized_parameter_distance(candidate, other, parameter_space)
        for other in selected
    )


def _normalized_parameter_distance(
    left: SearchCandidate,
    right: SearchCandidate,
    parameter_space: SignalParameterSpace,
) -> float:
    total = 0.0
    for parameter in parameter_space.search_parameters:
        low, high = parameter.bounds
        scale = high - low
        left_value = (float(left.parameters[parameter.name]) - low) / scale
        right_value = (float(right.parameters[parameter.name]) - low) / scale
        total += (left_value - right_value) ** 2
    return float(np.sqrt(total / len(parameter_space.search_parameters)))


def _make_optimizer(config: FalsificationConfig | DifferentialConfig, optimizers: Any) -> Any:
    if config.optimizer == "uniform_random":
        return optimizers.UniformRandom(min_cost=0.0)
    if config.optimizer == "dual_annealing":
        return optimizers.DualAnnealing(min_cost=0.0)
    raise ValueError("optimizer must be 'dual_annealing' or 'uniform_random'")


def _run_random(
    config: FalsificationConfig | DifferentialConfig,
    parameter_space: SignalParameterSpace,
    objective: RobustnessObjective | DifferentialRobustnessObjective,
) -> list[Any]:
    rng = np.random.default_rng(config.seed)
    bounds = list(parameter_space.bounds.values())
    evaluations: list[Any] = []
    budget = config.runs * config.max_iterations_per_signal_generation

    for _ in range(budget):
        values = [rng.uniform(low, high) for low, high in bounds]
        candidate = objective.evaluate_vector(values)
        evaluations.append(candidate)
        if candidate.accepted:
            break

    return evaluations
