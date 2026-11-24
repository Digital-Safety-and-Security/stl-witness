"""Parameter-space construction for signal-generator search."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from stl_witness import signals

from .config import VariableConfig
from .specification import IaStlSpecification

SignalGenerator = Callable[..., signals.Signal]

_TIME_BASE_PARAMETERS = {"t", "duration", "dt", "sample_rate", "start"}
_SIGNAL_GENERATORS: Mapping[str, SignalGenerator] = {
    "step": signals.step,
    "ramp": signals.ramp,
    "sinusoid": signals.sinusoid,
    "sinc": signals.sinc,
    "pulse": signals.pulse,
    "chirp": signals.chirp,
    "noise": signals.noise,
    "step_with_overshoot": signals.step_with_overshoot,
    "transient_disturbance": signals.transient_disturbance,
    "monotonic_convergence": signals.monotonic_convergence,
    "overshooting_convergence": signals.overshooting_convergence,
    "oscillatory_convergence": signals.oscillatory_convergence,
    "oscillatory_response": signals.oscillatory_response,
    "first_order_response": signals.first_order_response,
    "second_order_response": signals.second_order_response,
}


@dataclass(frozen=True)
class SearchParameter:
    """One numeric signal parameter optimized by Psy-TaLiRo."""

    name: str
    variable: str
    signal: str
    parameter: str
    bounds: tuple[float, float]


@dataclass(frozen=True)
class VariableSignalBinding:
    """Fully resolved signal configuration for one STL variable."""

    variable: str
    io_type: str
    signal: str
    generator: SignalGenerator
    fixed_parameters: Mapping[str, Any]


@dataclass(frozen=True)
class SignalParameterSpace:
    """Mapping between Psy-TaLiRo vectors and signal-generator parameters."""

    bindings: tuple[VariableSignalBinding, ...]
    search_parameters: tuple[SearchParameter, ...]

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return {parameter.name: parameter.bounds for parameter in self.search_parameters}

    @classmethod
    def from_spec(
        cls,
        spec: IaStlSpecification,
        variables: Mapping[str, VariableConfig],
    ) -> "SignalParameterSpace":
        bindings: list[VariableSignalBinding] = []
        search_parameters: list[SearchParameter] = []

        declared_io = {decl.name: decl.io_type for decl in spec.declarations}

        for variable in spec.variables:
            if variable not in variables:
                raise ValueError(f"missing signal configuration for variable '{variable}'")

            config = variables[variable]
            signal_name = config.signal
            generator = get_signal_generator(signal_name)
            fixed, searchable = _partition_parameters(variable, signal_name, generator, config)
            io_type = config.io or declared_io[variable]

            if io_type != declared_io[variable]:
                raise ValueError(
                    f"variable '{variable}' is declared as {declared_io[variable]} "
                    f"but configured as {io_type}"
                )

            bindings.append(
                VariableSignalBinding(
                    variable=variable,
                    io_type=io_type,
                    signal=signal_name,
                    generator=generator,
                    fixed_parameters=fixed,
                )
            )
            search_parameters.extend(searchable)

        if not search_parameters:
            raise ValueError("at least one signal parameter must have optimization bounds")

        return cls(tuple(bindings), tuple(search_parameters))

    @classmethod
    def from_specs(
        cls,
        specs: tuple[IaStlSpecification, ...],
        variables: Mapping[str, VariableConfig],
    ) -> "SignalParameterSpace":
        """Create a parameter space from the union of variables in multiple specs."""

        declarations: dict[str, str] = {}
        for spec in specs:
            for declaration in spec.declarations:
                previous = declarations.get(declaration.name)
                if previous is not None and previous != declaration.io_type:
                    raise ValueError(
                        f"variable '{declaration.name}' has inconsistent IO declarations: "
                        f"{previous} vs {declaration.io_type}"
                    )
                declarations[declaration.name] = declaration.io_type

        merged_spec = IaStlSpecification(
            formula="true",
            declarations=tuple(
                # Preserve first-seen declaration order across the supplied specs.
                next(
                    declaration
                    for spec in specs
                    for declaration in spec.declarations
                    if declaration.name == name
                )
                for name in declarations
            ),
        )
        return cls.from_spec(merged_spec, variables)

    def parameters_from_sample(self, sample: Mapping[str, float]) -> dict[str, dict[str, Any]]:
        """Expand a flat Psy-TaLiRo sample into per-variable signal parameters."""

        expanded = {
            binding.variable: dict(binding.fixed_parameters) for binding in self.bindings
        }
        for parameter in self.search_parameters:
            expanded[parameter.variable][parameter.parameter] = float(sample[parameter.name])
        return expanded

    def generate_trace(
        self,
        t,
        sample: Mapping[str, float],
    ) -> dict[str, signals.Signal]:
        """Generate all variable signals for one Psy-TaLiRo sample."""

        parameters = self.parameters_from_sample(sample)
        trace: dict[str, signals.Signal] = {}
        for binding in self.bindings:
            trace[binding.variable] = binding.generator(t, **parameters[binding.variable])
        return trace


def available_signal_classes() -> tuple[str, ...]:
    """Return names accepted as variable signal classes."""

    return tuple(sorted(_SIGNAL_GENERATORS))


def get_signal_generator(name: str) -> SignalGenerator:
    try:
        return _SIGNAL_GENERATORS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown signal class '{name}'. Available: {', '.join(available_signal_classes())}"
        ) from exc


def signal_parameter_defaults(generator: SignalGenerator) -> dict[str, Any]:
    """Return generator parameters, excluding explicit time-base controls."""

    defaults: dict[str, Any] = {}
    signature = inspect.signature(generator)
    for name, parameter in signature.parameters.items():
        if name in _TIME_BASE_PARAMETERS:
            continue
        if parameter.default is inspect.Parameter.empty:
            continue
        defaults[name] = parameter.default
    return defaults


def _partition_parameters(
    variable: str,
    signal_name: str,
    generator: SignalGenerator,
    config: VariableConfig,
) -> tuple[dict[str, Any], list[SearchParameter]]:
    defaults = signal_parameter_defaults(generator)
    unknown = sorted(set(config.parameters) - set(defaults))
    if unknown:
        raise ValueError(
            f"unknown parameter(s) for {signal_name}: {', '.join(unknown)}"
        )

    fixed = dict(defaults)
    searchable: list[SearchParameter] = []

    for parameter, value in config.parameters.items():
        bounds = _as_bounds(value)
        if bounds is None:
            fixed[parameter] = value
            continue

        fixed.pop(parameter, None)
        searchable.append(
            SearchParameter(
                name=f"{variable}.{parameter}",
                variable=variable,
                signal=signal_name,
                parameter=parameter,
                bounds=bounds,
            )
        )

    return fixed, searchable


def _as_bounds(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = value
    if isinstance(low, bool) or isinstance(high, bool):
        return None
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    low_f = float(low)
    high_f = float(high)
    if low_f >= high_f:
        raise ValueError("parameter bounds must be [low, high] with low < high")
    return low_f, high_f
