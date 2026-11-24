"""Parameterized signal generators for simulations and control examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_SAMPLE_RATE = 100.0
_EPS = 1e-12


@dataclass(frozen=True)
class Signal:
    """A generated signal and the metadata needed to reproduce it."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    name: str = "signal"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = _as_1d_float_array(self.t, "t")
        y = _as_1d_float_array(self.y, "y")
        if t.shape != y.shape:
            raise ValueError("t and y must have the same shape")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "parameters", dict(self.parameters))

    def __len__(self) -> int:
        return self.t.size

    def __iter__(self) -> Iterable[NDArray[np.float64]]:
        yield self.t
        yield self.y

    def __array__(
        self, dtype: np.dtype[Any] | None = None, copy: bool | None = None
    ) -> NDArray[np.float64]:
        values = np.asarray(self.y, dtype=dtype)
        return values.copy() if copy else values

    @property
    def sample_period(self) -> float:
        """Median sample period, or NaN for a single-sample signal."""

        if self.t.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.t)))

    @property
    def sample_rate(self) -> float:
        """Median sample rate in Hz, or NaN for a single-sample signal."""

        period = self.sample_period
        return float("nan") if period == 0 or np.isnan(period) else 1.0 / period

    def as_tuple(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(t, y)`` for APIs that expect separate arrays."""

        return self.t, self.y


def time_vector(
    duration: float,
    *,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    endpoint: bool = True,
) -> NDArray[np.float64]:
    """Create a uniformly sampled time vector.

    Parameters
    ----------
    duration:
        Nonnegative horizon length in seconds.
    dt:
        Sample period in seconds. Mutually exclusive with ``sample_rate``.
    sample_rate:
        Sample rate in Hz. Defaults to 100 Hz when neither ``dt`` nor
        ``sample_rate`` is provided.
    start:
        First sample time in seconds.
    endpoint:
        Include samples up to the duration boundary when it falls on the grid.
    """

    duration = _nonnegative("duration", duration)
    start = _finite("start", start)
    dt = _resolve_dt(dt, sample_rate)

    if duration == 0:
        return np.array([start], dtype=float)

    intervals = int(np.floor(duration / dt + _EPS))
    count = intervals + 1 if endpoint else intervals
    count = max(1, count)
    return start + np.arange(count, dtype=float) * dt


def step(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    delay: float | None = None,
    step_time: float | None = None,
    signal_duration: float | None = None,
    initial: float = 0.0,
    offset: float = 0.0,
) -> Signal:
    """Generate a step signal.

    ``amplitude`` is the increment applied after ``delay``. The value before
    the step is ``offset + initial`` and the value after the step is
    ``offset + initial + amplitude``. ``step_time`` is kept as a backwards
    compatible alias for ``delay``.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    delay = _resolve_delay(delay, legacy_name="step_time", legacy_value=step_time)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    initial = _finite("initial", initial)
    offset = _finite("offset", offset)
    if signal_duration is None:
        active = time >= delay
    else:
        active = _active_window(time, delay, signal_duration, include_end=True)
    values = offset + np.where(active, initial + amplitude, initial)
    return _signal(
        "step",
        time,
        values,
        amplitude=amplitude,
        delay=delay,
        step_time=delay,
        signal_duration=signal_duration,
        initial=initial,
        offset=offset,
    )


def ramp(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    slope: float = 1.0,
    target: float | None = None,
    delay: float | None = None,
    start_time: float | None = None,
    signal_duration: float | None = None,
    initial: float = 0.0,
    offset: float = 0.0,
) -> Signal:
    """Generate a ramp with configurable initial value, target, start, and slope.

    The signal holds ``offset + initial`` until ``delay``. When ``target`` is
    provided, the ramp moves toward ``offset + target`` at ``abs(slope)`` after
    ``delay`` and holds the target value once it is reached. ``signal_duration``
    may still cap the active ramping time, in which case the reached value is
    held even if the target has not been reached. When ``target`` is omitted,
    the legacy duration-based form ``offset + initial + slope * tau`` is used.
    ``start_time`` is kept as a backwards compatible alias for ``delay``.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    slope = _finite("slope", slope)
    target = None if target is None else _finite("target", target)
    delay = _resolve_delay(delay, legacy_name="start_time", legacy_value=start_time)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    initial = _finite("initial", initial)
    offset = _finite("offset", offset)
    tau = _elapsed_time(time, delay, signal_duration)
    if target is None:
        values = offset + initial + slope * tau
    else:
        values = offset + _targeted_ramp_values(
            tau,
            initial=initial,
            target=target,
            slope=slope,
        )
    return _signal(
        "ramp",
        time,
        values,
        slope=slope,
        target=target,
        delay=delay,
        start_time=delay,
        signal_duration=signal_duration,
        initial=initial,
        offset=offset,
    )


def sinusoid(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase: float = 0.0,
    offset: float = 0.0,
    delay: float | None = None,
    start_time: float | None = None,
    signal_duration: float | None = None,
) -> Signal:
    """Generate a sinusoid.

    ``frequency`` is specified in Hz and ``phase`` in radians. When
    ``signal_duration`` is provided, the sinusoid is emitted as a finite burst
    and returns to ``offset`` outside the active interval. ``start_time`` is
    kept as a backwards compatible alias for ``delay``.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    frequency = _nonnegative("frequency", frequency)
    phase = _finite("phase", phase)
    offset = _finite("offset", offset)
    delay = _resolve_delay(delay, legacy_name="start_time", legacy_value=start_time)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    active = _active_window(time, delay, signal_duration, include_end=True)
    tau = np.maximum(time - delay, 0.0)
    wave = amplitude * np.sin(2.0 * np.pi * frequency * tau + phase)
    values = np.where(active, offset + wave, offset)
    return _signal(
        "sinusoid",
        time,
        values,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        offset=offset,
        delay=delay,
        start_time=delay,
        signal_duration=signal_duration,
    )


def sinc(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    delay: float = 0.0,
    center: float = 0.0,
    width: float = 1.0,
    signal_duration: float | None = None,
    offset: float = 0.0,
) -> Signal:
    """Generate a delayed normalized sinc signal, ``sin(pi x) / (pi x)``.

    ``center`` is relative to ``delay``. When ``signal_duration`` is provided,
    the sinc is windowed to ``[delay, delay + signal_duration]``.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    delay = _finite("delay", delay)
    center = _finite("center", center)
    width = _positive("width", width)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    offset = _finite("offset", offset)
    active = _active_window(time, delay, signal_duration, include_end=True)
    values = offset + amplitude * np.sinc((time - (delay + center)) / width)
    values = np.where(active, values, offset)
    return _signal(
        "sinc",
        time,
        values,
        amplitude=amplitude,
        delay=delay,
        center=center,
        width=width,
        signal_duration=signal_duration,
        offset=offset,
    )


def pulse(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    delay: float | None = None,
    start_time: float | None = None,
    width: float = 1.0,
    period: float | None = None,
    duty_cycle: float | None = None,
    signal_duration: float | None = None,
    baseline: float = 0.0,
) -> Signal:
    """Generate a rectangular pulse train or a one-shot pulse.

    Set ``period`` for a repeating pulse. For periodic pulses, ``duty_cycle``
    may be used instead of an explicit ``width``. ``signal_duration`` bounds
    the total pulse-train duration after ``delay``. ``start_time`` is kept as a
    backwards compatible alias for ``delay``.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    delay = _resolve_delay(delay, legacy_name="start_time", legacy_value=start_time)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    baseline = _finite("baseline", baseline)

    if duty_cycle is not None:
        if period is None:
            raise ValueError("duty_cycle requires period")
        duty_cycle = _finite("duty_cycle", duty_cycle)
        if not 0.0 <= duty_cycle <= 1.0:
            raise ValueError("duty_cycle must be in [0, 1]")
        period = _positive("period", period)
        width = duty_cycle * period
    else:
        width = _positive("width", width)
        if period is not None:
            period = _positive("period", period)

    if period is not None and width - period > _EPS:
        raise ValueError("width must not exceed period")

    if period is None:
        active = (time >= delay) & (time < delay + width)
    else:
        phase = np.mod(time - delay, period)
        active = (time >= delay) & (phase < width)

    if signal_duration is not None:
        active &= time <= delay + signal_duration

    values = baseline + amplitude * active.astype(float)
    return _signal(
        "pulse",
        time,
        values,
        amplitude=amplitude,
        delay=delay,
        start_time=delay,
        width=width,
        period=period,
        duty_cycle=duty_cycle,
        signal_duration=signal_duration,
        baseline=baseline,
    )


def chirp(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    f0: float = 1.0,
    f1: float = 10.0,
    sweep_duration: float | None = None,
    method: str = "linear",
    phase: float = 0.0,
    offset: float = 0.0,
    delay: float | None = None,
    start_time: float | None = None,
    signal_duration: float | None = None,
) -> Signal:
    """Generate a swept-frequency sinusoid.

    Supported methods are ``"linear"``, ``"quadratic"``, and
    ``"logarithmic"``. Frequencies are specified in Hz and ``phase`` in
    radians. ``signal_duration`` emits a finite chirp burst that returns to
    ``offset`` after the active interval. ``start_time`` is kept as a backwards
    compatible alias for ``delay``.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    f0 = _nonnegative("f0", f0)
    f1 = _nonnegative("f1", f1)
    phase = _finite("phase", phase)
    offset = _finite("offset", offset)
    delay = _resolve_delay(delay, legacy_name="start_time", legacy_value=start_time)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    if sweep_duration is None:
        if signal_duration is not None:
            sweep_duration = max(signal_duration, _EPS)
        else:
            sweep_duration = max(float(time[-1] - delay), _EPS)
    sweep_duration = _positive("sweep_duration", sweep_duration)

    method_name = method.lower()
    tau_raw = time - delay
    tau = np.maximum(tau_raw, 0.0)
    tau_sweep = np.minimum(tau, sweep_duration)

    if method_name == "linear":
        rate = (f1 - f0) / sweep_duration
        cycles = f0 * tau_sweep + 0.5 * rate * tau_sweep**2
    elif method_name == "quadratic":
        cycles = f0 * tau_sweep + (f1 - f0) * tau_sweep**3 / (3.0 * sweep_duration**2)
    elif method_name in {"logarithmic", "log"}:
        if f0 <= 0.0 or f1 <= 0.0:
            raise ValueError("logarithmic chirp requires positive f0 and f1")
        ratio = f1 / f0
        if np.isclose(ratio, 1.0):
            cycles = f0 * tau_sweep
        else:
            cycles = f0 * sweep_duration * (ratio ** (tau_sweep / sweep_duration) - 1.0) / np.log(ratio)
    else:
        raise ValueError("method must be 'linear', 'quadratic', or 'logarithmic'")

    after_sweep = tau > sweep_duration
    cycles = np.asarray(cycles, dtype=float)
    cycles[after_sweep] += f1 * (tau[after_sweep] - sweep_duration)

    values = offset + amplitude * np.sin(2.0 * np.pi * cycles + phase)
    active = _active_window(time, delay, signal_duration, include_end=True)
    values[~active] = offset
    return _signal(
        "chirp",
        time,
        values,
        amplitude=amplitude,
        f0=f0,
        f1=f1,
        sweep_duration=sweep_duration,
        method=method_name,
        phase=phase,
        offset=offset,
        delay=delay,
        start_time=delay,
        signal_duration=signal_duration,
    )


def noise(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    mean: float = 0.0,
    distribution: str = "normal",
    delay: float = 0.0,
    signal_duration: float | None = None,
    seed: int | None = None,
) -> Signal:
    """Generate reproducible white noise.

    For ``distribution="normal"`` or ``"gaussian"``, ``amplitude`` is the
    standard deviation. For ``distribution="uniform"``, values are sampled from
    ``[mean - amplitude, mean + amplitude]``. Samples outside the delayed
    active interval are set to ``mean``.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _nonnegative("amplitude", amplitude)
    mean = _finite("mean", mean)
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    rng = np.random.default_rng(seed)
    distribution_name = distribution.lower()

    if distribution_name in {"normal", "gaussian"}:
        values = rng.normal(loc=mean, scale=amplitude, size=time.shape)
    elif distribution_name == "uniform":
        values = rng.uniform(low=mean - amplitude, high=mean + amplitude, size=time.shape)
    else:
        raise ValueError("distribution must be 'normal', 'gaussian', or 'uniform'")

    active = _active_window(time, delay, signal_duration, include_end=True)
    values = np.where(active, values, mean)

    return _signal(
        "noise",
        time,
        values,
        amplitude=amplitude,
        mean=mean,
        distribution=distribution_name,
        delay=delay,
        signal_duration=signal_duration,
        seed=seed,
    )


def step_with_overshoot(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    overshoot: float | None = None,
    overshoot_percent: float | None = None,
    settling_time: float = 2.0,
    settling_threshold: float = 0.02,
    delay: float = 0.0,
    signal_duration: float | None = None,
    initial: float = 0.0,
    offset: float = 0.0,
) -> Signal:
    """Generate a step response with a requested overshoot and settling time.

    ``overshoot`` is a fraction, so ``0.1`` means 10 percent. Use
    ``overshoot_percent`` when percentage notation is clearer. When
    ``signal_duration`` is provided, the response evolves for that many seconds
    after ``delay`` and then holds the reached value.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    overshoot_fraction = _resolve_overshoot(overshoot, overshoot_percent)
    settling_time = _positive("settling_time", settling_time)
    settling_threshold = _finite("settling_threshold", settling_threshold)
    if not 0.0 < settling_threshold < 1.0:
        raise ValueError("settling_threshold must be in (0, 1)")
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    initial = _finite("initial", initial)
    offset = _finite("offset", offset)

    damping_ratio = _damping_ratio_from_overshoot(overshoot_fraction)
    natural_frequency = -np.log(settling_threshold) / (damping_ratio * settling_time)
    tau = _elapsed_time(time, delay, signal_duration)
    unit_response = _second_order_unit_step(tau, damping_ratio, natural_frequency)
    values = offset + initial + amplitude * unit_response
    values[time < delay] = offset + initial

    return _signal(
        "step_with_overshoot",
        time,
        values,
        amplitude=amplitude,
        overshoot=overshoot_fraction,
        overshoot_percent=overshoot_fraction * 100.0,
        settling_time=settling_time,
        settling_threshold=settling_threshold,
        damping_ratio=damping_ratio,
        natural_frequency=natural_frequency,
        delay=delay,
        signal_duration=signal_duration,
        initial=initial,
        offset=offset,
    )


def transient_disturbance(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    nominal: float = 0.0,
    amplitude: float = 1.0,
    center: float = 1.0,
    width: float = 0.25,
    delay: float = 0.0,
    signal_duration: float | None = None,
) -> Signal:
    """Generate a smooth transient peak or valley around ``nominal``.

    The response uses a Gaussian disturbance,
    ``nominal + amplitude * exp(-0.5 * ((tau - center) / width)**2)``.
    Negative ``amplitude`` values create valleys. When ``signal_duration`` is
    provided, ``tau`` is capped after that many seconds and the reached value
    is held.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    nominal = _finite("nominal", nominal)
    amplitude = _finite("amplitude", amplitude)
    center = _nonnegative("center", center)
    width = _positive("width", width)
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    tau = _elapsed_time(time, delay, signal_duration)
    response = amplitude * np.exp(-0.5 * ((tau - center) / width) ** 2)
    values = nominal + response
    values[time < delay] = nominal
    return _signal(
        "transient_disturbance",
        time,
        values,
        nominal=nominal,
        amplitude=amplitude,
        center=center,
        width=width,
        delay=delay,
        signal_duration=signal_duration,
    )


def monotonic_convergence(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    initial: float = 0.0,
    target: float = 1.0,
    time_constant: float = 1.0,
    delay: float = 0.0,
    signal_duration: float | None = None,
) -> Signal:
    """Generate smooth monotonic convergence to ``target``.

    The response is ``target + (initial - target) * exp(-tau / time_constant)``.
    When ``signal_duration`` is provided, the response evolves for that many
    seconds after ``delay`` and then holds the reached value.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    initial = _finite("initial", initial)
    target = _finite("target", target)
    time_constant = _positive("time_constant", time_constant)
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    tau = _elapsed_time(time, delay, signal_duration)
    values = target + (initial - target) * np.exp(-tau / time_constant)
    values[time < delay] = initial
    return _signal(
        "monotonic_convergence",
        time,
        values,
        initial=initial,
        target=target,
        time_constant=time_constant,
        delay=delay,
        signal_duration=signal_duration,
    )


def overshooting_convergence(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    initial: float = 0.0,
    target: float = 1.0,
    time_constant: float = 1.0,
    overshoot: float = 0.2,
    peak_time: float = 1.0,
    delay: float = 0.0,
    signal_duration: float | None = None,
) -> Signal:
    """Generate smooth convergence with one overshoot lobe.

    The response is the sum of monotonic convergence and a decaying lobe:
    ``target + delta * (-exp(-tau / time_constant)
    + (overshoot + exp(-peak_time / time_constant))
    * (tau / peak_time) * exp(1 - tau / peak_time))``, where
    ``delta = target - initial``. The response is ``overshoot * delta`` beyond
    the target at ``tau = peak_time``.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    initial = _finite("initial", initial)
    target = _finite("target", target)
    time_constant = _positive("time_constant", time_constant)
    overshoot = _nonnegative("overshoot", overshoot)
    peak_time = _positive("peak_time", peak_time)
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    tau = _elapsed_time(time, delay, signal_duration)
    delta = target - initial
    lobe_scale = overshoot + np.exp(-peak_time / time_constant)
    lobe = (tau / peak_time) * np.exp(1.0 - tau / peak_time)
    values = target + delta * (-np.exp(-tau / time_constant) + lobe_scale * lobe)
    values[time < delay] = initial
    return _signal(
        "overshooting_convergence",
        time,
        values,
        initial=initial,
        target=target,
        time_constant=time_constant,
        overshoot=overshoot,
        peak_time=peak_time,
        lobe_scale=lobe_scale,
        delay=delay,
        signal_duration=signal_duration,
    )


def oscillatory_convergence(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    initial: float = 1.0,
    target: float = 0.0,
    damping_rate: float = 0.5,
    frequency: float = 1.0,
    delay: float = 0.0,
    signal_duration: float | None = None,
) -> Signal:
    """Generate smooth damped oscillatory convergence around ``target``.

    The response is
    ``target + (initial - target) * exp(-damping_rate * tau)
    * cos(2*pi*frequency*tau)``. When ``signal_duration`` is provided, the
    response evolves for that many seconds after ``delay`` and then holds the
    reached value.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    initial = _finite("initial", initial)
    target = _finite("target", target)
    damping_rate = _nonnegative("damping_rate", damping_rate)
    frequency = _nonnegative("frequency", frequency)
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    tau = _elapsed_time(time, delay, signal_duration)
    values = target + (initial - target) * np.exp(-damping_rate * tau) * np.cos(
        2.0 * np.pi * frequency * tau
    )
    values[time < delay] = initial
    return _signal(
        "oscillatory_convergence",
        time,
        values,
        initial=initial,
        target=target,
        damping_rate=damping_rate,
        frequency=frequency,
        delay=delay,
        signal_duration=signal_duration,
    )


def oscillatory_response(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    damping_ratio: float = 0.1,
    natural_frequency: float = 2.0 * np.pi,
    phase: float = 0.0,
    delay: float = 0.0,
    signal_duration: float | None = None,
    offset: float = 0.0,
) -> Signal:
    """Generate a damped oscillatory response around ``offset``.

    ``natural_frequency`` is specified in rad/s. When ``signal_duration`` is
    provided, the response evolves for that many seconds after ``delay`` and
    then holds the reached value.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    damping_ratio = _nonnegative("damping_ratio", damping_ratio)
    natural_frequency = _positive("natural_frequency", natural_frequency)
    phase = _finite("phase", phase)
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    offset = _finite("offset", offset)
    tau = _elapsed_time(time, delay, signal_duration)
    damped_frequency = natural_frequency * np.sqrt(max(0.0, 1.0 - damping_ratio**2))
    response = amplitude * np.exp(-damping_ratio * natural_frequency * tau) * np.cos(
        damped_frequency * tau + phase
    )
    values = offset + response
    values[time < delay] = offset
    return _signal(
        "oscillatory_response",
        time,
        values,
        amplitude=amplitude,
        damping_ratio=damping_ratio,
        natural_frequency=natural_frequency,
        damped_frequency=damped_frequency,
        phase=phase,
        delay=delay,
        signal_duration=signal_duration,
        offset=offset,
    )


def first_order_response(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    gain: float = 1.0,
    time_constant: float = 1.0,
    delay: float = 0.0,
    signal_duration: float | None = None,
    initial: float = 0.0,
    offset: float = 0.0,
) -> Signal:
    """Generate the step response of a first-order system.

    When ``signal_duration`` is provided, the response evolves for that many
    seconds after ``delay`` and then holds the reached value.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    gain = _finite("gain", gain)
    time_constant = _positive("time_constant", time_constant)
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    initial = _finite("initial", initial)
    offset = _finite("offset", offset)
    tau = _elapsed_time(time, delay, signal_duration)
    unit_response = 1.0 - np.exp(-tau / time_constant)
    values = offset + initial + gain * amplitude * unit_response
    values[time < delay] = offset + initial
    return _signal(
        "first_order_response",
        time,
        values,
        amplitude=amplitude,
        gain=gain,
        time_constant=time_constant,
        delay=delay,
        signal_duration=signal_duration,
        initial=initial,
        offset=offset,
    )


def second_order_response(
    t: ArrayLike | None = None,
    *,
    duration: float = 10.0,
    dt: float | None = None,
    sample_rate: float | None = None,
    start: float = 0.0,
    amplitude: float = 1.0,
    gain: float = 1.0,
    damping_ratio: float = 0.5,
    natural_frequency: float = 1.0,
    delay: float = 0.0,
    signal_duration: float | None = None,
    initial: float = 0.0,
    offset: float = 0.0,
) -> Signal:
    """Generate the step response of a canonical second-order system.

    The transfer function is ``gain * natural_frequency**2 /
    (s**2 + 2*damping_ratio*natural_frequency*s + natural_frequency**2)``.
    When ``signal_duration`` is provided, the response evolves for that many
    seconds after ``delay`` and then holds the reached value.
    """

    time = _resolve_time(t, duration, dt, sample_rate, start)
    amplitude = _finite("amplitude", amplitude)
    gain = _finite("gain", gain)
    damping_ratio = _nonnegative("damping_ratio", damping_ratio)
    natural_frequency = _positive("natural_frequency", natural_frequency)
    delay = _finite("delay", delay)
    signal_duration = _optional_nonnegative("signal_duration", signal_duration)
    initial = _finite("initial", initial)
    offset = _finite("offset", offset)
    tau = _elapsed_time(time, delay, signal_duration)
    unit_response = _second_order_unit_step(tau, damping_ratio, natural_frequency)
    values = offset + initial + gain * amplitude * unit_response
    values[time < delay] = offset + initial
    return _signal(
        "second_order_response",
        time,
        values,
        amplitude=amplitude,
        gain=gain,
        damping_ratio=damping_ratio,
        natural_frequency=natural_frequency,
        delay=delay,
        signal_duration=signal_duration,
        initial=initial,
        offset=offset,
    )


def _resolve_time(
    t: ArrayLike | None,
    duration: float,
    dt: float | None,
    sample_rate: float | None,
    start: float,
) -> NDArray[np.float64]:
    if t is not None:
        return _as_1d_float_array(t, "t")
    return time_vector(duration, dt=dt, sample_rate=sample_rate, start=start)


def _resolve_dt(dt: float | None, sample_rate: float | None) -> float:
    if dt is not None and sample_rate is not None:
        raise ValueError("dt and sample_rate are mutually exclusive")
    if dt is not None:
        return _positive("dt", dt)
    if sample_rate is None:
        sample_rate = DEFAULT_SAMPLE_RATE
    sample_rate = _positive("sample_rate", sample_rate)
    return 1.0 / sample_rate


def _resolve_delay(
    delay: float | None,
    *,
    legacy_name: str,
    legacy_value: float | None,
) -> float:
    if delay is not None and legacy_value is not None:
        raise ValueError(f"use delay or {legacy_name}, not both")
    if legacy_value is not None:
        return _finite(legacy_name, legacy_value)
    if delay is None:
        delay = 0.0
    return _finite("delay", delay)


def _elapsed_time(
    time: NDArray[np.float64],
    delay: float,
    signal_duration: float | None,
) -> NDArray[np.float64]:
    tau = np.maximum(time - delay, 0.0)
    if signal_duration is not None:
        tau = np.minimum(tau, signal_duration)
    return tau


def _targeted_ramp_values(
    tau: NDArray[np.float64],
    *,
    initial: float,
    target: float,
    slope: float,
) -> NDArray[np.float64]:
    if target == initial:
        return np.full_like(tau, initial, dtype=float)

    rate = abs(slope)
    if rate == 0.0:
        raise ValueError("slope must be nonzero when target differs from initial")

    direction = 1.0 if target > initial else -1.0
    values = initial + direction * rate * tau
    if direction > 0.0:
        return np.minimum(values, target)
    return np.maximum(values, target)


def _active_window(
    time: NDArray[np.float64],
    delay: float,
    signal_duration: float | None,
    *,
    include_end: bool,
) -> NDArray[np.bool_]:
    active = time >= delay
    if signal_duration is None:
        return active
    end_time = delay + signal_duration
    if include_end:
        return active & (time <= end_time)
    return active & (time < end_time)


def _second_order_unit_step(
    tau: NDArray[np.float64], damping_ratio: float, natural_frequency: float
) -> NDArray[np.float64]:
    tau = np.asarray(tau, dtype=float)
    zeta = damping_ratio
    omega_n = natural_frequency

    if zeta < 1.0 - 1e-9:
        root = np.sqrt(max(0.0, 1.0 - zeta**2))
        omega_d = omega_n * root
        phi = np.arctan2(root, zeta)
        return 1.0 - np.exp(-zeta * omega_n * tau) * np.sin(omega_d * tau + phi) / root

    if np.isclose(zeta, 1.0):
        return 1.0 - np.exp(-omega_n * tau) * (1.0 + omega_n * tau)

    root = np.sqrt(zeta**2 - 1.0)
    s1 = -omega_n * (zeta - root)
    s2 = -omega_n * (zeta + root)
    return 1.0 + (s2 * np.exp(s1 * tau) - s1 * np.exp(s2 * tau)) / (s1 - s2)


def _damping_ratio_from_overshoot(overshoot: float) -> float:
    if overshoot <= 0.0:
        return 1.0
    log_overshoot = np.log(overshoot)
    return float(-log_overshoot / np.sqrt(np.pi**2 + log_overshoot**2))


def _resolve_overshoot(
    overshoot: float | None, overshoot_percent: float | None
) -> float:
    if overshoot is not None and overshoot_percent is not None:
        raise ValueError("use overshoot or overshoot_percent, not both")
    if overshoot_percent is not None:
        overshoot_percent = _nonnegative("overshoot_percent", overshoot_percent)
        overshoot = overshoot_percent / 100.0
    if overshoot is None:
        overshoot = 0.1
    overshoot = _nonnegative("overshoot", overshoot)
    if overshoot >= 1.0:
        raise ValueError("overshoot must be less than 1.0")
    return overshoot


def _signal(
    name: str, t: NDArray[np.float64], y: NDArray[np.float64], **parameters: Any
) -> Signal:
    return Signal(t=t, y=y, name=name, parameters=parameters)


def _as_1d_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _optional_nonnegative(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _nonnegative(name, value)


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value
