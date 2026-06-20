"""Signal generation utilities for CPS and control engineering workflows."""

from .signals import (
    Signal,
    chirp,
    first_order_response,
    noise,
    oscillatory_response,
    pulse,
    ramp,
    second_order_response,
    sinc,
    sinusoid,
    step,
    step_with_overshoot,
    time_vector,
)

__all__ = [
    "Signal",
    "chirp",
    "first_order_response",
    "noise",
    "oscillatory_response",
    "pulse",
    "ramp",
    "second_order_response",
    "sinc",
    "sinusoid",
    "step",
    "step_with_overshoot",
    "time_vector",
]
