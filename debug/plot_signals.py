"""Generate representative signal plots for visual debugging.

Run from the repository root:

    python debug/plot_signals.py

Save instead of opening an interactive window:

    python debug/plot_signals.py --output debug/signals.png --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Matplotlib is required for this debug script. Install it with "
        "`python -m pip install matplotlib`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stl_witness import (  # noqa: E402
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


def representative_signals():
    """Return one representative signal from each generator."""

    t = time_vector(duration=6.0, dt=0.01)
    return [
        ("Step", step(t, amplitude=1.5, delay=1.0, signal_duration=2.5)),
        ("Ramp", ramp(t, slope=0.7, delay=0.75, signal_duration=2.5)),
        (
            "Sinusoid",
            sinusoid(
                t,
                amplitude=1.0,
                frequency=0.75,
                phase=0.2,
                delay=0.5,
                signal_duration=4.0,
            ),
        ),
        ("Sinc", sinc(t, amplitude=1.2, delay=1.0, center=2.0, width=0.6)),
        (
            "Pulse",
            pulse(t, amplitude=1.0, delay=0.5, period=1.2, duty_cycle=0.3, signal_duration=4.5),
        ),
        (
            "Chirp",
            chirp(
                t,
                amplitude=1.0,
                f0=0.2,
                f1=4.0,
                delay=0.5,
                sweep_duration=4.5,
                signal_duration=4.5,
            ),
        ),
        ("Noise", noise(t, amplitude=0.25, mean=0.0, delay=1.0, signal_duration=4.0, seed=7)),
        (
            "Step With Overshoot",
            step_with_overshoot(
                t,
                amplitude=1.0,
                overshoot_percent=20.0,
                settling_time=2.0,
                delay=0.5,
                signal_duration=4.5,
            ),
        ),
        (
            "Oscillatory Response",
            oscillatory_response(
                t,
                amplitude=1.0,
                damping_ratio=0.08,
                natural_frequency=2.0 * 3.141592653589793,
                delay=0.5,
                signal_duration=4.0,
            ),
        ),
        (
            "First-Order Response",
            first_order_response(
                t,
                amplitude=1.0,
                gain=1.2,
                time_constant=1.0,
                delay=0.5,
                signal_duration=4.5,
            ),
        ),
        (
            "Second-Order Response",
            second_order_response(
                t,
                amplitude=1.0,
                gain=1.0,
                damping_ratio=0.25,
                natural_frequency=3.0,
                delay=0.5,
                signal_duration=4.5,
            ),
        ),
    ]


def plot_all_signals(output: Path | None = None, show: bool = True) -> None:
    signals = representative_signals()
    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Representative stl-witness Signals", fontsize=16)

    for ax, (title, signal) in zip(axes.flat, signals):
        ax.plot(signal.t, signal.y, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[len(signals) :]:
        ax.axis("off")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one representative signal from each stl-witness generator."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving the generated figure, e.g. debug/signals.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive Matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_all_signals(output=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
