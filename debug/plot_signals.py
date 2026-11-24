"""Generate representative signal plots for visual debugging.

Run from the repository root:

    python debug/plot_signals.py

Save instead of opening an interactive window:

    python debug/plot_signals.py --output debug/signals.png --no-show
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stl_witness import (  # noqa: E402
    chirp,
    first_order_response,
    monotonic_convergence,
    noise,
    oscillatory_convergence,
    oscillatory_response,
    overshooting_convergence,
    pulse,
    ramp,
    second_order_response,
    sinc,
    sinusoid,
    step,
    step_with_overshoot,
    time_vector,
    transient_disturbance,
)


def representative_signals():
    """Return one representative signal from each generator."""

    t = time_vector(duration=6.0, dt=0.01)
    return [
        ("Step", step(t, amplitude=1.5, delay=1.0, signal_duration=2.5)),
        ("Ramp", ramp(t, slope=2, delay=1.5, signal_duration=0.5)),
        (
            "Pulse",
            pulse(t, amplitude=1.0, delay=0.5, period=1.2, duty_cycle=0.3, signal_duration=4.5),
        ),
        (
            "Transient Disturbance",
            transient_disturbance(
                t,
                nominal=5.5,
                amplitude=0.3,
                center=2.2,
                width=0.35,
                delay=0.5,
            ),
        ),
        (
            "Monotonic Convergence",
            monotonic_convergence(
                t,
                initial=1,
                target=1.5,
                time_constant=0.3,
                delay=1.5,
                signal_duration=4.5,
            ),
        ),
        (
            "Overshooting Convergence",
            overshooting_convergence(
                t,
                initial=0.0,
                target=1.0,
                time_constant=0.2,
                overshoot=1,
                peak_time=1.1,
                delay=0.5,
                signal_duration=1.5,
            ),
        ),
        (
            "Oscillatory Convergence",
            oscillatory_convergence(
                t,
                initial=0,
                target=1.2,
                damping_rate=0.9,
                frequency=0.8,
                delay=0.5,
                signal_duration=4.5,
            ),
        ),
        (
            "First-Order Response",
            first_order_response(
                t,
                amplitude=1.0,
                gain=2.2,
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
                gain=2.0,
                damping_ratio=0.6,
                natural_frequency=3.0,
                delay=0.5,
                signal_duration=4.5,
            ),
        ),
    ]


def plot_all_signals(output: Path | None = None, show: bool = True) -> None:
    signals = representative_signals()
    rows = 3
    cols = 3

    if plt is None:
        if output is None:
            output = REPO_ROOT / "debug" / "signals.svg"
        if output.suffix.lower() != ".svg":
            output = output.with_suffix(".svg")
        _plot_all_signals_svg(signals, output, rows=rows, cols=cols)
        print(
            "Matplotlib is not available; wrote a standard-library SVG fallback "
            f"to {output}."
        )
        return

    fig, axes = plt.subplots(rows, cols, figsize=(14, 13), constrained_layout=True)
    fig.suptitle("Signal Template Instantiations", fontsize=16)

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


def _plot_all_signals_svg(signals, output: Path, *, rows: int, cols: int) -> None:
    panel_width = 360
    panel_height = 230
    width = cols * panel_width
    height = rows * panel_height + 48
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50%" y="28" text-anchor="middle" font-family="Arial, sans-serif" '
        'font-size="20" font-weight="700">Representative stl-witness Signals</text>',
    ]

    for index, (title, signal) in enumerate(signals):
        row = index // cols
        col = index % cols
        x0 = col * panel_width + 42
        y0 = row * panel_height + 64
        plot_width = panel_width - 64
        plot_height = panel_height - 76
        parts.extend(_svg_panel(title, signal, x0, y0, plot_width, plot_height))

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def _svg_panel(title, signal, x0: int, y0: int, width: int, height: int) -> list[str]:
    x_values = signal.t
    y_values = signal.y
    x_min = float(x_values[0])
    x_max = float(x_values[-1])
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    y_pad = max((y_max - y_min) * 0.1, 0.1)
    y_min -= y_pad
    y_max += y_pad
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    def project_x(value: float) -> float:
        return x0 + (float(value) - x_min) * width / x_span

    def project_y(value: float) -> float:
        return y0 + height - (float(value) - y_min) * height / y_span

    points = " ".join(
        f"{project_x(x):.2f},{project_y(y):.2f}" for x, y in zip(x_values, y_values)
    )
    baseline_y = project_y(0.0) if y_min <= 0.0 <= y_max else None
    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 20}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="13" font-weight="700">'
        f"{html.escape(title)}</text>",
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
        'fill="#fafafa" stroke="#cccccc" stroke-width="1"/>',
    ]

    for fraction in (0.25, 0.5, 0.75):
        gx = x0 + width * fraction
        gy = y0 + height * fraction
        parts.append(
            f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0 + height}" '
            'stroke="#e0e0e0" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0 + width}" y2="{gy:.1f}" '
            'stroke="#e0e0e0" stroke-width="1"/>'
        )

    if baseline_y is not None:
        parts.append(
            f'<line x1="{x0}" y1="{baseline_y:.1f}" x2="{x0 + width}" '
            f'y2="{baseline_y:.1f}" stroke="#999999" stroke-width="1" '
            'stroke-dasharray="4 3"/>'
        )

    parts.append(
        f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{x0}" y="{y0 + height + 18}" font-family="Arial, sans-serif" '
        f'font-size="10">{x_min:.1f}s</text>'
    )
    parts.append(
        f'<text x="{x0 + width}" y="{y0 + height + 18}" text-anchor="end" '
        f'font-family="Arial, sans-serif" font-size="10">{x_max:.1f}s</text>'
    )
    parts.append(
        f'<text x="{x0}" y="{y0 - 6}" font-family="Arial, sans-serif" '
        f'font-size="10">{y_max:.2g}</text>'
    )
    parts.append(
        f'<text x="{x0}" y="{y0 + height + 32}" font-family="Arial, sans-serif" '
        f'font-size="10">{y_min:.2g}</text>'
    )
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one representative signal from each stl-witness generator."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path for saving the generated figure, e.g. debug/signals.png. "
            "If Matplotlib is unavailable, an SVG with the same stem is written."
        ),
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
