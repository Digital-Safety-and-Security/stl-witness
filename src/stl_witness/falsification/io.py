"""CSV and plotting utilities for witness traces."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from stl_witness.signals import Signal


def save_trace_csv(trace: Mapping[str, Signal], path: str | Path) -> Path:
    """Save a generated trace as CSV with columns ``time,<variables...>``."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    names = list(trace)
    if not names:
        raise ValueError("cannot save an empty trace")

    reference_time = trace[names[0]].t
    for name in names[1:]:
        if trace[name].t.shape != reference_time.shape or (trace[name].t != reference_time).any():
            raise ValueError("all signals must share the same time base to save CSV")

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", *names])
        for index, timestamp in enumerate(reference_time):
            writer.writerow([float(timestamp), *[float(trace[name].y[index]) for name in names]])

    return output


def plot_trace(
    trace: Mapping[str, Signal],
    *,
    output: str | Path | None = None,
    show: bool = False,
    title: str = "STL Witness Trace",
) -> Path | None:
    """Plot trace variables as stacked Matplotlib subplots."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install it with "
            "`python -m pip install matplotlib` or the project `plot` extra."
        ) from exc

    names = list(trace)
    fig, axes = plt.subplots(
        len(names),
        1,
        figsize=(10, max(2.4, 2.2 * len(names))),
        sharex=True,
        constrained_layout=True,
    )
    if len(names) == 1:
        axes = [axes]

    fig.suptitle(title)
    for ax, name in zip(axes, names):
        signal = trace[name]
        ax.plot(signal.t, signal.y, linewidth=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")

    output_path = Path(output) if output is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
