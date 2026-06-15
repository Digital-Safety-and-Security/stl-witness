"""Command-line entry point for IA-STL witness search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DifferentialConfig, FalsificationConfig, load_config
from .io import plot_trace, save_trace_csv
from .runner import run_differential, run_falsification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Search for satisfying, falsifying, or differential IA-STL witness traces."
        )
    )
    parser.add_argument("config", type=Path, help="Path to JSON witness-search config.")
    parser.add_argument("--csv", type=Path, help="Override CSV output path.")
    parser.add_argument("--png", type=Path, help="Save a Matplotlib PNG for found traces.")
    parser.add_argument("--show", action="store_true", help="Open Matplotlib windows.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.csv is not None or args.png is not None:
        config = _with_output_overrides(config, csv=args.csv, png=args.png)

    result = run_differential(config) if isinstance(config, DifferentialConfig) else run_falsification(config)
    summary = {}

    for target, target_result in result.results.items():
        best = target_result.best
        summary[target] = {
            "found": target_result.found,
            "robustness": None if best is None else best.robustness,
            "evaluations": len(target_result.evaluations),
        }

        if best is None or not target_result.found:
            continue

        csv_path = _target_path(config.output_csv, target, ".csv")
        png_path = _target_path(config.output_png, target, ".png")
        if csv_path is not None:
            save_trace_csv(best.trace, csv_path)
            summary[target]["csv"] = str(csv_path)
        if png_path is not None or args.show:
            plot_trace(
                best.trace,
                output=png_path,
                show=args.show,
                title=_plot_title(target, best.robustness),
            )
            if png_path is not None:
                summary[target]["png"] = str(png_path)

    print(json.dumps(summary, indent=2))
    return 0 if any(item["found"] for item in summary.values()) else 1


def _with_output_overrides(
    config: FalsificationConfig | DifferentialConfig,
    *,
    csv: Path | None,
    png: Path | None,
) -> FalsificationConfig | DifferentialConfig:
    common = {
        "variables": config.variables,
        "targets": config.targets,
        "horizon": config.horizon,
        "dt": config.dt,
        "sample_rate": config.sample_rate,
        "runs": config.runs,
        "iterations": config.iterations,
        "seed": config.seed,
        "optimizer": config.optimizer,
        "backend": config.backend,
        "output_csv": csv or config.output_csv,
        "output_png": png or config.output_png,
    }
    if isinstance(config, DifferentialConfig):
        return DifferentialConfig(phi1=config.phi1, phi2=config.phi2, **common)
    return FalsificationConfig(specification=config.specification, **common)


def _plot_title(target: str, robustness) -> str:
    if isinstance(robustness, dict):
        values = ", ".join(f"{name}={value:.6g}" for name, value in robustness.items())
        return f"{target.title()} Witness, {values}"
    return f"{target.title()} Witness, robustness={robustness:.6g}"


def _target_path(path: Path | None, target: str, suffix: str) -> Path | None:
    if path is None:
        return None
    if len(path.suffix) == 0:
        return path / f"{target}_trace{suffix}"
    return path.with_name(f"{path.stem}_{target}{path.suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
