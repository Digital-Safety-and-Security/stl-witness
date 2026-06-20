"""Command-line entry point for IA-STL witness search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DifferentialConfig, FalsificationConfig, load_config
from .io import plot_trace, save_trace_csv
from .runner import run_differential, run_falsification
from .specification import read_ia_stl_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Search for satisfying, falsifying, or differential IA-STL witness traces."
        )
    )
    parser.add_argument("config", type=Path, help="Path to JSON witness-search config.")
    parser.add_argument("--csv", type=Path, help="Override CSV output path.")
    parser.add_argument("--png", type=Path, help="Save a Matplotlib PNG for found traces.")
    parser.add_argument(
        "--witnesses",
        type=_positive_int_arg,
        help="Number of accepted witnesses to generate per target.",
    )
    parser.add_argument(
        "--max-iterations-per-signal-generation",
        "--iterations",
        dest="iterations",
        type=_positive_int_arg,
        help="Maximum falsification iterations for each witness-generation attempt.",
    )
    parser.add_argument("--show", action="store_true", help="Open Matplotlib windows.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if (
        args.csv is not None
        or args.png is not None
        or args.witnesses is not None
        or args.iterations is not None
    ):
        config = _with_overrides(
            config,
            csv=args.csv,
            png=args.png,
            witnesses=args.witnesses,
            iterations=args.iterations,
        )

    result = (
        run_differential(config)
        if isinstance(config, DifferentialConfig)
        else run_falsification(config)
    )
    constants = _constants_by_variable(config)
    summary = {}

    for target, target_result in result.results.items():
        best = target_result.best
        summary[target] = {
            "found": target_result.found,
            "requested_witnesses": config.witnesses,
            "witnesses_found": len(target_result.witnesses),
            "robustness": None if best is None else best.robustness,
            "evaluations": len(target_result.evaluations),
            "witnesses": [],
        }

        if not target_result.found:
            continue

        for index, witness in enumerate(target_result.witnesses, start=1):
            witness_number = index if config.witnesses > 1 else None
            csv_path = _target_path(config.output_csv, target, ".csv", witness_number)
            png_path = _target_path(config.output_png, target, ".png", witness_number)
            witness_summary = {
                "robustness": witness.robustness,
                "parameters": dict(witness.parameters),
            }

            if csv_path is not None:
                save_trace_csv(witness.trace, csv_path)
                witness_summary["csv"] = str(csv_path)
                if index == 1:
                    summary[target]["csv"] = str(csv_path)
            if png_path is not None or args.show:
                plot_trace(
                    witness.trace,
                    output=png_path,
                    show=args.show,
                    title=_plot_title(target, witness.robustness, witness_number),
                    constants=constants,
                    explanations=witness.explanations,
                )
                if png_path is not None:
                    witness_summary["png"] = str(png_path)
                    if index == 1:
                        summary[target]["png"] = str(png_path)

            summary[target]["witnesses"].append(witness_summary)

    print(json.dumps(summary, indent=2))
    return 0 if any(item["found"] for item in summary.values()) else 1


def _with_overrides(
    config: FalsificationConfig | DifferentialConfig,
    *,
    csv: Path | None,
    png: Path | None,
    witnesses: int | None,
    iterations: int | None,
) -> FalsificationConfig | DifferentialConfig:
    common = {
        "variables": config.variables,
        "targets": config.targets,
        "horizon": config.horizon,
        "dt": config.dt,
        "sample_rate": config.sample_rate,
        "runs": config.runs,
        "iterations": iterations if iterations is not None else config.iterations,
        "witnesses": witnesses if witnesses is not None else config.witnesses,
        "seed": config.seed,
        "optimizer": config.optimizer,
        "backend": config.backend,
        "output_csv": csv or config.output_csv,
        "output_png": png or config.output_png,
    }
    if isinstance(config, DifferentialConfig):
        return DifferentialConfig(phi1=config.phi1, phi2=config.phi2, **common)
    return FalsificationConfig(specification=config.specification, **common)


def _positive_int_arg(value: str) -> int:
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return integer


def _constants_by_variable(
    config: FalsificationConfig | DifferentialConfig,
) -> dict[str, tuple[float, ...]]:
    if isinstance(config, DifferentialConfig):
        return _merge_constants(
            read_ia_stl_file(config.phi1).constants_by_variable(),
            read_ia_stl_file(config.phi2).constants_by_variable(),
        )
    return read_ia_stl_file(config.specification).constants_by_variable()


def _merge_constants(
    *constant_sets: dict[str, tuple[float, ...]],
) -> dict[str, tuple[float, ...]]:
    merged: dict[str, list[float]] = {}
    seen: set[tuple[str, float]] = set()
    for constants in constant_sets:
        for variable, values in constants.items():
            for value in values:
                key = (variable, value)
                if key in seen:
                    continue
                seen.add(key)
                merged.setdefault(variable, []).append(value)
    return {
        variable: tuple(values)
        for variable, values in merged.items()
    }


def _plot_title(target: str, robustness, witness_number: int | None = None) -> str:
    label = f"{target.title()} Witness"
    if witness_number is not None:
        label = f"{label} {witness_number}"
    if isinstance(robustness, dict):
        values = ", ".join(f"{name}={value:.6g}" for name, value in robustness.items())
        return f"{label}, {values}"
    return f"{label}, robustness={robustness:.6g}"


def _target_path(
    path: Path | None,
    target: str,
    suffix: str,
    witness_number: int | None = None,
) -> Path | None:
    if path is None:
        return None
    witness_suffix = "" if witness_number is None else f"_{witness_number:03d}"
    if len(path.suffix) == 0:
        return path / f"{target}_trace{witness_suffix}{suffix}"
    return path.with_name(f"{path.stem}_{target}{witness_suffix}{path.suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
