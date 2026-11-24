import math
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stl_witness.falsification.config import DifferentialConfig, VariableConfig, load_config
from stl_witness.falsification.cli import _target_path
from stl_witness.falsification.io import save_trace_csv
from stl_witness.falsification.objective import (
    CandidateTrace,
    RobustnessObjective,
    differential_robustness_to_cost,
    is_differential_target_hit,
    is_target_hit,
    robustness_to_cost,
)
from stl_witness.falsification.parameters import SignalParameterSpace, available_signal_classes
from stl_witness.falsification.monitor import (
    _normalize_explanation_items,
    _normalize_explanations,
)
from stl_witness.falsification.runner import _select_target_witnesses
from stl_witness.falsification.specification import (
    extract_formula_constants,
    extract_formula_variables,
    normalize_formula_for_rtamt,
    read_ia_stl_file,
)


class FalsificationArchitectureTests(unittest.TestCase):
    def test_ia_stl_file_parses_declarations_and_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "request_response.stl"
            path.write_text(
                "\n".join(
                    [
                        "input request",
                        "output response",
                        "G(request >= 5 -> F_[0,2] response <= 4)",
                    ]
                ),
                encoding="utf-8",
            )

            spec = read_ia_stl_file(path)

        self.assertEqual(spec.variables, ("request", "response"))
        self.assertEqual(
            spec.rtamt_formula(),
            "out = always(request >= 5 implies eventually[0:2] response <= 4);",
        )

    def test_formula_variable_extraction_rejects_keywords(self) -> None:
        formula = "G(request >= 5 -> F_[0,2] response <= 4)"

        self.assertEqual(extract_formula_variables(formula), {"request", "response"})
        self.assertEqual(
            normalize_formula_for_rtamt(formula),
            "always(request >= 5 implies eventually[0:2] response <= 4)",
        )

    def test_formula_constant_extraction_groups_signal_thresholds(self) -> None:
        formula = "G(request >= 5 -> F_[0,2] response <= 4 and -1.5 < request)"

        constants = extract_formula_constants(formula, ("request", "response"))

        self.assertEqual(
            {(constant.variable, constant.value) for constant in constants},
            {("request", 5.0), ("request", -1.5), ("response", 4.0)},
        )

    def test_parameter_space_collects_bounds_and_fixed_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "request_response.stl"
            path.write_text(
                "input request\noutput response\nG(request >= 5 -> F_[0,2] response <= 4)",
                encoding="utf-8",
            )
            spec = read_ia_stl_file(path)

        space = SignalParameterSpace.from_spec(
            spec,
            {
                "request": VariableConfig(
                    signal="step",
                    parameters={"amplitude": [0, 10], "delay": [0, 2]},
                ),
                "response": VariableConfig(
                    signal="first_order_response",
                    parameters={
                        "amplitude": [0, 10],
                        "delay": [0, 2],
                        "time_constant": 0.5,
                    },
                ),
            },
        )

        self.assertIn("step", available_signal_classes())
        self.assertEqual(
            set(space.bounds),
            {"request.amplitude", "request.delay", "response.amplitude", "response.delay"},
        )
        self.assertEqual(space.bounds["request.amplitude"], (0.0, 10.0))
        params = space.parameters_from_sample(
            {
                "request.amplitude": 7,
                "request.delay": 1,
                "response.amplitude": 3,
                "response.delay": 1.5,
            }
        )
        self.assertEqual(params["response"]["time_constant"], 0.5)

    def test_parameter_space_can_merge_two_spec_variable_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            phi1_path = root / "phi1.stl"
            phi2_path = root / "phi2.stl"
            phi1_path.write_text("input request\nG(request >= 1)", encoding="utf-8")
            phi2_path.write_text("output response\nF(response <= 2)", encoding="utf-8")
            phi1 = read_ia_stl_file(phi1_path)
            phi2 = read_ia_stl_file(phi2_path)

        space = SignalParameterSpace.from_specs(
            (phi1, phi2),
            {
                "request": VariableConfig(
                    signal="step",
                    parameters={"amplitude": [0, 10]},
                ),
                "response": VariableConfig(
                    signal="ramp",
                    parameters={"slope": [0, 5], "target": [1, 4]},
                ),
            },
        )

        self.assertEqual(
            set(space.bounds),
            {"request.amplitude", "response.slope", "response.target"},
        )

    def test_robustness_costs_encode_targets(self) -> None:
        self.assertLess(robustness_to_cost(0.5, "satisfy"), 0.0)
        self.assertGreater(robustness_to_cost(-0.5, "satisfy"), 0.0)
        self.assertGreater(robustness_to_cost(math.inf, "satisfy"), 0.0)
        self.assertLess(robustness_to_cost(-0.5, "falsify"), 0.0)
        self.assertTrue(is_target_hit(0.5, "satisfy"))
        self.assertFalse(is_target_hit(math.inf, "satisfy"))
        self.assertTrue(is_target_hit(-0.5, "falsify"))

    def test_differential_robustness_costs_encode_quadrants(self) -> None:
        both = {"phi1": 0.5, "phi2": 0.25}
        phi1_only = {"phi1": 0.5, "phi2": -0.25}
        phi2_only = {"phi1": -0.5, "phi2": 0.25}
        neither = {"phi1": -0.5, "phi2": -0.25}

        self.assertTrue(is_differential_target_hit(both, "both"))
        self.assertTrue(is_differential_target_hit(phi1_only, "phi1_only"))
        self.assertTrue(is_differential_target_hit(phi2_only, "phi2_only"))
        self.assertTrue(is_differential_target_hit(neither, "neither"))
        self.assertLess(differential_robustness_to_cost(phi1_only, "phi1_only"), 0.0)
        self.assertGreater(
            differential_robustness_to_cost({"phi1": math.inf, "phi2": 1.0}, "both"),
            0.0,
        )

    def test_differential_config_loads_two_specs_and_all_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "phi1.stl").write_text("input request\nG(request >= 1)", encoding="utf-8")
            (root / "phi2.stl").write_text("input request\nF(request <= 2)", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mode": "differential",
                        "phi1": "phi1.stl",
                        "phi2": "phi2.stl",
                        "dt": 0.1,
                        "variables": {
                            "request": {
                                "signal": "step",
                                "parameters": {"amplitude": [0, 3]},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertIsInstance(config, DifferentialConfig)
        self.assertEqual(config.targets, ("both", "phi1_only", "phi2_only", "neither"))
        self.assertEqual(config.dt, 0.1)
        self.assertIsNone(config.sample_rate)

    def test_config_loads_witness_count_and_iteration_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "specification": "request_response.stl",
                        "max_iterations_per_signal_generation": 17,
                        "witnesses": 3,
                        "variables": {},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.iterations, 17)
        self.assertEqual(config.max_iterations_per_signal_generation, 17)
        self.assertEqual(config.witnesses, 3)

    def test_config_loads_trace_diagnostics_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "specification": "request_response.stl",
                        "trace_diagnostics": False,
                        "variables": {},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertFalse(config.trace_diagnostics)

    def test_objective_can_skip_trace_diagnostics(self) -> None:
        class MonitorWithoutDiagnostics:
            def evaluate(self, trace) -> float:
                return 1.0

            def explain(self, trace, polarity):  # pragma: no cover - must not be called
                raise AssertionError("trace diagnostics should be disabled")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "request.stl"
            path.write_text("input request\nG(request >= 1)", encoding="utf-8")
            spec = read_ia_stl_file(path)

        space = SignalParameterSpace.from_spec(
            spec,
            {
                "request": VariableConfig(
                    signal="step",
                    parameters={"amplitude": [0, 10]},
                )
            },
        )
        objective = RobustnessObjective(
            target="satisfy",
            parameter_space=space,
            monitor=MonitorWithoutDiagnostics(),
            horizon=1.0,
            dt=1.0,
            sample_rate=None,
            trace_diagnostics=False,
        )

        candidate = objective.evaluate({"request.amplitude": 2.0})

        self.assertTrue(candidate.accepted)
        self.assertEqual(candidate.explanations, {})

    def test_witness_selection_prefers_parameter_space_variability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "request.stl"
            path.write_text("input request\nG(request >= 1)", encoding="utf-8")
            spec = read_ia_stl_file(path)

        space = SignalParameterSpace.from_spec(
            spec,
            {
                "request": VariableConfig(
                    signal="step",
                    parameters={"amplitude": [0, 10], "delay": [0, 10]},
                )
            },
        )
        near_best = CandidateTrace(
            target="falsify",
            robustness=-4.0,
            cost=-4.0,
            accepted=True,
            parameters={"request.amplitude": 0.1, "request.delay": 0.1},
            trace={},
        )
        best = CandidateTrace(
            target="falsify",
            robustness=-5.0,
            cost=-5.0,
            accepted=True,
            parameters={"request.amplitude": 0.0, "request.delay": 0.0},
            trace={},
        )
        diverse = CandidateTrace(
            target="falsify",
            robustness=-3.0,
            cost=-3.0,
            accepted=True,
            parameters={"request.amplitude": 10.0, "request.delay": 10.0},
            trace={},
        )

        witnesses = _select_target_witnesses(
            [near_best, best, diverse],
            "falsify",
            space,
            count=2,
        )

        self.assertEqual(witnesses, (best, diverse))

    def test_indexed_target_paths_are_used_for_multiple_witnesses(self) -> None:
        self.assertEqual(
            _target_path(Path("outputs/witness.csv"), "falsify", ".csv", 2),
            Path("outputs/witness_falsify_002.csv"),
        )
        self.assertEqual(
            _target_path(Path("outputs"), "falsify", ".csv", 2),
            Path("outputs/falsify_trace_002.csv"),
        )

    def test_explanation_index_intervals_are_mapped_to_signal_time(self) -> None:
        from stl_witness import step

        signal = step([0.0, 0.5, 1.0, 1.5], amplitude=1.0)

        intervals, points = _normalize_explanation_items([[1, 2], [3, 3]], signal)

        self.assertEqual(intervals, ((0.5, 1.0),))
        self.assertEqual(points, (1.5,))

    def test_explanation_normalization_keeps_output_variables(self) -> None:
        from stl_witness import step

        trace = {
            "T": step([0.0, 1.0, 2.0], amplitude=70.0),
            "cooling": step([0.0, 1.0, 2.0], amplitude=1.0),
        }
        explanations = _normalize_explanations(
            {"T": [[1, 2]], "cooling": [[2, 2]]},
            trace,
            ("T", "cooling"),
            "violation",
        )

        self.assertIn("T", explanations)
        self.assertIn("cooling", explanations)
        self.assertEqual(explanations["cooling"][0].points, (2.0,))

    def test_csv_export_writes_time_and_variables(self) -> None:
        from stl_witness import step

        trace = {
            "request": step([0.0, 1.0], amplitude=2.0),
            "response": step([0.0, 1.0], amplitude=3.0),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output = save_trace_csv(trace, Path(tmpdir) / "trace.csv")
            content = output.read_text(encoding="utf-8")

        self.assertIn("time,request,response", content)
        self.assertIn("0.0,2.0,3.0", content)


if __name__ == "__main__":
    unittest.main()
