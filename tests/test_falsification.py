import math
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stl_witness.falsification.config import DifferentialConfig, VariableConfig, load_config
from stl_witness.falsification.io import save_trace_csv
from stl_witness.falsification.objective import (
    differential_robustness_to_cost,
    is_differential_target_hit,
    is_target_hit,
    robustness_to_cost,
)
from stl_witness.falsification.parameters import SignalParameterSpace, available_signal_classes
from stl_witness.falsification.specification import (
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
                    parameters={"slope": [0, 5]},
                ),
            },
        )

        self.assertEqual(
            set(space.bounds),
            {"request.amplitude", "response.slope"},
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
