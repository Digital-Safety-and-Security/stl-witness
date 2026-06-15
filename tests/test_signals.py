import math
import unittest

import numpy as np

from stl_witness import (
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


class SignalGenerationTests(unittest.TestCase):
    def test_time_vector_uses_default_or_explicit_sampling(self) -> None:
        default_t = time_vector(0.02)
        explicit_t = time_vector(0.02, sample_rate=100.0)

        np.testing.assert_allclose(default_t, [0.0, 0.01, 0.02])
        np.testing.assert_allclose(default_t, explicit_t)

    def test_signal_supports_tuple_and_numpy_array_protocols(self) -> None:
        sig = Signal(t=np.array([0.0, 1.0]), y=np.array([2.0, 3.0]), name="demo")

        self.assertEqual(len(sig), 2)
        self.assertEqual(sig.sample_period, 1.0)
        np.testing.assert_allclose(np.asarray(sig), [2.0, 3.0])
        np.testing.assert_allclose(sig.as_tuple()[0], [0.0, 1.0])

    def test_step_applies_amplitude_at_step_time(self) -> None:
        sig = step([0.0, 0.5, 1.0], amplitude=3.0, step_time=0.5, initial=-1.0)

        np.testing.assert_allclose(sig.y, [-1.0, 2.0, 2.0])
        self.assertEqual(sig.parameters["amplitude"], 3.0)

    def test_step_supports_delay_and_finite_duration(self) -> None:
        sig = step([0.0, 1.0, 2.0, 3.0], delay=1.0, signal_duration=1.0)

        np.testing.assert_allclose(sig.y, [0.0, 1.0, 1.0, 0.0])
        self.assertEqual(sig.parameters["delay"], 1.0)
        self.assertEqual(sig.parameters["signal_duration"], 1.0)

    def test_ramp_starts_after_start_time(self) -> None:
        sig = ramp([0.0, 1.0, 2.0], slope=2.0, start_time=1.0, initial=5.0)

        np.testing.assert_allclose(sig.y, [5.0, 5.0, 7.0])

    def test_ramp_delay_and_duration_hold_final_value(self) -> None:
        sig = ramp([0.0, 1.0, 2.0, 3.0], slope=2.0, delay=1.0, signal_duration=1.0)

        np.testing.assert_allclose(sig.y, [0.0, 0.0, 2.0, 2.0])

    def test_sinusoid_uses_hz_and_radians(self) -> None:
        sig = sinusoid([0.0, 0.25, 0.5], amplitude=2.0, frequency=1.0)

        np.testing.assert_allclose(sig.y, [0.0, 2.0, 0.0], atol=1e-12)

    def test_sinusoid_delay_and_duration_create_finite_burst(self) -> None:
        sig = sinusoid(
            [0.0, 0.25, 0.5, 0.75],
            amplitude=1.0,
            frequency=1.0,
            offset=10.0,
            delay=0.25,
            signal_duration=0.25,
        )

        np.testing.assert_allclose(sig.y, [10.0, 10.0, 11.0, 10.0], atol=1e-12)

    def test_sinc_reaches_amplitude_at_center(self) -> None:
        sig = sinc([-1.0, 0.0, 1.0], amplitude=4.0, center=0.0, width=1.0)

        self.assertAlmostEqual(sig.y[1], 4.0)
        self.assertAlmostEqual(sig.y[0], 0.0, places=12)

    def test_pulse_supports_periodic_duty_cycle(self) -> None:
        sig = pulse(
            [0.0, 0.25, 0.5, 1.0, 1.25, 1.5],
            amplitude=2.0,
            period=1.0,
            duty_cycle=0.5,
        )

        np.testing.assert_allclose(sig.y, [2.0, 2.0, 0.0, 2.0, 2.0, 0.0])

    def test_pulse_signal_duration_bounds_pulse_train(self) -> None:
        sig = pulse(
            [0.0, 0.25, 0.5, 0.75, 1.25, 1.5],
            amplitude=2.0,
            delay=0.25,
            period=0.5,
            duty_cycle=0.5,
            signal_duration=0.75,
        )

        np.testing.assert_allclose(sig.y, [0.0, 2.0, 0.0, 2.0, 0.0, 0.0])

    def test_chirp_starts_at_phase_and_is_reproducible(self) -> None:
        sig = chirp([0.0, 0.5, 1.0], f0=1.0, f1=2.0, sweep_duration=1.0, phase=math.pi / 2.0)

        self.assertAlmostEqual(sig.y[0], 1.0)
        self.assertEqual(sig.name, "chirp")
        self.assertEqual(sig.parameters["method"], "linear")

    def test_chirp_delay_and_duration_return_to_offset(self) -> None:
        sig = chirp(
            [0.0, 0.5, 1.0, 1.5],
            f0=1.0,
            f1=2.0,
            delay=0.5,
            signal_duration=0.5,
            offset=3.0,
        )

        self.assertEqual(sig.y[0], 3.0)
        self.assertNotEqual(sig.y[1], 0.0)
        self.assertAlmostEqual(sig.y[-1], 3.0)

    def test_noise_seed_is_reproducible(self) -> None:
        first = noise(duration=0.1, dt=0.01, seed=7)
        second = noise(duration=0.1, dt=0.01, seed=7)

        np.testing.assert_allclose(first.y, second.y)

    def test_noise_delay_and_duration_use_mean_outside_active_window(self) -> None:
        sig = noise(
            [0.0, 1.0, 2.0, 3.0],
            amplitude=0.25,
            mean=5.0,
            delay=1.0,
            signal_duration=1.0,
            seed=3,
        )

        self.assertEqual(sig.y[0], 5.0)
        self.assertNotEqual(sig.y[1], 5.0)
        self.assertNotEqual(sig.y[2], 5.0)
        self.assertEqual(sig.y[3], 5.0)

    def test_first_order_response_matches_one_time_constant_value(self) -> None:
        sig = first_order_response([0.0, 1.0], amplitude=2.0, time_constant=1.0)

        np.testing.assert_allclose(sig.y, [0.0, 2.0 * (1.0 - math.exp(-1.0))])

    def test_first_order_response_duration_holds_reached_value(self) -> None:
        sig = first_order_response(
            [0.0, 1.0, 2.0, 3.0],
            delay=1.0,
            signal_duration=1.0,
            time_constant=1.0,
        )
        expected = 1.0 - math.exp(-1.0)

        np.testing.assert_allclose(sig.y, [0.0, 0.0, expected, expected])

    def test_second_order_critical_response_matches_formula(self) -> None:
        sig = second_order_response(
            [0.0, 1.0],
            amplitude=3.0,
            damping_ratio=1.0,
            natural_frequency=2.0,
        )
        expected = 3.0 * (1.0 - math.exp(-2.0) * 3.0)

        self.assertAlmostEqual(sig.y[0], 0.0)
        self.assertAlmostEqual(sig.y[1], expected)

    def test_step_with_overshoot_hits_requested_overshoot_approximately(self) -> None:
        sig = step_with_overshoot(
            duration=8.0,
            dt=0.005,
            amplitude=1.0,
            overshoot_percent=10.0,
            settling_time=2.0,
        )

        self.assertGreater(np.max(sig.y), 1.08)
        self.assertLess(np.max(sig.y), 1.12)
        self.assertAlmostEqual(sig.y[-1], 1.0, places=3)

    def test_oscillatory_response_decays(self) -> None:
        sig = oscillatory_response([0.0, 1.0, 2.0], amplitude=1.0, damping_ratio=0.5)

        self.assertGreater(abs(sig.y[0]), abs(sig.y[-1]))

    def test_validation_rejects_invalid_sampling(self) -> None:
        with self.assertRaises(ValueError):
            time_vector(1.0, dt=0.1, sample_rate=10.0)

        with self.assertRaises(ValueError):
            pulse(duration=1.0, period=1.0, width=2.0)

        with self.assertRaises(ValueError):
            ramp(delay=1.0, start_time=2.0)


if __name__ == "__main__":
    unittest.main()
