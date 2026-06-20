# stl-witness

`stl-witness` provides parameterized signal generators for cyber-physical systems
research, control engineering examples, simulations, and benchmark traces.

The core implementation depends only on NumPy. SciPy and python-control are
declared as optional extras for downstream workflows that want to combine these
signals with additional analysis or simulation tooling.

## Signals

- `step`
- `ramp`
- `sinusoid`
- `sinc`
- `pulse`
- `chirp`
- `noise`
- `step_with_overshoot`
- `oscillatory_response`
- `first_order_response`
- `second_order_response`

Each generator returns a `Signal` object with:

- `t`: one-dimensional NumPy time array
- `y`: one-dimensional NumPy signal array
- `name`: signal family
- `parameters`: generation metadata for reproducibility

`Signal` also behaves naturally with NumPy through `np.asarray(signal)`, which
returns the `y` samples.

## Example

```python
import numpy as np
from stl_witness import second_order_response, sinusoid, time_vector

t = time_vector(duration=8.0, dt=0.01)

reference = sinusoid(t, amplitude=0.5, frequency=0.25, offset=1.0)
plant = second_order_response(
    t,
    amplitude=1.0,
    delay=0.5,
    damping_ratio=0.35,
    natural_frequency=3.0,
)

error = reference.y - plant.y
print(np.max(np.abs(error)))
```

## Time Bases

All generators accept either an explicit time vector:

```python
sig = step([0.0, 0.1, 0.2, 0.3], amplitude=2.0, step_time=0.2)
```

or time-base parameters:

```python
sig = chirp(duration=5.0, sample_rate=200.0, f0=0.5, f1=20.0)
```

If neither `dt` nor `sample_rate` is provided, generators use a default sample
rate of 100 Hz.

## Delays And Active Durations

Generators accept `delay` to shift the signal onset:

```python
sig = step(t, amplitude=2.0, delay=1.5)
```

Generators that have a meaningful finite active interval also accept
`signal_duration`. For ramps and response-like signals, the value reached at the
end of the active interval is held:

```python
sig = ramp(t, slope=0.5, delay=1.0, signal_duration=3.0)
```

For burst-like signals such as sinusoids, chirps, pulse trains, and noise,
samples outside the active interval return to their baseline (`offset`,
`baseline`, or `mean`).

Older names such as `step_time` and `start_time` are still accepted as aliases
for `delay`.

## STL Witness Search

`stl-witness` also includes an integration layer that combines:

- Psy-TaLiRo for parameter search
- RTAMT discrete-time STL robustness monitoring and explanations
- the signal generators in this package for trace construction

An `.stl` file must declare every formula variable as `input` or `output`,
followed by the formula:

```text
input request
output response
G(request >= 5 -> F_[0,2] response <= 4)
```

Use a JSON config to bind each declared variable to a signal class and provide
parameter bounds. Parameters written as `[low, high]` are optimized by
Psy-TaLiRo; scalar/string/null values are fixed.

```json
{
  "specification": "specs/request_response.stl",
  "target": "both",
  "horizon": 8.0,
  "dt": 0.01,
  "runs": 1,
  "max_iterations_per_signal_generation": 200,
  "witnesses": 3,
  "optimizer": "dual_annealing",
  "variables": {
    "request": {
      "io": "input",
      "signal": "step",
      "parameters": {
        "amplitude": [0.0, 10.0],
        "delay": [0.0, 3.0],
        "signal_duration": 3.0
      }
    },
    "response": {
      "io": "output",
      "signal": "first_order_response",
      "parameters": {
        "amplitude": [0.0, 10.0],
        "delay": [0.0, 3.0],
        "time_constant": [0.1, 2.0]
      }
    }
  },
  "output_csv": "outputs/witness.csv",
  "output_png": "outputs/witness.png"
}
```

Run the search with:

```powershell
stl-witness-falsify witness_config.json
```

A satisfying witness requires finite robustness strictly greater than zero, so
`+inf` robustness is treated as vacuous and rejected. A falsifying witness
requires robustness strictly less than zero. Found traces can be written to CSV
and optionally plotted with Matplotlib.

PNG plots include red horizontal lines for numeric constants that appear as
direct signal thresholds in the STL formula. When RTAMT explanations are
available, violation explanations are shown as translucent red time bands or
red vertical point markers, while satisfaction explanations use pastel green.

Use `witnesses` to request K accepted traces per target. Each witness-generation
attempt is capped by `max_iterations_per_signal_generation`; older configs may
still use `iterations` as an alias. When K is greater than one, the search uses
deterministic seed offsets and selects accepted witnesses greedily to increase
their normalized parameter-space distance. Output files receive stable witness
suffixes such as `_001`, `_002`, and `_003`.

## Differential Analysis

Differential mode compares two IA-STL specifications, `phi1` and `phi2`, over
the same generated signal trace. It searches for up to four witnesses:

- `both`: satisfies `phi1` and `phi2`
- `phi1_only`: satisfies `phi1` and falsifies `phi2`
- `phi2_only`: satisfies `phi2` and falsifies `phi1`
- `neither`: falsifies both `phi1` and `phi2`

The objective combines both robustness values by maximizing the weakest required
margin. For example, `phi1_only` optimizes the minimum of `robustness(phi1)` and
`-robustness(phi2)`.

```json
{
  "mode": "differential",
  "phi1": "specs/phi1.stl",
  "phi2": "specs/phi2.stl",
  "horizon": 8.0,
  "dt": 0.01,
  "runs": 1,
  "max_iterations_per_signal_generation": 200,
  "witnesses": 2,
  "targets": "all",
  "variables": {
    "request": {
      "io": "input",
      "signal": "step",
      "parameters": {
        "amplitude": [0.0, 10.0],
        "delay": [0.0, 3.0]
      }
    },
    "response": {
      "io": "output",
      "signal": "first_order_response",
      "parameters": {
        "amplitude": [0.0, 10.0],
        "delay": [0.0, 3.0],
        "time_constant": [0.1, 2.0]
      }
    }
  },
  "output_csv": "outputs/differential.csv",
  "output_png": "outputs/differential.png"
}
```

Run it with the same CLI:

```powershell
stl-witness-falsify differential_config.json
```

For the example above, found traces are written as target-specific files such as
`outputs/differential_phi1_only.csv` and
`outputs/differential_phi1_only.png`.

## Development

Run the tests with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```
