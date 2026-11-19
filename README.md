# Compact MLP inference in synthesizable RTL

Research project by **Tatiana Caetano Praxedes de Moura and João Vitor Carvalho Lima**,
students of the CI Digital Residency Program — CEPEDI.

Vibration-only rotating-machine classification with a 4–8–5 MLP and two INT16
implementations: a folded core and a parallel core. Five classes: Normal, BPFI,
BPFO, Misalignment and Unbalance. This repository contains curated sources and
the model parameters and a small numerical test set.

## Contents

| Directory / files | Purpose |
|---|---|
| `rtl/` | Folded and parallel cores, datapath, controller and weight ROM |
| `python/` | Feature extraction, training, evaluation, quantization and verification |
| `model/parameters.py` | Class names, preprocessing constants and fixed-point scales |
| `model/weights_rom.mem` | 85 signed INT16 weights and biases encoded for the RTL ROM |
| `tests/` | Streaming testbench and 450 golden vectors, ten per acquisition, plus 1,625 arithmetic stress vectors |
| `config/` | Relative dataset manifest and three acquisition-level test folds |
| `requirements.txt` | Python dependencies for feature extraction and training |

The golden vectors are an evenly spaced subset within each of the 45 acquisitions.
Weights and vectors are retained because they are required to run the numerical
verification without downloading the dataset. Simulator databases, reports,
intermediate training files and the dataset are excluded from version control.

## Quick RTL verification (no dataset needed)

Install Python and Icarus Verilog (`iverilog` and `vvp` in PATH). From this directory:

```bash
python python/run_checks.py
```

The four runs compare hidden activations, accumulators, class, score and alarms,
and exercise receiver pauses and reset. Expect `STREAM_PASS` for each run.
The fixture checks numerical behavior, not independent diagnostic accuracy.
To use an authenticated Cadence environment instead:

```bash
python python/run_checks.py --simulator xrun
```

Outputs go only to ignored `build/`. The full development dataset contains 28,080
vectors; the small committed fixture is intentionally sufficient for a quick check.

## FPGA implementation

```bash
python python/prepare_quartus.py
```

Open `build/quartus/folded/mlp.qpf` or `build/quartus/parallel/mlp.qpf` in Quartus.
Reference environment: Quartus Prime Lite 20.1, Cyclone V 5CEBA4F23C7, seed 1,
20 MHz. The projects use virtual interfaces and characterize register-to-register
timing; they are not complete board pin assignments or external timing contracts.
The generator applies the frozen fractional scales. Do not instantiate a core
with historical default parameters instead of the scales in `model/parameters.py`.

## Data and complete evaluation

The dataset is not redistributed. Obtain the source data described by Jung et al.,
*Vibration, acoustic, temperature, and motor current dataset of rotating machine
under varying operating conditions for fault diagnosis*, Data in Brief 48 (2023),
109049, DOI: [10.1016/j.dib.2023.109049](https://doi.org/10.1016/j.dib.2023.109049).
Use the vibration subset and the paths listed in `config/vibration_manifest.csv`.
For example, the supplied directory must contain `vibration/0Nm_BPFI_03.mat`.
Dataset rights remain with their respective owners.

The recorded Python environment uses Python 3.13 and the versions in
`requirements.txt`. Training may take substantial time.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python python/dual_housing_features.py --dataset /path/to/dataset
```

Run the controlled nested study (three acquisition folds, three held-out loads,
seeds 17/42/73), consolidate using inner scores, then evaluate integer precision:

```bash
python python/evaluate_model_capacity.py
python python/consolidate_final_model.py
python python/evaluate_precision.py
```

The study fixes 4096-sample harmonic-ratio features, compares 8/16 hidden neurons
and alpha 0.01/0.1/1, and uses balanced L-BFGS fitting. Scaling and selection use
training acquisitions only. The final development fit selects 8 neurons, alpha 1,
seed 42; selection does not rank external test accuracy. Fractional scales are
13/12/12/11 for inputs/first-layer weights/second-layer weights/hidden activations.
The regenerated final ROM is in `results/candidate_precision/final/int16/`;
compare it to `model/weights_rom.mem` before replacing any frozen artifacts.

The earlier INT8 evaluation and baseline scripts remain available for historical
comparisons: `dual_housing_evaluation.py` (six split/seed combinations), followed
by `consolidation_baselines.py --split acquisition` and `--split load`. They use
the earlier feature/configuration grid and are not the current release pipeline.

These scripts write ignored `results/`; they do not replace committed `model/`.
The frozen final model has no independent held-out accuracy: it uses all
development acquisitions. To regenerate its complete RTL vectors from extracted
features and verify them:

```bash
python python/export_vectors.py
python python/run_checks.py --vectors build/full_vectors.txt
```

## Model and interface

Four inputs are standardized log ratios of second/first and third/first harmonic
band powers at two synchronous vibration housings. The window is 4096 samples
(160 ms at 25.6 kHz). `model/parameters.py` supplies means and scales. Preprocessing
is software; FFT/acquisition latency is not included in RTL latency.

Inputs are accepted on a rising edge when `in_valid && in_ready`. Capture occurs
at that handshake. Outputs remain stable while `out_valid && !out_ready` and are
consumed when both are asserted on a rising edge. Reset cancels a transaction.
The signed winning score is not a probability. Alarm outputs are prototype
policies, not validated safety functions. These cores have no UART.

## Reported evidence and limits

The release uses signed 16-bit inputs, weights and hidden activations, with
36-bit accumulators. The public winning score remains a saturated signed 8-bit
value. Core defaults are historical; use the explicit parameters generated by
`prepare_quartus.py`. The testbench supplies the released parameters.

| Quantity | Folded | Parallel |
|---|---:|---:|
| Latency at 20 MHz | 116 cycles / 5.80 us | 3 cycles / 0.15 us |
| Acceptance interval | 118 cycles | 5 cycles |
| ALMs | 544 | 642 |
| Registers | 601 | 208 |
| DSP blocks | 1 | 59 |
| Slow 85 C Fmax | 37.36 MHz | 27.52 MHz |
| Estimated power, matched 100k results/s | 198.72 mW | 200.92 mW |
| Estimated energy, matched demand | 1.9872 uJ/result | 2.0092 uJ/result |

The revised RTL was checked with Icarus Verilog, and post-fit activity with
ModelSim. Earlier Xcelium runs apply to the prior INT8 revision, not this INT16
release. To exercise both cores with arithmetic edge cases:

```bash
python python/run_checks.py --vectors tests/stress_vectors.txt
```

Across three seeds, acquisition/unseen-load INT16 mean accuracies are
89.33%/91.32%; Normal recall is 77.62%/90.77%. Seven classifications differ from
floating point across 168,480 evaluations. Some individual fault conditions
remain unrecognized. Numerical fidelity is distinct from diagnostic adequacy.
The dataset was explored repeatedly and contains only three Normal acquisitions
on one testbed; these results do not establish industrial reliability.

An optimized single-thread C baseline on an Intel Core i5-13420H achieved about
26.52 million inferences/s, exceeding both FPGA core rates at 20 MHz. The source
is `software/mlp_benchmark.c` (Windows QPC timing, GCC):

```bash
gcc -O3 -march=native -std=c11 software/mlp_benchmark.c -o build/mlp_benchmark.exe
build/mlp_benchmark.exe model/weights_rom.mem build/full_vectors.txt tests/stress_vectors.txt 13 12 12 11
```

CPU measurements exclude preprocessing and I/O; FPGA rates are derived from RTL
cycles with post-fit timing checks. CPU energy was not measured. Power estimates
use post-fit activity without SDF, a 20-MHz clock, 25 C ambient and typical 1.10 V.
High activity-data confidence is not an error bound: the 1.09% matched-power
difference does not demonstrate physical energy savings. Static power dominates
at the actual 6.25-result/s feature cadence. No physical board was used.

No Cadence/Intel libraries, generated vendor netlists, simulator databases, VCD
traces, training checkpoints or dataset are redistributed. Detailed experiment
logs remain in the authors' archive. JSON summaries are generated locally in
ignored `results/` or `build/`; none are required as committed source files.

## Acknowledgments

The authors thank CEPEDI for support through the CI Digital Residency Program.
This project was supported by the Ministry of Science, Technology and Innovations
(MCTI), with funding under Law No. 8,248 of October 23, 1991, within the Projects
of National Interest in Information and Communication Technologies coordinated
by Softex, under Residência em TIC 41 (DOU 01245.002558/2024-03).
