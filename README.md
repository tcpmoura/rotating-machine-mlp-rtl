# Compact MLP inference in synthesizable RTL

Research project by **Tatiana Caetano Praxedes de Moura and João Vitor Carvalho Lima**,
students of the CI Digital Residency Program — CEPEDI.

Vibration-only rotating-machine classification with a 4–8–5 MLP and two INT8
implementations: a folded core and a parallel core. Five classes: Normal, BPFI,
BPFO, Misalignment and Unbalance. This repository contains curated sources and
the model parameters and a small numerical test set.

## Contents

| Directory / files | Purpose |
|---|---|
| `rtl/` | Folded and parallel cores, datapath, controller and weight ROM |
| `python/` | Feature extraction, training, evaluation, quantization and verification |
| `model/parameters.py` | Class names, preprocessing constants and fixed-point scales |
| `model/weights_rom.mem` | 85 signed INT8 weights and biases encoded for the RTL ROM |
| `tests/` | Streaming testbench and 450 golden vectors, ten per acquisition |
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

Run the full nested MLP evaluation for each combination of `--split acquisition`
or `--split load` and `--seed 17`, `42` or `73` (six commands):

```bash
python python/dual_housing_evaluation.py --split acquisition --seed 17
```

`--resume` reuses compatible checkpoints. Then run the software controls and
select/final-fit using internal scores:

```bash
python python/consolidation_baselines.py --split acquisition
python python/consolidation_baselines.py --split load
python python/consolidate_final_model.py
```

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

At 20 MHz the folded/parallel result latencies are 116/3 cycles; acceptance
intervals are 118/5 cycles. Reported resources are 370/475 ALMs and 1/60 DSPs.
The two full-regression runs per architecture were repeated in Xcelium 24.09-s013.
The release includes a declaration-order portability correction and comments
clarifying feature inputs; earlier fitted reports refer to the preserved
declaration-order predecessor. Arithmetic/control expressions were not changed.

MLP INT8 mean accuracy is 90.70% across acquisition splits and 87.38% on unseen
loads (three seeds). Normal-class recall is 77.55% and 66.74%, respectively.
The RBF SVM baseline reaches 91.19% and 93.03% mean accuracy on the same splits.
The evaluation splits acquisitions before expanding their windows, fits scaling
on training data, and selects model configurations by inner cross-validation.
The experiments reuse a dataset explored during development. These are
exploratory results on one testbed, not industrial reliability certification.
Normal-class and local quantization limitations remain. SVM performs better on
the reported aggregate classification metrics.

Power/energy summaries are tool estimates from post-fit activity, not measurements
on a physical FPGA. High activity-data confidence is not an error bound. Energy
depends on workload and the treatment of idle time. No Cadence/Intel libraries,
generated vendor netlists, VCD traces or licensed tool binaries are redistributed.
Complete simulation/power logs remain in the authors' working archive; this
minimal repository is not that archive.

## Acknowledgments

The authors thank CEPEDI for support through the CI Digital Residency Program.
This project was supported by the Ministry of Science, Technology and Innovations
(MCTI), with funding under Law No. 8,248 of October 23, 1991, within the Projects
of National Interest in Information and Communication Technologies coordinated
by Softex, under Residência em TIC 41 (DOU 01245.002558/2024-03).
