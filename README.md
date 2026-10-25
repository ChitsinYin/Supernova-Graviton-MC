# Supernova Graviton Monte Carlo

ROOT-based phase-space Monte Carlo code for gravitational-wave / graviton production in a supernova core.

The repository implements thermal initial-state sampling, exact ROOT phase-space generation, event-by-event physics reweighting, and the final spectrum
$\omega \frac{dP}{d\omega}$.

## Implemented channels

- `gamma e -> h e` — electron photoproduction
- `gamma p -> h p` — proton photoproduction
- `e- e- -> h e- e-` — electron-electron bremsstrahlung
- `e- Z -> h e- Z` — electron-heavy-target bremsstrahlung
- `e- e+ -> h gamma` — electron-positron annihilation

## Repository structure

```text
data/                     Supernova profile inputs
src/                      ROOT/C++ generators and spectrum reweighters
ROOT_PHSP_Tutorial_docs/  Detailed technical tutorial and validation figures
```

The calculation is organized in two steps:

```text
thermal sampling + ROOT PHSP generation
                ↓
          reusable ROOT pool
                ↓
       physics reweighting
                ↓
     omega dP/domega spectrum
```

## Requirements

- ROOT 6
- C++17-compatible compiler

## Quick example: electron photoproduction

From the repository root:

```bash
g++ -O3 -std=c++17 \
  src/generate_gammae_phsp.cpp \
  $(root-config --cflags --libs) \
  -o generate_gammae_phsp

g++ -O3 -std=c++17 \
  src/weight_gammae_spectrum.cpp \
  $(root-config --cflags --libs) \
  -o weight_gammae_spectrum
```

Generate a phase-space pool:

```bash
./generate_gammae_phsp \
  data/sn_profile_data.csv \
  gammae_phsp.root \
  5000000 \
  20260910 \
  18.0
```

Compute the spectrum:

```bash
./weight_gammae_spectrum \
  gammae_phsp.root \
  gammae_spectrum.csv
```

The ROOT file contains the generated kinematic event pool, while the CSV file contains the weighted spectrum and Monte Carlo statistical uncertainty.

The other channels follow the same **generate -> reweight** workflow. Channel-specific commands, physics conventions, and validation plots are documented in:

[`ROOT_PHSP_Tutorial_docs/ROOT_PHSP_Tutorial.pdf`](ROOT_PHSP_Tutorial_docs/ROOT_PHSP_Tutorial.pdf)

## Output format

The spectrum CSV files contain

```text
omega_GeV
omega_dP_domega_GeV2
mc_error_GeV2
raw_event_count
effective_weighted_events
```

