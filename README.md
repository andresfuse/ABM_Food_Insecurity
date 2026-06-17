# Food Access Agent-Based Model (ABM)

An agent-based model built on [Mesa](https://mesa.readthedocs.io/) that simulates
how households in a city choose where to live and where to shop for food month
by month, and how those choices interact with food security, housing security,
neighborhood composition, and store openings/closings over time.

## Overview

Each simulated month, the model:

1. Lets food-insecure or house-insecure households search for a better outcome
   (a cheaper food basket, a new set of stores, or a new home).
2. Solves a small integer program (via [PuLP](https://github.com/coin-or/pulp))
   for each household to choose which stores to visit and how much to buy at
   each one, given its food budget, maximum number of stores, and an affinity
   score combining price, racial composition, and store type.
3. Updates households' savings, food consumption, and food/housing security
   classification.
4. Once a year, re-evaluates neighborhood choice using a discrete-choice
   ("Bruch-style") utility model over census tracts, and updates store sales,
   openings, and closings.

Results are collected with Mesa's `DataCollector` at the model level: agent
state, store state, housing unit state, and the neighborhood choice
probabilities computed each year.

## Project structure

```
.
├── main.py        # Entry point: selects a seed, builds and runs the model, saves results
├── runner.py       # Loads input data, reconciles it with a chosen simulation seed
├── model.py        # FoodAccessModel: scheduling, neighborhood choice, data collection
├── household.py     # Household agent: income, food/housing security, store selection
├── stores.py        # Store agent: location, prices, sales, opening/closing
└── data/             # Not included in this repository — see below
```

## Data requirements

This repository does not include any data. For data privacy reasons, this must be requested by contacting the researchers. To run the model, create a `data/`
folder alongside the scripts with the following files:

| Path | Format | Description |
|---|---|---|
| `data/persons.csv` | CSV | Synthetic population, person-level records |
| `data/households.xlsx` (sheet `data`) | Excel | Synthetic population, household-level records |
| `data/housing_units.xlsx` (sheet `data`) | Excel | Housing unit stock (market value, rent, bedrooms, vacancy, tract) |
| `data/stores_prices.xlsx` (sheet `data`) | Excel | Average food prices and standard deviations by store and food category |
| `data/city_limits.shp` (+ companion `.dbf`/`.shx`/`.prj`) | Shapefile | City boundary, used to place new stores when old ones close |
| `data/seed_runs/` | CSV triples | Prior baseline simulation outputs (`*agents*.csv`, `*housing units*.csv`, `*stores*.csv`) used to initialize the model at month 18 instead of starting from scratch |

`main.py` randomly selects one seed run from `data/seed_runs/` (filtered by a
`True_0_0_0_0` tag in the filename, matching a specific policy configuration)
and uses its agents/housing units/stores files as the household, housing, and
store starting state.

Output is written to `../results/` relative to the scripts, as a CSV named
with the run date and policy parameters.

## Requirements

- Python 3.9+
- `mesa`, `pandas`, `numpy`, `geopandas`, `shapely`, `pulp`, `scipy`

```bash
pip install mesa pandas numpy geopandas shapely pulp scipy
```

## Running

```bash
python main.py
```

This will:
1. Pick a random seed run from `data/seed_runs/`.
2. Build the model from the base population/housing/stores data plus the seed.
3. Run 24 monthly steps.
4. Write a console log to `log_console_<model_label>_<date>.txt`.
5. Save the model-level results to `../results/results_<date>_..._<model_label>.csv`.

## Policy parameters

`main.py` exposes a few parameters that control policy scenarios passed down
to `Household`:

- `stamps_active` — whether SNAP benefits are active in the simulation.
- `perc` (`umbral`) — adjustment to the SNAP income eligibility threshold.
- `aj_bt0` / `aj_btelig` — adjustments to the SNAP benefit regression intercept
  and eligibility coefficient.
- `umbral_sec8` — adjustment to the Section 8 / public housing eligibility
  threshold.
