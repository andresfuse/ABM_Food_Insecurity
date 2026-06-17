"""Food Access ABM - main run script.

Selects a simulation seed from a set of baseline runs, builds the model from
the base synthetic population plus the seed state, runs it for a fixed number
of monthly steps, and saves the resulting model-level data to ../results.

All input files are expected under data/ (sibling folder, not included in
this repository):

- data/persons.csv - synthetic population, person-level records
- data/households.xlsx (sheet "data") - synthetic population, household-level records
- data/housing_units.xlsx (sheet "data") - housing unit stock
- data/stores_prices.xlsx (sheet "data") - average food prices by store
- data/city_limits.shp - city boundary shapefile
- data/seed_runs/ - prior baseline simulation outputs (agents/housing units/stores
  CSV triples) used to initialize the model at month 18
"""

import os
import time
import glob
import re
import random
from datetime import datetime

import runner

DATA_DIR = "data"

persons_data = os.path.join(DATA_DIR, "persons.csv")
household_data = os.path.join(DATA_DIR, "households.xlsx")
housing_data = os.path.join(DATA_DIR, "housing_units.xlsx")
stores_prices = os.path.join(DATA_DIR, "stores_prices.xlsx")
phily_shapefile = os.path.join(DATA_DIR, "city_limits.shp")
seed_runs_dir = os.path.join(DATA_DIR, "seed_runs")

print("Selecting the simulation seed")
print(time.time())

csv_files = glob.glob(f"{seed_runs_dir}/*.csv")
candidate_files = [f for f in csv_files if re.search(r"agents", f, re.IGNORECASE)
                    and re.search(r"True_0_0_0_0", f, re.IGNORECASE)]
agents_path = random.choice(candidate_files)
housing_units_path = agents_path.replace("agents", "housing units")
stores_path = agents_path.replace("agents", "stores")
match = re.search(r"main\d+(\.\d+)?", agents_path)
model_label = match.group(0) if match else "main_unknown"

number_steps = 24
today = datetime.now().strftime("%d%m%Y")
log_console = f"log_console_{model_label}_{today}.txt"
stamps_active = False
perc = 0
aj_bt0 = 0
aj_btelig = 0
umbral_sec8 = 0

if __name__ == "__main__":
    print("Initializing the model")
    print(time.time())
    initial_time = time.time()
    rn = runner.Runner(persons_data, household_data, agents_path, housing_data, housing_units_path, stores_path,
                        stores_prices, phily_shapefile, log_console, number_steps, stamps_active, perc,
                        aj_bt0, aj_btelig, umbral_sec8)
    data = rn.run(number_steps)

    output_dir = os.path.join("..", "results")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"results_{today}_{stamps_active}_{perc}_{aj_bt0}_{aj_btelig}_{umbral_sec8}_{model_label}.csv"
    )
    final_time = time.time()
    data.to_csv(save_path)
    print((final_time - initial_time) / 3600)
