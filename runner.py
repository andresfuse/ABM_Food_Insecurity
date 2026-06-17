import pandas as pd
import time
import sys
from model import FoodAccessModel


class Runner:
    """Loads input data, reconciles a chosen simulation seed with the base
    synthetic population, and builds/runs the FoodAccessModel.

    household_data_modelo / housing_data_modelo / stores_data come from a
    previously generated simulation run (the "seed") that is used to
    initialize the model at month 18, instead of starting the population
    from scratch.
    """

    def __init__(self, persons_data, household_data, household_data_modelo, housing_data, housing_data_modelo,
                 stores_data, stores_prices, phily_shapefile,
                 log_consola, number_steps, activar_estampas, umbral, aj_bt0, aj_btelig, umbral_sec8):

        print("Loading datasets")
        print(time.time())
        self.persons = pd.read_csv(persons_data)
        self.household_original = pd.read_excel(household_data, sheet_name="data")
        self.household_modelo = pd.read_csv(household_data_modelo)
        self.household_modelo = self.household_modelo[self.household_modelo["anio"] == 18].reset_index()

        self.housing = pd.read_excel(housing_data, sheet_name="data")
        self.housing_modelo = pd.read_csv(housing_data_modelo)
        self.housing_modelo = self.housing_modelo[self.housing_modelo["anio"] == 18].reset_index()
        self.stores_modelo = pd.read_csv(stores_data)
        self.stores_modelo = self.stores_modelo[self.stores_modelo["anio"] == 18].reset_index()
        self.stores_prices = pd.read_excel(stores_prices, sheet_name="data")

        self.log_consola = log_consola
        self.number_steps = number_steps

        print("Reconciling input files with the selected seed")
        print(time.time())
        self.housing = self.housing.drop(columns=["VACANCY"], errors="ignore")
        self.housing = self.housing[self.housing["ID"].isin(self.housing_modelo["ID"])].reset_index()
        self.housing = self.housing.merge(
            self.housing_modelo[["ID", "For_sell", "Vacancy", "Mrent"]],
            on="ID",
            how="left"
        )

        self.housing = self.housing.rename(columns={
            "Mrent": "MRENT",
            "Vacancy": "VACANCY"
        })

        columns_to_remove = ["basket_clust", "RENT", "TRACTCE10", "HH_ID"]
        self.household_original = self.household_original.drop(columns=columns_to_remove, errors="ignore")
        agentes_modelo_selected = self.household_modelo[["ID", "Basket_id", "Rentm", "TRACTCE", "HU_id",
                                                           "Food_sec", "House_sec", "Moving", "Months_since_moving",
                                                           "Stores_searching", "Months_searching"]]

        self.household_original = self.household_original.merge(
            agentes_modelo_selected,
            on="ID",
            how="left"
        )

        self.household_original = self.household_original.rename(columns={
            "Basket_id": "basket_clust",
            "Rentm": "RENT",
            "TRACTCE": "TRACTCE10",
            "HU_id": "HH_ID",
            "Food_sec": "FOOD_SEC",
            "House_sec": "HOUSE_SEC"
        })

        hu_selected = self.housing[["ID", "INTPTLA", "INTPTLO"]]

        self.household_original = self.household_original.merge(
            hu_selected,
            left_on="HH_ID",
            right_on="ID",
            how="left"
        )

        self.household_original["INTPTLA"] = self.household_original.apply(
            lambda row: row["INTPTLA_x"] if pd.isna(row["HH_ID"]) else row["INTPTLA_y"], axis=1
        )

        self.household_original["INTPTLO"] = self.household_original.apply(
            lambda row: row["INTPTLO_x"] if pd.isna(row["HH_ID"]) else row["INTPTLO_y"], axis=1
        )

        self.household_original.drop(columns=["INTPTLA_x", "INTPTLA_y", "INTPTLO_x", "INTPTLO_y"], inplace=True)
        self.household_original = self.household_original.drop(columns=["ID_y"], errors="ignore")
        self.household_original = self.household_original.rename(columns={"ID_x": "ID"})

        # Stores in the expected model format
        self.stores_modelo["place_id"] = self.stores_modelo["ID"]
        self.stores_modelo["name"] = self.stores_modelo["ID"]
        self.stores_modelo["address"] = self.stores_modelo["ID"]

        self.stores_modelo = self.stores_modelo.rename(columns={
            "LAT": "INTPTLAT",
            "LON": "INTPTLON",
            "Race": "final_race",
            "Type": "final_type"
        })

        # Per-household chosen stores, quantities and prices from the seed run
        self.new_store_selection = self.household_modelo[["ID", "Stores_ids", "Stores_prices", "Stores_quantities", "Stores_others"]].copy()
        self.new_store_selection = self.new_store_selection[self.new_store_selection["Stores_quantities"] != "[]"]

        self.new_store_selection["ids"] = self.new_store_selection["ID"]
        self.new_store_selection = self.new_store_selection.rename(columns={
            "Stores_ids": "q_val",
            "Stores_quantities": "w_val",
            "Stores_prices": "stores_prices",
            "Stores_others": "other_costs"
        })

        print("Building the model")
        print(time.time())
        self.model = FoodAccessModel(
            self.household_original,
            self.housing,
            self.persons,
            self.stores_modelo,
            self.stores_prices,
            phily_shapefile, self.number_steps, activar_estampas, umbral,
            self.new_store_selection, aj_bt0, aj_btelig, umbral_sec8)

    def run(self, n_steps):
        with open(self.log_consola, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            try:
                for i in range(n_steps):
                    start_time = time.time()
                    print("-----------------------------------------------------------------------------------")
                    print(f"starting step {i + 19}")
                    self.model.step()
                    print(f"finishing step {i + 19}")
                    print("-----------------------------------------------------------------------------------")
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("Step time: %s minutes" % (elapsed_time / 60))
                    print("Remaining time: %s minutes" % ((elapsed_time * (n_steps - i)) / 60))
                return self.model.datacollector.get_model_vars_dataframe()
            finally:
                sys.stdout = original_stdout
