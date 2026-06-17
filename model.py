from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from household import Household
from stores import Stores
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import math
import ast
import time
import statistics
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore")

HOUSING_UNITS = 0
INITIAL_STORES = 0
INITIAL_STORES_PRICES = 0
N_AGENTS = 0
ACTUAL_STORES = 0
ACTUAL_STORES_PRICES = 0
ID_MAX_STORES = 0
PHILY = 0

# Globals used for neighborhood selection
tracts_lists = []
black_composition = []
white_composition = []
asian_composition = []
other_composition = []
hispanic_composition = []
median_income = []
p20_rent = []
min_rent = []
old_median_income = []
old_p20_rent = []
old_min_rent = []
bruch_dic = {'Id': [], 'Bruch': []}
NUMBER_STEPS = 0

# Policy activation state, shared across households via globals
ACTIVAR_ESTAMPAS = True
UMBRAL = 0          # threshold for SNAP eligibility adjustment
AJ_BT0 = 0           # adjustment to the regression intercept for stamp value
AJ_BTELIG = 0        # adjustment to the eligibility coefficient for stamp value
UMBRAL_SEC8 = 0      # adjustment to the Section 8 eligibility threshold


class FoodAccessModel(Model):
    """Mesa model coordinating households and stores for the food-access ABM."""

    def __init__(self, households_data, housing_units_data, persons_data, stores_data, stores_prices,
                 phily_shapefile, number_steps, activar_estampas, umbral, stores_selection, aj_bt0, aj_btelig, umbral_sec8):
        super().__init__()

        global HOUSING_UNITS, INITIAL_STORES, INITIAL_STORES_PRICES, N_AGENTS
        global ACTUAL_STORES_PRICES, ID_MAX_STORES, PHILY, tracts_lists, NUMBER_STEPS
        global ACTIVAR_ESTAMPAS, UMBRAL, AJ_BT0, AJ_BTELIG, UMBRAL_SEC8

        HOUSING_UNITS = housing_units_data
        NUMBER_STEPS = number_steps

        INITIAL_STORES = stores_data
        INITIAL_STORES_PRICES = stores_prices
        ACTUAL_STORES_PRICES = stores_prices
        PHILY = phily_shapefile
        tracts_lists = HOUSING_UNITS["TRACTCE10"].unique().astype(int)
        HOUSING_UNITS["TRACTCE10"] = HOUSING_UNITS["TRACTCE10"].astype(int)

        N_AGENTS = households_data.shape[0]

        self.schedule_agents = RandomActivation(self)
        self.schedule_stores = RandomActivation(self)
        print("Building schedules")
        print(time.time())

        # Stores creation ------------------------------------------------------------------------
        ID_MAX_STORES = stores_data["ID"].max()
        for _, row in stores_data.iterrows():
            store_id = row["ID"]
            ID_MAX_STORES = max(ID_MAX_STORES, store_id)
            new_store = Stores(self, store_id, row["ID_data"], row["place_id"], row["name"], row["address"],
                                row["final_race"], row["final_type"], row["INTPTLAT"], row["INTPTLON"], stores_prices)
            self.schedule_stores.add(new_store)

        print("Stores added")
        print(time.time())

        # Households creation ----------------------------------------------------------------------
        total_agents = 0
        for _, row in households_data.iterrows():
            hh_id = row["ID"]
            hh_size = row["HHSIZE"]
            hh_type = row["HHTYPE"]
            inctot = row["HHINCOME"]
            owner = row["OWNERSHIP"]
            rent = row["RENTM"]
            race = row["race_max"]  # 1: white, 2: black, 3: asian, 4: hispanic, 5: other
            hu_id = None if pd.isna(row["HH_ID"]) else row["HH_ID"]
            maxstores = row["num_stores"]
            food_budget = row["budget"]
            ind_consumption = row["consumption"]
            food_stamp = row["FOODSTMP"]
            tract = row["TRACTCE10"]
            food_sec = row["FOOD_SEC"]
            house_sec = row["HOUSE_SEC"]
            months_after_moving = row["Months_since_moving"]
            moving = row["Moving"]
            store_searching = row["Stores_searching"]
            months_after_store_searching = row["Months_searching"]

            housing_units = HOUSING_UNITS[HOUSING_UNITS["ID"] == hu_id]
            if not housing_units.empty:
                HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == hu_id, "MRENT"] = housing_units["MRENT"].iloc[0] if rent <= 100 else rent
            lat = row["INTPTLA"]
            lon = row["INTPTLO"]

            stores_prices_data = stores_selection[stores_selection["ids"] == hh_id].reset_index(drop=True)

            if not stores_prices_data.empty:
                stores_ids = np.array(list(ast.literal_eval(stores_prices_data.loc[0, "q_val"])))
                stores_quantities = np.array(list(ast.literal_eval(stores_prices_data.loc[0, "w_val"])))
                stores_prices_arr = np.array([float(v) for v in stores_prices_data.loc[0, "stores_prices"].strip('[]').split()])
                other_costs = np.array([float(v) for v in stores_prices_data.loc[0, "other_costs"].strip('[]').split()])
            else:
                stores_ids = []
                stores_quantities = []
                stores_prices_arr = []
                other_costs = []

            new_household = Household(self, hh_id, tract, hh_size, hh_type, inctot, owner, rent, food_stamp, race, hu_id,
                                       maxstores, food_budget, ind_consumption, lat, lon, persons_data[persons_data["ID"] == hh_id],
                                       stores_data, stores_prices, ACTIVAR_ESTAMPAS, UMBRAL, AJ_BT0, AJ_BTELIG, HOUSING_UNITS, UMBRAL_SEC8,
                                       stores_ids, stores_quantities, stores_prices_arr, other_costs, food_sec, house_sec,
                                       months_after_moving, moving, store_searching, months_after_store_searching)

            self.schedule_agents.add(new_household)

            total_agents += 1
            if total_agents % 7500 == 0:
                print(f"Initialized {100 * total_agents / households_data.shape[0]:.2f}% of agents ({total_agents} agents)")
                print(time.time())

        print("Households added")
        print(time.time())

        ACTIVAR_ESTAMPAS = activar_estampas
        UMBRAL = umbral
        AJ_BT0 = aj_bt0
        AJ_BTELIG = aj_btelig
        UMBRAL_SEC8 = umbral_sec8

        self.datacollector = DataCollector(
            model_reporters={"Day": day, "Race_by_tract": tract_race_distribution, "Rent_by_rate": tract_rental_distribution,
                              "Agents_dictionary": dictionary_agents, "Stores": dictionary_stores,
                              "Housing_units": housing_units_export, "Bruch_dic": bruch_dic_export}
        )
        self.running = True

    # Mesa step -----------------------------------------------------------------------------------
    def step(self):
        global ACTUAL_STORES, ACTUAL_STORES_PRICES, INITIAL_STORES_PRICES, HOUSING_UNITS
        global tracts_lists, black_composition, white_composition, asian_composition
        global other_composition, hispanic_composition, bruch_dic
        global ACTIVAR_ESTAMPAS, UMBRAL, AJ_BT0, AJ_BTELIG, UMBRAL_SEC8

        print("Collecting data")
        ACTUAL_STORES = self.update_actual_stores()
        ACTUAL_STORES_PRICES = self.get_stores_prices()
        bruch_dic = {'Id': [], 'Bruch': []}
        print("Food insecure households: " + str(self.count_hh_fdinsecure()))
        print("House insecure households: " + str(self.count_hh_hhinsecure()))
        print("Households without an optimal store: " + str(self.count_without_stores()))
        on_sale = (HOUSING_UNITS["For_sell"] == True).sum()
        on_rent = (HOUSING_UNITS["For_sell"] == False).sum()
        print("Units for sale: " + str(on_sale) + ", units for rent: " + str(on_rent))
        print("Vacant units: " + str((HOUSING_UNITS["VACANCY"] == True).sum()))
        print(time.time())
        self.tract_composition_calculates()
        print(time.time())

        count = 0
        time_model = self.schedule_agents.time + 18
        if time == 24:
            agents = [agent for agent in self.schedule_agents.agents]
            for i in agents:
                i.activate_policies(ACTIVAR_ESTAMPAS, UMBRAL, AJ_BT0, AJ_BTELIG, UMBRAL_SEC8)

        if time_model % 12 == 0:  # Annual update
            print("Running annual update")
            agents = [agent for agent in self.schedule_agents.agents]
            for i in agents:
                old_huid = i.get_huid()
                self.neighborhood_selection(i)
                new_huid = i.get_huid()
                if old_huid != new_huid:
                    basket_id = i.get_basket_id()
                    new_basket = i.update_food_basket(i.get_race(), basket_id + 10)
                    i.change_basket(new_basket)
                    actual_budget = i.get_food_budget()
                    i.store_selection(ACTUAL_STORES, ACTUAL_STORES_PRICES, i.get_basket(), actual_budget)
                    if i.get_census_tract() is not None and not np.any(i.get_stores_ids()):
                        while not np.any(i.get_stores_ids()) and i.get_basket_id() > 1:
                            basket_id = i.get_basket_id()
                            new_basket = i.update_food_basket(i.get_race(), basket_id - 1)
                            i.change_basket(new_basket)
                            i.store_selection(ACTUAL_STORES, ACTUAL_STORES_PRICES, i.get_basket(), actual_budget)

                count += 1
                if count % 5000 == 0:
                    print("Reviewed: " + str(count) + " of " + str(len(agents)))
                    print(time.time())

        self.validate_households()
        self.validate_stores_selections()
        self.validate_household_searching()

        print("Running one step for all agents after validations")
        self.schedule_agents.step()
        self.update_stores_sales()
        self.schedule_stores.step()

        if time_model % 12 == 0:
            print("Computing store sales and openings/closings")
            self.sales_opening_and_closing()
        print("Collecting model data")
        self.datacollector.collect(self)
        print("step --------------------------------")
        print(time.time())

    # Store dynamics --------------------------------------------------------------------------------
    def sales_opening_and_closing(self):
        global ACTUAL_STORES_PRICES, ID_MAX_STORES, PHILY
        stores_list = [agent for agent in self.schedule_stores.agents]
        agg_sales = [i.get_agg_sales() for i in stores_list]
        perc7 = np.percentile(agg_sales, 7)

        stores_races, stores_types, stores_ids_data = [], [], []
        for i in stores_list:
            if i.get_agg_sales() < perc7:
                stores_races.append(i.get_race())
                stores_types.append(i.get_type())
                stores_ids_data.append(i.get_id_data())
                i.close_store()
            i.modify_agg_sales(0)

        for i in range(len(stores_races)):
            ID_MAX_STORES += 1
            n_lat, n_lon = generate_random_point(PHILY)
            new_store = Stores(self, ID_MAX_STORES, stores_ids_data[i], None, None, None,
                                stores_races[i], stores_types[i], n_lat, n_lon, INITIAL_STORES_PRICES)
            self.schedule_stores.add(new_store)

    def validate_stores_selections(self):
        print("Validating household store selections")
        agents = [agent for agent in self.schedule_agents.agents]
        food_insecure = self.count_hh_fdinsecure()
        count = 0
        max_iterations = 10
        for i in agents:
            if i.get_food_sec() == False or not np.any(i.get_stores_ids()):
                if i.get_census_tract() is not None and i.get_store_searching() == False:
                    perc_increase = 0.1
                    actual_budget = i.get_food_budget()
                    exitvar = False
                    iteration_count = 0
                    i.update_store_searching(True)
                    while (not np.any(i.get_stores_ids()) or exitvar == False) and perc_increase < 0.6:
                        iteration_count += 1
                        if iteration_count > max_iterations:
                            break

                        rent_adj = i.get_monthly_rent() if i.get_public_housing() == 0 else i.get_monthly_rent_fx()
                        expenses_with_food = (i.get_income() / 12) - i.get_other_expenses() - rent_adj + i.get_stamp_value() - actual_budget

                        if expenses_with_food > 0:
                            expenses_without_food = (i.get_income() / 12) - i.get_other_expenses() - rent_adj + i.get_stamp_value()
                            if (expenses_without_food - actual_budget * (1 + perc_increase)) > 0:
                                new_budget = actual_budget * (1 + perc_increase)
                                i.change_food_budget(new_budget)
                                i.store_selection(ACTUAL_STORES, ACTUAL_STORES_PRICES, i.get_basket(), i.get_food_budget())
                                perc_increase += 0.1
                                exitvar = True
                        else:
                            basket_id = i.get_basket_id()
                            if basket_id > 1:
                                new_basket = i.update_food_basket(i.get_race(), basket_id - 1)
                                i.change_basket(new_basket)
                                i.store_selection(ACTUAL_STORES, ACTUAL_STORES_PRICES, i.get_basket(), actual_budget)
                                exitvar = True
                            else:
                                if i.get_moving() == False:
                                    self.neighborhood_selection(i)
                                    i.store_selection(ACTUAL_STORES, ACTUAL_STORES_PRICES, i.get_basket(), i.get_food_budget())
                                exitvar = True
                                break

                count += 1
                if count % 1000 == 0:
                    print("Reviewed: " + str(count) + " of " + str(food_insecure))
                    print(time.time())

    def validate_household_searching(self):
        print("Validating housing search eligibility")
        agents = [agent for agent in self.schedule_agents.agents]
        for agent in agents:
            months = agent.get_months_since_moving()
            moving = agent.get_moving()
            store_searching = agent.get_store_searching()
            months_searching = agent.get_months_after_store_searching()

            if moving == True:
                if months < 3:
                    agent.update_months_since_moving(months + 1)
                else:
                    agent.update_months_since_moving(0)
                    agent.update_moving(False)

            if store_searching == True:
                if months_searching < 3:
                    agent.update_months_after_store_searching(months_searching + 1)
                else:
                    agent.update_months_since_moving(0)
                    agent.update_store_searching(False)

    def validate_households(self):
        print("Validating households")
        agents = [agent for agent in self.schedule_agents.agents]
        house_insecure = self.count_hh_hhinsecure()
        count = 0
        for i in agents:
            if i.get_house_sec() == False and i.get_moving() == False:
                self.neighborhood_selection(i)
                count += 1
                if count % 1000 == 0:
                    print("Pending review: " + str(count) + " of " + str(house_insecure))
                    print(time.time())

    def update_actual_stores(self):
        stores_list = [agent for agent in self.schedule_stores.agents]
        return pd.DataFrame({
            "ID": [i.get_id() for i in stores_list],
            "INTPTLAT": [i.get_lat() for i in stores_list],
            "INTPTLON": [i.get_lon() for i in stores_list],
            "final_race": [i.get_type() for i in stores_list],
            "final_type": [i.get_race() for i in stores_list]
        })

    def get_stores_prices(self):
        stores_list = [agent for agent in self.schedule_stores.agents]
        return pd.DataFrame({
            "Dairy": np.random.normal(loc=[i.get_dairy_avg() for i in stores_list], scale=[i.get_dairy_sd() for i in stores_list]),
            "Fruit": np.random.normal(loc=[i.get_fruit_avg() for i in stores_list], scale=[i.get_fruit_sd() for i in stores_list]),
            "Grains": np.random.normal(loc=[i.get_grains_avg() for i in stores_list], scale=[i.get_grains_sd() for i in stores_list]),
            "Proteins": np.random.normal(loc=[i.get_protein_avg() for i in stores_list], scale=[i.get_protein_sd() for i in stores_list]),
            "Prepared": np.random.normal(loc=[i.get_prep_avg() for i in stores_list], scale=[i.get_prep_sd() for i in stores_list]),
            "Other": np.random.normal(loc=[i.get_other_avg() for i in stores_list], scale=[i.get_other_sd() for i in stores_list]),
            "Veggies": np.random.normal(loc=[i.get_veg_avg() for i in stores_list], scale=[i.get_veg_sd() for i in stores_list])
        })

    # Neighborhood / housing choice ------------------------------------------------------------------
    def tract_composition_calculates(self):
        global HOUSING_UNITS, tracts_lists, black_composition, white_composition, asian_composition
        global hispanic_composition, median_income, p20_rent, min_rent
        global old_median_income, old_p20_rent, old_min_rent

        old_median_income = median_income
        old_p20_rent = p20_rent
        old_min_rent = min_rent
        black_composition = []
        white_composition = []
        asian_composition = []
        hispanic_composition = []
        median_income = []
        p20_rent = []
        min_rent = []
        for i in tracts_lists:
            df = self.calculate_race_composition(i)
            black_composition.append(df["BLACK"].values[0])
            white_composition.append(df["WHITE"].values[0])
            asian_composition.append(df["ASIAN"].values[0])
            hispanic_composition.append(df["HISPANIC"].values[0])
            median_income.append(self.get_ct_median_income(i))
            p20_rent.append(self.get_ct_p20_rent(i))
            min_rent.append(self.get_min_rent(i))

    def calculate_race_composition(self, tract):
        df = pd.DataFrame({"WHITE": [0], "BLACK": [0], "ASIAN": [0], "HISPANIC": [0], "OTHER": [0]})
        agents = [agent for agent in self.schedule_agents.agents]
        for i in agents:
            if i.get_census_tract() == tract:
                race_value = i.get_race()
                race = ("WHITE" if race_value == 1 else
                        ("BLACK" if race_value == 2 else
                         ("ASIAN" if race_value == 3 else
                          ("HISPANIC" if race_value == 4 else "OTHER"))))
                df[race] += 1
        df["TOTAL"] = df[["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"]].sum(axis=1)
        df[["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"]] = df[["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"]].div(df["TOTAL"], axis=0)
        return df

    def calculate_access(self, stores, hh, race):
        gdf_hh = gpd.GeoDataFrame(hh, geometry='geometry', crs='EPSG:4326')
        gdf_stores = gpd.GeoDataFrame(stores, geometry='geometry', crs='EPSG:4326')
        gdf_hh_buffer = gdf_hh.buffer(500)
        close_stores = gdf_hh_buffer.apply(lambda buffer_hh: gdf_stores[gdf_stores["RACE"].isin(race)].within(buffer_hh).sum(), axis=1)
        return True if close_stores > 0 else False

    def housing_units_selling_check(self):
        global HOUSING_UNITS
        rent = HOUSING_UNITS["MRENT"] * 12
        mk_value = HOUSING_UNITS["MARKET_VALUE"]
        flip_rate = mk_value / rent
        HOUSING_UNITS["For_sell"] = flip_rate > 15

    def neighborhood_selection(self, agent):
        global tracts_lists, black_composition, white_composition, asian_composition
        global hispanic_composition, median_income, p20_rent, min_rent
        global HOUSING_UNITS, ACTUAL_STORES, ACTUAL_STORES_PRICES, bruch_dic

        moving = agent.get_moving()

        if moving == False:
            betas = [5.983, -0.421, -3.083, 5.146, -0.583, 0.359, 0.223, -2.144, -0.364, 8.121, -3.704, 1.386, -0.765,
                     -1.611, -1.049, 14.985, -17.687, -2.242, 2.010, -3.002, 6.126, -1.935, -1.94, 4.367, -2.746,
                     -1.488, 2.698, 0.102, -1.666, 3.036, -3.16, 5.805, -10.838, -12.251, 22.061, 0.35, -0.989, -1.215, 0.002,
                     0.479, -0.016, 2.075, -4.654, -3.858, 5.176, -0.031, 0.031]

            hhincome = agent.get_income() / 12
            validation = [i for i, j in enumerate(min_rent) if hhincome > j]
            filled = False

            if len(validation) == 0:
                public_by_tract = self.public_housing_units_by_tract(HOUSING_UNITS)
                validation = [i for i, j in enumerate(public_by_tract) if j > 0]
                search_tracts = list(tracts_lists[validation])
                if len(search_tracts) == 0:
                    filled = True
            else:
                renting = False if agent.get_owner() == 1 else True
                available_tracts = self.housing_units_available(HOUSING_UNITS, renting=renting)
                search_tracts = list(tracts_lists[validation])
                search_tracts = [tract for tract, available in zip(search_tracts, available_tracts) if available > 0]

            new_ctid = agent.get_census_tract()
            if new_ctid is not None:
                search_tracts.append(new_ctid)
                search_tracts = list(set(search_tracts))

            utilities = [self.utility_neighborhood_model(agent, betas, x)[0] for x in search_tracts]
            utilities = np.array(utilities)
            max_utility = np.max(utilities)
            adjusted_utilities = utilities - max_utility

            if new_ctid is not None:
                search_tracts_np = np.array(search_tracts)
                indices = np.where(search_tracts_np == new_ctid)[0]
                ctid_index = None if indices.size == 0 else indices[0]

            probabilities = np.exp(adjusted_utilities) / np.sum(np.exp(adjusted_utilities))
            new_pos = np.argmax(probabilities)
            max_value = max(probabilities)

            new_row = {'Id': agent.get_id(), 'Bruch': dict(zip(search_tracts, probabilities))}
            bruch_dic = add_row(bruch_dic, new_row)

            self.housing_units_selling_check()
            probabilities_for_iter = list(probabilities)
            max_iterations = len(tracts_lists) + 1

            if new_ctid is not None:
                iterations = 0
                while filled == False and iterations < max_iterations:
                    if ctid_index is not None and new_pos == ctid_index:
                        agent.change_optimal_house(True)
                        filled = True
                        break
                    if agent.get_owner() == 1:
                        if agent.get_savings() > 0:
                            potential_savings = agent.get_savings() + HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == agent.get_huid(), "MARKET_VALUE"].values[0]
                            potential_housing_units = HOUSING_UNITS.loc[
                                (HOUSING_UNITS["TRACTCE10"] == HOUSING_UNITS["TRACTCE10"].unique()[new_pos]) &
                                (HOUSING_UNITS["NUM_BEDRMS"].isin([agent.get_hh_size() - 1, agent.get_hh_size() - 2])) &
                                (HOUSING_UNITS["VACANCY"] == True) & (HOUSING_UNITS["For_sell"] == True) &
                                (HOUSING_UNITS["MARKET_VALUE"] <= potential_savings) &
                                (HOUSING_UNITS["MRENT"] <= agent.get_monthly_rent())]
                        else:
                            potential_housing_units = HOUSING_UNITS.loc[
                                (HOUSING_UNITS["TRACTCE10"] == HOUSING_UNITS["TRACTCE10"].unique()[new_pos]) &
                                (HOUSING_UNITS["NUM_BEDRMS"].isin([agent.get_hh_size() - 1, agent.get_hh_size() - 2])) &
                                (HOUSING_UNITS["VACANCY"] == True) & (HOUSING_UNITS["For_sell"] == False) &
                                (HOUSING_UNITS["MRENT"] <= agent.get_monthly_rent())]
                    else:
                        potential_housing_units = HOUSING_UNITS.loc[
                            (HOUSING_UNITS["TRACTCE10"] == HOUSING_UNITS["TRACTCE10"].unique()[new_pos]) &
                            (HOUSING_UNITS["NUM_BEDRMS"].isin([agent.get_hh_size() - 1, agent.get_hh_size() - 2])) &
                            (HOUSING_UNITS["VACANCY"] == True) & (HOUSING_UNITS["For_sell"] == False) &
                            (HOUSING_UNITS["MRENT"] <= agent.get_monthly_rent())]

                    if not potential_housing_units.empty:
                        choosed = potential_housing_units.sample(n=1)
                        old_id = agent.get_huid()
                        HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == old_id, "VACANCY"] = True
                        self.developer_dynamics(agent, old_id)
                        new_home_id = choosed["ID"].values[0]
                        HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == new_home_id, "VACANCY"] = False

                        if agent.get_owner() == 1:
                            old_market_value = HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == old_id, "MARKET_VALUE"].values[0]
                            new_market_value = HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == new_home_id, "MARKET_VALUE"].values[0]
                            agent.change_savings(old_market_value)
                            agent.change_savings(-new_market_value)

                        agent.change_house(new_home_id,
                                            choosed["TRACTCE10"].values[0],
                                            choosed["INTPTLA"].values[0], choosed["INTPTLO"].values[0],
                                            choosed["MRENT"].values[0])
                        filled = True
                        agent.classify_house_insecure(True)
                        agent.change_optimal_house(True)
                        agent.update_moving(True)
                        agent.update_store_searching(False)
                        agent.update_months_after_store_searching(0)
                        agent.store_selection(ACTUAL_STORES, ACTUAL_STORES_PRICES, agent.get_basket(), agent.get_food_budget())
                    else:
                        if len(probabilities_for_iter) == 0:
                            filled = True
                            agent.classify_house_insecure(False)
                            agent.change_optimal_house(False)
                            agent.update_moving(True)
                            break
                        else:
                            probabilities_for_iter.remove(max_value)
                            if len(probabilities_for_iter) == 0:
                                agent.update_moving(True)
                                filled = True
                                if (agent.get_income() / 12) > agent.get_monthly_rent():
                                    agent.change_optimal_house(False)
                                    break
                                else:
                                    agent.classify_house_insecure(False)
                                    agent.change_optimal_house(False)
                                    break
                            else:
                                new_pos = np.argmax(probabilities_for_iter)
                                max_value = max(probabilities_for_iter)

                    iterations += 1
                    if filled == False and iterations == max_iterations:
                        agent.classify_house_insecure(False)
                        agent.change_optimal_house(False)
                        filled = True
                        break

    def developer_dynamics(self, agent, old_id):
        global HOUSING_UNITS
        if old_id is not None:
            owner_hu = HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == old_id, "RENT_OWN"].values[0]
            rent = HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == old_id, "MRENT"].values[0] * 12
            mk_value = HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == old_id, "MARKET_VALUE"].values[0]
            flip_rate = mk_value / rent

            if owner_hu == "dev":
                if flip_rate > 15:
                    HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == old_id, "For_sell"] = True
                    base = HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == agent.get_huid(), "MARKET_VALUE"]
                    HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == agent.get_huid(), "MARKET_VALUE"] = base * 1.1
                    base = HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == agent.get_huid(), "MRENT"]
                    HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == agent.get_huid(), "MRENT"] = base * 1.1
                else:
                    HOUSING_UNITS.loc[HOUSING_UNITS["ID"] == old_id, "For_sell"] = False

    def public_housing_units_by_tract(self, houses):
        tract_lists = houses["TRACTCE10"].unique()
        return [houses[(houses["TRACTCE10"] == i) & (houses["RENT_OWN"] == "public")].shape[0] for i in tract_lists]

    def housing_units_available(self, houses, renting):
        tract_lists = houses["TRACTCE10"].unique()
        houses_list = []
        for i in tract_lists:
            if renting:
                count = houses[(houses["TRACTCE10"] == i) & (houses["For_sell"] == False) & (houses["VACANCY"] == True)].shape[0]
            else:
                count = houses[(houses["TRACTCE10"] == i) & (houses["VACANCY"] == True)].shape[0]
            if count > 0:
                houses_list.append(count)
        return houses_list

    def utility_neighborhood_model(self, agent, betas, ctid):
        """Computes the Bruch-style neighborhood utility for an agent/tract pair."""
        global tracts_lists, black_composition, white_composition, asian_composition
        global hispanic_composition, median_income, p20_rent

        black = 1 if agent.get_race() == 2 else 0
        white = 1 if agent.get_race() == 1 else 0
        hispanic = 1 if agent.get_race() == 4 else 0
        asian = 1 if agent.get_race() == 3 else 0

        dij = 1 if agent.get_census_tract() == ctid else 0
        pos = np.where(tracts_lists == ctid)[0][0]
        incomer = agent.get_income() / median_income[pos]
        pricer = p20_rent[pos] / (agent.get_income() / 12)

        b = black_composition[pos]
        w = white_composition[pos]
        a = asian_composition[pos]
        h = hispanic_composition[pos]

        components = [dij, b, b ** 2, black * b, hispanic * b, hispanic * b ** 2, h, h ** 2, hispanic * h, hispanic * h ** 2,
                      black * h, black * h ** 2, a, a ** 2, asian * a, asian * a ** 2, w ** 2, white * w ** 2,
                      dij * b, dij * b ** 2, dij * black * b, dij * black * b ** 2, dij * hispanic * b, dij * hispanic * b ** 2,
                      dij * hispanic, dij * hispanic ** 2, dij * hispanic * h, dij * hispanic * h ** 2,
                      dij * black * h, dij * black * h ** 2, dij * a, dij * a ** 2, dij * asian * a, dij * asian * a ** 2,
                      dij * w ** 2, dij * white * w ** 2, incomer, incomer ** 2, dij * incomer, dij * incomer ** 2,
                      pricer, pricer ** 2, dij * pricer, dij * pricer ** 2, p20_rent[pos] / 1000,
                      dij * median_income[pos] / 1000]

        utility_estimate = sum(x * y for x, y in zip(components, betas))
        return utility_estimate, components, incomer, pricer

    def get_ct_median_income(self, ctid):
        global old_median_income
        income = [i.get_income() for i in self.schedule_agents.agents if i.get_census_tract() == ctid]
        pos = np.where(tracts_lists == ctid)[0][0]
        return np.median(np.array(income)) if len(income) > 0 else old_median_income[pos]

    def get_ct_p20_rent(self, ctid):
        global old_p20_rent
        rent = [i.get_monthly_rent() for i in self.schedule_agents.agents
                if i.get_census_tract() == ctid and i.get_monthly_rent() > 0]
        pos = np.where(tracts_lists == ctid)[0][0]
        return np.percentile(np.array(rent), 20) if len(rent) > 0 else old_p20_rent[pos]

    def get_min_rent(self, ctid):
        global old_min_rent
        rent = [i.get_monthly_rent() for i in self.schedule_agents.agents
                if i.get_census_tract() == ctid and i.get_monthly_rent() > 0]
        pos = np.where(tracts_lists == ctid)[0][0]
        return np.min(np.array(rent)) if len(rent) > 0 else old_min_rent[pos]

    # Store sales -----------------------------------------------------------------------------------
    def update_stores_sales(self):
        agents = [agent for agent in self.schedule_agents.agents]
        ids_stores, sales = [], []
        for i in agents:
            stores_selected = i.get_stores_ids()
            prices = i.get_stores_prices()
            quantities = i.get_stores_quantities()
            ids_stores.extend(np.array(stores_selected))
            sales.extend(np.array([x * y for x, y in zip(prices, quantities)]))

        agg_sales = pd.DataFrame({"ID": ids_stores, "Sales": sales})
        stores = [agent for agent in self.schedule_stores.agents]
        for j in stores:
            store_sales = agg_sales[agg_sales["ID"] == j.get_id()]
            j.update_monthly_sales(store_sales["Sales"].values[0] if len(store_sales) > 0 else 0)

    def count_hh_fdinsecure(self):
        return sum(1 for i in self.schedule_agents.agents if i.get_food_sec() == False)

    def count_hh_hhinsecure(self):
        return sum(1 for i in self.schedule_agents.agents if i.get_house_sec() == False)

    def count_without_stores(self):
        return sum(1 for i in self.schedule_agents.agents if len(i.get_stores_ids()) == 0)


# Data collector routines ---------------------------------------------------------------------------
def day(model):
    return model.schedule_agents.time


def tract_race_distribution(model):
    races_by_tract = {}
    tracts = [agent.tract for agent in model.schedule_agents.agents]
    races = [agent.race for agent in model.schedule_agents.agents]
    for t in set(tracts):
        races_counts = [0, 0, 0, 0, 0]  # white, black, asian, hispanic, other
        for i in range(len(races)):
            if tracts[i] == t:
                races_counts[races[i] - 1] += 1
        races_by_tract[t] = races_counts
    return races_by_tract


def tract_rental_distribution(model):
    rental_by_tract = {}
    tracts = [agent.tract for agent in model.schedule_agents.agents]
    rentm = [agent.rent for agent in model.schedule_agents.agents]
    for t in set(tracts):
        datos = [rentm[i] for i in range(len(rentm)) if tracts[i] == t and rentm[i] is not None and rentm[i] > 0]
        if datos:
            rental_by_tract[t] = [statistics.mean(datos), statistics.stdev(datos), statistics.median(datos),
                                   min(datos), max(datos), np.percentile(datos, 25), np.percentile(datos, 75),
                                   np.percentile(datos, 90)]
        else:
            rental_by_tract[t] = [None] * 8
    return rental_by_tract


def dictionary_agents(model):
    agents = [agent for agent in model.schedule_agents.agents]
    agents_dictionary = {
        "ID": [], "Race": [], "HH_ID": [], "Old_hhid": [], "Basket_id": [], "Census_tract": [],
        "Optimal_selection": [], "Food_security": [], "House_security": [], "Food_consumption": [],
        "Current_budget": [], "Savings": [], "Rentm": [], "Income": [], "Food_quants": [],
        "Initial_consumption": [], "Other_expenses": [], "Potential_budget": [], "Renter": [],
        "Stamp_value": [], "Moving": [], "Months_since_moving": [], "Stores_searching": [],
        "Months_searching": [], "Stores_ids": [], "Stores_prices": [], "Stores_quantities": [],
        "Stores_others": [], "Public_housing": [], "Rent_adj": [], "LAT": [], "LON": []
    }

    for i in agents:
        agents_dictionary["ID"].append(i.get_id())
        agents_dictionary["Race"].append(i.get_race())
        agents_dictionary["HH_ID"].append(i.get_huid())
        agents_dictionary["Old_hhid"].append(i.get_old_huid())
        agents_dictionary["Basket_id"].append(i.get_basket_id())
        agents_dictionary["Census_tract"].append(i.get_census_tract())
        agents_dictionary["Optimal_selection"].append(i.get_optimal_selection())
        agents_dictionary["Food_security"].append(i.get_food_sec())
        agents_dictionary["House_security"].append(i.get_house_sec())
        agents_dictionary["Food_consumption"].append(i.get_consumption())
        agents_dictionary["Current_budget"].append(i.get_food_budget())
        agents_dictionary["Savings"].append(i.get_savings())
        agents_dictionary["Rentm"].append(i.get_monthly_rent())
        agents_dictionary["Income"].append(i.get_income())
        agents_dictionary["Food_quants"].append(np.sum(i.get_stores_quantities()))
        agents_dictionary["Initial_consumption"].append(i.get_initial_consumption())
        agents_dictionary["Other_expenses"].append(i.get_other_expenses())
        agents_dictionary["Potential_budget"].append(i.get_potential_budget())
        agents_dictionary["Renter"].append(i.get_owner())
        agents_dictionary["Stamp_value"].append(i.get_stamp_value())
        agents_dictionary["Moving"].append(i.get_moving())
        agents_dictionary["Months_since_moving"].append(i.get_months_since_moving())
        agents_dictionary["Stores_searching"].append(i.get_store_searching())
        agents_dictionary["Months_searching"].append(i.get_months_after_store_searching())
        agents_dictionary["Stores_ids"].append(i.get_stores_ids())
        agents_dictionary["Stores_prices"].append(i.get_stores_prices())
        agents_dictionary["Stores_quantities"].append(i.get_stores_quantities())
        agents_dictionary["Stores_others"].append(i.get_stores_others_costs())
        agents_dictionary["Public_housing"].append(i.get_public_housing())
        agents_dictionary["Rent_adj"].append(i.get_monthly_rent_fx())
        agents_dictionary["LAT"].append(i.get_lat())
        agents_dictionary["LON"].append(i.get_lon())
    return agents_dictionary


def dictionary_stores(model):
    agents = [agent for agent in model.schedule_stores.agents]
    return {
        "ID": [i.get_id() for i in agents],
        "ID_data": [i.get_id_data() for i in agents],
        "Race": [i.get_race() for i in agents],
        "Type": [i.get_type() for i in agents],
        "Agg_sales": [i.get_agg_sales() for i in agents],
        "Latitude": [i.get_lat() for i in agents],
        "Longitude": [i.get_lon() for i in agents]
    }


def housing_units_export(model):
    global HOUSING_UNITS
    return {
        "ID": HOUSING_UNITS['ID'].tolist(),
        "VACANCY": HOUSING_UNITS['VACANCY'].tolist(),
        "For_sell": HOUSING_UNITS['For_sell'].tolist(),
        "MRENT": HOUSING_UNITS['MRENT'].tolist()
    }


def bruch_dic_export(model):
    global bruch_dic
    return bruch_dic


def generate_random_point(shapefile):
    gdf = gpd.read_file(shapefile)
    minx, miny, maxx, maxy = gdf.total_bounds
    random_lat = np.random.uniform(miny, maxy)
    random_lon = np.random.uniform(minx, maxx)
    return random_lat, random_lon


def add_row(dic, row):
    dic['Id'].append(row['Id'])
    dic['Bruch'].append(row['Bruch'])
    return dic
