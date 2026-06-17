import numpy as np
import math
from pulp import LpProblem, LpVariable, LpMaximize, LpInteger, LpContinuous, LpStatus, value, lpSum
from math import radians, sin, cos, sqrt, atan2
import statistics as st
from mesa import Agent

HOUSING_UNITS = 0


class Household(Agent):
    """Household agent: income, food security, housing security and store
    selection logic.

    Store selection is solved as a small integer program (PuLP) that picks
    which stores to visit and how much to buy at each one, given a food
    budget, a maximum number of stores and a per-store "affinity" score
    (price, racial composition and store type).
    """

    def __init__(self, model, hh_id, tract, hh_size, hh_type, inctot, rentown, rent, food_stamp,
                 race, houseid, maxstores, food_budget, ind_consumption, lat, lon, persons_data,
                 stores_data, avg_prices, activar_estampas, umbral, aj_bt0, aj_btelig, housing_units, umbral_sec8,
                 stores_ids, stores_quantities, stores_prices, other_costs, food_sec, house_sec,
                 months_after_moving, moving, store_searching, months_after_store_searching):
        super().__init__(hh_id, model)

        global HOUSING_UNITS
        HOUSING_UNITS = housing_units

        self.hh_id = hh_id
        self.tract = tract
        self.old_tract = tract
        self.hh_size = hh_size
        self.hh_type = hh_type
        self.inctot = inctot
        self.race = race
        self.gross_inctot = inctot
        self.inctot = self.income_after_taxes(inctot, hh_type)
        self.other_expenses = self.estimate_other_expenses(self.race, self.inctot / 12, self.hh_size)  # monthly
        self.snap_eligible = self.snap_eligibility(self.hh_size, self.gross_inctot / 12, 0)
        self.savings = 0
        self.rentown = rentown
        self.rent = rent

        self.food_stamp = food_stamp
        self.months_after_moving = months_after_moving
        self.moving = moving
        self.store_searching = store_searching
        self.months_after_store_searching = months_after_store_searching

        self.sec8 = self.rent_subsidies_eligibility(self.hh_size, self.gross_inctot, umbral_sec8)
        self.pubhouse = self.estimate_pub_housing(self.hh_size, self.gross_inctot, self.rentown,
                                                    self.race, self.sec8)
        stmpval = self.estimate_stamps(self.hh_size, self.gross_inctot,
                                        self.rentown, self.race,
                                        self.pubhouse, self.snap_eligible, activar_estampas, 0, 0) / 12
        self.stmpval = stmpval

        self.houseid = houseid
        self.old_houseid = houseid
        self.persons = persons_data
        self.maxstores = maxstores
        self.food_budget = food_budget
        self.potential_food_budget = 0
        self.initial_food_buget = self.food_budget
        self.ind_consumption = ind_consumption
        self.initial_consumption = self.ind_consumption
        self.poverty_rate = self.define_poverty_line(self.hh_size, self.inctot)  # 2023 Federal Poverty Level
        self.basket, self.basket_id = self.define_basket(self.race, self.poverty_rate)
        self.food_consumption = 0

        self.lat = lat
        self.lon = lon

        self.rentfix = self.estimate_fixed_rent(HOUSING_UNITS, self.rent, self.pubhouse, self.houseid, self.gross_inctot)

        self.food_sec = food_sec
        self.house_sec = house_sec
        self.optimal_house = True

        self.stores_ids = stores_ids
        self.stores_prices = stores_prices
        self.stores_quantities = stores_quantities
        self.initial_stores_quantities = self.stores_quantities
        self.other_costs = other_costs

    # Step -------------------------------------------------------------------------------------
    def step(self):
        rent = self.rent if self.pubhouse == 0 else self.rentfix
        basket_budget = sum(x * y for x, y in zip(self.stores_prices, self.stores_quantities)) + sum(self.other_costs)
        basket_budget_potential = sum(x * y for x, y in zip(self.stores_prices, self.initial_stores_quantities)) + sum(self.other_costs)
        if basket_budget is None or basket_budget == "" or not np.any(self.stores_ids):
            basket_budget = self.food_budget
            self.classify_food_insecure(False)
        if rent is None or rent == "" or self.houseid is None:
            rent = 0
            self.classify_house_insecure(False)
        self.potential_food_budget = basket_budget_potential
        utilities = (self.inctot / 12) + self.stmpval - basket_budget - rent - self.other_expenses
        if (self.inctot / 12) >= rent:
            if utilities < 0:
                self.classify_food_insecure(False)
            else:
                self.food_consumption = basket_budget
                self.classify_food_insecure(True)
        else:
            self.classify_house_insecure(False)
            utilities2 = (self.inctot / 12) - basket_budget + self.stmpval - self.other_expenses
            if utilities2 < 0:
                self.classify_food_insecure(False)
            else:
                self.classify_food_insecure(True)
        self.change_savings(utilities)

    def activate_policies(self, activar_estampas, umbral, aj_bt0, aj_btelig, umbral_sec8):
        self.snap_eligible = self.snap_eligibility(self.hh_size, self.gross_inctot / 12, umbral)
        self.sec8 = self.rent_subsidies_eligibility(self.hh_size, self.gross_inctot, umbral_sec8)
        self.pubhouse = self.estimate_pub_housing(self.hh_size, self.gross_inctot, self.rentown,
                                                    self.race, self.sec8)
        stmpval = self.estimate_stamps(self.hh_size, self.gross_inctot,
                                        self.rentown, self.race,
                                        self.pubhouse, self.snap_eligible, activar_estampas, aj_bt0, aj_btelig) / 12
        self.stmpval = stmpval
        self.rentfix = self.estimate_fixed_rent(HOUSING_UNITS, self.rent, self.pubhouse, self.houseid, self.gross_inctot)

    # Store selection ---------------------------------------------------------------------------
    def store_selection(self, stores, stores_prices, weights, food_budget):
        if self.get_census_tract() is not None:
            affinity, prices_list, other_costs = self.calculate_affinity(stores, stores_prices, weights)
            q, w, _, status = self.choose_store(stores, self.maxstores, food_budget, self.ind_consumption,
                                                 affinity, prices_list, other_costs)

            if status == 'Optimal':
                q_val = {k: v for k, v in q.items() if v == 1}
                w_val = {k: v for k, v in w.items() if v > 0.1}
                self.stores_ids = list(q_val.keys())
                self.stores_prices = prices_list[np.array(list(q_val.keys()))]
                self.stores_quantities = list(w_val.values())
                self.other_costs = np.array(list(other_costs[np.array(list(q_val.keys()))]))
            else:
                self.stores_ids = []
                self.stores_prices = []
                self.stores_quantities = []
                self.other_costs = []
                self.classify_food_insecure(False)

    def calculate_affinity(self, stores, stores_prices, weights):
        distances = stores.apply(lambda row: euclidean_distance(row["INTPTLAT"], row["INTPTLON"], self.lat, self.lon), axis=1)
        ref_ethnicity = self.race
        prices = stores_prices.applymap(abs)
        weights = np.array(weights) / np.sum(weights)
        final_prices = prices.values.dot(weights)
        prices_stores = final_prices
        # Travel cost assuming $0.16/mile fuel + $28.78/250 miles maintenance
        other_costs = (distances * 0.000621371) * 0.16 * 2 + ((distances * 0.000621371) / 49.7) * 28.78
        share_ethnicity = [1 if x == ref_ethnicity else 0 for x in stores['final_race']]
        prices_list = [1 - (x / max(prices_stores)) for x in prices_stores]
        diversity = [1 if x == "Supermarket" else (0.85 if x == "Grocery store" else 0.7) for x in stores["final_type"]]
        affinity = [st.mean(t) for t in zip(prices_list, share_ethnicity, diversity)]
        return affinity, prices_stores, other_costs

    def choose_store(self, df, max_stores, income, foodreq, affinity, stores_prices, other_costs):
        prob = LpProblem("Store_selection", LpMaximize)

        store_vars = LpVariable.dicts("Store", df.index, lowBound=0, upBound=1, cat=LpInteger)
        quantities_vars = LpVariable.dicts("Quantity", df.index, lowBound=0, cat=LpContinuous)

        prob += lpSum([(store_vars[i] * affinity[i] - 0.25 * quantities_vars[i] * affinity[i] / foodreq) for i in df.index])

        prob += lpSum([store_vars[i] for i in df.index]) <= max_stores
        prob += lpSum([quantities_vars[i] * stores_prices[i] for i in df.index]) <= income - lpSum([store_vars[i] * other_costs[i] for i in df.index])
        prob += lpSum([quantities_vars[i] for i in df.index]) >= foodreq
        for i in df.index:
            prob += quantities_vars[i] >= store_vars[i]
            prob += quantities_vars[i] <= store_vars[i] * foodreq / (max_stores - 0.3)

        prob.solve()

        chosen_stores = {i: store_vars[i].value() for i in df.index}
        chosen_quantities = {i: quantities_vars[i].value() for i in df.index}
        total_affinity = value(prob.objective)

        if LpStatus[prob.status] == 'Optimal':
            return chosen_stores, chosen_quantities, total_affinity, LpStatus[prob.status]
        if chosen_stores:
            return chosen_stores, chosen_quantities, total_affinity, LpStatus[prob.status]
        return None

    # Attribute estimation ------------------------------------------------------------------------
    def delim_consumption(self, race, size):
        if race == 1:
            consumption = sum(np.random.normal(loc=981.0718, scale=15.61896, size=size))
        elif race == 2:
            consumption = sum(np.random.normal(loc=892.4134, scale=19.23031, size=size))
        elif race == 3:
            consumption = sum(np.random.normal(loc=1220.4159, scale=40.99786, size=size))
        elif race == 4:
            consumption = sum(np.random.normal(loc=1028.2093, scale=16.49776, size=size))
        else:
            consumption = sum(np.random.normal(loc=1007.5341, scale=47.72992, size=size))
        return consumption

    def delim_num_stores(self, race):
        if race == 1:
            num_stores = round(np.random.normal(loc=3.419949, scale=0.06149978), 0)
        elif race == 2:
            num_stores = round(np.random.normal(loc=3.476082, scale=0.13090013), 0)
        elif race == 3:
            num_stores = round(np.random.normal(loc=3.797720, scale=0.22782226), 0)
        elif race == 4:
            num_stores = round(np.random.normal(loc=3.303682, scale=0.22347340), 0)
        else:
            num_stores = round(np.random.normal(loc=3.351857, scale=0.15890548), 0)
        return num_stores

    def delim_budget(self, race, income, hhsize):
        if race == 1:
            budget = 108.9030385 + 0.0150158 * income + 39.4970226 * hhsize
        elif race == 2:
            budget = 92.654624 + 0.009800 * income + 43.774172 * hhsize
        elif race == 3:
            budget = 77.757515 + 0.011635 * income + 66.884581 * hhsize
        elif race == 4:
            budget = 121.878139 + 0.012932 * income + 42.130580 * hhsize
        else:
            budget = 64.348037 + 0.003637 * income + 72.735986 * hhsize
        return budget

    def estimate_other_expenses(self, race, income, hhsize):
        if race == 1:
            response = 81.790653 + income * 0.047406 + 111.498510 * hhsize
        elif race == 2:
            response = 148.695137 + income * 0.057930 + 2.016360 * hhsize
        elif race == 3:
            response = 85.66394 + income * 0.06957 + 55.15325 * hhsize
        elif race == 4:
            response = 109.09546 + income * 0.05832 + 68.84791 * hhsize
        else:
            response = 135.08378 + income * 0.03120 + 121.71024 * hhsize
        return response

    def estimate_fixed_rent(self, hu, rent, pubhouse, houseid, hhincome):
        estimate = 0
        if houseid is not None:
            mrent = hu.loc[hu["ID"] == houseid, "MRENT"].values[0]
            beds = hu.loc[hu["ID"] == houseid, "NUM_BEDRMS"].values[0]
            if pubhouse == 1:
                rent_threshold = self.rent_sec8_threshold(beds)
                if rent == 0 or rent is None:
                    max_rent_benefit = min(0.4 * hhincome / 12, rent_threshold, mrent)
                    estimate = mrent - max_rent_benefit
                else:
                    max_rent_benefit = min(0.4 * hhincome / 12, rent_threshold, rent)
                    estimate = rent - max_rent_benefit
        return estimate

    def rent_sec8_threshold(self, beds):
        if beds <= 1:
            return 941
        if beds == 2:
            return 1157
        if beds == 3:
            return 1481
        return 1657

    def estimate_pub_housing(self, hhsize, income, ownershp, race, sec8):
        if race == 2:
            race_estimate = 0.6468543
        elif race == 3:
            race_estimate = 0.09132154
        elif race == 4:
            race_estimate = 0.2149494
        elif race == 5:
            race_estimate = 0.6863667
        else:
            race_estimate = 0
        estimate = -21.10982 - 0.1936444 * math.log(income) + 19.8582 * (ownershp - 1) + race_estimate - 0.05413686 * hhsize + 1.618889 * sec8
        return 1 if estimate >= 0.15 else 0

    def rent_subsidies_eligibility(self, hhsize, income, umbral_sec8):
        if hhsize == 1:
            income_limit = income < 23348 * (1 + umbral_sec8)
        elif hhsize == 2:
            income_limit = income < 26683 * (1 + umbral_sec8)
        elif hhsize == 3:
            income_limit = income < 30018 * (1 + umbral_sec8)
        elif hhsize == 4:
            income_limit = income < 33354 * (1 + umbral_sec8)
        elif hhsize == 5:
            income_limit = income < 36051 * (1 + umbral_sec8)
        elif hhsize == 6:
            income_limit = income < 38747 * (1 + umbral_sec8)
        elif hhsize == 7:
            income_limit = income < 41373 * (1 + umbral_sec8)
        else:
            income_limit = income < 44070 * (1 + umbral_sec8)
        return income_limit

    def estimate_stamps(self, hhsize, income, ownershp, race, pubhous, snap_eligible, activar_estampas, aj_bt0, aj_btelig):
        if race == 2:
            race_estimate = 161.8687
        elif race == 3:
            race_estimate = -116.5559
        elif race == 4:
            race_estimate = -20.87629
        elif race == 5:
            race_estimate = 241.6672
        else:
            race_estimate = 0
        stamp_value = (1030.7096 * (1 + aj_bt0) + 209.2892 * hhsize - 131.061 * math.log(income)
                       + 226.2091 * (ownershp - 1) + 638.1825 * pubhous + race_estimate
                       + 390.8014 * (1 + aj_btelig) * snap_eligible)
        stamp_value = stamp_value if stamp_value > 0 else 0
        return stamp_value if activar_estampas else 0

    def snap_eligibility(self, hhsize, income, umbral):
        if hhsize == 1:
            resp = income < 2430 * (1 + umbral)
        elif hhsize == 2:
            resp = income < 3288 * (1 + umbral)
        elif hhsize == 3:
            resp = income < 4144 * (1 + umbral)
        elif hhsize == 4:
            resp = income < 5000 * (1 + umbral)
        elif hhsize == 5:
            resp = income < 5858 * (1 + umbral)
        elif hhsize == 6:
            resp = income < 6714 * (1 + umbral)
        elif hhsize == 7:
            resp = income < 7570 * (1 + umbral)
        elif hhsize == 8:
            resp = income < 8428 * (1 + umbral)
        elif hhsize == 9:
            resp = income < 9286 * (1 + umbral)
        elif hhsize == 10:
            resp = income < 10144 * (1 + umbral)
        else:
            resp = income < (10144 + 858 * (hhsize - 10)) * (1 + umbral)
        return resp

    def define_poverty_line(self, hhsize, hhincome):
        if hhsize == 1:
            response = hhincome / 14580
        elif hhsize == 2:
            response = hhincome / 19720
        elif hhsize == 3:
            response = hhincome / 24860
        elif hhsize == 4:
            response = hhincome / 30000
        elif hhsize == 5:
            response = hhincome / 35140
        elif hhsize == 6:
            response = hhincome / 40280
        elif hhsize == 7:
            response = hhincome / 45420
        elif hhsize == 8:
            response = hhincome / 50560
        else:
            response = hhincome / (50560 + (hhsize - 8) * 5140)
        return response

    def categorize_poverty_line(self, value):
        if value < 0.5:
            return 1
        if value < 1.0:
            return 2
        if value < 1.5:
            return 3
        if value < 2.5:
            return 4
        if value < 3.5:
            return 5
        return 6

    def restart_initial_values(self):
        self.stores_quantities = self.initial_stores_quantities
        self.food_budget = self.initial_food_buget

    def income_after_taxes(self, income, hhtype):
        federal_tax = 0
        if hhtype in [4, 6, 9]:
            if income <= 11600:
                federal_tax = income * 0.1
            if 11600 < income <= 47150:
                federal_tax = income * 0.12
            if 47150 < income <= 100525:
                federal_tax = income * 0.22
            if 100525 < income <= 191950:
                federal_tax = income * 0.24
            if 191950 < income <= 243725:
                federal_tax = income * 0.32
            if 243725 < income <= 609350:
                federal_tax = income * 0.35
            else:
                federal_tax = income * 0.37
        elif hhtype in [2, 3]:
            if income <= 11600:
                federal_tax = income * 0.1
            if 11600 < income <= 47150:
                federal_tax = income * 0.12
            if 47150 < income <= 100525:
                federal_tax = income * 0.22
            if 100525 < income <= 191950:
                federal_tax = income * 0.24
            if 191950 < income <= 243725:
                federal_tax = income * 0.32
            if 243725 < income <= 365600:
                federal_tax = income * 0.35
            else:
                federal_tax = income * 0.37
        elif hhtype == 1:
            if income <= 23200:
                federal_tax = income * 0.1
            if 23200 < income <= 94300:
                federal_tax = income * 0.12
            if 94300 < income <= 201050:
                federal_tax = income * 0.22
            if 201050 < income <= 383900:
                federal_tax = income * 0.24
            if 383900 < income <= 487450:
                federal_tax = income * 0.32
            if 487450 < income <= 731200:
                federal_tax = income * 0.35
            else:
                federal_tax = income * 0.37
        else:
            if income <= 16550:
                federal_tax = income * 0.1
            if 16550 < income <= 63100:
                federal_tax = income * 0.12
            if 63100 < income <= 100500:
                federal_tax = income * 0.22
            if 100500 < income <= 191950:
                federal_tax = income * 0.24
            if 191950 < income <= 243700:
                federal_tax = income * 0.32
            if 243700 < income <= 609350:
                federal_tax = income * 0.35
            else:
                federal_tax = income * 0.37
        state_tax = income * 0.0307
        city_tax = income * 0.0375
        return income - federal_tax - state_tax - city_tax

    def define_basket(self, race, poverty_line):
        if race == 1:
            probabilidades = np.array([
                [17.6, 40.0, 58.8, 80.0, 100.0],
                [26.2, 46.7, 58.4, 80.4, 100.0],
                [19.1, 41.7, 55.8, 80.2, 100.0],
                [16.7, 37.1, 54.1, 79.0, 100.0],
                [15.6, 34.9, 49.5, 80.3, 100.0],
                [16.4, 31.9, 47.7, 80.3, 100.0]
            ])
            pesos = np.array([
                [0.0339, 0.0215, 0.0339, 0.0434, 0.715, 0.102, 0.05],
                [0.437, 0.0622, 0.0732, 0.0851, 0.154, 0.111, 0.0774],
                [0.0719, 0.0795, 0.0722, 0.148, 0.0872, 0.134, 0.407],
                [0.0815, 0.0947, 0.0904, 0.0889, 0.371, 0.178, 0.0947],
                [0.0645, 0.0854, 0.164, 0.256, 0.0612, 0.272, 0.097]
            ])
        elif race == 2:
            probabilidades = np.array([
                [10.8, 51.1, 82.0, 100.0],
                [9.4, 45.1, 78.6, 100.0],
                [15.3, 45.8, 78.4, 100.0],
                [14.5, 50.0, 81.0, 100.0],
                [9.1, 39.6, 77.5, 100.0],
                [14.0, 38.5, 79.6, 100.0]
            ])
            pesos = np.array([
                [0.401, 0.0553, 0.087, 0.113, 0.137, 0.112, 0.0612],
                [0.0218, 0.0361, 0.0524, 0.0854, 0.611, 0.128, 0.0647],
                [0.0305, 0.134, 0.151, 0.136, 0.135, 0.19, 0.224],
                [0.0265, 0.0285, 0.108, 0.472, 0.113, 0.139, 0.114]
            ])
        elif race == 3:
            probabilidades = np.array([
                [31.3, 50.0, 100.0],
                [30.2, 53.5, 100.0],
                [40.7, 59.3, 100.0],
                [47.9, 68.1, 100.0],
                [44.6, 66.2, 100.0],
                [38.8, 58.8, 100.0]
            ])
            pesos = np.array([
                [0.047, 0.075, 0.112, 0.092, 0.516, 0.089, 0.068],
                [0.33, 0.095, 0.125, 0.104, 0.16, 0.079, 0.077],
                [0.042, 0.137, 0.261, 0.168, 0.115, 0.117, 0.161]
            ])
        elif race == 4:
            probabilidades = np.array([
                [28.4, 48.3, 88.8, 100.0],
                [23.9, 37.3, 81.1, 100.0],
                [24.0, 42.0, 80.0, 100.0],
                [26.9, 43.8, 84.2, 100.0],
                [29.4, 41.9, 80.6, 100.0],
                [30.1, 43.7, 78.3, 100.0]
            ])
            pesos = np.array([
                [0.0352, 0.0544, 0.0696, 0.0887, 0.568, 0.122, 0.0621],
                [0.0467, 0.049, 0.179, 0.385, 0.0934, 0.138, 0.109],
                [0.365, 0.0843, 0.0969, 0.106, 0.175, 0.0985, 0.0744],
                [0.0529, 0.192, 0.129, 0.102, 0.131, 0.183, 0.21]
            ])
        else:
            probabilidades = np.array([
                [54.5, 59.1, 100.0],
                [42.9, 57.1, 100.0],
                [24.5, 49.1, 100.0],
                [44.6, 66.2, 100.0],
                [38.2, 64.7, 100.0],
                [34.7, 53.5, 100.0]
            ])
            pesos = np.array([
                [0.0289, 0.0325, 0.0545, 0.0809, 0.628, 0.121, 0.054],
                [0.373, 0.0595, 0.0747, 0.011, 0.174, 0.111, 0.0973],
                [0.0264, 0.0722, 0.139, 0.229, 0.131, 0.207, 0.195]
            ])

        random_value = np.random.uniform(0, 100)
        cluster_assigned = len(probabilidades[0])
        for i, prob in enumerate(probabilidades[self.categorize_poverty_line(poverty_line) - 1]):
            if random_value <= prob:
                cluster_assigned = i + 1
                break
        return pesos[cluster_assigned - 1], cluster_assigned

    # State mutation ------------------------------------------------------------------------------
    def change_house(self, houseid, tract, lat, lon, rent):
        self.old_tract = self.tract
        self.old_houseid = self.houseid
        self.tract = tract
        self.houseid = houseid
        self.lat = lat
        self.lon = lon
        self.rent = rent

    def update_food_basket(self, race, actual_basket):
        if race == 1:
            pesos = np.array([
                [0.0339, 0.0215, 0.0339, 0.0434, 0.715, 0.102, 0.05],
                [0.437, 0.0622, 0.0732, 0.0851, 0.154, 0.111, 0.0774],
                [0.0719, 0.0795, 0.0722, 0.148, 0.0872, 0.134, 0.407],
                [0.0815, 0.0947, 0.0904, 0.0889, 0.371, 0.178, 0.0947],
                [0.0645, 0.0854, 0.164, 0.256, 0.0612, 0.272, 0.097]
            ])
        elif race == 2:
            pesos = np.array([
                [0.401, 0.0553, 0.087, 0.113, 0.137, 0.112, 0.0612],
                [0.0218, 0.0361, 0.0524, 0.0854, 0.611, 0.128, 0.0647],
                [0.0305, 0.134, 0.151, 0.136, 0.135, 0.19, 0.224],
                [0.0265, 0.0285, 0.108, 0.472, 0.113, 0.139, 0.114]
            ])
        elif race == 3:
            pesos = np.array([
                [0.047, 0.075, 0.112, 0.092, 0.516, 0.089, 0.068],
                [0.33, 0.095, 0.125, 0.104, 0.16, 0.079, 0.077],
                [0.042, 0.137, 0.261, 0.168, 0.115, 0.117, 0.161]
            ])
        elif race == 4:
            pesos = np.array([
                [0.0352, 0.0544, 0.0696, 0.0887, 0.568, 0.122, 0.0621],
                [0.0467, 0.049, 0.179, 0.385, 0.0934, 0.138, 0.109],
                [0.365, 0.0843, 0.0969, 0.106, 0.175, 0.0985, 0.0744],
                [0.0529, 0.192, 0.129, 0.102, 0.131, 0.183, 0.21]
            ])
        else:
            pesos = np.array([
                [0.0289, 0.0325, 0.0545, 0.0809, 0.628, 0.121, 0.054],
                [0.373, 0.0595, 0.0747, 0.011, 0.174, 0.111, 0.0973],
                [0.0264, 0.0722, 0.139, 0.229, 0.131, 0.207, 0.195]
            ])
        self.change_basket_id(min(actual_basket, len(pesos)))
        return pesos[min(actual_basket - 1, len(pesos) - 1)]

    def change_foodsec(self, value):
        self.foodsec = value

    def change_savings(self, value):
        self.savings = self.savings + value

    def change_food_budget(self, value):
        self.food_budget = value

    def change_basket_id(self, new_id):
        self.basket_id = new_id

    def change_basket(self, new_basket):
        self.basket = new_basket

    # Accessors -------------------------------------------------------------------------------------
    def get_months_since_moving(self): return self.months_after_moving
    def get_moving(self): return self.moving
    def update_months_since_moving(self, value): self.months_after_moving = value
    def update_moving(self, value): self.moving = value
    def get_months_after_store_searching(self): return self.months_after_store_searching
    def get_store_searching(self): return self.store_searching
    def update_months_after_store_searching(self, value): self.months_after_store_searching = value
    def update_store_searching(self, value): self.store_searching = value
    def get_lat(self): return self.lat
    def get_lon(self): return self.lon
    def get_id(self): return self.hh_id
    def get_census_tract(self): return self.tract
    def get_hh_size(self): return self.hh_size
    def get_race(self): return self.race
    def get_income(self): return self.inctot
    def get_owner(self): return self.rentown
    def get_monthly_rent(self): return self.rent
    def get_monthly_rent_fx(self): return self.rentfix
    def get_huid(self): return self.houseid
    def get_old_huid(self): return self.old_houseid
    def get_savings(self): return self.savings
    def get_food_budget(self): return self.food_budget
    def get_stores_ids(self): return self.stores_ids
    def get_stores_prices(self): return self.stores_prices
    def get_stores_quantities(self): return self.stores_quantities
    def get_stores_others_costs(self): return self.other_costs
    def get_basket_id(self): return self.basket_id
    def get_basket(self): return self.basket
    def classify_food_insecure(self, response): self.food_sec = response
    def classify_house_insecure(self, response): self.house_sec = response
    def get_food_sec(self): return self.food_sec
    def get_house_sec(self): return self.house_sec
    def get_consumption(self): return self.food_consumption
    def change_optimal_house(self, value): self.optimal_house = value
    def get_optimal_selection(self): return self.optimal_house
    def get_initial_consumption(self): return self.initial_consumption
    def get_other_expenses(self): return self.other_expenses
    def get_potential_budget(self): return self.potential_food_budget
    def get_stamp_value(self): return self.stmpval
    def get_public_housing(self): return self.pubhouse


def euclidean_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two lat/lon points."""
    R = 6371000
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = radians(lat1), radians(lon1), radians(lat2), radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
