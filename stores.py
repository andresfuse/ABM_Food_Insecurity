from mesa import Agent


class Stores(Agent):
    """Store agent holding location, classification and food price statistics.

    Prices for each food category are stored as (avg, sd) pairs and are used
    by households to draw randomized prices when estimating store affinity.
    """

    def __init__(self, model, store_id, store_id_data, place_id, name, address,
                 store_race, store_type, lat, lon, stores_prices):
        super().__init__(store_id, model)

        self.store_id = store_id
        self.store_id_data = store_id_data
        self.place_id = place_id
        self.name = name
        self.address = address
        self.store_race = store_race
        self.store_type = store_type
        self.monthly_sales = 0
        self.agg_sales = 0
        self.lat = lat
        self.lon = lon

        self.dairy_avg = self.set_prices("Dairy", "avg", stores_prices)
        self.dairy_sd = self.set_prices("Dairy", "sd", stores_prices)
        self.fruit_avg = self.set_prices("Fruit", "avg", stores_prices)
        self.fruit_sd = self.set_prices("Fruit", "sd", stores_prices)
        self.grains_avg = self.set_prices("Grains", "avg", stores_prices)
        self.grains_sd = self.set_prices("Grains", "sd", stores_prices)
        self.protein_avg = self.set_prices("Proteins", "avg", stores_prices)
        self.protein_sd = self.set_prices("Proteins", "sd", stores_prices)
        self.prep_avg = self.set_prices("Prep meals", "avg", stores_prices)
        self.prep_sd = self.set_prices("Prep meals", "sd", stores_prices)
        self.other_avg = self.set_prices("Others", "avg", stores_prices)
        self.other_sd = self.set_prices("Others", "sd", stores_prices)
        self.veg_avg = self.set_prices("Vegetables", "avg", stores_prices)
        self.veg_sd = self.set_prices("Vegetables", "sd", stores_prices)

    def set_prices(self, food_type, metric, df):
        column = "avgs" if metric == "avg" else "sds"
        return df.loc[(df["types"] == food_type) & (df["ID"] == self.store_id_data), column].values[0]

    def update_monthly_sales(self, new_sales):
        self.monthly_sales = new_sales

    def close_store(self):
        self.model.schedule_stores.remove(self)

    def step(self):
        self.agg_sales = self.agg_sales + self.monthly_sales

    # Getters --------------------------------------------------------------
    def get_dairy_avg(self): return self.dairy_avg
    def get_dairy_sd(self): return self.dairy_sd
    def get_fruit_avg(self): return self.fruit_avg
    def get_fruit_sd(self): return self.fruit_sd
    def get_grains_avg(self): return self.grains_avg
    def get_grains_sd(self): return self.grains_sd
    def get_protein_avg(self): return self.protein_avg
    def get_protein_sd(self): return self.protein_sd
    def get_prep_avg(self): return self.prep_avg
    def get_prep_sd(self): return self.prep_sd
    def get_other_avg(self): return self.other_avg
    def get_other_sd(self): return self.other_sd
    def get_veg_avg(self): return self.veg_avg
    def get_veg_sd(self): return self.veg_sd
    def get_lat(self): return self.lat
    def get_lon(self): return self.lon
    def get_type(self): return self.store_type
    def get_race(self): return self.store_race
    def get_id(self): return self.store_id
    def get_id_data(self): return self.store_id_data
    def get_agg_sales(self): return self.agg_sales
    def modify_agg_sales(self, value): self.agg_sales = value
