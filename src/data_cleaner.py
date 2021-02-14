import pandas as pd
from datetime import datetime
from tqdm import tqdm

class DataCleaner:
    def __init__(self, path_to_data, path_to_store="../data/store.csv" ):
        self.data = pd.DataFrame()

        self.integer_features = ["DayOfWeek",
                            "Promo",
                            "SchoolHoliday",
                            "CompetitionDistance",
                            "CompetitionOpenSinceMonth",
                            "CompetitionOpenSinceYear"]
        self.category_features = ["Store", "StateHoliday", "StoreType", "Assortment", "PromoInterval"]

        self.read_csv(path_to_data, path_to_store)


    def read_csv(self, path_to_data, path_to_store="data/store.csv"):
        train = pd.read_csv( path_to_data, low_memory=False )
        store = pd.read_csv( path_to_store, low_memory=False )
        self.data = self.merge_store_data(train,store)
        self.drop_customers()
        self.set_types()

    def merge_store_data(self, data, store):
        return pd.merge(data, store, on="Store")

    def drop_customers(self):
        self.data.drop("Customers", axis=1, inplace=True)

    def set_types(self):
        #Create new category for PromoInterval
        self.data.PromoInterval.fillna(value="No Interval", inplace=True)
        #Set types:
        for f in self.integer_features:
            self.data[f] = self.data[f].astype("int64", errors="ignore")
        for f in self.category_features:
            self.data[f] = self.data[f].astype("category", errors="ignore")

    def drop_null_sales(self):
        null_map = self.data.Sales.isnull()
        null_index = self.data[null_map].index
        self.data.drop(index=null_index, inplace=True)

    def drop_zero_sales(self):
        zero_map = (self.data.Sales == 0)
        zero_index = self.data[zero_map].index
        self.data.drop(index=zero_index, inplace=True)
        #Store has to be open to sell anyway...
        self.data.drop("Open", axis=1, inplace=True)

    def convert_date(self):
        self.data.loc[:,"Date"] = pd.to_datetime(self.data.Date)
        #Expand the date columns to numerical values:
        self.data.loc[:,"Year"] = self.data.Date.apply(lambda x: x.year)
        self.data.loc[:,"Month"] = self.data.Date.apply(lambda x: x.month)
        self.data.loc[:,"Day"] = self.data.Date.apply(lambda x: x.day)
        #Set date as index:
        self.data.index = self.data.Date
        #Drop redundant Date column:
        self.data.drop("Date", axis=1, inplace=True)

    def aggregate_promotion_features(self):
        promo_feature = []
        print("Generate new feature: Elapsed days since continuous promotion...")
        for index, row in tqdm(self.data.iterrows()):
            promo_feature.append( self.timedelta_promo(index,
                                                  row.Promo2SinceWeek,
                                                  row.Promo2SinceYear,
                                                  promo_bool=row.Promo2 ) )
        self.data["aggregated_promo2"] = promo_feature
        self.data.drop(["Promo2", "Promo2SinceYear", "Promo2SinceWeek"], axis=1, inplace=True)


    @staticmethod
    def timedelta_promo(current, promo_week, promo_year, promo_bool=1 ):
        if promo_bool == 1:
            date = datetime.strptime(f"{int(promo_year):04d}-{int(promo_week):02d}-1", '%Y-%W-%w')
            return (current - date).days
        else:
            return 0

    def new_weekend_feature(self):
        print("Add is_weekend flag feature...")
        self.data["is_weekend"] = self.data.DayOfWeek.apply(self.is_weekend)

    @staticmethod
    def is_weekend(x):
        if x in [5,6,7,1]:
            return 1
        else:
            return 0

    def aggregated_competition_features(self):
        days_since = []
        print("Generate new feature: Elapsed days since competition...")
        for index, row in tqdm(self.data.iterrows()):
            if row.CompetitionOpenSinceMonth == row.CompetitionOpenSinceMonth:
                delta = index - datetime.strptime(f"{int(row.CompetitionOpenSinceYear):04d}-{int(row.CompetitionOpenSinceMonth):02d}", '%Y-%m')
                days_since.append(delta.days)
            else:
                days_since.append(-1)
        self.data["days_since_competition"] = days_since

    def manual_high_sale_day_feature(self):
        print("Generate high sale day of year flag feature...")
        self.data["HighSaleDay"] = self.data.index
        self.data["HighSaleDay"] = self.data["HighSaleDay"].apply(lambda x: x.timetuple().tm_yday)
        self.data["HighSaleDay"] = self.data["HighSaleDay"].apply(self.isHighSale)

    @staticmethod
    def isHighSale(DayOfYear):
        # high_sale = [6, 20, 34, 48, 63, 76, 90, 104, 121,
        #                     153, 167, 181, 196, 209, 224, 238,
        #                     252, 266, 280, 294, 307, 314, 321,
        #                     328, 335, 342, 350, 357, 360]
        high_sale = [121, 181, 321, 328, 335, 350, 357]
        if DayOfYear in high_sale:
            return 2
        elif DayOfYear + 1 in high_sale:
            return 1
        elif DayOfYear - 1 in high_sale:
            return 1
        else:
            return 0

    def handle_nulls(self):
        self.data.drop("CompetitionOpenSinceMonth", axis=1, inplace=True)
        self.data.drop("CompetitionOpenSinceYear", axis=1, inplace=True)

        self.fill_with_mean("CompetitionDistance")
        self.fill_with_mode("SchoolHoliday")
        self.fill_with_mode("StateHoliday")
        self.fill_with_mode("Promo")
        self.fill_with_mode("DayOfWeek")


    def fill_with_mode(self, column):
        self.data[column].fillna( value=self.data[column].mode()[0], inplace=True )

    def fill_with_mean(self, column):
        self.data[column].fillna( value=self.data[column].mean(), inplace=True)

    def get_clean_data(self):
        self.drop_zero_sales()
        self.drop_null_sales()
        self.convert_date()
        self.aggregate_promotion_features()
        self.new_weekend_feature()
        self.aggregated_competition_features()
        self.manual_high_sale_day_feature()
        self.handle_nulls()
        return self.data


if __name__ == "__main__":
    dc = DataCleaner("data")
    data = dc.get_clean_data()
    print(data.info())