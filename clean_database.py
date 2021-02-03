import pandas as pd
import os
from datetime import datetime

class DataCleaner:
    def __init__(self, path ):
        self.data = pd.DataFrame()
        self.read_csv(path)

    def read_csv(self, path):
        train = pd.read_csv( os.path.join(path, "train.csv"), low_memory=False )
        store = pd.read_csv( os.path.join(path, "store.csv"), low_memory=False )
        self.data = pd.merge(train, store, on="Store")
        self.data.drop("Customers", axis=1, inplace=True)
        #Set types:
        self.data.PromoInterval.fillna(value="No Inteval", inplace=True)
        self.set_types()

    def set_types(self):
        integer_features = ["DayOfWeek",
                            "Promo",
                            "SchoolHoliday",
                            "CompetitionDistance",
                            "CompetitionOpenSinceMonth",
                            "CompetitionOpenSinceYear"]
        category_features = ["Store", "StateHoliday", "StoreType", "Assortment", "PromoInterval"]
        for f in integer_features:
            self.data[f] = self.data[f].astype("int64", errors="ignore")
        for f in category_features:
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

    def new_promotion_feature(self):
        promo_feature = []
        for index, row in self.data.iterrows():
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
        self.new_promotion_feature()
        self.handle_nulls()
        return self.data


if __name__ == "__main__":
    dc = DataCleaner("data")
    dc.drop_zero_sales()
    dc.drop_null_sales()
    dc.convert_date()
    dc.new_promotion_feature()
    dc.handle_nulls()
    print(dc.data.info())