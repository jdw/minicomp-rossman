import pandas as pd
import os

class DataCleaner:
    def __init__(self, path ):
        self.data = pd.DataFrame()
        self.read_csv(path)

    def read_csv(self, path):
        train = pd.read_csv( os.path.join(path, "train.csv"), low_memory=False )
        store = pd.read_csv( os.path.join(path, "store.csv"), low_memory=False )
        self.data = pd.merge(train, store, on="Store")
        self.data.drop("Customers", axis=1, inplace=True)

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

    def handle_nulls(self):
        self.fill_with_mode("DayOfWeek")
        self.fill_with_mode("Promo2")
        self.fill_with_mode("StateHoliday")
        self.fill_with_mode("SchoolHoliday")

        #I am not happy here! Filling in 150k values!
        self.fill_with_mode("CompetitionOpenSinceMonth")
        self.fill_with_mode("CompetitionOpenSinceYear")
        self.fill_with_mode("Promo2SinceWeek") #~50% of the whole data == Promo2 is false
        self.fill_with_mode("PromoInterval") #~50% of the whole data == Promo2 is false

        self.fill_with_mean("CompetitionDistance")
        #We should rework this:
        self.fill_with_mean("Promo2SinceYear") #~50% of the whole data == Promo2 is false

        self.data.Promo.fillna( self.data.Promo2, inplace=True )

    def fill_with_mode(self, column):
        self.data[column].fillna( value=self.data[column].mode()[0], inplace=True )

    def fill_with_mean(self, column):
        self.data[column].fillna( value=self.data[column].mean(), inplace=True)

    def get_clean_data(self):
        self.drop_zero_sales()
        self.drop_null_sales()
        self.convert_date()
        self.handle_nulls()
        return self.data


if __name__ == "__main__":
    dc = DataCleaner("data")
    dc.drop_zero_sales()
    dc.drop_null_sales()
    dc.convert_date()
    dc.handle_nulls()
    print(dc.data.info())