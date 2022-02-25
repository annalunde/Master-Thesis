from datetime import timedelta

import pandas as pd
import numpy as np
import math
from decouple import config


class Analyser:
    def __init__(self, data_path, data_path_returns):
        self.data_path = data_path
        self.data_path_returns = data_path_returns

    def register_returns(self):
        # populate the data with returns
        df = pd.read_csv(self.data_path)
        df["Return"] = 0
        return_index = 1

        # groupby by column day
        days = pd.to_datetime(df["Requested Pickup Time"]).dt.date.dropna()
        days = days.drop_duplicates()
        for day in days:
            df_day = df[pd.to_datetime(df["Requested Pickup Time"]).dt.date == day]
            for idx1, r1 in df_day.iterrows():
                for idx2, r2 in df_day.iterrows():
                    if (
                        r2["Rider ID"] == r1["Rider ID"]
                        and r1["Origin Lat"] == r2["Destination Lat"]
                        and r1["Origin Lng"] == r2["Destination Lng"]
                        and r1["Destination Lat"] == r2["Origin Lat"]
                        and r1["Destination Lng"] == r2["Origin Lng"]
                        and r1["Request Status"] == "Completed"
                        and r2["Request Status"] == "Completed"
                    ):
                        df.loc[idx1, "Return"] = return_index
                        df.loc[idx2, "Return"] = return_index
                        return_index += 1
        df.to_csv(config("data_processed_path_return"))

    def return_probability(self):
        df = pd.read_csv(self.data_path_returns)

        print(df.head())

        # probability of return out of total requests
        df["Return T/F"] = np.where(df["Return"] > 0, True, False)
        number_of_trips_with_return = df[df["Return T/F"] == True].shape[0] / 2
        number_of_trips = df["Return T/F"].shape[0] - number_of_trips_with_return
        print(
            "Percentage of trips with return: ",
            number_of_trips_with_return * 100 / number_of_trips,
        )

        # probability of return where the request is ordered the same day as requested pickup/dropoff
        df['Request Creation Time'] = pd.to_datetime(df['Request Creation Time'],
                                                     format="%Y-%m-%d %H:%M:%S")
        df['Requested Pickup Time'] = pd.to_datetime(df['Requested Pickup Time'],
                                                     format="%Y-%m-%d %H:%M:%S")
        df['Requested Dropoff Time'] = pd.to_datetime(df['Requested Dropoff Time'],
                                                      format="%Y-%m-%d %H:%M:%S")

        df["Requested Pickup/Dropoff Time"] = (df["Requested Pickup Time"]).fillna(df["Requested Dropoff Time"])
        df['Date Creation'] = df['Request Creation Time'].dt.date
        df['Time Creation'] = df['Request Creation Time'].dt.hour
        df['Date Pickup/Dropoff'] = df['Requested Pickup/Dropoff Time'].dt.date
        #df_same_day = df[df['Date Creation'] == df['Date Pickup/Dropoff']]
        df["Return T/F"] = np.where(((df["Return"] > 0) & (df['Date Creation'] == df['Date Pickup/Dropoff'])), True, False)
        number_of_trips_with_return = df[df["Return T/F"] == True].shape[0] / 2
        number_of_trips = df["Return T/F"].shape[0] - number_of_trips_with_return
        print(
            "Percentage of trips with return: ",
            number_of_trips_with_return * 100 / number_of_trips,
        )

        df = df[(df["Return T/F"] == True)]
        df_same = df[df['Date Creation'] == df['Date Pickup/Dropoff']]
        df_same.to_csv(config("data_processed_path_return_timediff"))

        count_total = 0
        count_not_ordered_at_same_time = 0
        for idx1, r1 in df_same.iterrows():
            for idx2, r2 in df_same.iterrows():
                if r1["Return"] == r2["Return"] and r1['Request Creation Time'] - r2['Request Creation Time'] > timedelta(minutes=10):
                    count_not_ordered_at_same_time += 1
            count_total += 1
            print(count_total)

        print("Percentage: ", count_not_ordered_at_same_time/(count_total/2))

        # what is the average time between returns?
        df["Requested Pickup Time"] = pd.to_datetime(df["Requested Pickup Time"])
        df = df[(df["Return T/F"] == True)]
        df["return_time_diff"] = (
            df["Requested Pickup Time"]
            .sub(df.groupby("Return")["Requested Pickup Time"].transform("min"))
            .apply(f)
        )
        df.to_csv(config("data_processed_path_return_timediff"))

        mean_diff = df["return_time_diff"].mean()
        print("Average number of seconds between returns: ", mean_diff)


def f(x):
    ts = x.total_seconds()
    if math.isnan(ts):
        return 0
    return ts


def main():
    analyser = None

    try:
        analyser = Analyser(
            data_path=config("data_processed_path"),
            data_path_returns=config("data_processed_path_return"),
        )
        # print("Creating dataset with returns")
        # analyser.register_returns()
        print("Analysing return probability distributions")
        analyser.return_probability()

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
