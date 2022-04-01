import pandas as pd
from decouple import config
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AnalyserInitial:
    def __init__(self):
        pass

    def total_filter(self):
        df = pd.read_csv(config("data_processed_path"))
        #df = df.loc[(df["Request Status"] == "Completed") | (df["Request Status"] == "Seat Unavilable")]

        df["Requested Pickup Time"] = pd.to_datetime(
            df["Requested Pickup Time"], format="%Y-%m-%d %H:%M:%S"
        )
        df["Requested Dropoff Time"] = pd.to_datetime(
            df["Requested Dropoff Time"], format="%Y-%m-%d %H:%M:%S"
        )
        df["Request Creation Time"] = pd.to_datetime(
            df["Request Creation Time"], format="%Y-%m-%d %H:%M:%S"
        )

        df["Requested Pickup/Dropoff Time"] = (
            df["Requested Pickup Time"]).fillna(df["Requested Dropoff Time"])

        df['Date Pickup/Dropoff'] = df['Requested Pickup/Dropoff Time'].dt.date


        # count how many request for each date and which are created before 10 am
        # [total, intial, reopt]
        initial_days = dict()
        for index, row in df.iterrows():
            time = datetime(row['Date Pickup/Dropoff'].year,
                            row['Date Pickup/Dropoff'].month,
                            row['Date Pickup/Dropoff'].day, 10)
            if row['Request Creation Time'] < time and (row['Request Status'] == "Completed" or row['Request Status'] == "Seat Unavilable"):
                if row['Date Pickup/Dropoff'] in initial_days:
                    initial_days[row['Date Pickup/Dropoff']][0] += 1
                    initial_days[row['Date Pickup/Dropoff']][1] += 1
                else:
                    initial_days[row['Date Pickup/Dropoff']] = [1, 1, 0]
            if row['Request Creation Time'] >= time:
                if row['Date Pickup/Dropoff'] in initial_days:
                    initial_days[row['Date Pickup/Dropoff']][0] += 1
                    initial_days[row['Date Pickup/Dropoff']][2] += 1
                else:
                    initial_days[row['Date Pickup/Dropoff']] = [1, 0, 1]

        total = [value[0] for key, value in initial_days.items() if value[0] > 50]

        sns.displot(data=total, kind="hist", bins=3)
        plt.xlabel('Counts per day')
        plt.show()

        # small (50 - 150)
        print("SMALL")
        small_initial = [value[1] for key, value in initial_days.items() if 50 < value[0] <= 150]
        print("Small initial")
        print("Min", min(small_initial))
        print("Max", max(small_initial))
        print("Mean", np.mean([i for i in small_initial]))
        print()
        small_reopt = [value[2] for key, value in initial_days.items() if 50 < value[0] <= 150]
        print("Small reopt")
        print("Min", min(small_reopt))
        print("Max", max(small_reopt))
        print("Mean", np.mean([i for i in small_reopt]))
        print()
        print("Small dates")
        small_dates = [key for key, value in initial_days.items() if 50 < value[0] <= 150]
        print(small_dates)
        print()

        # medium (150 - 230)
        print("MEDIUM")
        medium_initial = [value[1] for key, value in initial_days.items() if 150 < value[0] <= 230]
        print("Medium initial")
        print("Min", min(medium_initial))
        print("Max", max(medium_initial))
        print("Mean", np.mean([i for i in medium_initial]))
        print()
        medium_reopt = [value[2] for key, value in initial_days.items() if 150 < value[0] <= 230]
        print("Medium reopt")
        print("Min", min(medium_reopt))
        print("Max", max(medium_reopt))
        print("Mean", np.mean([i for i in medium_reopt]))
        print()
        print("Medium dates")
        medium_dates = [key for key, value in initial_days.items() if 150 < value[0] <= 230]
        print(medium_dates)
        print()

        # large (230 - 314)
        print("LARGE")
        large_initial = [value[1] for key, value in initial_days.items() if value[0] > 230]
        print("Large initial")
        print("Min", min(large_initial))
        print("Max", max(large_initial))
        print("Mean", np.mean([i for i in large_initial]))
        print()
        large_reopt = [value[2] for key, value in initial_days.items() if value[0] > 230]
        print("Large reopt")
        print("Min", min(large_reopt))
        print("Max", max(large_reopt))
        print("Mean", np.mean([i for i in large_reopt]))
        print()
        print("Large dates")
        large_dates = [key for key, value in initial_days.items() if value[0] > 230]
        print(large_dates)
        print()

    def initial_filter(self):
        df = pd.read_csv(config("data_processed_path"))
        #df = df.loc[(df["Request Status"] == "Completed") | (df["Request Status"] == "Seat Unavilable")]

        df["Requested Pickup Time"] = pd.to_datetime(
            df["Requested Pickup Time"], format="%Y-%m-%d %H:%M:%S"
        )
        df["Requested Dropoff Time"] = pd.to_datetime(
            df["Requested Dropoff Time"], format="%Y-%m-%d %H:%M:%S"
        )
        df["Request Creation Time"] = pd.to_datetime(
            df["Request Creation Time"], format="%Y-%m-%d %H:%M:%S"
        )

        df["Requested Pickup/Dropoff Time"] = (
            df["Requested Pickup Time"]).fillna(df["Requested Dropoff Time"])

        df['Date Pickup/Dropoff'] = df['Requested Pickup/Dropoff Time'].dt.date


        # count how many request for each date and which are created before 10 am
        # [intial, reopt]
        initial_days = dict()
        for index, row in df.iterrows():
            time = datetime(row['Date Pickup/Dropoff'].year,
                            row['Date Pickup/Dropoff'].month,
                            row['Date Pickup/Dropoff'].day, 10)
            if row['Request Creation Time'] < time and (row['Request Status'] == "Completed" or row['Request Status'] == "Seat Unavilable"):
                if row['Date Pickup/Dropoff'] in initial_days:
                    initial_days[row['Date Pickup/Dropoff']][0] += 1
                else:
                    initial_days[row['Date Pickup/Dropoff']] = [1, 0]
            if row['Request Creation Time'] >= time:
                if row['Date Pickup/Dropoff'] in initial_days:
                    initial_days[row['Date Pickup/Dropoff']][1] += 1
                else:
                    initial_days[row['Date Pickup/Dropoff']] = [0, 1]

        total = [value[0] for key, value in initial_days.items() if value[0] > 0]

        sns.displot(data=total, kind="hist", bins=3)
        plt.xlabel('Counts per day')
        plt.show()

        # small (47 - 110)
        print("SMALL")
        small_initial = [value[0] for key, value in initial_days.items() if 46 < value[0] <= 110]
        print("Small initial")
        print("Min", min(small_initial))
        print("Max", max(small_initial))
        print("Mean", np.mean([i for i in small_initial]))
        print()
        small_reopt = [value[1] for key, value in initial_days.items() if 46 < value[0] <= 110]
        print("Small reopt")
        print("Min", min(small_reopt))
        print("Max", max(small_reopt))
        print("Mean", np.mean([i for i in small_reopt]))
        print()
        print("Small dates")
        small_dates = [(key, value[0]) for key, value in initial_days.items() if 46 < value[0] <= 110]
        print(small_dates)
        print()

        # medium (110 - 170)
        print("MEDIUM")
        medium_initial = [value[0] for key, value in initial_days.items() if 110 < value[0] <= 170]
        print("Medium initial")
        print("Min", min(medium_initial))
        print("Max", max(medium_initial))
        print("Mean", np.mean([i for i in medium_initial]))
        print()
        medium_reopt = [value[1] for key, value in initial_days.items() if 110 < value[0] <= 170]
        print("Medium reopt")
        print("Min", min(medium_reopt))
        print("Max", max(medium_reopt))
        print("Mean", np.mean([i for i in medium_reopt]))
        print()
        print("Medium dates")
        medium_dates = [(key, value[0]) for key, value in initial_days.items() if 110 < value[0] <= 170]
        print(medium_dates)
        print()

        # large (170 - 233)
        print("LARGE")
        large_initial = [value[0] for key, value in initial_days.items() if value[0] > 170]
        print("Large initial")
        print("Min", min(large_initial))
        print("Max", max(large_initial))
        print("Mean", np.mean([i for i in large_initial]))
        print()
        large_reopt = [value[1] for key, value in initial_days.items() if value[0] > 170]
        print("Large reopt")
        print("Min", min(large_reopt))
        print("Max", max(large_reopt))
        print("Mean", np.mean([i for i in large_reopt]))
        print()
        print("Large dates")
        large_dates = [(key, value[0]) for key, value in initial_days.items() if value[0] > 170]
        print(large_dates)
        print()


def main():
    preprocessor = None

    try:
        preprocessor = AnalyserInitial()
        #preprocessor.total_filter()
        preprocessor.initial_filter()

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()