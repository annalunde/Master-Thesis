import pandas as pd
import os
from decouple import config
from datetime import datetime, timedelta
import numpy as np


class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def process_data_TT(self):
        files = [
            os.path.join(self.data_path, filename)
            for filename in os.listdir(self.data_path)
            if filename.startswith("Statistik")
        ]
        print(files)
        data = []
        for file in files:
            data.append(pd.read_csv(file, index_col=None, header=0, sep=";"))
        df = pd.concat(data, axis=0, ignore_index=True)
        chosen_columns = [
            "Id",
            "Hämt-tid",
            "Lämna-tid",
            "Resetyp",
            "Krav",
            "H-lat",
            "H-lon",
            "L-lat",
            "L-lon",
        ]
        df = df[chosen_columns]
        df = df.drop(df[df["Resetyp"] != "FRI"].index)
        df = df.drop("Resetyp", 1)

        df = df.rename(
            columns={
                "Id": "Request ID",
                "Hämt-tid": "Requested Pickup Time",
                "Lämna-tid": "Requested Dropoff Time",
                "H-lat": "Origin Lat",
                "H-lon": "Origin Lng",
                "L-lat": "Destination Lat",
                "L-lon": "Destination Lng",
            }
        )
        df.insert(0, "Request Creation Time", df["Requested Pickup Time"])
        df.insert(1, "Wheelchair", 0)
        df["Wheelchair"] = np.where(df["Krav"].str.contains("WHEL"), 1, 0)
        df = df.drop("Krav", 1)
        df.insert(3, "Rider ID", None)
        df.insert(4, "Number of Passengers", 1)
        df.insert(len(df.columns), "Reason For Travel", None)
        date_time = []
        date = df["Requested Pickup Time"].values.tolist()
        time = df["Requested Dropoff Time"].values.tolist()

        for i in range(len(df.index)):
            date_time.append(date[i][0:10] + " " + time[i])

        df["Requested Dropoff Time"] = date_time
        print(df.columns)
        print(df.head())
        df.to_csv("data_TT.csv")

    def process_data_RAT(self):
        files = [
            os.path.join(self.data_path, filename)
            for filename in os.listdir(self.data_path)
            if filename.startswith("Ride Requests-")
        ]
        data = []
        for f in files:
            data.append(pd.read_csv(f, index_col=None, header=0))
        df = pd.concat(data, axis=0, ignore_index=True)

        # convert yes no to 1 and 0
        df["Wheelchair"] = df["Wheelchair Accessible"].map({"Yes": 1, "No": 0})

        # convert to solely wheelchair and solely passengers
        df["Number of Passengers"] = df["Number of Passengers"] - df["Wheelchair"]

        # convert to correct datetime format
        time_columns = ["Request Creation Time", "Requested Pickup Time", "Actual Pickup Time",
                        "Requested Dropoff Time",
                        "Actual Dropoff Time",
                        "Cancellation Time",
                        "No Show Time", "Original Planned Pickup Time"]
        for col in time_columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime(
                "%Y-%m-%d %H:%M:%S")

        chosen_columns = [
            "Request Creation Time",
            "Wheelchair",
            "Request ID",
            "Request Status",
            "Rider ID",
            "Ride ID",
            "Number of Passengers",
            "Requested Pickup Time",
            "Actual Pickup Time",
            "Requested Dropoff Time",
            "Actual Dropoff Time",
            "Cancellation Time",
            "No Show Time",
            "Origin Zone",
            "Origin Lat",
            "Origin Lng",
            "Destination Zone",
            "Destination Lat",
            "Destination Lng",
            "Reason For Travel",
            "Original Planned Pickup Time",
        ]
        data = df[chosen_columns]
        df['Origin Zone'] = df['Origin Zone'].astype(str)
        df['Destination Zone'] = df['Destination Zone'].astype(str)
        data = data.loc[df['Origin Zone'] != "8. Nes"]
        data = data.loc[df['Origin Zone'] != "Nes"]
        data = data.loc[df['Destination Zone'] != "8. Nes"]
        data = data.loc[df['Destination Zone'] != "Nes"]
        data.to_csv(config("data_processed_path"))

    def generate_one_day_completed(self, date):
        df = pd.read_csv(config("data_processed_path"))
        df = df.loc[(df["Request Status"] == "Completed") | (df["Request Status"] == "Seat Unavilable")]

        df["Requested Pickup Time"] = pd.to_datetime(
            df["Requested Pickup Time"], format="%Y-%m-%d %H:%M:%S"
        )
        df["Requested Dropoff Time"] = pd.to_datetime(
            df["Requested Dropoff Time"], format="%Y-%m-%d %H:%M:%S"
        )
        df["Request Creation Time"] = pd.to_datetime(
            df["Request Creation Time"], format="%Y-%m-%d %H:%M:%S"
        )

        valid_date = datetime(date[0], date[1], date[2])
        next_day = valid_date + timedelta(days=1)

        df_filtered = df[
            (
                (df["Requested Pickup Time"] > valid_date)
                & (df["Requested Pickup Time"] < next_day)
            )
            | (
                (df["Requested Dropoff Time"] > valid_date)
                & (df["Requested Dropoff Time"] < next_day)
            )
        ]

        # Filter out the requests that arrived before 10 o'clock of the specific date and return these
        time = datetime(date[0], date[1], date[2], 10)

        df_filtered_before_10 = df_filtered[
            (df_filtered["Request Creation Time"] < time)
        ]

        df_filtered_before_10.to_csv(config("test_data_construction"))


def main():
    preprocessor = None

    try:
        preprocessor = Preprocessor(
            data_path=config("data_initial_path"))
        print("Preprocessing data RAT: ")
        # preprocessor.process_data_RAT()
        # print("Preprocessing data TT: ")
        # preprocessor.process_data_TT()
        preprocessor.generate_one_day_completed(date=[2021, 5, 10])

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
