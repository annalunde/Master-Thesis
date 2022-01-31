import pandas as pd
import os
from decouple import config
from datetime import datetime
from datetime import timedelta
import numpy as np


class ModelPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def process_RAT_for_initial_model(self, date):
        df = pd.read_csv(self.data_path)
        df.drop("Unnamed: 0", inplace=True, axis=1)
        df.drop("Rider ID", inplace=True, axis=1)
        df.drop(
            "Reason For Travel", inplace=True, axis=1
        )  # here 26168 was null out of 26891
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

        # Filter out the requests that arrived before 8 o'clock of the specific date and return these
        time = datetime(date[0], date[1], date[2], 8)

        df_filtered_before_8 = df_filtered[
            (df_filtered["Request Creation Time"] <= time)
        ]

        return df_filtered_before_8

    def process_RAT_events_for_reoptimization_model(self, date):
        df = pd.read_csv(self.data_path)
        df.drop("Unnamed: 0", inplace=True, axis=1)
        df.drop("Rider ID", inplace=True, axis=1)
        df.drop(
            "Reason For Travel", inplace=True, axis=1
        )  # here 26168 was null out of 26891
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

        # Filter out the requests that arrived after 8 o'clock of the specific date and return these
        time = datetime(date[0], date[1], date[2], 8)

        df_filtered_after_8 = df_filtered[
            (df_filtered["Request Creation Time"] >= time)
        ]

        return df_filtered_after_8

    def add_time_windows(self, df, filename):
        df["T_S_L_P"] = (df["Requested Pickup Time"] - timedelta(minutes=5)).fillna(
            df["Requested Dropoff Time"] - timedelta(hours=2)
        )
        df["T_S_L_D"] = (df["Requested Dropoff Time"] - timedelta(minutes=5)).fillna(
            df["Requested Pickup Time"] - timedelta(minutes=15)
        )
        df["T_S_U_P"] = (df["Requested Pickup Time"] + timedelta(minutes=5)).fillna(
            df["Requested Dropoff Time"] + timedelta(minutes=15)
        )
        df["T_S_U_D"] = (df["Requested Dropoff Time"] + timedelta(minutes=5)).fillna(
            df["Requested Pickup Time"] + timedelta(hours=2)
        )
        df["T_H_L_P"] = (df["Requested Pickup Time"] - timedelta(minutes=15)).fillna(
            df["Requested Dropoff Time"] - timedelta(hours=2)
        )
        df["T_H_L_D"] = (df["Requested Dropoff Time"] - timedelta(minutes=15)).fillna(
            df["Requested Pickup Time"] - timedelta(minutes=15)
        )
        df["T_H_U_P"] = (df["Requested Pickup Time"] + timedelta(minutes=15)).fillna(
            df["Requested Dropoff Time"] - timedelta(minutes=15)
        )
        df["T_H_U_D"] = (df["Requested Dropoff Time"] + timedelta(minutes=15)).fillna(
            df["Requested Pickup Time"] + timedelta(hours=2)
        )

        df.to_csv(filename)
        return df


def main():
    preprocessor = None

    try:
        preprocessor = ModelPreprocessor(data_path=config("data_path_RAT"))

        initial_model_data = preprocessor.process_RAT_for_initial_model(
            date=[2021, 5, 10]
        )
        preprocessor.add_time_windows(
            initial_model_data, filename="test_data_initial_model.csv"
        )

        reopt_model_data = preprocessor.process_RAT_events_for_reoptimization_model(
            date=[2021, 5, 10]
        )
        preprocessor.add_time_windows(
            reopt_model_data, filename="test_data_events.csv"
        )

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
