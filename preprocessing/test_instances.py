import pandas as pd
import os
from decouple import config
from datetime import datetime, timedelta
import numpy as np


class Instance:
    def __init__(self):
        pass

    def generate_one_day_completed(self, date, file_path, seat_unavailable):
        df = pd.read_csv(config("data_processed_path"))
        if seat_unavailable:
            df = df.loc[(df["Request Status"] == "Completed") | (df["Request Status"] == "Seat Unavailable")]
        else:
            df = df.loc[(df["Request Status"] == "Completed")]

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

        df_filtered_before_10.to_csv(config(file_path))


def main():
    instance = None

    try:
        instance = Instance()

        seat_unavailable = True      # True means include both completed and seat unavailable

        # SMALL TEST INSTANCES
        instance.generate_one_day_completed(date=[2021, 7, 3], file_path="test_instance_small_1_20210703",
                                            seat_unavailable=seat_unavailable)
        instance.generate_one_day_completed(date=[2021, 7, 24], file_path="test_instance_small_2_20210724",
                                            seat_unavailable=seat_unavailable)
        instance.generate_one_day_completed(date=[2021, 9, 18], file_path="test_instance_small_3_20210918",
                                            seat_unavailable=seat_unavailable)

        # MEDIUM TEST INSTANCES
        instance.generate_one_day_completed(date=[2021, 7, 6], file_path="test_instance_medium_1_20210706",
                                            seat_unavailable=seat_unavailable)
        instance.generate_one_day_completed(date=[2021, 8, 30], file_path="test_instance_medium_2_20210830",
                                            seat_unavailable=seat_unavailable)
        instance.generate_one_day_completed(date=[2021, 10, 15], file_path="test_instance_medium_3_20211015",
                                            seat_unavailable=seat_unavailable)

        # LARGE TEST INSTANCES
        instance.generate_one_day_completed(date=[2021, 10, 5], file_path="test_instance_large_1_20211005",
                                            seat_unavailable=seat_unavailable)
        instance.generate_one_day_completed(date=[2021, 10, 14], file_path="test_instance_large_2_20211014",
                                            seat_unavailable=seat_unavailable)
        instance.generate_one_day_completed(date=[2022, 1, 12], file_path="test_instance_large_3_20220112",
                                            seat_unavailable=seat_unavailable)

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
