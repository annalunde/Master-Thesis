from decouple import config
import pandas as pd

class NewRequests:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_and_drop_random_request(self):
        df = pd.read_csv(self.data_path, index_col=0)
        df = df[df['Request Status'].isin(["Completed"])]
        df['Request Creation Time'] = pd.to_datetime(df['Request Creation Time'],
                                                             format="%Y-%m-%d %H:%M:%S")
        df['Requested Pickup Time'] = pd.to_datetime(df['Requested Pickup Time'],
                                                             format="%Y-%m-%d %H:%M:%S")
        df['Requested Dropoff Time'] = pd.to_datetime(df['Requested Dropoff Time'],
                                                              format="%Y-%m-%d %H:%M:%S")

        # request arrives at same day as it is requested to be served
        df["Requested Pickup/Dropoff Time"] = (df["Requested Pickup Time"]).fillna(
            df["Requested Dropoff Time"])
        df['Date Creation'] = df['Request Creation Time'].dt.date
        df['Time Creation'] = df['Request Creation Time'].dt.hour
        df['Date Pickup/Dropoff'] = df['Requested Pickup/Dropoff Time'].dt.date
        df_same_day = df[df['Date Creation'] == df['Date Pickup/Dropoff']]

        # requests arrives after 10am
        df_same_day_after_10 = df_same_day[
            (df_same_day['Time Creation'] >= 10)
        ]

        # get random request
        random_request = df_same_day_after_10.sample()
        random_request.drop(columns=['Request Creation Time',
                                     'Requested Pickup Time',
                                     'Actual Pickup Time',
                                     'Requested Dropoff Time',
                                     'Actual Dropoff Time',
                                     'Original Planned Pickup Time',
                                     'Requested Pickup/Dropoff Time',
                                     'Date Creation',
                                     'Time Creation',
                                     'Date Pickup/Dropoff',
                                     'Request ID',
                                     'Request Status',
                                     'Rider ID',
                                     'Ride ID',
                                     'Cancellation Time',
                                     'No Show Time',
                                     'Origin Zone',
                                     'Destination Zone',
                                     'Reason For Travel'], inplace=True)

        # drop the request
        df_same_day_after_10_updated = df_same_day_after_10.drop(random_request.index)

        # write updated dataframe to csv
        df_same_day_after_10_updated.to_csv(config("data_simulator_path"))

        return random_request


def main():
    new_request = None

    try:
        new_request = NewRequests(data_path=config("data_simulator_path"))
        #new_request = NewRequests(data_path=config("data_processed_path"))
        random_request = new_request.get_and_drop_random_request()
        print(random_request)

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()