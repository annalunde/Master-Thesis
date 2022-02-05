import pandas as pd
import os
from decouple import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AnalyserDisruptions:
    def __init__(self, data_path):
        self.data_path = data_path

    # NEW REQUEST
    # ha ankomstrate per tidsintervall (i minutter)
    # gjelder for dagens requests

    # DELAY
    # hva defineres som en delay? legge til at man kan endre hvor mange minutter som defineres som en delay
    # analyse av antall delays per totalt antall requests
    # average tid i mellom hver delay
    # ha ankomstrate per tidsintervall (i timer)
    # distrubisjon av lengde av hver delay

    # CANCEL
    # kanselleringen må skje samme dag for at det skal ha en betydelse for oss?
    # må vi ikke egt også ha de dagene hvor det ikke skjer noe?

    def cancel(self):
        df = pd.read_csv(self.data_path)
        df_cancel = df[df['Request Status'].isin(["Cancel", "Late Cancel"])]
        df_cancel = df_cancel.dropna(subset=['Original Planned Pickup Time', 'Cancellation Time'])
        df_cancel['Cancellation Time'] = pd.to_datetime(df_cancel['Cancellation Time'], format="%Y-%m-%d %H:%M:%S")
        df_cancel['Original Planned Pickup Time'] = pd.to_datetime(df_cancel['Original Planned Pickup Time'], format="%Y-%m-%d %H:%M:%S")
        df_cancel['Date'] = df_cancel['Cancellation Time'].dt.date
        df_cancel['Time'] = df_cancel['Cancellation Time'].dt.hour
        df_cancel['Date Pickup'] = df_cancel['Original Planned Pickup Time'].dt.date
        df_cancel = df_cancel.sort_values(by=['Cancellation Time'])
        df_cancel['Diff'] = df_cancel['Cancellation Time'].diff()
        df_cancel['Diff Original and Cancel'] = df_cancel['Original Planned Pickup Time'] - df_cancel['Cancellation Time']
        df_cancel['Date Pickup'] = df_cancel['Original Planned Pickup Time'].dt.date

        diff_dict = dict()
        for index, row in df_cancel.iterrows():
            if row['Date'] == row['Date Pickup'] and row['Cancellation Time'] <= row['Original Planned Pickup Time']: # kansellering skjer samme dag
                if row['Date'] in diff_dict:
                    diff_dict[row['Date']].append([row['Time'], row['Diff'].total_seconds()/60])
                else:
                    diff_dict[row['Date']] = []

        time_dict = dict()
        for value in diff_dict.values():
            for v in value:
                if v[0] in time_dict:
                    time_dict[v[0]].append(v[1])
                else:
                    time_dict[v[0]] = [v[1]]

        hour_intervals = []
        avg_diff = []
        total_diff = []

        print("Hour interval", '\t', "Average interval")
        for key, value in time_dict.items():
            mean = np.mean([c for c in value])
            hour_intervals.append(key)
            avg_diff.append(mean)
            total_diff += value
            print(key, '\t', mean)
            print(value)

        total_mean = np.mean([c for c in total_diff])
        print("Average number of minutes between cancellations:",total_mean)
        plt.bar(hour_intervals,avg_diff)
        plt.xlabel('Hour of the day')
        plt.ylabel('Minutes between cancellations')
        plt.show()

        # probability density function waiting time before no show is known
        waiting_times = []
        for index,row in df_cancel.iterrows():
            if row['Date'] == row['Date Pickup']:       # kansellering skjer samme dag
                if row['Diff Original and Cancel'].total_seconds()/60 >= 0:
                    waiting_times.append(row['Diff Original and Cancel'].total_seconds()/60)

        sns.displot(data=waiting_times, kind="kde")
        plt.xlabel('Number of minutes before planned pickup')
        plt.show()


    def event_per_total(self):
        df = pd.read_csv(self.data_path)
        counts_normalized = df['Request Status'].value_counts(normalize=True)
        counts = df['Request Status'].value_counts()
        print(counts_normalized)
        print(counts)

    # NO SHOW

    def no_show(self):
        df = pd.read_csv(self.data_path)
        df_no_show = df[df['Request Status']=="No Show"]
        df_no_show['No Show Time'] = pd.to_datetime(df_no_show['No Show Time'], format="%Y-%m-%d %H:%M:%S")
        df_no_show['Original Planned Pickup Time'] = pd.to_datetime(df_no_show['Original Planned Pickup Time'], format="%Y-%m-%d %H:%M:%S")
        df_no_show['Date'] = df_no_show['No Show Time'].dt.date
        df_no_show['Time'] = df_no_show['No Show Time'].dt.hour
        df_no_show = df_no_show.sort_values(by=['No Show Time'])
        df_no_show['Diff'] = df_no_show['No Show Time'].diff()
        df_no_show['Wait'] = df_no_show['No Show Time'] - df_no_show['Original Planned Pickup Time']

        diff_dict = dict()
        for index, row in df_no_show.iterrows():
            if row['Date'] in diff_dict:
                diff_dict[row['Date']].append([row['Time'], row['Diff'].total_seconds()/60])
            else:
                diff_dict[row['Date']] = []

        time_dict = dict()
        for value in diff_dict.values():
            for v in value:
                if v[0] in time_dict:
                    time_dict[v[0]].append(v[1])
                else:
                    time_dict[v[0]] = [v[1]]

        hour_intervals = []
        avg_diff = []
        total_diff = []

        print("Hour interval", '\t', "Average interval")
        for key, value in time_dict.items():
            mean = np.mean([c for c in value])
            hour_intervals.append(key)
            avg_diff.append(mean)
            total_diff += value
            print(key, '\t', mean)

        total_mean = np.mean([c for c in total_diff])
        print("Average number of minutes between no shows:",total_mean)
        plt.bar(hour_intervals,avg_diff)
        plt.xlabel('Hour of the day')
        plt.ylabel('Minutes between no show')
        plt.show()

        # probability density function waiting time before no show is known
        waiting_times = []
        for index,row in df_no_show.iterrows():
            if row['Wait'].total_seconds()/60 >= 0:
                waiting_times.append(row['Wait'].total_seconds()/60)

        sns.displot(data=waiting_times, kind="kde")
        plt.xlabel('Waiting time in minutes')
        plt.show()

def main():
    analyser = None

    try:
        analyser = AnalyserDisruptions(
            data_path=config("data_processed_path"))
        #analyser.event_per_total()
        #analyser.no_show()
        analyser.cancel()

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
