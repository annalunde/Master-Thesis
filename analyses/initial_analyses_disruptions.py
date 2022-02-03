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
    # ha ankomstrate per tidsintervall (i timer)

    # NO SHOW
    # analyse av antall no shows per totalt antall requests
    # average tid i mellom hver no show
    # ha ankomstrate per tidsintervall (i timer)
    # distribusjon ventetid sjåfør

    # DELAY
    # hva defineres som en delay? legge til at man kan endre hvor mange minutter som defineres som en delay
    # analyse av antall delays per totalt antall requests
    # average tid i mellom hver delay
    # ha ankomstrate per tidsintervall (i timer)
    # distrubisjon av lengde av hver delay

    # CANCEL
    # kanselleringen må skje samme dag som den originale requesten for å bli tatt med - teller kun på systemet da
    # analyse av antall cancels per totalt antall requests
    # average tid i mellom hver cancel
    # ha ankomstrate per tidsintervall (i timer)
    # distrubisjon av lengde av hver cancel

    def event_per_total(self):
        df = pd.read_csv(self.data_path)
        counts_normalized = df['Request Status'].value_counts(normalize=True)
        counts = df['Request Status'].value_counts()
        print(counts_normalized)
        print(counts)

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
        print("Total average interval:",total_mean)
        plt.bar(hour_intervals,avg_diff)
        plt.xlabel('Hour interval')
        plt.ylabel('Minutes')
        plt.show()

        # probability density function waiting time before no show is known
        waiting_times = []
        for index,row in df_no_show.iterrows():
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
        analyser.no_show()
    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
