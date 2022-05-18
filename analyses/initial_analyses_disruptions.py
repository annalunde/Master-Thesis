from datetime import timedelta
import pandas as pd
import os

import scipy
from scipy.stats import gamma, beta
from decouple import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter


class AnalyserDisruptions:
    def __init__(self, data_path):
        self.data_path = data_path

    # NEW REQUEST
    # ha ankomstrate per tidsintervall (i minutter)
    # gjelder for dagens requests - skal fraktes samme dag som requested kommer inn
    # distribution of how long in advance the new request is revealed

    def new_request(self):
        df_request = pd.read_csv(self.data_path)
        df_request['Request Creation Time'] = pd.to_datetime(
            df_request['Request Creation Time'], format="%Y-%m-%d %H:%M:%S")
        df_request['Requested Pickup Time'] = pd.to_datetime(
            df_request['Requested Pickup Time'], format="%Y-%m-%d %H:%M:%S")
        df_request['Requested Dropoff Time'] = pd.to_datetime(
            df_request['Requested Dropoff Time'], format="%Y-%m-%d %H:%M:%S")

        # request arrives at same day as it is requested to be served
        df_request["Requested Pickup/Dropoff Time"] = (
            df_request["Requested Pickup Time"]).fillna(df_request["Requested Dropoff Time"])
        df_request['Date Creation'] = df_request['Request Creation Time'].dt.date
        df_request['Time Creation'] = df_request['Request Creation Time'].dt.hour
        df_request['Date Pickup/Dropoff'] = df_request['Requested Pickup/Dropoff Time'].dt.date
        df_request = df_request[df_request['Date Creation']
                                == df_request['Date Pickup/Dropoff']]
        # requests arrives after 10am
        df_request = df_request[
            (df_request['Time Creation'] >= 10)
        ]

        # calculating the time interval between new requests
        df_request = df_request.sort_values(by=['Request Creation Time'])
        df_request['Diff'] = df_request['Request Creation Time'].diff()

        # number of requested pickups vs requested dropoffs
        df_pickup = df_request.dropna(subset=['Requested Pickup Time'])
        df_dropoff = df_request.dropna(subset=['Requested Dropoff Time'])
        print("Total number of requests to be served on the same day as request is made: ",
              df_request.shape[0])
        print("Percentage of requests with requested pickup time: ",
              (df_pickup.shape[0]*100)/df_request.shape[0])
        print("Percentage of requests with requested dropoff time: ",
              (df_dropoff.shape[0] * 100) / df_request.shape[0])

        # number of minutes between the request creation and the requested pickup/dropoff time
        df_pickup['Diff Creation and Requested Pickup'] = df_pickup['Requested Pickup Time'] - \
            df_pickup['Request Creation Time']
        df_dropoff['Diff Creation and Requested Dropoff'] = df_dropoff['Requested Dropoff Time'] - \
            df_dropoff['Request Creation Time']

        # ARRIVAL RATES
        # sort on date and time, and count how many requests for each date-time pair
        date_time_dict = dict()
        for index, row in df_pickup.iterrows():
            if row['Diff Creation and Requested Pickup'].total_seconds()/60 <= 480:
                if (row['Date Creation'], row['Time Creation']) in date_time_dict:
                    date_time_dict[(row['Date Creation'],
                                    row['Time Creation'])] += 1
                else:
                    date_time_dict[(row['Date Creation'],
                                    row['Time Creation'])] = 1

        for index, row in df_dropoff.iterrows():
            if row['Diff Creation and Requested Dropoff'].total_seconds()/60 <= 480:
                if (row['Date Creation'], row['Time Creation']) in date_time_dict:
                    date_time_dict[(row['Date Creation'],
                                    row['Time Creation'])] += 1
                else:
                    date_time_dict[(row['Date Creation'],
                                    row['Time Creation'])] = 1

        # group by time across all dates and create list of how many requests per date
        time_interval_dict = dict()
        for key, value in date_time_dict.items():
            if key[1] in time_interval_dict:
                time_interval_dict[key[1]].append(value)
            else:
                time_interval_dict[key[1]] = [value]

        # add 0 values for those days that did not have incoming requests for certain time intervals
        number_of_days = df_request['Date Creation'].nunique()
        for key, value in time_interval_dict.items():
            while len(value) < number_of_days:
                time_interval_dict[key].append(0)

        # calculate average arrival rate for each time interval
        time_intervals = []
        arrival_rates = []

        print("Hour interval", '\t', "Arrival rate")
        for key, value in time_interval_dict.items():
            if key >= 10 and key < 19:
                mean = np.mean([c for c in value])
                time_intervals.append(key)
                arrival_rates.append(mean)
                print(key, '\t', mean)

        plt.bar(time_intervals, arrival_rates)
        plt.xlabel('Hour of the day')
        plt.ylabel('Average number of requests in time interval')
        plt.show()

        # probability density function time between request creation time and requested pickup time
        print("Requested Pickup Time")
        waiting_times = []
        for index, row in df_pickup.iterrows():
            if row['Diff Creation and Requested Pickup'].total_seconds() / 60 <= 480:
                waiting_times.append(
                    row['Diff Creation and Requested Pickup'].total_seconds()/60)
        sns.displot(data=waiting_times, kind="hist", bins=100)
        plt.xlabel('Number of minutes between creation and requested pickup time')
        plt.show()

        under_60 = [i for i in waiting_times if i < 60]
        print("Mean waiting time pick-up", len(under_60)/len(waiting_times))

        f = Fitter(waiting_times, distributions=[
                   'gamma', 'lognorm', "norm"])
        f.fit()
        print(f.summary())
        print(f.get_best(method='sumsquare_error'))
        plt.show()

        # find best parameters for distribution
        xlin = np.linspace(0, 500, 50)

        fit_shape, fit_loc, fit_scale = gamma.fit(waiting_times)
        print([fit_shape, fit_loc, fit_scale])
        #plt.hist(waiting_times, bins=50, density=True)
        plt.plot(xlin, gamma.pdf(xlin, a=fit_shape,
                 loc=fit_loc, scale=fit_scale), '#A0BCD4')
        plt.xlabel('Minutes between creation and requested pick-up')
        plt.ylabel('Probability density')
        plt.show()

        # probability density function time between request creation time and requested dropoff time
        print("Requested Dropoff Time")
        waiting_times = []
        for index, row in df_dropoff.iterrows():
            if row['Diff Creation and Requested Dropoff'].total_seconds() / 60 <= 480:
                waiting_times.append(
                    row['Diff Creation and Requested Dropoff'].total_seconds() / 60)
        sns.displot(data=waiting_times, kind="hist", bins=100)
        plt.xlabel(
            'Minutes between creation and requested drop-off')
        plt.show()

        f = Fitter(waiting_times, distributions=[
                   'gamma', 'lognorm', "norm"])
        f.fit()
        print(f.summary())
        print(f.get_best(method='sumsquare_error'))
        plt.show()

        # find best parameters for distribution
        xlin = np.linspace(0, 500, 50)

        fit_shape, fit_loc, fit_scale = gamma.fit(waiting_times)
        print([fit_shape, fit_loc, fit_scale])
        #plt.hist(waiting_times, bins=50, density=True)
        plt.plot(xlin, gamma.pdf(xlin, a=fit_shape,
                 loc=fit_loc, scale=fit_scale), '#A0BCD4')
        plt.xlabel('Minutes between creation and requested drop-off')
        plt.ylabel('Probability density')
        plt.show()

    # DELAY
    # hva defineres som en delay? legge til at man kan endre hvor mange minutter som defineres som en delay
    # har kun data på delays mellom original planned pickup time and actual pickup time - ikke får dropoff
    # antar at delay kun er at man er for sein, ikke for tidlig
    # ha ankomstrate per tidsintervall
    # distrubisjon av lengde av hver delay

    def delay(self, minutes):
        df = pd.read_csv(self.data_path)
        df_delay = df[df['Request Status'].isin(["Completed"])]

        # define which rows are considered to be delays
        df_delay['Original Planned Pickup Time'] = pd.to_datetime(
            df_delay['Original Planned Pickup Time'], format="%Y-%m-%d %H:%M:%S")
        df_delay['Actual Pickup Time'] = pd.to_datetime(
            df_delay['Actual Pickup Time'], format="%Y-%m-%d %H:%M:%S")
        df_delay['Delay'] = df_delay['Actual Pickup Time'] - \
            df_delay['Original Planned Pickup Time']
        df_delay = df_delay[df_delay['Delay'] >= timedelta(minutes=minutes)]

        # calculating the time interval between new delays
        df_delay['Date Actual Pickup'] = df_delay['Actual Pickup Time'].dt.date
        df_delay['Time Actual Pickup'] = df_delay['Actual Pickup Time'].dt.hour
        df_delay = df_delay.sort_values(by=['Actual Pickup Time'])
        df_delay['Diff'] = df_delay['Actual Pickup Time'].diff()

        '''
        diff_dict = dict()
        for index, row in df_delay.iterrows():
            if row['Date Actual Pickup'] in diff_dict:
                diff_dict[row['Date Actual Pickup']].append([row['Time Actual Pickup'], row['Diff'].total_seconds() / 60])
            else:
                diff_dict[row['Date Actual Pickup']] = []

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
        print("Average number of minutes between delays:", total_mean)
        plt.bar(hour_intervals, avg_diff)
        plt.xlabel('Hour of the day')
        plt.ylabel('Minutes between delays')
        plt.show()
        '''

        # ARRIVAL RATES
        # sort on date and time, and count how many requests for each date-time pair
        date_time_dict = dict()
        for index, row in df_delay.iterrows():
            if (row['Date Actual Pickup'], row['Time Actual Pickup']) in date_time_dict:
                date_time_dict[(row['Date Actual Pickup'],
                                row['Time Actual Pickup'])] += 1
            else:
                date_time_dict[(row['Date Actual Pickup'],
                                row['Time Actual Pickup'])] = 1

        # group by time across all dates and create list of how many requests per date
        time_interval_dict = dict()
        for key, value in date_time_dict.items():
            if key[1] in time_interval_dict:
                time_interval_dict[key[1]].append(value)
            else:
                time_interval_dict[key[1]] = [value]

        # add 0 values for those days that did not have incoming requests for certain time intervals
        number_of_days = df_delay['Date Actual Pickup'].nunique()
        for key, value in time_interval_dict.items():
            while len(value) < number_of_days:
                time_interval_dict[key].append(0)

        # calculate average arrival rate for each time interval
        time_intervals = []
        arrival_rates = []

        print("Hour interval", '\t', "Arrival rate")
        for key, value in time_interval_dict.items():
            if key >= 10:
                mean = np.mean([c for c in value])
                time_intervals.append(key)
                arrival_rates.append(mean)
                print(key, '\t', mean)

        plt.bar(time_intervals, arrival_rates)
        plt.xlabel('Hour of the day')
        plt.ylabel('Average number of delays in time interval')
        plt.show()

        waiting_times = []
        for index, row in df_delay.iterrows():
            waiting_times.append(row['Delay'].total_seconds() / 60)

        print("max waiting", max(waiting_times))
        sns.displot(data=waiting_times, kind="hist", bins=100)
        plt.xlabel('Delay (minutes)')
        plt.show()

        # find best distribution
        f = Fitter(waiting_times, distributions=[
                   'lognorm', "beta", "norm"])
        f.fit()
        print(f.summary())
        print(f.get_best(method='sumsquare_error'))
        plt.show()

        # find best parameters for distribution
        xlin = np.linspace(0, 140, 50)

        fit_a, fit_b, fit_loc, fit_scale = beta.fit(waiting_times)
        print([fit_a, fit_b, fit_loc, fit_scale])
        #plt.hist(waiting_times, bins=50, density=True)
        plt.plot(xlin, beta.pdf(xlin, a=fit_a,
                 b=fit_b, loc=fit_loc, scale=fit_scale), '#A0BCD4')
        plt.xlabel('Duration of delay in minutes')
        plt.ylabel('Probability density')
        plt.show()

    # CANCEL
    # kanselleringen må skje samme dag for at det skal ha en betydelse for oss?
    # må vi ikke egt også ha de dagene hvor det ikke skjer noe?
    def cancel(self, minutes):
        df = pd.read_csv(self.data_path)
        df_cancel = df[df['Request Status'].isin(["Cancel", "Late Cancel"])]
        df_cancel = df_cancel.dropna(
            subset=['Original Planned Pickup Time', 'Cancellation Time'])
        df_cancel['Cancellation Time'] = pd.to_datetime(
            df_cancel['Cancellation Time'], format="%Y-%m-%d %H:%M:%S")
        df_cancel['Original Planned Pickup Time'] = pd.to_datetime(
            df_cancel['Original Planned Pickup Time'], format="%Y-%m-%d %H:%M:%S")
        df_cancel['Date'] = df_cancel['Cancellation Time'].dt.date
        df_cancel['Time'] = df_cancel['Cancellation Time'].dt.hour
        df_cancel['Date Pickup'] = df_cancel['Original Planned Pickup Time'].dt.date
        df_cancel = df_cancel[(df_cancel['Date'] == df_cancel['Date Pickup']) & (
            df_cancel['Cancellation Time'] <= df_cancel['Original Planned Pickup Time'])]
        df_cancel = df_cancel.sort_values(by=['Cancellation Time'])
        df_cancel['Diff'] = df_cancel['Cancellation Time'].diff()
        df_cancel['Diff Original and Cancel'] = df_cancel['Original Planned Pickup Time'] - \
            df_cancel['Cancellation Time']

        '''
        diff_dict = dict()
        for index, row in df_cancel.iterrows():
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
        print("Average number of minutes between cancellations:",total_mean)
        plt.bar(hour_intervals,avg_diff)
        plt.xlabel('Hour of the day')
        plt.ylabel('Minutes between cancellations')
        plt.show()
        '''

        # ARRIVAL RATES
        # sort on date and time, and count how many requests for each date-time pair
        date_time_dict = dict()
        for index, row in df_cancel.iterrows():
            if row['Diff Original and Cancel'].total_seconds()/60 <= 480:
                if (row['Date'], row['Time']) in date_time_dict:
                    date_time_dict[(row['Date'], row['Time'])] += 1
                else:
                    date_time_dict[(row['Date'], row['Time'])] = 1

        # group by time across all dates and create list of how many requests per date
        time_interval_dict = dict()
        for key, value in date_time_dict.items():
            if key[1] in time_interval_dict:
                time_interval_dict[key[1]].append(value)
            else:
                time_interval_dict[key[1]] = [value]

        # add 0 values for those days that did not have incoming requests for certain time intervals
        number_of_days = df_cancel['Date'].nunique()
        for key, value in time_interval_dict.items():
            while len(value) < number_of_days:
                time_interval_dict[key].append(0)

        # calculate average arrival rate for each time interval
        time_intervals = []
        arrival_rates = []

        print("Hour interval", '\t', "Arrival rate")
        for key, value in time_interval_dict.items():
            if key >= 10:
                mean = np.mean([c for c in value])
                time_intervals.append(key)
                arrival_rates.append(mean)
                print(key, '\t', mean)

        plt.bar(time_intervals, arrival_rates)
        plt.xlabel('Hour of the day')
        plt.ylabel('Average number of cancels in time interval')
        plt.show()

        # probability density function time between planned pickup and cancellation
        waiting_times = []
        for index, row in df_cancel.iterrows():
            if row['Diff Original and Cancel'].total_seconds() / 60 <= 480:
                waiting_times.append(
                    row['Diff Original and Cancel'].total_seconds()/60)

        under = [i for i in waiting_times if i < minutes]
        print("Percentage under", minutes, len(under)/len(waiting_times))

        sns.displot(data=waiting_times, kind="kde")
        sns.displot(data=waiting_times, kind="hist")
        plt.xlabel('Number of minutes before planned pickup')
        plt.show()
        print(min(waiting_times))
        print(np.mean(waiting_times))

        # find best distribution
        f = Fitter(waiting_times, distributions=[
            'lognorm', "beta", "norm"])
        f.fit()
        print(f.summary())
        print(f.get_best(method='sumsquare_error'))
        plt.show()

        # find best parameters for distribution
        xlin = np.linspace(0, 500, 50)

        fit_a, fit_b, fit_loc, fit_scale = beta.fit(waiting_times)
        print([fit_a, fit_b, fit_loc, fit_scale])
        # plt.hist(waiting_times, bins=50, density=True)
        plt.plot(xlin, beta.pdf(xlin, a=fit_a,
                                b=fit_b, loc=fit_loc, scale=fit_scale), '#A0BCD4')
        plt.xlabel('Minutes between cancellation and planned pick-up')
        plt.ylabel('Probability density')
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
        df_no_show = df[df['Request Status'] == "No Show"]
        df_no_show['No Show Time'] = pd.to_datetime(
            df_no_show['No Show Time'], format="%Y-%m-%d %H:%M:%S")
        df_no_show['Original Planned Pickup Time'] = pd.to_datetime(
            df_no_show['Original Planned Pickup Time'], format="%Y-%m-%d %H:%M:%S")
        df_no_show['Date'] = df_no_show['No Show Time'].dt.date
        df_no_show['Time'] = df_no_show['No Show Time'].dt.hour
        df_no_show = df_no_show[df_no_show['No Show Time']
                                >= df_no_show['Original Planned Pickup Time']]
        df_no_show = df_no_show.sort_values(by=['No Show Time'])
        df_no_show['Diff'] = df_no_show['No Show Time'].diff()
        df_no_show['Wait'] = df_no_show['No Show Time'] - \
            df_no_show['Original Planned Pickup Time']

        '''
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
        '''

        # ARRIVAL RATES
        # sort on date and time, and count how many requests for each date-time pair
        date_time_dict = dict()
        for index, row in df_no_show.iterrows():
            if (row['Date'], row['Time']) in date_time_dict:
                date_time_dict[(row['Date'], row['Time'])] += 1
            else:
                date_time_dict[(row['Date'], row['Time'])] = 1

        # group by time across all dates and create list of how many requests per date
        time_interval_dict = dict()
        for key, value in date_time_dict.items():
            if key[1] in time_interval_dict:
                time_interval_dict[key[1]].append(value)
            else:
                time_interval_dict[key[1]] = [value]

        # add 0 values for those days that did not have incoming requests for certain time intervals
        number_of_days = df_no_show['Date'].nunique()
        for key, value in time_interval_dict.items():
            while len(value) < number_of_days:
                time_interval_dict[key].append(0)

        # calculate average arrival rate for each time interval
        time_intervals = []
        arrival_rates = []

        print("Hour interval", '\t', "Arrival rate")
        for key, value in time_interval_dict.items():
            if key >= 10:
                mean = np.mean([c for c in value])
                time_intervals.append(key)
                arrival_rates.append(mean)
                print(key, '\t', mean)

        plt.bar(time_intervals, arrival_rates)
        plt.xlabel('Hour of the day')
        plt.ylabel('Average number of no shows in time interval')
        plt.show()

        # probability density function waiting time before no show is known
        waiting_times = []
        for index, row in df_no_show.iterrows():
            waiting_times.append(row['Wait'].total_seconds()/60)

        sns.displot(data=waiting_times, kind="kde")
        plt.xlabel('Waiting time in minutes')
        plt.show()


def main():
    analyser = None

    try:
        analyser = AnalyserDisruptions(
            data_path=config("data_processed_path"))
        # analyser.event_per_total()
        # analyser.no_show()
        analyser.cancel(60)
        # analyser.new_request()
        # analyser.delay(5)

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
