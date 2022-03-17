from datetime import datetime

import numpy as np
from simulation.simulation_config import *

class Poisson:
    def __init__(self):
        pass

    def disruption_times(self, arrival_rates, sim_clock, disruption_type):

        time_intervals = [10, 11, 12, 13, 14, 15, 16, 17, 18]

        timestamps = []

        for time_step in range(start_poisson, end_poisson):
            time_step_arrivals = np.random.poisson(arrival_rates[time_step])
            for t in range(0, time_step_arrivals):
                timestamps.append(np.random.uniform(time_intervals[time_step], time_intervals[time_step]+1))

        timestamps.sort()

        # convert timestamps to datetime
        disruption_timestamps = []
        for timestamp in timestamps:
            hours = int(timestamp)
            minutes = int((timestamp-hours)*60)
            seconds = int(((timestamp-hours)*60-minutes)*60)
            date_time = datetime(sim_clock.year, sim_clock.month, sim_clock.day,
                                hours, minutes, seconds)
            disruption_timestamps.append((disruption_type, date_time))

        # We plot the resulting inhomogeneous Poisson process with its
        # intensity and its ticks over time
        #plot_point_process(in_poi)

        return disruption_timestamps


def main():
    poisson = None

    try:
        sim_clock = datetime.strptime("2021-05-10 10:00:00", "%Y-%m-%d %H:%M:%S")
        poisson = Poisson()
        disruption_time = poisson.disruption_times(arrival_rate_cancel, sim_clock, 'cancel')
        print(disruption_time)
        print(len(disruption_time))

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()