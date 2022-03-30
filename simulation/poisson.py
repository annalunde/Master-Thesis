from datetime import datetime
import numpy.random as rnd
from config.simulation_config import *


class Poisson:
    def __init__(self):
        pass

    def disruption_times(self, arrival_rates, sim_clock, disruption_type):

        time_intervals = [10, 11, 12, 13, 14, 15, 16, 17, 18]

        timestamps = []

        for time_step in range(start_poisson, end_poisson):
            time_step_arrivals = rnd.poisson(arrival_rates[time_step])
            for t in range(0, time_step_arrivals):
                timestamps.append(rnd.uniform(
                    time_intervals[time_step], time_intervals[time_step]+1))

        timestamps.sort()

        # convert timestamps to datetime
        disruption_timestamps = []
        for timestamp in timestamps:
            hours = int(timestamp)
            minutes = int((timestamp-hours)*60)
            seconds = int(((timestamp-hours)*60-minutes)*60)
            date_time = datetime(sim_clock.year, sim_clock.month, sim_clock.day,
                                 hours, minutes, seconds)
            if date_time >= sim_clock:
                disruption_timestamps.append((disruption_type, date_time))

        return disruption_timestamps
