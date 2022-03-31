from datetime import datetime
from numpy.random import poisson, uniform
from config.simulation_config import *


class Poisson:
    def __init__(self):
        pass

    def disruption_times(self, arrival_rates, sim_clock, disruption_type):

        timestamps = [uniform(time_intervals[time_step], time_intervals[time_step]+1)
                      for time_step in range(start_poisson, end_poisson)
                      for t in range(0, poisson(arrival_rates[time_step]))]


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

