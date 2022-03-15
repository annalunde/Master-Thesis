from datetime import datetime

import numpy as np
from tick.base import TimeFunction
from tick.plot import plot_point_process
from tick.hawkes import SimuInhomogeneousPoisson
from simulation.simulation_config import *

class Poisson:
    def __init__(self):
        pass

    def disruption_times(self, arrival_rates, sim_clock, disruption_type):

        run_time = 18

        T = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=float)

        tf = TimeFunction((T, arrival_rates), inter_mode=TimeFunction.InterConstRight)

        # We define a 1 dimensional inhomogeneous Poisson process with the
        # intensity function seen above
        in_poi = SimuInhomogeneousPoisson([tf], end_time=run_time, verbose=False)

        # We activate intensity tracking and launch simulation
        in_poi.track_intensity(0.1)
        in_poi.simulate()

        # convert timestamps to datetime
        disruption_timestamps = []
        for timestamps in in_poi.timestamps:
            for timestamp in timestamps:
                hours = int(timestamp)
                minutes = int((timestamp-hours)*60)
                seconds = int(((timestamp-hours)*60-minutes)*60)
                date_time = datetime(sim_clock.year, sim_clock.month, sim_clock.day,
                                    hours, minutes, seconds)
                disruption_timestamps.append((disruption_type, date_time))

        # We plot the resulting inhomogeneous Poisson process with its
        # intensity and its ticks over time
        plot_point_process(in_poi)

        return disruption_timestamps


def main():
    poisson = None

    try:
        sim_clock = datetime.strptime("2021-05-10 10:00:00", "%Y-%m-%d %H:%M:%S")
        poisson = Poisson()
        disruption_time = poisson.disruption_times(arrival_rate_cancel, sim_clock, 'cancel')
        print(disruption_time)

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()