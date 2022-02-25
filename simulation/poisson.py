from datetime import datetime

import numpy as np
from tick.base import TimeFunction
from tick.plot import plot_point_process
from tick.hawkes import SimuInhomogeneousPoisson

class Poisson:
    def __init__(self, arrival_rates, sim_clock):
        self.arrival_rates = arrival_rates
        self.sim_clock = sim_clock

    def disruption_time(self):

        run_time = 18

        T = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=float)

        tf = TimeFunction((T, self.arrival_rates), inter_mode=TimeFunction.InterConstRight)

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
                date_time = datetime(self.sim_clock.year, self.sim_clock.month, self.sim_clock.day,
                                    hours, minutes, seconds)
                disruption_timestamps.append(date_time)

        # if no disruptions are found - set sim clock to at the end of the day
        disruption_time = datetime(self.sim_clock.year, self.sim_clock.month, self.sim_clock.day,
                                    18, 1, 00)

        # find first disruption time, but later than simulation clock
        for timestamp in disruption_timestamps:
            if timestamp >= self.sim_clock:
                disruption_time = timestamp
                break

        # We plot the resulting inhomogeneous Poisson process with its
        # intensity and its ticks over time
        #plot_point_process(in_poi)

        return disruption_time


def main():
    poisson = None

    try:
        arrival_rates = np.array([14.728110599078342, 8.193548387096774, 6.2949308755760365, 5.557603686635945,
                                  5.428571428571429, 4.331797235023042, 2.824884792626728, 0.2350230414746544,
                                  0.0001], dtype=float)
        sim_clock = datetime.strptime("2021-05-10 10:00:00", "%Y-%m-%d %H:%M:%S")
        poisson = Poisson(arrival_rates, sim_clock)
        disruption_time = poisson.disruption_time()
        print(disruption_time)

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()