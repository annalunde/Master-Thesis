from poisson import *
from simulation_config import *

class Simulator:
    def __init__(self, sim_clock):
        self.sim_clock = sim_clock

    def get_disruption_time(self, sim_clock):
        # get disruption times for each disruption type
        request = Poisson(arrival_rate_request, sim_clock).disruption_time()
        delay = Poisson(arrival_rate_delay, sim_clock).disruption_time()
        cancel = Poisson(arrival_rate_cancel, sim_clock).disruption_time()
        no_show = Poisson(arrival_rate_no_show, sim_clock).disruption_time()

        # calculate actual delay and no show

        # return the earliest disruption time
        return min([request, delay, cancel, no_show])

    def new_request(self):

    def delay(self):

    def cancel(self):

    def no_show(self):

def main():
    simulator = None

    try:
        simulator = Simulator()

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()