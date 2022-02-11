from poisson import *
from simulation_config import *

class Simulator:
    def __init__(self, sim_clock, current_route_plan):
        self.sim_clock = sim_clock
        self.current_route_plan = current_route_plan

    def get_disruption(self):
        # get disruption times for each disruption type
        request = Poisson(arrival_rate_request, self.sim_clock).disruption_time()
        initial_delay = Poisson(arrival_rate_delay, self.sim_clock).disruption_time()
        cancel = Poisson(arrival_rate_cancel, self.sim_clock).disruption_time()
        initial_no_show = Poisson(arrival_rate_no_show, self.sim_clock).disruption_time()

        # calculate actual delay and no show
        # + see if there are remaining nodes that can be delayed, cancelled or no showed
        #delay_rid, actual_delay = self.delay(initial_delay)
        #cancel_rid = self.cancel(cancel)
        no_show_rid, actual_no_show = self.no_show(initial_no_show)

        # identify earliest disruption time and corresponding disruption type
        disruption_types = ['request']
        disruption_times = [request]
        '''
        if delay_rid > 0:
            disruption_types.append('delay')
            disruption_times.append(actual_delay)
        if cancel_rid > 0:
            disruption_types.append('cancel')
            disruption_times.append(cancel)
        '''
        if no_show_rid > 0:
            disruption_types.append('no show')
            disruption_times.append(actual_no_show)
        disruption_index = disruption_times.index(min(disruption_times))
        disruption_type = disruption_types[disruption_index]
        disruption_time = disruption_times[disruption_index]

        # call on function corresponding to disruption type to get disruption type, time/updated sim_clock, data
        # VIKTIG Å HUSKE AT UPDATED SIM CLOCK ER DET SAMME SOM NY DISRUPTION TIME
        if disruption_type=='request':
            #return disruption_type, disruption_time, self.new_request()
            return True
        elif disruption_type=='delay':
            #return disruption_type, disruption_time, delay_rid
            return True
        elif disruption_type=='cancel':
            #return disruption_type, disruption_time, cancel_rid
            return True
        else:
            return disruption_type, disruption_time, no_show_rid

    #def new_request(self):
        # return new request data

    #def delay(self, initial_delay):
        # return rid, actual_disruption_time

    #def cancel(self):
        # return rid

    def no_show(self, initial_no_show):
        pickup_rids = []
        planned_pickup_times = []

        # potential no shows - pickup nodes with planned pickup after initial_no_show
        for row in self.current_route_plan:
            for col in range(1, len(row)):
                temp_rid = row[col][0]
                temp_planned_time = row[col][1]
                if not temp_rid % int(temp_rid) and temp_planned_time >= initial_no_show:
                    pickup_rids.append(temp_rid)
                    planned_pickup_times.append(temp_planned_time)

        # check whether there are any no shows, if not, what to do?
        if len(pickup_rids) > 0:
            actual_disruption_time = min(planned_pickup_times)
            rid = pickup_rids[planned_pickup_times.index(actual_disruption_time)]
            return rid, actual_disruption_time
        else:
            return -1, initial_no_show


def main():
    simulator = None

    try:
        route_plan = [
            [1, (1, datetime.strptime("2021-05-10 13:02:00", "%Y-%m-%d %H:%M:%S")), (1.5, datetime.strptime("2021-05-10 13:10:00", "%Y-%m-%d %H:%M:%S"))],
            [2, (2, datetime.strptime("2021-05-10 14:15:00", "%Y-%m-%d %H:%M:%S")), (2.5, datetime.strptime("2021-05-10 14:12:00", "%Y-%m-%d %H:%M:%S"))]
                      ]
        current_route_plan = np.array(route_plan)
        sim_clock = datetime.strptime("2021-05-10 13:00:00", "%Y-%m-%d %H:%M:%S")
        simulator = Simulator(sim_clock, current_route_plan)
        disruption_type, disruption_time, disruption_data = simulator.get_disruption()
        print(disruption_type)
        print(disruption_time)
        print(disruption_data)
        sim_clock = disruption_time

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()