from datetime import timedelta
from heuristic.construction.heuristic_config import *
from heuristic.construction.construction import ConstructionHeuristic
from simulation.poisson import *
from simulation.new_requests import *
from simulation.simulation_config import *
import random
from scipy.stats import gamma

class Simulator:
    def __init__(self, sim_clock):
        self.sim_clock = sim_clock
        self.poisson = Poisson()
        self.disruptions_stack = self.create_disruption_stack()

    def create_disruption_stack(self):
        # get disruption times for each disruption type
        request = self.poisson.disruption_times(arrival_rate_request, self.sim_clock, 'request')
        delay = self.poisson.disruption_times(arrival_rate_delay, self.sim_clock, 'delay')
        cancel = self.poisson.disruption_times(arrival_rate_cancel, self.sim_clock, 'cancel')
        initial_no_show = self.poisson.disruption_times(arrival_rate_no_show, self.sim_clock, 'no show')
        disruption_stack = request + delay + cancel + initial_no_show
        disruption_stack.sort(reverse=True, key=lambda x: x[1])
        return disruption_stack

    def get_disruption(self, current_route_plan, data_path):
        # get disruption from stack
        disruption = self.disruptions_stack.pop()
        disruption_type = disruption[0]
        disruption_time = disruption[1]

        # find which disruption type it is
        if disruption_type == 'request':
            add_request, disruption_info = self.new_request(disruption_time, data_path)
            if add_request < 0:
                disruption_type = 'no disruption'
                disruption_info = None

        elif disruption_type == 'delay':
            delay_vehicle_index, delay_rid_index, duration_delay = self.delay(disruption_time, current_route_plan)
            disruption_info = (delay_vehicle_index, delay_rid_index, duration_delay)
            if delay_rid_index < 0:
                disruption_type = 'no disruption'
                disruption_info = None

        elif disruption_type == 'cancel':
            cancel_vehicle_index, cancel_pickup_rid_index, cancel_dropoff_rid_index = self.cancel(disruption_time, current_route_plan)
            disruption_info = (cancel_vehicle_index, cancel_pickup_rid_index, cancel_dropoff_rid_index)
            if cancel_pickup_rid_index < 0:
                disruption_type = 'no disruption'
                disruption_info = None

        else:
            no_show_vehicle_index, no_show_pickup_rid_index, no_show_dropoff_rid_index, actual_no_show = self.no_show(disruption_time, current_route_plan)
            disruption_info = (no_show_vehicle_index, no_show_pickup_rid_index, no_show_dropoff_rid_index)
            next_disruption_time = self.disruptions_stack[-1][1] if len(self.disruptions_stack) > 0 else datetime.strptime("2021-05-10 19:00:00", "%Y-%m-%d %H:%M:%S")
            if no_show_pickup_rid_index < 0 or actual_no_show >= next_disruption_time:
                disruption_type = 'no disruption'
                disruption_info = None
            else:
                disruption_time = actual_no_show

        # update the sim_clock
        self.sim_clock = disruption_time
        return disruption_type, disruption_time, disruption_info

    def new_request(self, request_arrival, data_path):
        # return new request data
        random_number = np.random.rand()
        if random_number > percentage_dropoff:
            # request has requested pickup time - draw random time
            requested_pickup_time = request_arrival + timedelta(minutes=gamma.rvs(pickup_fit_shape, pickup_fit_loc, pickup_fit_scale))

            # if request arrival is after 17.45 or the requested pickup time is after 17.45, the request is ignored
            if request_arrival > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00) \
                    or requested_pickup_time > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00):
                return -1, -1

            else:
                # get random request
                random_request = NewRequests(data_path).get_and_drop_random_request()

                # update creation time to request disruption time
                random_request['Request Creation Time'] = request_arrival

                # update requested pickup time and set requested dropoff time to NaN
                random_request['Requested Pickup Time'] = requested_pickup_time
                random_request['Requested Dropoff Time'] = None

                return 1, random_request

        else:
            # request has requested dropoff time - draw random time
            requested_dropoff_time = request_arrival + timedelta(minutes=gamma.rvs(dropoff_fit_shape, dropoff_fit_loc, dropoff_fit_scale))

            # if request arrival is after 17.45 or requested dropoff time is after 18.00, the request is ignored
            if request_arrival > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00)\
                    or requested_dropoff_time > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 18, 00, 00):
                return -1, -1

            else:
                # get random request
                random_request = NewRequests(data_path).get_and_drop_random_request()

                # update creation time to request disruption time
                random_request['Request Creation Time'] = request_arrival

                # update requested pickup time and set requested dropoff time to NaN
                random_request['Requested Pickup Time'] = None
                random_request['Requested Dropoff Time'] = requested_dropoff_time

                return 1, random_request

    def delay(self, initial_delay, current_route_plan):
        # draw duration of delay
        delay = timedelta(minutes=gamma.rvs(delay_fit_shape, delay_fit_loc, delay_fit_scale))

        rids_indices = []
        planned_departure_times = []
        vehicle_indices = []

        # potential delays - nodes with planned pickup time after initial_delay
        vehicle_index = 0
        for row in current_route_plan:
            for col in range(1, len(row)):
                temp_planned_time = row[col][1] - timedelta(minutes=S)
                if temp_planned_time >= initial_delay:
                    rids_indices.append(col)
                    planned_departure_times.append(temp_planned_time)
                    vehicle_indices.append(vehicle_index)
            vehicle_index += 1

        # check whether there are any delays, if not, another disruption type will be chosen
        # if yes, pick the delay with earliest planned departure time
        if len(rids_indices) > 0:
            temp_actual_disruption_time = min(planned_departure_times)
            rid_index = rids_indices[planned_departure_times.index(temp_actual_disruption_time)]
            vehicle_index = vehicle_indices[planned_departure_times.index(temp_actual_disruption_time)]
            return vehicle_index, rid_index, delay
        else:
            return -1, -1, -1

    def cancel(self, cancel, current_route_plan):
        indices = []

        # potential cancellations - pickup nodes with planned pickup after disruption time of cancellation
        vehicle_index = 0
        for row in current_route_plan:
            for col in range(1, len(row)):
                temp_rid = row[col][0]
                temp_planned_time = row[col][1] - timedelta(minutes=S)
                if not temp_rid % int(temp_rid) and temp_planned_time >= cancel:
                    for i in range(col, len(row)):
                        if row[i][0] == temp_rid + 0.5:
                            indices.append((vehicle_index, col, i))
            vehicle_index += 1

        # check whether there are any cancellations, if not, another disruption will be chosen
        # if yes, pick a random pickup node as the cancellation
        if len(indices) > 0:
            index = random.choice(indices)
            return index[0], index[1], index[2]
        else:
            return -1, -1, -1

    def no_show(self, initial_no_show, current_route_plan):
        indices = []
        planned_pickup_times = []

        # potential no shows - pickup nodes with planned pickup after initial_no_show
        vehicle_index = 0
        for row in current_route_plan:
            for col in range(1, len(row)):
                temp_rid = row[col][0]
                temp_planned_time = row[col][1] - timedelta(minutes=S)
                if not temp_rid % int(temp_rid) and temp_planned_time >= initial_no_show:
                    for i in range(col, len(row)):
                        if row[i][0] == temp_rid + 0.5:
                            indices.append((vehicle_index, col, i))
                            planned_pickup_times.append(temp_planned_time)
            vehicle_index += 1

        # check whether there are any no shows, if not, another disruption type will be chosen
        # if yes, pick the no show with earliest planned pickup time
        if len(indices) > 0:
            actual_disruption_time = min(planned_pickup_times)
            index = indices[planned_pickup_times.index(actual_disruption_time)]
            return index[0], index[1], index[2], actual_disruption_time
        else:
            return -1, -1, -1, -1


def main():
    simulator = None

    try:

        # CONSTRUCTION OF INITIAL SOLUTION
        df = pd.read_csv(config("test_data_construction"))
        constructor = ConstructionHeuristic(requests=df.head(20), vehicles=V)
        print("Constructing initial solution")
        current_route_plan, initial_objective, infeasible_set = constructor.construct_initial()

        num_new_requests = 0
        num_delay = 0
        num_cancel = 0
        num_no_show = 0
        num_no_disruption = 0

        # SIMULATION
        # første runde av simulator må kjøre med new requests fra data_processed_path for å få fullstendig antall
        # requests første runde, deretter skal rundene kjøre med data_simulator_path for å få updated data
        print("Start simulation")
        sim_clock = datetime.strptime("2021-05-10 10:00:00", "%Y-%m-%d %H:%M:%S")
        simulator = Simulator(sim_clock)
        first_iteration = True

        while len(simulator.disruptions_stack) > 0:
            if not first_iteration:
                disruption_type, disruption_time, disruption_info = simulator.get_disruption(current_route_plan, config(
                    "data_simulator_path"))
            else:
                disruption_type, disruption_time, disruption_info = simulator.get_disruption(current_route_plan, config(
                    "data_processed_path"))
                first_iteration = False
            #print("Disruption type", disruption_type)
            #print("Disruption time", disruption_time)
            #print("Disruption info", disruption_info)
            #print()

            if disruption_type == "request":
                num_new_requests += 1
            elif disruption_type == "delay":
                num_delay += 1
            elif disruption_type == 'cancel':
                num_cancel += 1
            elif disruption_type == 'no show':
                num_no_show += 1
            else:
                num_no_disruption += 1

        print("New requests", num_new_requests)
        print("Delay", num_delay)
        print("Cancel", num_cancel)
        print("No show", num_no_show)
        print("No disruption", num_no_disruption)

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()