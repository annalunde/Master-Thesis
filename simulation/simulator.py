from datetime import timedelta
from poisson import *
from new_requests import *
from simulation_config import *
import random
from scipy.stats import gamma

class Simulator:
    def __init__(self, sim_clock, current_route_plan, data_path):
        self.sim_clock = sim_clock
        self.current_route_plan = current_route_plan
        self.data_path = data_path

    def get_disruption(self):
        # get disruption times for each disruption type
        request = Poisson(arrival_rate_request, self.sim_clock).disruption_time()
        initial_delay = Poisson(arrival_rate_delay, self.sim_clock).disruption_time()
        cancel = Poisson(arrival_rate_cancel, self.sim_clock).disruption_time()
        initial_no_show = Poisson(arrival_rate_no_show, self.sim_clock).disruption_time()

        # calculate actual delay and no show
        # + see if there are remaining nodes that can be delayed, cancelled or no showed
        delay_vehicle_index, delay_rid_index, duration_delay, actual_delay = self.delay(initial_delay)
        cancel_vehicle_index, cancel_pickup_rid_index, cancel_dropoff_rid_index = self.cancel(cancel)
        no_show_vehicle_index, no_show_pickup_rid_index, no_show_dropoff_rid_index, actual_no_show = self.no_show(initial_no_show)

        # identify earliest disruption time and corresponding disruption type
        disruption_types = ['request']
        disruption_times = [request]

        if delay_rid_index > -1:
            disruption_types.append('delay')
            disruption_times.append(actual_delay)
        if cancel_pickup_rid_index > -1:
            disruption_types.append('cancel')
            disruption_times.append(cancel)
        if no_show_pickup_rid_index > -1:
            disruption_types.append('no show')
            disruption_times.append(actual_no_show)

        disruption_index = disruption_times.index(min(disruption_times))
        disruption_type = disruption_types[disruption_index]
        disruption_time = disruption_times[disruption_index]

        # call on function corresponding to disruption type to get disruption type, time/updated sim_clock, data
        # VIKTIG Å HUSKE AT UPDATED SIM CLOCK ER DET SAMME SOM NY DISRUPTION TIME
        if disruption_type == 'request':
            return disruption_type, disruption_time, self.new_request(request)
        elif disruption_type == 'delay':
            # disruption_data is tuple with (delayed node rid, which vehicle delayed node on, delay in minutes)
            return disruption_type, disruption_time, (delay_vehicle_index, delay_rid_index, duration_delay)
        elif disruption_type == 'cancel':
            # disruption data is tuple with (cancelled pickup rid, cancelled dropoff rid,
            # which vehicle cancelled node on)
            return disruption_type, disruption_time, (cancel_vehicle_index, cancel_pickup_rid_index, cancel_dropoff_rid_index)
        else:
            # disruption data is tuple with (no show pickup rid, no show dropoff rid, which vehicle no show node on)
            return disruption_type, disruption_time, (no_show_vehicle_index, no_show_pickup_rid_index, no_show_dropoff_rid_index)

    def new_request(self, request):
        # return new request data
        random_number = np.random.rand()
        if random_number > percentage_dropoff:
            # request has requested pickup time - draw random time
            requested_pickup_time = request + timedelta(minutes=gamma.rvs(pickup_fit_shape, pickup_fit_loc, pickup_fit_scale))

            # get random request
            random_request = NewRequests(self.data_path).get_and_drop_random_request()

            # update creation time to request disruption time
            random_request['Request Creation Time'] = request

            # update requested pickup time and set requested dropoff time to NaN
            random_request['Requested Pickup Time'] = requested_pickup_time
            random_request['Requested Dropoff Time'] = None

            return random_request

        else:
            # request has requested dropoff time - draw random time
            requested_dropoff_time = request + timedelta(minutes=gamma.rvs(dropoff_fit_shape, dropoff_fit_loc, dropoff_fit_scale))

            # get random request
            random_request = NewRequests(self.data_path).get_and_drop_random_request()

            # update creation time to request disruption time
            random_request['Request Creation Time'] = request

            # update requested pickup time and set requested dropoff time to NaN
            random_request['Requested Pickup Time'] = None
            random_request['Requested Dropoff Time'] = requested_dropoff_time

            return random_request

    def delay(self, initial_delay):
        # draw duration of delay
        delay = timedelta(minutes=gamma.rvs(delay_fit_shape, delay_fit_loc, delay_fit_scale))

        rids_indices = []
        planned_departure_times = []
        vehicle_indices = []

        # potential delays - nodes with planned departure time after initial_delay
        vehicle_index = 0
        for row in self.current_route_plan:
            for col in range(0, len(row)):
                temp_planned_time = row[col][1]
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
            actual_disruption_time = temp_actual_disruption_time + delay
            return vehicle_index, rid_index, delay, actual_disruption_time
        else:
            return -1, -1, -1, -1

    def cancel(self, cancel):
        indices = []

        # potential cancellations - pickup nodes with planned pickup after disruption time of cancellation
        vehicle_index = 0
        for row in self.current_route_plan:
            for col in range(0, len(row)):
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

    def no_show(self, initial_no_show):
        indices = []
        planned_pickup_times = []

        # potential no shows - pickup nodes with planned pickup after initial_no_show
        vehicle_index = 0
        for row in self.current_route_plan:
            for col in range(0, len(row)):
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
        current_route_plan = [
            [(1, datetime.strptime("2021-05-10 12:02:00", "%Y-%m-%d %H:%M:%S")), (3, datetime.strptime("2021-05-10 12:10:00", "%Y-%m-%d %H:%M:%S")),
             (1.5, datetime.strptime("2021-05-10 13:15:00", "%Y-%m-%d %H:%M:%S")), (3.5, datetime.strptime("2021-05-10 14:12:00", "%Y-%m-%d %H:%M:%S"))],
            [(2, datetime.strptime("2021-05-10 13:15:00", "%Y-%m-%d %H:%M:%S")), (2.5, datetime.strptime("2021-05-10 14:12:00", "%Y-%m-%d %H:%M:%S"))]
                      ]
        sim_clock = datetime.strptime("2021-05-10 12:30:00", "%Y-%m-%d %H:%M:%S")
        # første runde av simulator må kjøre med new requests fra data_processed_path for å få fullstendig antall
        # requests første runde, deretter skal rundene kjøre med data_simulator_path for å få updated data
        simulator = Simulator(sim_clock, current_route_plan, config("data_processed_path"))
        disruption_type, disruption_time, disruption_data = simulator.get_disruption()
        print(disruption_type)
        print(disruption_time)
        print(disruption_data)
        sim_clock = disruption_time

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()