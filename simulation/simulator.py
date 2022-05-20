from config.main_config import *
from simulation.poisson import *
from config.simulation_config import *
import random
from scipy.stats import gamma, beta
from numpy.random import rand, seed
import pandas as pd
from decouple import config


class Simulator:
    def __init__(self, sim_clock):
        self.sim_clock = sim_clock
        self.poisson = Poisson()
        self.disruptions_stack = self.create_disruption_stack()

    def create_disruption_stack(self):
        """
        Get disruption times for each disruption type, indexed as follows:
            request: 0
            delay: 1
            cancel: 2
            no show: 3
            no disruption: 4
        """
        request = self.poisson.disruption_times(
            arrival_rate_request, self.sim_clock, 0)
        delay = self.poisson.disruption_times(
            arrival_rate_delay, self.sim_clock, 1)
        cancel = self.poisson.disruption_times(
            arrival_rate_cancel, self.sim_clock, 2)
        initial_no_show = self.poisson.disruption_times(
            arrival_rate_no_show, self.sim_clock, 3)
        disruption_stack = request + delay + cancel + initial_no_show
        disruption_stack.sort(reverse=True, key=lambda x: x[1])
        return disruption_stack

    def get_disruption(self, current_route_plan, data_path):
        # get disruption from stack
        disruption = self.disruptions_stack.pop()
        disruption_type = disruption[0]
        disruption_time = disruption[1]

        # find which disruption type it is
        if disruption_type == 0:
            add_request, disruption_info = self.new_request(
                disruption_time, data_path)
            if add_request < 0:
                disruption_type = 4
                disruption_info = None

        elif disruption_type == 1:
            delay_vehicle_index, delay_rid_index, duration_delay, delay_rid = self.delay(
                disruption_time, current_route_plan)
            disruption_info = (delay_vehicle_index,
                               delay_rid_index, duration_delay, delay_rid)
            if delay_rid_index < 0:
                disruption_type = 4
                disruption_info = None

        elif disruption_type == 2:
            cancel_vehicle_index, cancel_pickup_rid_index, cancel_dropoff_rid_index, node_p, node_d = self.cancel(
                disruption_time, current_route_plan)
            disruption_info = (
                cancel_vehicle_index, cancel_pickup_rid_index, cancel_dropoff_rid_index, node_p, node_d)
            if cancel_pickup_rid_index < 0:
                disruption_type = 4
                disruption_info = None

        else:
            no_show_vehicle_index, no_show_pickup_rid_index, no_show_dropoff_rid_index, node_p, node_d, actual_no_show = self.no_show(
                disruption_time, current_route_plan)
            disruption_info = (
                no_show_vehicle_index, no_show_pickup_rid_index, no_show_dropoff_rid_index, node_p, node_d)
            next_disruption_time = self.disruptions_stack[-1][1] if len(
                self.disruptions_stack) > 0 else datetime.strptime("2021-05-10 19:00:00", "%Y-%m-%d %H:%M:%S")
            if no_show_pickup_rid_index < 0 or actual_no_show >= next_disruption_time:
                disruption_type = 4
                disruption_info = None
            else:
                disruption_time = actual_no_show

        # update the sim_clock
        self.sim_clock = disruption_time

        return disruption_type, disruption_time, disruption_info

    def new_request(self, request_arrival, data_path):
        seed(int(request_arrival.timestamp()))
        random.seed(int(request_arrival.timestamp()))

        # return new request data
        random_number = rand()
        if random_number > percentage_dropoff:
            # request has requested pickup time - draw random time
            requested_pickup_time = request_arrival + \
                timedelta(minutes=gamma.rvs(pickup_fit_shape,
                          pickup_fit_loc, pickup_fit_scale))

            # if request arrival is after 17.45 or the requested pickup time is after 17.45, the request is ignored
            if request_arrival > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00) \
                    or requested_pickup_time > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00):
                return -1, -1

            else:
                # get random request
                random_request = self.get_and_drop_random_request(
                    data_path, request_arrival)

                # update creation time to request disruption time
                random_request['Request Creation Time'] = request_arrival

                # update requested pickup time and set requested dropoff time to NaN
                random_request['Requested Pickup Time'] = requested_pickup_time
                random_request['Requested Dropoff Time'] = pd.NaT

                return 1, random_request

        else:
            # request has requested dropoff time - draw random time
            requested_dropoff_time = request_arrival + \
                timedelta(minutes=gamma.rvs(dropoff_fit_shape,
                          dropoff_fit_loc, dropoff_fit_scale))

            # if request arrival is after 17.45 or requested dropoff time is after 18.00, the request is ignored
            if request_arrival > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00)\
                    or requested_dropoff_time > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 18, 00, 00):
                return -1, -1

            else:
                # get random request
                random_request = self.get_and_drop_random_request(
                    data_path, request_arrival)

                # update creation time to request disruption time
                random_request['Request Creation Time'] = request_arrival

                # update requested pickup time and set requested dropoff time to NaN
                random_request['Requested Pickup Time'] = pd.NaT
                random_request['Requested Dropoff Time'] = requested_dropoff_time

                return 1, random_request

    def delay(self, initial_delay, current_route_plan):
        seed(int(initial_delay.timestamp()))
        random.seed(int(initial_delay.timestamp()))

        # draw duration of delay
        delay = timedelta(minutes=beta.rvs(
            delay_fit_a, delay_fit_b, delay_fit_loc, delay_fit_scale))

        # potential delays - nodes with planned pickup time after initial_delay
        vehicle_index = 0
        possible_nodes = []
        for vehicle_route in current_route_plan:
            for idx, node in enumerate(vehicle_route):
                if node[0] == 0:
                    continue
                s = S_W if node[5]["Wheelchair"] else S_P
                temp_planned_time = node[1] - timedelta(minutes=s)
                if temp_planned_time >= initial_delay:
                    possible_nodes.append(
                        (idx, temp_planned_time, vehicle_index, node[0]))
            vehicle_index += 1

        # check whether there are any delays, if not, another disruption type will be chosen
        # if yes, pick the delay with earliest planned time
        if len(possible_nodes) > 0:
            possible_nodes.sort(key=lambda x: x[1])
            return possible_nodes[0][2], possible_nodes[0][0], delay, possible_nodes[0][3]
        else:
            return -1, -1, -1, -1

    def cancel(self, cancel, current_route_plan):
        seed(int(cancel.timestamp()))
        random.seed(int(cancel.timestamp()))

        # draw duration of delay
        cancel_time = timedelta(minutes=beta.rvs(
            cancel_fit_a, cancel_fit_b, cancel_fit_loc, cancel_fit_scale))
        indices = []

        # potential cancellations - pickup nodes with planned pickup after disruption time of cancellation + cancel_time
        vehicle_index = 0
        for row in current_route_plan:
            for col in range(0, len(row)):
                if row[col][0] == 0:
                    continue
                temp_rid = row[col][0]
                s = S_W if row[col][5]["Wheelchair"] else S_P
                temp_planned_time = row[col][1] - timedelta(minutes=s)
                if not temp_rid % int(temp_rid) and temp_planned_time >= cancel + cancel_time:
                    for i in range(col, len(row)):
                        if row[i][0] == temp_rid + 0.5:
                            indices.append(
                                (vehicle_index, col, i, temp_rid, row[i][0], temp_planned_time))
            vehicle_index += 1

        # check whether there are any cancellations, if not, another disruption will be chosen
        # if yes, pick a random pickup node as the cancellation
        if len(indices) > 0:
            indices.sort(reverse=False, key=lambda x: x[5])
            return indices[0][0], indices[0][1], indices[0][2], indices[0][3], indices[0][4]
        else:
            return -1, -1, -1, -1, -1

    def no_show(self, initial_no_show, current_route_plan):

        indices, planned_pickup_times = [], []

        # potential no shows - pickup nodes with planned pickup after initial_no_show
        vehicle_index = 0
        possible_noshows = []
        for vehicle_route in current_route_plan:
            for idx, node in enumerate(vehicle_route):
                if node[0] == 0:
                    continue
                temp_rid = node[0]
                s = S_W if node[5]["Wheelchair"] else S_P
                temp_planned_time = node[1] - timedelta(minutes=s)
                if not temp_rid % int(temp_rid) and temp_planned_time >= initial_no_show:
                    for i in vehicle_route[idx:]:
                        if node[0] == temp_rid + 0.5:
                            possible_noshows.append(
                                (temp_planned_time, vehicle_index, idx, i, temp_rid, vehicle_route[i][0]))
            vehicle_index += 1

        # check whether there are any no shows, if not, another disruption type will be chosen
        # if yes, pick the no show with earliest planned pickup time
        if len(possible_noshows) > 0:
            possible_noshows.sort(key=lambda x: x[0])
            actual_disruption_time = possible_noshows[0][0]
            index = possible_noshows[0][1:]
            return index[0], index[1], index[2], index[3], index[4], actual_disruption_time
        else:
            return -1, -1, -1, -1, -1, -1

    def get_and_drop_random_request(self, data_path, request_arrival):

        df_same_day_after_10 = pd.read_csv(data_path, index_col=0)

        # get random request
        random_request = df_same_day_after_10.sample(
            random_state=int(request_arrival.timestamp()))
        random_request.drop(columns=['Request Creation Time',
                                     'Requested Pickup Time',
                                     'Actual Pickup Time',
                                     'Requested Dropoff Time',
                                     'Actual Dropoff Time',
                                     'Requested Pickup/Dropoff Time',
                                     'Date Creation',
                                     'Time Creation',
                                     'Date Pickup/Dropoff',
                                     'Request ID',
                                     'Request Status',
                                     'Rider ID',
                                     'Ride ID',
                                     'Cancellation Time',
                                     'No Show Time',
                                     'Origin Zone',
                                     'Destination Zone',
                                     'Reason For Travel'], inplace=True)

        return random_request
