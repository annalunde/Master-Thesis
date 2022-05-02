from config.main_config import *
from simulation.poisson import *
from config.simulation_config import *
import random
from scipy.stats import gamma, beta
from numpy.random import rand, seed
import pandas as pd
from decouple import config

from heuristic.construction.construction import ConstructionHeuristic

class Simulator:
    def __init__(self, sim_clock):
        self.sim_clock = sim_clock
        self.poisson = Poisson()
        self.disruptions_stack = self.create_disruption_stack()
        seed(int((self.sim_clock - timedelta(hours=self.sim_clock.hour, minutes=self.sim_clock.minute,
                                             seconds=self.sim_clock.second)).timestamp()))
        random.seed(int((self.sim_clock - timedelta(hours=self.sim_clock.hour, minutes=self.sim_clock.minute,
                                                    seconds=self.sim_clock.second)).timestamp()))

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

    def get_disruption(self, current_route_plan, data_path, first_iteration):
        # get disruption from stack
        disruption = self.disruptions_stack.pop()
        disruption_type = disruption[0]
        disruption_time = disruption[1]

        # find which disruption type it is
        if disruption_type == 0:
            add_request, disruption_info = self.new_request(
                disruption_time, data_path, first_iteration)
            if add_request < 0:
                disruption_type = 4
                disruption_info = None

        elif disruption_type == 1:
            delay_vehicle_index, delay_rid_index, duration_delay = self.delay(
                disruption_time, current_route_plan)
            disruption_info = (delay_vehicle_index,
                               delay_rid_index, duration_delay)
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

    def new_request(self, request_arrival, data_path, first_iteration):
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
                random_request = self.get_and_drop_random_request(data_path, first_iteration)

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
                random_request = self.get_and_drop_random_request(data_path, first_iteration)

                # update creation time to request disruption time
                random_request['Request Creation Time'] = request_arrival

                # update requested pickup time and set requested dropoff time to NaN
                random_request['Requested Pickup Time'] = pd.NaT
                random_request['Requested Dropoff Time'] = requested_dropoff_time

                return 1, random_request

    def delay(self, initial_delay, current_route_plan):
        # draw duration of delay
        delay = timedelta(minutes=beta.rvs(
            delay_fit_a, delay_fit_b, delay_fit_loc, delay_fit_scale))

        rids_indices, planned_times, vehicle_indices = [], [], []

        # potential delays - nodes with planned pickup time after initial_delay
        vehicle_index = 0
        for row in current_route_plan:
            for col in range(1, len(row)):
                s = S_W if row[col][5]["Wheelchair"] else S_P
                temp_planned_time = row[col][1] - timedelta(minutes=s)
                if temp_planned_time >= initial_delay:
                    rids_indices.append(col)
                    planned_times.append(temp_planned_time)
                    vehicle_indices.append(vehicle_index)
            vehicle_index += 1

        # check whether there are any delays, if not, another disruption type will be chosen
        # if yes, pick the delay with earliest planned time
        if len(rids_indices) > 0:
            temp_actual_disruption_time = min(planned_times)
            rid_index = rids_indices[planned_times.index(
                temp_actual_disruption_time)]
            vehicle_index = vehicle_indices[planned_times.index(
                temp_actual_disruption_time)]
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
                s = S_W if row[col][5]["Wheelchair"] else S_P
                temp_planned_time = row[col][1] - timedelta(minutes=s)
                if not temp_rid % int(temp_rid) and temp_planned_time >= cancel:
                    for i in range(col, len(row)):
                        if row[i][0] == temp_rid + 0.5:
                            indices.append(
                                (vehicle_index, col, i, temp_rid, row[i][0]))
            vehicle_index += 1

        # check whether there are any cancellations, if not, another disruption will be chosen
        # if yes, pick a random pickup node as the cancellation
        if len(indices) > 0:
            index = random.choice(indices)
            return index[0], index[1], index[2], index[3], index[4]
        else:
            return -1, -1, -1, -1, -1

    def no_show(self, initial_no_show, current_route_plan):
        indices, planned_pickup_times = [], []

        # potential no shows - pickup nodes with planned pickup after initial_no_show
        vehicle_index = 0
        for row in current_route_plan:
            for col in range(1, len(row)):
                temp_rid = row[col][0]
                s = S_W if row[col][5]["Wheelchair"] else S_P
                temp_planned_time = row[col][1] - timedelta(minutes=s)
                if not temp_rid % int(temp_rid) and temp_planned_time >= initial_no_show:
                    for i in range(col, len(row)):
                        if row[i][0] == temp_rid + 0.5:
                            indices.append(
                                (vehicle_index, col, i, temp_rid, row[i][0]))
                            planned_pickup_times.append(temp_planned_time)
            vehicle_index += 1

        # check whether there are any no shows, if not, another disruption type will be chosen
        # if yes, pick the no show with earliest planned pickup time
        if len(indices) > 0:
            actual_disruption_time = min(planned_pickup_times)
            index = indices[planned_pickup_times.index(actual_disruption_time)]
            return index[0], index[1], index[2], index[3], index[4], actual_disruption_time
        else:
            return -1, -1, -1, -1, -1, -1

    def get_and_drop_random_request(self, data_path, first_iteration):
        if first_iteration:
            df = pd.read_csv(data_path, index_col=0)
            df['Request Creation Time'] = pd.to_datetime(df['Request Creation Time'],
                                                         format="%Y-%m-%d %H:%M:%S")
            df['Requested Pickup Time'] = pd.to_datetime(df['Requested Pickup Time'],
                                                         format="%Y-%m-%d %H:%M:%S")
            df['Requested Dropoff Time'] = pd.to_datetime(df['Requested Dropoff Time'],
                                                          format="%Y-%m-%d %H:%M:%S")

            # request arrives at same day as it is requested to be served
            df["Requested Pickup/Dropoff Time"] = (df["Requested Pickup Time"]).fillna(
                df["Requested Dropoff Time"])
            df['Date Creation'] = df['Request Creation Time'].dt.date
            df['Time Creation'] = df['Request Creation Time'].dt.hour
            df['Date Pickup/Dropoff'] = df['Requested Pickup/Dropoff Time'].dt.date
            df_same_day = df[df['Date Creation'] == df['Date Pickup/Dropoff']]

            # requests arrives after 10am
            df_same_day_after_10 = df_same_day[
                (df_same_day['Time Creation'] >= 10)
            ]
        else:
            df_same_day_after_10 = pd.read_csv(data_path, index_col=0)

        # get random request
        random_request = df_same_day_after_10.sample()
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

        # drop the request
        df_same_day_after_10_updated = df_same_day_after_10.drop(
            random_request.index)

        # write updated dataframe to csv
        df_same_day_after_10_updated.to_csv(config("data_simulator_path"))

        return random_request


def main():
    sim_clock = datetime.strptime("2021-05-10 10:00:00", "%Y-%m-%d %H:%M:%S")
    simulator = Simulator(sim_clock)
    print(simulator.disruptions_stack)
    print(simulator.create_disruption_stack())
    print(len(simulator.disruptions_stack))

    # CONSTRUCTION OF INITIAL SOLUTION
    df = pd.read_csv(config("test_data_construction"))
    constructor = ConstructionHeuristic(requests=df, vehicles=V)
    print("Constructing initial solution")
    initial_route_plan, initial_objective, initial_infeasible_set = constructor.construct_initial()

    first_iteration = True

    while len(simulator.disruptions_stack) > 0:
        if not first_iteration:
            disruption_type, disruption_time, disruption_info = simulator.get_disruption(
                initial_route_plan, config("data_simulator_path"), first_iteration)
        else:
            disruption_type, disruption_time, disruption_info = simulator.get_disruption(
                initial_route_plan, config("data_processed_path"), first_iteration)
            first_iteration = False

        if disruption_type == 0:
            print("New request", disruption_time, disruption_info.iloc[0]["Requested Pickup Time"], disruption_info.iloc[0]["Origin Lat"])

        elif disruption_type == 1:
            print("Delay", disruption_time, disruption_info)

        elif disruption_type == 2:
            print("Cancel", disruption_time, disruption_info)

        else:
            print("No show", disruption_time, disruption_info)

if __name__ == "__main__":
    main()