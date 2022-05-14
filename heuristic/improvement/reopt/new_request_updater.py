import pandas as pd
import numpy as np
from copy import copy
from math import radians
from functools import reduce
from config.main_config import *
from sklearn.metrics.pairwise import haversine_distances
from heuristic.improvement.reopt.reopt_repair_generator import ReOptRepairGenerator

pd.options.mode.chained_assignment = None


class NewRequestUpdater:
    def __init__(self, constructor):
        self.vehicles = [i for i in range(V)]
        self.introduced_vehicles = set()
        self.requests = constructor.requests.copy(deep=False)
        self.n = len(self.requests.index)
        self.num_nodes_and_depots = len(self.vehicles) + 2 * self.n
        self.current_objective = timedelta(0)
        self.T_ij = np.array(constructor.T_ij, copy=True)
        self.infeasible_set = copy(constructor.infeasible_set)
        self.re_opt_repair_generator = ReOptRepairGenerator(self, False)
        self.preprocessed = copy(constructor.preprocessed)
        self.alpha = constructor.alpha
        self.beta = constructor.beta
        self.gamma = constructor.gamma

    def set_parameters(self, new_request):
        updated_new_request = self.compute_pickup_time(new_request)
        self.requests = self.requests.append(updated_new_request)
        self.n = len(self.requests)
        self.num_nodes_and_depots = len(self.vehicles) + 2 * self.n
        self.T_ij = self.travel_matrix(self.requests)
        self.preprocessed = self.preprocess_requests(updated_new_request)

    def compute_pickup_time(self, new_request):
        if pd.isnull(new_request.iloc[0]["Requested Pickup Time"]):
            origin_lat_lon = [(np.deg2rad(new_request.iloc[0]["Origin Lat"]),
                              np.deg2rad(new_request.iloc[0]["Origin Lng"]))]
            destination_lat_lon = [(np.deg2rad(new_request.iloc[0]["Destination Lat"]),
                                   np.deg2rad(new_request.iloc[0]["Destination Lng"]))]
            D_ij = haversine_distances(
                origin_lat_lon, destination_lat_lon) * 6371
            speed = 20
            travel_time = (timedelta(hours=(D_ij[0][0] / speed)
                                     ).total_seconds())*(1+F/2)

            # rush hour modelling:
            if not (new_request.iloc[0]["Requested Dropoff Time"].weekday() == 5):
                if new_request.iloc[0]["Requested Dropoff Time"].hour >= 15 and new_request.iloc[0]["Requested Dropoff Time"].hour < 17:
                    travel_time = travel_time * R_F

            new_request["Requested Pickup Time"].iloc[0] = new_request.iloc[0]["Requested Dropoff Time"] - timedelta(
                seconds=travel_time)
        return new_request

    def temp_travel_time(self, to_id, from_id, fraction, temp_T_ij):
        return timedelta(seconds=(1+F/2) * temp_T_ij[to_id, from_id]) if fraction else timedelta(seconds=temp_T_ij[to_id, from_id])

    def preprocess_requests(self, new_request):
        # link requests that are too close in time and space for the same vehicle to serve both requests:
        travel_time = self.T_ij
        request_time = self.requested_time_matrix()
        infeasible = []
        for i in range(len(self.preprocessed)):
            tests = [i, i+self.n]
            for k in tests:
                for j in [self.n-1, 2 * self.n-1]:
                    if request_time[k][j] is not None and timedelta(seconds=travel_time[k][j]) - 2*U_D > request_time[k][j]:
                        n_j = j+1 - self.n if j >= self.n else j+1
                        n_k = k - self.n if k >= self.n else k
                        self.preprocessed[n_k].add(n_j)
                        infeasible.append(n_k+1)
        return np.append(self.preprocessed, [set(infeasible)], axis=0)

    def greedy_insertion_new_request(self, current_route_plan, current_infeasible_set, new_request, sim_clock, vehicle_clocks, i, current_objective):
        rid = len(self.requests.index)
        route_plan = list(map(list, current_route_plan))
        infeasible_set = copy(current_infeasible_set)
        request = self.requests.iloc[-1]
        request["Requested Pickup Time"] = request["Requested Pickup Time"] + \
            i*U_D

        self.re_opt_repair_generator.greedy = True
        route_plan, new_objective, infeasible_set, vehicle_clocks = self.re_opt_repair_generator.generate_insertions(
            route_plan=route_plan, request=request, rid=rid, infeasible_set=infeasible_set, initial_route_plan=None,
            index_removed=None, sim_clock=sim_clock, vehicle_clocks=vehicle_clocks,
            objectives=False, delayed=(False, None, None), still_delayed_nodes=[],
            prev_objective=current_objective)

        rejection = False if (rid, request) not in infeasible_set else True
        infeasible_set = [] if rejection else infeasible_set
        # update current objective
        self.current_objective = new_objective
        self.re_opt_repair_generator.greedy = False
        return route_plan, self.current_objective, infeasible_set, vehicle_clocks, rejection, rid

    def new_objective(self, new_routeplan, new_infeasible_set, greedy):
        total_deviation, total_travel_time = timedelta(
            minutes=0), timedelta(minutes=0)
        total_infeasible = len(new_infeasible_set)
        for vehicle, vehicle_route in enumerate(new_routeplan):
            if len(vehicle_route) >= 2:
                for i in range(len(vehicle_route) - 1):
                    sn = vehicle_route[i][0]
                    en = vehicle_route[i+1][0]
                    sn_mod = sn % int(sn) if sn else 0
                    en_mod = en % int(en)
                    start_id = int(
                        sn - 0.5 - 1 + self.n if sn_mod else sn - 1) if sn else 2 * self.n + vehicle
                    end_id = int(en - 0.5 - 1 + self.n if en_mod else en - 1)
                    total_travel_time += self.travel_time(
                        start_id, end_id, False)

            pen_dev = [j if j > timedelta(
                0) else -j for j in [i[2] for i in vehicle_route if i[2] is not None]]
            total_deviation += reduce(
                lambda a, b: a+b, [i-P_S_R if i > P_S_R else timedelta(0) for i in pen_dev]) if pen_dev else timedelta(0)
        updated = self.alpha*total_travel_time + self.beta * \
            total_deviation + self.gamma*total_infeasible
        return updated, total_travel_time, total_deviation, self.gamma*total_infeasible

    def total_objective(self, current_objective, cumulative_objective, cumulative_recalibration, cumulative_rejected, rejection):
        cum_rej = cumulative_rejected if not rejection else cumulative_rejected - 1
        total_objective = current_objective + \
            cumulative_objective + cumulative_recalibration + self.gamma*cum_rej
        return total_objective, self.gamma*cum_rej

    def travel_matrix(self, df):
        # Lat and lon for each request
        origin_lat_lon = list(
            zip(np.deg2rad(df["Origin Lat"]), np.deg2rad(df["Origin Lng"]))
        )
        destination_lat_lon = list(
            zip(np.deg2rad(df["Destination Lat"]),
                np.deg2rad(df["Destination Lng"]))
        )
        request_lat_lon = origin_lat_lon + destination_lat_lon

        vehicle_lat_lon = []

        # Origins for each vehicle
        vehicle_lat_lon = [(radians(59.946829115276145), radians(
            10.779841653639243)) for i in range(len(self.vehicles))]

        # Positions
        lat_lon = request_lat_lon + vehicle_lat_lon

        # Distance matrix
        D_ij = haversine_distances(lat_lon, lat_lon) * 6371

        # Travel time matrix
        speed = 20

        T_ij = np.empty(
            shape=(self.num_nodes_and_depots,
                   self.num_nodes_and_depots), dtype=timedelta
        )

        for i in range(self.num_nodes_and_depots):
            for j in range(self.num_nodes_and_depots):
                T_ij[i][j] = (
                    timedelta(hours=(D_ij[i][j] / speed)
                              ).total_seconds()
                )

        # rush hour modelling:
        if not (df.iloc[0]["Requested Pickup Time"].weekday() == 5):
            for k in range(self.n):
                for l in range(self.n):
                    if df.iloc[k]["Requested Pickup Time"].hour >= 15 and df.iloc[k]["Requested Pickup Time"].hour < 17 and df.iloc[l]["Requested Pickup Time"].hour >= 15 and df.iloc[l]["Requested Pickup Time"].hour < 17:
                        T_ij[k][l] = T_ij[k][l] * R_F
                        T_ij[k + self.n][l] = T_ij[k + self.n][l] * R_F
                        T_ij[k][l + self.n] = T_ij[k][l + self.n] * R_F
                        T_ij[k + self.n][l + self.n] = T_ij[k +
                                                            self.n][l + self.n] * R_F

        return T_ij

    def requested_time_matrix(self):
        # Requested time matrix
        R_ij = np.empty(
            shape=(2*self.n,
                   2*self.n), dtype=timedelta
        )
        nat_pickup = np.isnat(self.requests["Requested Pickup Time"])
        nat_dropoff = np.isnat(self.requests["Requested Dropoff Time"])

        for i in range(2*self.n):
            for j in range(2*self.n):
                if i == j:
                    continue
                if j == i + self.n:
                    continue
                if i == j - self.n:
                    continue
                if i < self.n:
                    i_time = self.requests.iloc[i]["Requested Pickup Time"] if not nat_pickup.iloc[
                        i] else self.requests.iloc[i]["Requested Dropoff Time"] - timedelta(seconds=self.T_ij[i][i+self.n])
                else:
                    i_time = self.requests.iloc[i - self.n]["Requested Dropoff Time"] if not nat_dropoff.iloc[
                        i - self.n] else self.requests.iloc[i - self.n]["Requested Pickup Time"] + timedelta(seconds=self.T_ij[i][i-self.n])
                if j < self.n:
                    j_time = self.requests.iloc[j]["Requested Pickup Time"] if not nat_pickup.iloc[
                        j] else self.requests.iloc[j]["Requested Dropoff Time"] - timedelta(seconds=self.T_ij[j][j+self.n])
                else:
                    j_time = self.requests.iloc[j - self.n]["Requested Dropoff Time"] if not nat_dropoff.iloc[
                        j - self.n] else self.requests.iloc[j - self.n]["Requested Pickup Time"] + timedelta(seconds=self.T_ij[j][j-self.n])

                if pd.to_datetime(j_time) >= pd.to_datetime(i_time):
                    R_ij[i][j] = (pd.to_datetime(j_time) -
                                  pd.to_datetime(i_time))
                else:
                    R_ij[i][j] = None
        return R_ij

    def travel_time(self, to_id, from_id, fraction):
        return timedelta(seconds=(1+F/2) * self.T_ij[to_id, from_id]) if fraction else timedelta(seconds=self.T_ij[to_id, from_id])

    def get_max_travel_time(self, to_id, from_id):
        return timedelta(seconds=(1+F) * self.T_ij[to_id, from_id])

    def get_delta_objective(self, new_routeplan, infeasible_set, current_objective):
        return current_objective - self.new_objective(new_routeplan, infeasible_set, False)
