import pandas as pd
import numpy as np
from math import radians
from decouple import config
from functools import reduce
from config.main_config import *
from heuristic.construction.insertion_generator import InsertionGenerator
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import haversine_distances
pd.options.mode.chained_assignment = None


class ConstructionHeuristic:
    def __init__(self, requests, vehicles):
        self.vehicles = [i for i in range(vehicles)]
        self.n = len(requests.index)
        self.num_nodes_and_depots = vehicles + 2 * self.n
        self.temp_requests = self.compute_pickup_time(requests)
        self.requests = self.temp_requests.sort_values(
            "Requested Pickup Time").reset_index(drop=True)
        self.requests["Requested Pickup Time"] = pd.to_datetime(
            self.requests["Requested Pickup Time"], format="%Y-%m-%d %H:%M:%S"
        )
        self.requests["Requested Dropoff Time"] = pd.to_datetime(
            self.requests["Requested Dropoff Time"], format="%Y-%m-%d %H:%M:%S"
        )
        self.requests["Request Creation Time"] = pd.to_datetime(
            self.requests["Request Creation Time"], format="%Y-%m-%d %H:%M:%S"
        )
        self.current_objective = timedelta(0)
        self.T_ij = self.travel_matrix(self.requests)
        self.introduced_vehicles = set()
        self.infeasible_set = []
        self.insertion_generator = InsertionGenerator(self)
        self.preprocessed = self.preprocess_requests()
        self.gamma = alpha * 4 * \
            timedelta(seconds=np.amax(self.T_ij)) + beta * \
            timedelta(minutes=15) * 2 * (self.n / V)

    def compute_pickup_time(self, requests):
        requests["Requested Pickup Time"] = pd.to_datetime(
            requests["Requested Pickup Time"], format="%Y-%m-%d %H:%M:%S"
        )
        requests["Requested Dropoff Time"] = pd.to_datetime(
            requests["Requested Dropoff Time"], format="%Y-%m-%d %H:%M:%S"
        )

        temp_T_ij = self.travel_matrix(requests)

        nat_pickup = np.isnat(requests["Requested Pickup Time"])

        for i in range(self.n):
            if nat_pickup.iloc[i]:

                travel_time = self.temp_travel_time(
                    i, self.n + i, True, temp_T_ij)

                # rush hour modelling:
                if not (requests.iloc[i]["Requested Dropoff Time"].weekday() == 5):
                    if requests.iloc[i]["Requested Dropoff Time"].hour >= 15 and requests.iloc[i]["Requested Dropoff Time"].hour < 17:
                        travel_time = travel_time * R_F

                requests["Requested Pickup Time"].iloc[i] = requests["Requested Dropoff Time"].iloc[i] - travel_time

        return requests

    def temp_travel_time(self, to_id, from_id, fraction, temp_T_ij):
        return timedelta(seconds=(1+F/2) * temp_T_ij[to_id, from_id]) if fraction else timedelta(seconds=temp_T_ij[to_id, from_id])

    def preprocess_requests(self):
        # link requests that are too close in time and space for the same vehicle to serve both requests:
        travel_time = self.T_ij
        request_time = self.requested_time_matrix()
        P_ij = [set() for _ in range(self.n)]

        for i in range(2 * self.n):
            n_i = i - self.n if i >= self.n else i
            for j in range(2 * self.n):
                if request_time[i][j] is not None and timedelta(seconds=travel_time[i][j]) - 2*U_D > request_time[i][j]:
                    n_j = j - self.n if j >= self.n else j
                    P_ij[n_i].add(n_j+1)
                    P_ij[n_j].add(n_i+1)
        return np.array(P_ij)

    def construct_initial(self):
        rid = 1
        unassigned_requests = self.requests.copy(deep=False)
        self.introduced_vehicles.add(self.vehicles.pop(0))
        route_plan = [[]]

        prev_objective = timedelta(0)

        for i in range(unassigned_requests.shape[0]):
            # while not unassigned_requests.empty:
            request = unassigned_requests.iloc[i]

            route_plan, new_objective = self.insertion_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, prev_objective=prev_objective)

            # update current objective
            self.current_objective = new_objective

            rid += 1
        return route_plan, self.current_objective, self.infeasible_set

    def new_objective(self, new_routeplan, new_infeasible_set):
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
                lambda a, b: a+b, [i-P_S_C if i > P_S_C else timedelta(0) for i in pen_dev]) if pen_dev else timedelta(0)

        updated = alpha*total_travel_time + beta * \
            total_deviation + self.gamma*total_infeasible
        return updated

    def print_objective(self, new_routeplan, new_infeasible_set):
        total_deviation, total_travel_time = timedelta(
            minutes=0), timedelta(minutes=0)
        total_infeasible = len(new_infeasible_set)
        for vehicle, vehicle_route in enumerate(new_routeplan):
            if len(vehicle_route) >= 2:
                for i in range(len(vehicle_route) - 1):
                    sn = vehicle_route[i][0]
                    en = vehicle_route[i + 1][0]
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
                lambda a, b: a+b, [i-P_S_C if i > P_S_C else timedelta(0) for i in pen_dev]) if pen_dev else timedelta(0)

        objective = alpha*total_travel_time + beta * \
            total_deviation + self.gamma*total_infeasible

        print("Objective", objective)
        print("Total travel time", total_travel_time)
        print("Total deviation", total_deviation)
        print("Total infeasible", total_infeasible)

    def total_objective(self, current_objective, current_infeasible, cumulative_objective, cumulative_recalibration):
        total_objective = current_objective + cumulative_objective - len(current_infeasible) * self.gamma \
            + cumulative_recalibration
        return total_objective

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
                        T_ij[k][l] = T_ij[k][l]*R_F
                        T_ij[k+self.n][l] = T_ij[k+self.n][l]*R_F
                        T_ij[k][l+self.n] = T_ij[k][l+self.n]*R_F
                        T_ij[k+self.n][l+self.n] = T_ij[k+self.n][l+self.n]*R_F

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

    @staticmethod
    def recalibrate_solution(route_plan):
        return [[(node[0], node[1], timedelta(0), node[3], node[4], node[5]) for node in vehicle_route] for vehicle_route in route_plan]

    def get_delta_objective(self, new_routeplan, infeasible_set, current_objective):
        return current_objective - self.new_objective(new_routeplan, infeasible_set)
