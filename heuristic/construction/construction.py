import pandas as pd
import numpy as np
from tqdm import tqdm
from math import radians
from decouple import config
from config.construction_config import *
from heuristic.construction.insertion_generator import InsertionGenerator
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import haversine_distances
pd.options.mode.chained_assignment = None


class ConstructionHeuristic:
    def __init__(self, requests, vehicles):
        self.vehicles = [i for i in range(vehicles)]
        self.n = len(requests.index)
        self.num_nodes_and_depots = 2 * vehicles + 2 * self.n
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
                requests["Requested Pickup Time"].iloc[i] = requests["Requested Dropoff Time"].iloc[i] - self.temp_travel_time(
                    i, self.n + i, True, temp_T_ij)

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
        for i in tqdm(range(unassigned_requests.shape[0]), colour='#39ff14'):
            # while not unassigned_requests.empty:
            request = unassigned_requests.iloc[i]

            route_plan, new_objective = self.insertion_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid)

            # update current objective
            self.current_objective = new_objective

            rid += 1
        return route_plan, self.current_objective, self.infeasible_set

    def new_objective(self, new_routeplan, new_infeasible_set):
        total_deviation, total_travel_time = timedelta(
            minutes=0), timedelta(minutes=0)
        total_infeasible = timedelta(minutes=len(new_infeasible_set))
        for vehicle_route in new_routeplan:
            if len(vehicle_route) >= 2:
                diff = (pd.to_datetime(
                    vehicle_route[-1][1]) - pd.to_datetime(vehicle_route[0][1])) / pd.Timedelta(minutes=1)
                total_travel_time += timedelta(minutes=diff)
            for n, t, d, p, w, _ in vehicle_route:
                if d is not None:
                    d = d if d > timedelta(0) else -d
                    pen_dev = d - P_S if d > P_S else timedelta(0)
                    total_deviation += pen_dev

        updated = alpha*total_travel_time + beta * \
            total_deviation + gamma*total_infeasible
        return updated

    def print_new_objective(self, new_routeplan, new_infeasible_set):
        total_deviation, total_travel_time = timedelta(
            minutes=0), timedelta(minutes=0)
        total_infeasible = timedelta(minutes=len(new_infeasible_set))
        for vehicle_route in new_routeplan:
            diff = (pd.to_datetime(
                vehicle_route[-1][1]) - pd.to_datetime(vehicle_route[0][1])) / pd.Timedelta(minutes=1)
            total_travel_time += timedelta(minutes=diff)
            for n, t, d, p, w, _ in vehicle_route:
                if d is not None:
                    d = d if d > timedelta(0) else -d
                    total_deviation += d
        print("Total travel time", total_travel_time)
        print("Total deviation", total_deviation)
        print("Total infeasible", total_infeasible)

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
        for i in range(len(self.vehicles)):
            vehicle_lat_lon.append(
                (radians(59.946829115276145), radians(10.779841653639243))
            )

        # Destinations for each vehicle
        for i in range(len(self.vehicles)):
            vehicle_lat_lon.append(
                (radians(59.946829115276145), radians(10.779841653639243))
            )

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
