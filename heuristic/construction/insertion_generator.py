import copy
import math
import numpy as np
import pandas
import sklearn.metrics
from math import radians
from heuristic_config import *
from decouple import config
from datetime import datetime, timedelta

"""
NOTE to self: we only try to add it after the first node that is closest in time
"""


class InsertionGenerator:
    def __init__(self, construction_heuristic):
        self.heuristic = construction_heuristic

    def generate_insertions(self, route_plan, request, rid):
        possible_insertions = {}  # dict: delta objective --> route plan
        for introduced_vehicle in self.heuristic.introduced_vehicles:
            # generate all possible insertions

            if not route_plan[introduced_vehicle]:
                # it is trivial to add the new request
                temp_route_plan = copy.deepcopy(route_plan)
                if not pandas.isnull(request["Requested Pickup Time"]):
                    temp_route_plan[introduced_vehicle] = self.add_initial_nodes_pickup(request=request, introduced_vehicle=introduced_vehicle, rid=rid, vehicle_route=temp_route_plan[introduced_vehicle]
                                                                                        )
                else:
                    temp_route_plan[introduced_vehicle] = self.add_initial_nodes_dropoff(request=request, introduced_vehicle=introduced_vehicle, rid=rid, vehicle_route=temp_route_plan[introduced_vehicle]
                                                                                         )
                # calculate change in objective
                change_objective = self.heuristic.delta_objective(
                    temp_route_plan)
                possible_insertions[change_objective] = temp_route_plan

            else:
                # the vehicle already has other nodes in its route
                # will be set to True if both pickup and dropoff of the request have been added
                feasible_request = False
                activated_checks = False  # will be set to True if there is a test that fails
                temp_route_plan = copy.deepcopy(route_plan)

                if not pandas.isnull(request["Requested Pickup Time"]):
                    vehicle_route = route_plan[introduced_vehicle]

                    # check if there are any infeasible matches with current request
                    preprocessed_check_activated = self.preprocess_check(
                        rid=rid, vehicle_route=vehicle_route)

                    if not preprocessed_check_activated:
                        dropoff_time = request["Requested Pickup Time"] + self.heuristic.travel_time(
                            rid-1, self.heuristic.n + rid-1, True)

                        start_idx = 0
                        vehicle_route = temp_route_plan[introduced_vehicle]
                        for idx, (node, time, deviation, passenger, wheelchair) in enumerate(vehicle_route):
                            if time <= request["Requested Pickup Time"]:
                                start_idx = idx

                        s_p_node, s_p_time, s_p_d, s_p_p, s_p_w = vehicle_route[start_idx]
                        if start_idx == len(vehicle_route) - 1:
                            # there is no other end node, and we only need to check the travel time from start to the node
                            s_d = s_p_node % int(s_p_node)
                            start_id = int(
                                s_p_node - 0.5 - 1 + self.heuristic.n if s_d else s_p_node - 1)
                            s_p_travel_time = self.heuristic.travel_time(
                                rid-1, start_id, True)

                            if s_p_time - timedelta(minutes=(D-s_p_d)) + s_p_travel_time <= request["Requested Pickup Time"]:
                                push_back = s_p_time + s_p_travel_time - request["Requested Pickup Time"] if request["Requested Pickup Time"] - \
                                    s_p_time - s_p_travel_time < timedelta(0) else 0

                                # update backward
                                if push_back:
                                    temp_route_plan[introduced_vehicle], activated_checks = self.update_backward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_p=start_p, push_back=push_back, activated_checks=activated_checks)

                                # check max ride time between nodes
                                activated_checks = self.check_max_ride_time(
                                    vehicle_route=temp_route_plan[introduced_vehicle], activated_checks=activated_checks)

                                # check capacities
                                activated_checks = self.check_capacities(
                                    vehicle_route=temp_route_plan[introduced_vehicle], request=request, rid=rid, start_id=start_idx+1, dropoff_id=start_idx+2, activated_checks=activated_checks)

                                if not activated_checks:
                                    # add pickup node
                                    pickup_id, vehicle_route = self.add_node(
                                        vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=request["Requested Pickup Time"], pickup=True, rid=rid, node_idx=start_idx)

                                    # add dropoff node
                                    dropoff_id, vehicle_route = self.add_node(
                                        vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=dropoff_time, pickup=False, rid=rid, node_idx=start_idx+1)

                                    feasible_request = True

                                    self.check_remove(rid, request)

                                    # calculate change in objective
                                    change_objective = self.heuristic.delta_objective(
                                        temp_route_plan)
                                    possible_insertions[change_objective] = temp_route_plan
                        else:
                            end_node_p, e_p_time, e_p_d, e_p_p, e_p_w = vehicle_route[start_idx+1]

                            end_idx = 0
                            vehicle_route = temp_route_plan[introduced_vehicle]
                            for idx, (node, time, deviation, passenger, wheelchair) in enumerate(vehicle_route):
                                if time <= dropoff_time:
                                    end_idx = idx

                            start_node_d, start_time_d, start_deviation_d, start_pass_d, start_wheelchair_d = vehicle_route[
                                end_idx]
                            if end_idx == len(vehicle_route) - 1:
                                # there is no other end node, and we only need to check the travel time from start to the node
                                end_node_d = None
                            else:
                                end_node_d, end_time_d, end_deviation_d, end_pass_d, end_wheelchair_d = vehicle_route[
                                    end_idx + 1]

                            # try to add pickup and dropff
                            s_p = s_p_node % int(s_p_node)
                            e_p = end_node_p % int(end_node_p)
                            s_d = start_node_d % int(start_node_d)
                            e_d = end_node_d % int(
                                end_node_d) if end_node_d else None
                            start_id_p = int(
                                s_p_node - 0.5 - 1 + self.heuristic.n if s_p else s_p_node - 1)
                            end_id_p = int(
                                end_node_p - 0.5 - 1 + self.heuristic.n if e_p else end_node_p - 1)
                            start_id_d = int(
                                start_node_d - 0.5 - 1 + self.heuristic.n if s_d else start_node_d - 1)
                            if e_d:
                                end_id_d = int(
                                    end_node_d - 0.5 - 1 + self.heuristic.n if e_d else end_node_d - 1)

                            s_p_travel_time = self.heuristic.travel_time(
                                rid-1, start_id_p, True)
                            p_e_travel_time = self.heuristic.travel_time(
                                rid-1, end_id_p, True)
                            s_d_travel_time = self.heuristic.travel_time(
                                rid-1 + self.heuristic.n, start_id_d, True)
                            d_e_travel_time = self.heuristic.travel_time(
                                rid-1 + self.heuristic.n, end_id_d, True) if e_d else None

                            if s_p_time - timedelta(minutes=(D-s_p_d)) + s_p_travel_time <= request["Requested Pickup Time"] and request["Requested Pickup Time"] + timedelta(minutes=S) + p_e_travel_time <= end_time_p + timedelta(minutes=(D-e_p_d)) and start_time_d - timedelta(minutes=(D-start_deviation_d)) + s_d_travel_time <= dropoff_time:
                                push_forward_p = request["Requested Pickup Time"] + \
                                    timedelta(minutes=S) + p_e_travel_time - end_time_p if end_time_p - request["Requested Pickup Time"] - \
                                    timedelta(minutes=S) - p_e_travel_time < timedelta(0) else 0
                                push_back_p = s_p_time + s_p_travel_time - request["Requested Pickup Time"] if request["Requested Pickup Time"] - \
                                    s_p_time - s_p_travel_time < timedelta(0) else 0
                                push_back_d = start_time_d + s_d_travel_time - dropoff_time if dropoff_time - \
                                    start_time_d - s_d_travel_time < timedelta(0) else 0
                                if end_node_d:
                                    if dropoff_time + timedelta(minutes=S) + d_e_travel_time <= end_time_d + timedelta(minutes=(D-end_deviation_d)):
                                        push_forward_d = dropoff_time + \
                                            timedelta(minutes=S) + d_e_travel_time - end_time_d if end_time_d - dropoff_time - \
                                            timedelta(minutes=S) - d_e_travel_time < timedelta(0) else 0
                                    else:
                                        activated_checks = True

                                # update forward
                                if push_forward_p:
                                    temp_route_plan[introduced_vehicle], activated_checks = self.update_forward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=start_idx, push_forward=push_forward_p, activated_checks=activated_checks)
                                if end_node_d:
                                    if push_forward_d:
                                        temp_route_plan[introduced_vehicle], activated_checks = self.update_forward(
                                            vehicle_route=temp_route_plan[introduced_vehicle], start_idx=end_idx, push_forward=push_forward_d, activated_checks=activated_checks)

                                # update backward
                                if push_back_p:
                                    temp_route_plan[introduced_vehicle], activated_checks = self.update_backward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=start_idx, push_back=push_back_p, activated_checks=activated_checks)
                                if push_back_d:
                                    temp_route_plan[introduced_vehicle], activated_checks = self.update_backward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=end_idx, push_back=push_back_d, activated_checks=activated_checks)

                                # check max ride time between nodes
                                activated_checks = self.check_max_ride_time(
                                    vehicle_route=temp_route_plan[introduced_vehicle], activated_checks=activated_checks)

                                # check capacities
                                activated_checks = self.check_capacities(
                                    vehicle_route=temp_route_plan[introduced_vehicle], request=request, rid=rid, start_id=start_idx+1, dropoff_id=end_idx+1, activated_checks=activated_checks)

                                if not activated_checks:
                                    # add pickup node
                                    pickup_id, vehicle_route = self.add_node(
                                        vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=request["Requested Pickup Time"], pickup=True, rid=rid, node_idx=start_idx)

                                    # add dropoff node
                                    dropoff_id, vehicle_route = self.add_node(
                                        vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=dropoff_time, pickup=False, rid=rid, node_idx=end_idx)

                                    feasible_request = True

                                    self.check_remove(rid, request)

                                    # calculate change in objective
                                    change_objective = self.heuristic.delta_objective(
                                        temp_route_plan)
                                    possible_insertions[change_objective] = temp_route_plan

                        if feasible_request:
                            temp_route_plan[introduced_vehicle] = self.update_capacities(
                                vehicle_route=temp_route_plan[introduced_vehicle], request=request, rid=rid, start_id=pickup_id, dropoff_id=dropoff_id)

                else:  # the request has defined dropoff time
                    vehicle_route = route_plan[introduced_vehicle]
                    pickup_time = request["Requested Dropoff Time"] - self.heuristic.travel_time(
                        rid-1, self.heuristic.n + rid-1, True)

                    # check if there are any infeasible matches with current request
                    preprocessed_check_activated = self.preprocess_check(
                        rid=rid, vehicle_route=vehicle_route)

                    if not preprocessed_check_activated:

                        for idx, (node, time, deviation, passenger, wheelchair) in enumerate(vehicle_route):
                            if time <= pickup_time:
                                start_idx = idx

                        s_p_node, time, deviation_s_p, p, w = vehicle_route[start_idx]
                        if start_idx == len(vehicle_route) - 1:
                            # there is no other end node, and we only need to check the travel time from start to the node
                            s_p = s_p_node % int(s_p_node)
                            start_id = int(
                                s_p_node - 0.5 - 1 + self.heuristic.n if s_p else s_p_node - 1)
                            s_n_travel_time = self.heuristic.travel_time(
                                rid-1, start_id, True)
                            if time - timedelta(minutes=(D-deviation_s_p)) + s_n_travel_time <= pickup_time:
                                push_back = time + s_n_travel_time - \
                                    pickup_time if pickup_time - time - \
                                    s_n_travel_time < timedelta(0) else 0

                                # update backward
                                if push_back:
                                    temp_route_plan[introduced_vehicle], activated_checks = self.update_backward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_p=start_p, push_back=push_back, activated_checks=activated_checks)

                                # check max ride time between nodes
                                activated_checks = self.check_max_ride_time(
                                    vehicle_route=temp_route_plan[introduced_vehicle], activated_checks=activated_checks)

                                # check capacities
                                activated_checks = self.check_capacities(
                                    vehicle_route=temp_route_plan[introduced_vehicle], request=request, rid=rid, start_id=start_idx+1, dropoff_id=start_idx+2, activated_checks=activated_checks)

                                if not activated_checks:
                                    # add pickup node
                                    pickup_id, vehicle_route = self.add_node(
                                        vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=pickup_time, pickup=True, rid=rid, node_idx=start_idx)

                                    # add dropoff node
                                    dropoff_id, vehicle_route = self.add_node(
                                        vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=request["Requested Dropoff Time"], pickup=False, rid=rid, node_idx=start_idx+1)

                                    self.check_remove(rid, request)

                                    # calculate change in objective
                                    change_objective = self.heuristic.delta_objective(
                                        temp_route_plan)
                                    possible_insertions[change_objective] = temp_route_plan
                        else:
                            end_node_p, end_time_p, end_deviation_p, end_pass_p, end_wheel_p = vehicle_route[
                                start_idx+1]

                            end_idx = 0
                            vehicle_route = temp_route_plan[introduced_vehicle]
                            for idx, (node, time, deviation, passenger, wheelchair) in enumerate(vehicle_route):
                                if time <= request["Requested Dropoff Time"]:
                                    end_idx = idx

                            start_node_d, start_time_d, start_deviation_d, start_pass_d, start_wheelchair_d = vehicle_route[
                                end_idx]
                            if end_idx == len(vehicle_route)-1:
                                end_node_d = None

                            else:
                                end_node_d, end_time_d, end_deviation_d, end_pass_d, end_wheelchair_d = vehicle_route[
                                    end_idx + 1]

                            # try to add pickup and dropoff node
                            s_p = s_p_node % int(s_p_node)
                            e_p = end_node_p % int(end_node_p)
                            s_d = start_node_d % int(start_node_d)
                            e_d = end_node_d % int(
                                end_node_d) if end_node_d else None
                            start_id_p = int(
                                s_p_node - 0.5 - 1 + self.heuristic.n if s_p else s_p_node - 1)
                            end_id_p = int(
                                end_node_p - 0.5 - 1 + self.heuristic.n if e_p else end_node_p - 1)
                            start_id_d = int(
                                start_node_d - 0.5 - 1 + self.heuristic.n if s_d else start_node_d - 1)
                            if e_d:
                                end_id_d = int(
                                    end_node_d - 0.5 - 1 + self.heuristic.n if e_d else end_node_d - 1)

                            s_p_travel_time = self.heuristic.travel_time(
                                rid-1, start_id_p, True)
                            p_e_travel_time = self.heuristic.travel_time(
                                rid-1, end_id_p, True)
                            s_d_travel_time = self.heuristic.travel_time(
                                rid-1 + self.heuristic.n, start_id_d, True)
                            if e_d:
                                d_e_travel_time = self.heuristic.travel_time(
                                    rid-1 + self.heuristic.n, end_id_d, True)

                            if time - timedelta(minutes=(D-deviation_s_p)) + s_p_travel_time <= pickup_time and pickup_time + timedelta(minutes=S) + p_e_travel_time <= end_time_p + timedelta(minutes=(D-end_deviation_p)) and start_time_d - timedelta(minutes=(D-start_deviation_d)) + s_d_travel_time <= request["Requested Dropoff Time"]:
                                push_forward_p = pickup_time + \
                                    timedelta(minutes=S) + p_e_travel_time - end_time_p if end_time_p - pickup_time - \
                                    timedelta(minutes=S) - p_e_travel_time < timedelta(0) else 0
                                push_back_p = s_p_time + s_p_travel_time - pickup_time if pickup_time - \
                                    s_p_time - s_p_travel_time < timedelta(0) else 0
                                push_back_d = time + s_n_travel_time - request["Requested Dropoff Time"] if request["Requested Dropoff Time"] - \
                                    time - s_n_travel_time < timedelta(0) else 0

                                if e_d:
                                    if request["Requested Dropoff Time"] + timedelta(minutes=S) + d_e_travel_time <= end_time + timedelta(minutes=(D-end_deviation)):
                                        push_forward_d = request["Requested Dropoff Time"] + \
                                            timedelta(minutes=S) + n_e_travel_time - end_time if end_time - \
                                            request["Requested Dropoff Time"] - \
                                            timedelta(minutes=S) - n_e_travel_time < timedelta(0) else 0
                                    else:
                                        activated_checks = True

                                # update forward
                                if push_forward_p:
                                    temp_route_plan[introduced_vehicle], activated_checks = self.update_forward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=start_idx, push_forward=push_forward_p, activated_checks=activated_checks)
                                if e_d:
                                    if push_forward_d:
                                        temp_route_plan[introduced_vehicle], activated_checks = self.update_forward(
                                            vehicle_route=temp_route_plan[introduced_vehicle], start_idx=end_idx, push_forward=push_forward_d, activated_checks=activated_checks)

                                # update backward
                                if push_back_p:
                                    temp_route_plan[introduced_vehicle], activated_checks = self.update_backward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=start_idx, push_back=push_back_p, activated_checks=activated_checks)
                                if push_back_d:
                                    temp_route_plan[introduced_vehicle], activated_checks = self.update_backward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=end_idx, push_back=push_back_d, activated_checks=activated_checks)

                                # check max ride time between nodes
                                activated_checks = self.check_max_ride_time(
                                    vehicle_route=temp_route_plan[introduced_vehicle], activated_checks=activated_checks)

                                # check capacities
                                activated_checks = self.check_capacities(
                                    vehicle_route=temp_route_plan[introduced_vehicle], request=request, rid=rid, start_id=start_idx+1, dropoff_id=end_idx+1, activated_checks=activated_checks)

                                if not activated_checks:
                                    # add pickup node
                                    pickup_id, vehicle_route = self.add_node(
                                        vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=pickup_time, pickup=True, rid=rid, node_idx=start_idx)

                                    # add dropoff node
                                    dropoff_id, vehicle_route = self.add_node(
                                        vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=request["Requested Dropoff Time"], pickup=False, rid=rid, node_idx=end_idx)

                                    self.check_remove(rid, request)

                                    # calculate change in objective
                                    change_objective = self.heuristic.delta_objective(
                                        temp_route_plan)
                                    possible_insertions[change_objective] = temp_route_plan

                        # update capacity between pickup and dropoff and check capacity
                        if feasible_request:
                            temp_route_plan[introduced_vehicle] = self.update_capacities(
                                vehicle_route=temp_route_plan[introduced_vehicle], request=request, rid=rid, start_id=pickup_id, dropoff_id=dropoff_id)

        # check if no possible insertions have been made and introduce a new vehicle
        if not len(possible_insertions):
            if self.heuristic.vehicles:
                temp_route_plan = copy.deepcopy(route_plan)
                new_vehicle = self.heuristic.vehicles.pop(0)
                temp_route_plan.append([])
                self.heuristic.introduced_vehicles.add(new_vehicle)
                if not pandas.isnull(request["Requested Pickup Time"]):
                    temp_route_plan[new_vehicle] = self.add_initial_nodes_pickup(
                        request=request, introduced_vehicle=new_vehicle, rid=rid, vehicle_route=temp_route_plan[new_vehicle])

                else:
                    temp_route_plan[introduced_vehicle] = self.add_initial_nodes_dropoff(
                        request=request, introduced_vehicle=new_vehicle, rid=rid, vehicle_route=temp_route_plan[new_vehicle])

                # calculate change in objective
                change_objective = self.heuristic.delta_objective(
                    temp_route_plan)
                possible_insertions[change_objective] = temp_route_plan

            # if no new vehicles available, append the request in an infeasible set
            else:
                self.heuristic.infeasible_set.append((rid, request))

        return possible_insertions[min(possible_insertions.keys())] if len(possible_insertions) else route_plan, min(possible_insertions.keys()) if len(possible_insertions) else timedelta(0)

    def check_remove(self, rid, request):
        if (rid, request) in self.heuristic.infeasible_set:
            self.heuristic.infeasible_set.remove((rid, request))

    def update_backward(self, vehicle_route, start_idx, push_back, activated_checks):
        for idx in range(start_idx, -1, -1):
            n, t, d, p, w = vehicle_route[idx]
            if abs(d) + push_back > D:
                self.heuristic.infeasible_set.append(
                    (rid, request))
                activated_checks = True
                break
            t = t - timedelta(minutes=push_back)
            d = d - push_back
            vehicle_route[idx] = (n, t, d, p, w)
        return vehicle_route, activated_checks

    def update_forward(self, vehicle_route, start_idx, push_forward, activated_checks):
        for idx, (n, t, d, p, w) in enumerate(vehicle_route[start_idx+1:]):
            if abs(d) + push_forward > D:
                self.heuristic.infeasible_set.append(
                    (rid, request))
                activated_checks = True
                break
            t = t + timedelta(minutes=push_forward)
            d = d + push_forward
            vehicle_route[idx] = (n, t, d, p, w)
        return vehicle_route, activated_checks

    def check_max_ride_time(self, vehicle_route, activated_checks):
        nodes = [int(n) for n, t, d, p, w in vehicle_route]
        nodes.remove(0)
        nodes_set = []
        [nodes_set.append(i) for i in nodes if i not in nodes_set]
        for n in nodes_set:
            p_idx = next(i for i, (node, *_)
                         in enumerate(vehicle_route) if node == n)
            d_idx = next(i for i, (node, *_)
                         in enumerate(vehicle_route) if node == n+0.5)
            pn, pickup_time, pd, pp, pw = vehicle_route[p_idx]
            dn, dropoff_time, dd, dp, dw = vehicle_route[d_idx]
            total_time = (dropoff_time - pickup_time).seconds
            max_time = self.heuristic.get_max_travel_time(
                n-1, n-1 + self.heuristic.n)
            if total_time > max_time.total_seconds():
                self.heuristic.infeasible_set.append((
                    rid, request))
                activated_checks = True
                break
        return activated_checks

    def update_capacities(self, vehicle_route, start_id, dropoff_id, request, rid):
        for idx, (n, t, d, p, w) in enumerate(vehicle_route[start_id+1:dropoff_id]):
            p = p + request["Passengers"]
            w = w + request["Wheelchair"]
            vehicle_route[idx] = (n, t, d, p, w)
        return vehicle_route

    def check_capacities(self, vehicle_route, start_id, dropoff_id, request, rid, activated_checks):
        for idx, (n, t, d, p, w) in enumerate(vehicle_route[start_id+1:dropoff_id]):
            if p + request["Passengers"] > P or w + request["Wheelchair"] > W:
                self.heuristic.infeasible_set.append(
                    (rid, request))
                activated_checks = True
                break
        return activated_checks

    def add_initial_nodes_pickup(self, request, introduced_vehicle, rid, vehicle_route):
        service_time = request["Requested Pickup Time"] - self.heuristic.travel_time(
            rid-1, 2*self.heuristic.n + introduced_vehicle, True)
        vehicle_route.append(
            (0, service_time, 0, 0, 0))
        vehicle_route.append(
            (rid,
                request["Requested Pickup Time"], 0, request["Number of Passengers"], request["Wheelchair"])
        )
        travel_time = self.heuristic.travel_time(
            rid-1, self.heuristic.n + rid - 1, True)
        vehicle_route.append(
            (rid + 0.5,
                request["Requested Pickup Time"]+travel_time, 0, 0, 0)
        )
        return vehicle_route

    def add_initial_nodes_dropoff(self, request, introduced_vehicle, rid, vehicle_route):
        travel_time = self.heuristic.travel_time(
            rid-1, self.heuristic.n + rid - 1, True)
        service_time = request["Requested Dropoff Time"] - travel_time - self.heuristic.travel_time(
            rid-1, 2*self.heuristic.n + introduced_vehicle, True)
        vehicle_route.append(
            (0, service_time, 0, 0, 0))
        vehicle_route.append(
            (rid, request["Requested Dropoff Time"]-travel_time, 0,
                request["Number of Passengers"], request["Wheelchair"])
        )
        vehicle_route.append(
            (rid + 0.5, request["Requested Dropoff Time"], 0, 0, 0)
        )
        return vehicle_route

    def preprocess_check(self, rid, vehicle_route):
        preprocessed_check_activated = False
        if self.heuristic.preprocessed[rid-1]:
            nodes = [int(n) for n, t, d, p, w in vehicle_route]
            for n in nodes:
                if n in self.heuristic.preprocessed[rid-1]:
                    # the new request cannot exist in the same route as another request already in the route
                    preprocessed_check_activated = True
                    break
        return preprocessed_check_activated

    def add_node(self, vehicle_route, request, time, pickup, rid, node_idx):
        p = vehicle_route[node_idx][3] + \
            request["Number of Passengers"] if pickup else vehicle_route[node_idx][3] - \
            request["Number of Passengers"]
        w = vehicle_route[node_idx][4] + \
            request["Wheelchair"] if pickup else vehicle_route[node_idx][4] - \
            request["Wheelchair"]
        if pickup:
            vehicle_route.insert(
                node_idx+1, (rid, time, 0, p, w))
        else:
            vehicle_route.insert(
                node_idx+1, (rid+0.5, time, 0, p, w))
        return node_idx+1, vehicle_route
