import copy
import math
import numpy as np
import pandas
import sklearn.metrics
from math import radians
from alns import *
from decouple import config
from datetime import datetime, timedelta
from heuristic.construction.heuristic_config import *

"""
NOTE: we only try to add it after the first node that is closest in time
"""


class RepairGenerator:
    def __init__(self, construction_heuristic, infeasible_set):
        self.heuristic = construction_heuristic
        self.infeasible_set = infeasible_set

    def generate_insertions(self, route_plan, request, rid):
        # regne ut introduced vehicles

        possible_insertions = {}  # dict: delta objective --> route plan
        for introduced_vehicle in self.heuristic.introduced_vehicles:
            # generate all possible insertions

            if not route_plan[introduced_vehicle]:
                # it is trivial to add the new request
                temp_route_plan = copy.deepcopy(route_plan)
                if not pandas.isnull(request["Requested Pickup Time"]):
                    temp_route_plan[introduced_vehicle] = self.add_initial_nodes_pickup(request=request, introduced_vehicle=introduced_vehicle, rid=rid, vehicle_route=temp_route_plan[introduced_vehicle]
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
                        test_vehicle_route = copy.deepcopy(vehicle_route)
                        for idx, (node, time, deviation, passenger, wheelchair, _) in enumerate(vehicle_route):
                            if time <= request["Requested Pickup Time"]:
                                start_idx = idx

                        s_p_node, s_p_time, s_p_d, s_p_p, s_p_w, _ = vehicle_route[start_idx]
                        if start_idx == len(vehicle_route) - 1:
                            # there is no other end node, and we only need to check the travel time from start to the node
                            s_p = s_p_node % int(s_p_node)
                            start_id = int(
                                s_p_node - 0.5 - 1 + self.heuristic.n if s_p else s_p_node - 1)
                            s_p_travel_time = self.heuristic.travel_time(
                                rid-1, start_id, True)

                            if s_p_time + (L_D-s_p_d) + s_p_travel_time <= request["Requested Pickup Time"]:
                                push_back = s_p_time + s_p_travel_time - request["Requested Pickup Time"] if request["Requested Pickup Time"] - \
                                    s_p_time - s_p_travel_time < timedelta(0) else 0

                                # check backward
                                if push_back:
                                    activated_checks = self.check_backward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=start_idx, push_back=push_back, activated_checks=activated_checks, rid=rid, request=request)

                                # check capacities
                                activated_checks = self.check_capacities(
                                    vehicle_route=temp_route_plan[introduced_vehicle], request=request, rid=rid,
                                    start_id=start_idx + 1, dropoff_id=start_idx + 2,
                                    activated_checks=activated_checks)

                                if not activated_checks:

                                    # update backward to test vehicle route
                                    if push_back:
                                        test_vehicle_route = self.update_backward(
                                            vehicle_route=test_vehicle_route, start_idx=start_idx, push_back=push_back, activated_checks=activated_checks, rid=rid, request=request)

                                    # add pickup node to test vehicle route
                                    pickup_id, test_vehicle_route = self.add_node(
                                        vehicle_route=test_vehicle_route, request=request, time=request["Requested Pickup Time"], pickup=True, rid=rid, node_idx=start_idx)

                                    # add dropoff node to test vehicle route
                                    dropoff_id, test_vehicle_route = self.add_node(
                                        vehicle_route=test_vehicle_route, request=request, time=dropoff_time, pickup=False, rid=rid, node_idx=start_idx+1)

                                    # check max ride time between nodes on test vehicle route
                                    activated_checks = self.check_max_ride_time(
                                        vehicle_route=test_vehicle_route,
                                        activated_checks=activated_checks, rid=rid, request=request)

                                    if not activated_checks:
                                        # can update temp route plan
                                        # update backward
                                        if push_back:
                                            temp_route_plan[introduced_vehicle] = self.update_backward(
                                                vehicle_route=temp_route_plan[introduced_vehicle], start_idx=start_idx,
                                                push_back=push_back, activated_checks=activated_checks, rid=rid,
                                                request=request)

                                        # add pickup node
                                        pickup_id, vehicle_route = self.add_node(
                                            vehicle_route=temp_route_plan[introduced_vehicle], request=request,
                                            time=request["Requested Pickup Time"], pickup=True, rid=rid,
                                            node_idx=start_idx)

                                        # add dropoff node
                                        dropoff_id, vehicle_route = self.add_node(
                                            vehicle_route=temp_route_plan[introduced_vehicle], request=request, time=dropoff_time,
                                            pickup=False, rid=rid, node_idx=start_idx + 1)

                                        feasible_request = True

                                        self.check_remove(rid, request)

                                        # calculate change in objective
                                        change_objective = self.heuristic.delta_objective(
                                            temp_route_plan)
                                        possible_insertions[change_objective] = temp_route_plan
                        else:
                            e_p_node, e_p_time, e_p_d, e_p_p, e_p_w, _ = vehicle_route[start_idx + 1]
                            s_p = s_p_node % int(s_p_node)
                            e_p = e_p_node % int(e_p_node)
                            start_id_p = int(
                                s_p_node - 0.5 - 1 + self.heuristic.n if s_p else s_p_node - 1)
                            end_id_p = int(
                                e_p_node - 0.5 - 1 + self.heuristic.n if e_p else e_p_node - 1)

                            s_p_travel_time = self.heuristic.travel_time(
                                rid - 1, start_id_p, True)
                            p_e_travel_time = self.heuristic.travel_time(
                                rid - 1, end_id_p, True)

                            if s_p_time + (L_D - s_p_d) + s_p_travel_time <= request["Requested Pickup Time"] and \
                                request["Requested Pickup Time"] + timedelta(
                                    minutes=S) + p_e_travel_time <= e_p_time + (U_D - e_p_d):
                                push_forward_p = request["Requested Pickup Time"] + timedelta(
                                    minutes=S) + p_e_travel_time - e_p_time if e_p_time - request["Requested Pickup Time"] - \
                                    timedelta(
                                    minutes=S) - p_e_travel_time < timedelta(
                                    0) else 0
                                push_back_p = s_p_time + s_p_travel_time - request["Requested Pickup Time"] if request["Requested Pickup Time"] - s_p_time - s_p_travel_time < timedelta(
                                    0) else 0

                                # check forward
                                if push_forward_p:
                                    activated_checks = self.check_forward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=start_idx,
                                        push_forward=push_forward_p, activated_checks=activated_checks, rid=rid,
                                        request=request)

                                # check backward
                                if push_back_p:
                                    activated_checks = self.check_backward(
                                        vehicle_route=temp_route_plan[introduced_vehicle], start_idx=start_idx,
                                        push_back=push_back_p, activated_checks=activated_checks, rid=rid,
                                        request=request)

                                if not activated_checks:
                                    # update forward
                                    if push_forward_p:
                                        test_vehicle_route = self.update_forward(
                                            vehicle_route=test_vehicle_route, start_idx=start_idx,
                                            push_forward=push_forward_p, activated_checks=activated_checks, rid=rid,
                                            request=request)

                                    # update backward
                                    if push_back_p:
                                        test_vehicle_route = self.update_backward(
                                            vehicle_route=test_vehicle_route, start_idx=start_idx,
                                            push_back=push_back_p, activated_checks=activated_checks, rid=rid,
                                            request=request)

                                    # add pickup node to test vehicle route
                                    pickup_id, test_vehicle_route = self.add_node(
                                        vehicle_route=test_vehicle_route, request=request,
                                        time=request["Requested Pickup Time"], pickup=True, rid=rid,
                                        node_idx=start_idx)

                                    s_p_node, s_p_time, s_p_d, s_p_p, s_p_w, _ = test_vehicle_route[
                                        start_idx]
                                    e_p_node, e_p_time, e_p_d, e_p_p, e_p_w, _ = test_vehicle_route[
                                        start_idx + 1]
                                    end_idx = 0
                                    for idx, (node, time, deviation, passenger, wheelchair, _) in enumerate(test_vehicle_route):
                                        if time <= dropoff_time:
                                            end_idx = idx

                                    s_d_node, s_d_time, s_d_d, s_d_p, s_d_w, _ = test_vehicle_route[
                                        end_idx]

                                    if end_idx == len(test_vehicle_route) - 1:
                                        # there is no other end node, and we only need to check the travel time from start to the node
                                        e_d_node = None
                                    else:
                                        e_d_node, e_d_time, e_d_d, e_d_p, e_d_w, _ = test_vehicle_route[
                                            end_idx + 1]

                                    s_d = s_d_node % int(s_d_node)
                                    e_d = e_d_node % int(
                                        e_d_node) if e_d_node else None

                                    start_id_d = int(
                                        s_d_node - 0.5 - 1 + self.heuristic.n if s_d else s_d_node - 1)
                                    if e_d_node:
                                        end_id_d = int(
                                            e_d_node - 0.5 - 1 + self.heuristic.n if e_d else e_d_node - 1)

                                    s_d_travel_time = self.heuristic.travel_time(
                                        rid - 1 + self.heuristic.n, start_id_d, True)
                                    d_e_travel_time = self.heuristic.travel_time(
                                        rid - 1 + self.heuristic.n, end_id_d, True) if e_d_node else None

                                    if s_p_time + (
                                            L_D - s_p_d) + s_p_travel_time <= request["Requested Pickup Time"] and request["Requested Pickup Time"] + timedelta(
                                            minutes=S) + p_e_travel_time <= e_p_time + (U_D - e_p_d) and s_d_time + (
                                            L_D - s_d_d) + s_d_travel_time <= dropoff_time:
                                        push_back_d = s_d_time + s_d_travel_time - dropoff_time if \
                                            dropoff_time - \
                                            s_d_time - s_d_travel_time < timedelta(
                                                0) else 0
                                        if e_d_node:
                                            if dropoff_time + timedelta(
                                                    minutes=S) + d_e_travel_time <= e_d_time + (
                                                    U_D - e_d_d):
                                                push_forward_d = dropoff_time + \
                                                    timedelta(
                                                        minutes=S) + d_e_travel_time - e_d_time if e_d_time - \
                                                    dropoff_time - \
                                                    timedelta(
                                                        minutes=S) - d_e_travel_time < timedelta(
                                                        0) else 0
                                            else:
                                                activated_checks = True
                                                push_forward_d = None

                                        if e_d_node:
                                            if push_forward_d:
                                                activated_checks = self.check_forward(
                                                    vehicle_route=test_vehicle_route,
                                                    start_idx=end_idx, push_forward=push_forward_d,
                                                    activated_checks=activated_checks, rid=rid, request=request)

                                        # check backward
                                        if push_back_d:
                                            activated_checks = self.check_backward(
                                                vehicle_route=test_vehicle_route, start_idx=start_idx,
                                                push_back=push_back_d, activated_checks=activated_checks,
                                                rid=rid,
                                                request=request)

                                        if not activated_checks:
                                            # update forward
                                            if e_d_node:
                                                if push_forward_d:
                                                    test_vehicle_route = self.update_forward(
                                                        vehicle_route=test_vehicle_route, start_idx=start_idx,
                                                        push_forward=push_forward_d, activated_checks=activated_checks,
                                                        rid=rid,
                                                        request=request)

                                            # update backward
                                            if push_back_d:
                                                test_vehicle_route = self.update_backward(
                                                    vehicle_route=test_vehicle_route, start_idx=start_idx,
                                                    push_back=push_back_d, activated_checks=activated_checks,
                                                    rid=rid,
                                                    request=request)

                                            # add dropoff node to test vehicle route
                                            dropoff_id, test_vehicle_route = self.add_node(
                                                vehicle_route=test_vehicle_route, request=request,
                                                time=dropoff_time, pickup=False, rid=rid,
                                                node_idx=end_idx)

                                            # check capacities
                                            activated_checks = self.check_capacities(
                                                vehicle_route=test_vehicle_route, request=request,
                                                rid=rid,
                                                start_id=start_idx + 1, dropoff_id=end_idx + 1,
                                                activated_checks=activated_checks)

                                            # check max ride time between nodes
                                            activated_checks = self.check_max_ride_time(
                                                vehicle_route=test_vehicle_route,
                                                activated_checks=activated_checks, rid=rid, request=request)

                                            if not activated_checks:
                                                # add pickup node
                                                pickup_id, vehicle_route = self.add_node(
                                                    vehicle_route=temp_route_plan[introduced_vehicle], request=request,
                                                    time=request["Requested Pickup Time"], pickup=True, rid=rid,
                                                    node_idx=start_idx)

                                                # add dropoff node
                                                dropoff_id, vehicle_route = self.add_node(
                                                    vehicle_route=temp_route_plan[introduced_vehicle],
                                                    request=request,
                                                    time=dropoff_time, pickup=False, rid=rid,
                                                    node_idx=end_idx)

                                                feasible_request = True

                                                self.check_remove(rid, request)

                                                # calculate change in objective
                                                change_objective = self.heuristic.delta_objective(
                                                    temp_route_plan)
                                                possible_insertions[change_objective] = temp_route_plan

                        # update capacity between pickup and dropoff
                        if feasible_request:
                            temp_route_plan[introduced_vehicle] = self.update_capacities(
                                vehicle_route=temp_route_plan[introduced_vehicle], request=request, rid=rid,
                                start_id=pickup_id, dropoff_id=dropoff_id)

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

                # calculate change in objective
                change_objective = self.heuristic.delta_objective(
                    temp_route_plan)
                possible_insertions[change_objective] = temp_route_plan

            # if no new vehicles available, append the request in an infeasible set
            else:
                if (rid, request) not in self.infeasible_set:
                    self.infeasible_set.append((rid, request))

        return possible_insertions[min(possible_insertions.keys())] if len(possible_insertions) else route_plan, min(possible_insertions.keys()) if len(possible_insertions) else timedelta(0)

    def check_remove(self, rid, request):
        if (rid, request) in self.infeasible_set:
            self.infeasible_set.remove((rid, request))

    def check_backward(self, vehicle_route, start_idx, push_back, activated_checks, rid, request):
        for idx in range(start_idx, -1, -1):
            n, t, d, p, w, _ = vehicle_route[idx]
            if d is not None and d - push_back < L_D and (rid, request) not in self.infeasible_set:
                activated_checks = True
                break
        return activated_checks

    def update_backward(self, vehicle_route, start_idx, push_back, activated_checks, rid, request):
        for idx in range(start_idx, -1, -1):
            n, t, d, p, w, r = vehicle_route[idx]
            if d is not None:
                vehicle_route[idx] = (n, t, d, p, w, r)
        return vehicle_route

    def check_forward(self, vehicle_route, start_idx, push_forward, activated_checks, rid, request):
        idx = start_idx + 1
        for n, t, d, p, w, _ in vehicle_route[start_idx+1:]:
            # since updating happens at start_idx + 1, there is no need to check for depot
            if d + push_forward > U_D and (rid, request) not in self.infeasible_set:
                activated_checks = True
                break
        return activated_checks

    def update_forward(self, vehicle_route, start_idx, push_forward, activated_checks, rid, request):
        idx = start_idx + 1
        for n, t, d, p, w, r in vehicle_route[start_idx+1:]:
            # since updating happens at start_idx + 1, there is no need to check for depot
            t = t + push_forward
            d = d + push_forward
            vehicle_route[idx] = (n, t, d, p, w, r)
            idx += 1
        return vehicle_route

    def check_max_ride_time(self, vehicle_route, activated_checks, rid, request):
        nodes = [int(n) for n, t, d, p, w, _ in vehicle_route]
        nodes.remove(0)
        nodes_set = []
        [nodes_set.append(i) for i in nodes if i not in nodes_set]
        for n in nodes_set:
            p_idx = next(i for i, (node, *_)
                         in enumerate(vehicle_route) if node == n)
            d_idx = next(i for i, (node, *_)
                         in enumerate(vehicle_route) if node == n+0.5)
            pn, pickup_time, pd, pp, pw, _ = vehicle_route[p_idx]
            dn, dropoff_time, dd, dp, dw, _ = vehicle_route[d_idx]
            total_time = (dropoff_time - pickup_time).seconds
            max_time = self.heuristic.get_max_travel_time(
                n-1, n-1 + self.heuristic.n)
            if total_time > max_time.total_seconds():
                activated_checks = True
                break
        return activated_checks

    def update_capacities(self, vehicle_route, start_id, dropoff_id, request, rid):
        idx = start_id+1
        for n, t, d, p, w, _ in vehicle_route[start_id+1:dropoff_id]:
            p = p + request["Number of Passengers"]
            w = w + request["Wheelchair"]
            vehicle_route[idx] = (n, t, d, p, w, _)
            idx += 1
        return vehicle_route

    def check_capacities(self, vehicle_route, start_id, dropoff_id, request, rid, activated_checks):
        for n, t, d, p, w, _ in vehicle_route[start_id+1:dropoff_id]:
            if p + request["Number of Passengers"] > P and (rid, request) not in self.infeasible_set or w + request["Wheelchair"] > W and (rid, request) not in self.infeasible_set:
                activated_checks = True
                break
        return activated_checks

    def add_initial_nodes_pickup(self, request, introduced_vehicle, rid, vehicle_route):
        service_time = request["Requested Pickup Time"] - self.heuristic.travel_time(
            rid-1, 2*self.heuristic.n + introduced_vehicle, True)
        vehicle_route.append(
            (0, service_time, None, 0, 0, None))
        vehicle_route.append(
            (rid,
                request["Requested Pickup Time"], timedelta(0), request["Number of Passengers"], request["Wheelchair"], request)
        )
        travel_time = self.heuristic.travel_time(
            rid-1, self.heuristic.n + rid - 1, True)
        vehicle_route.append(
            (rid + 0.5,
                request["Requested Pickup Time"]+travel_time, timedelta(0), 0, 0, request)
        )
        return vehicle_route

    def preprocess_check(self, rid, vehicle_route):
        preprocessed_check_activated = False
        if self.heuristic.preprocessed[rid-1]:
            nodes = [int(n) for n, t, d, p, w, _ in vehicle_route]
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
                node_idx+1, (rid, time, timedelta(0), p, w, request))
        else:
            vehicle_route.insert(
                node_idx+1, (rid+0.5, time, timedelta(0), p, w, request))
        return node_idx+1, vehicle_route
