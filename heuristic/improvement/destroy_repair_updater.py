import math
import copy
import numpy as np
import numpy.random as rnd
from itertools import groupby
from operator import itemgetter
from datetime import datetime, timedelta
from config.initial_improvement_config import *


class Destroy_Repair_Updater:
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def update_solution(self, route_plan, index_removed_requests, removed_requests):
        updated_solution = copy.deepcopy(route_plan)
        # (row, counter) --> sequences([(node,row,col),...])
        index_removed_requests = self.filter_indexes(index_removed_requests)

        for row, c in index_removed_requests.keys():
            new_dict = {}
            vehicle_route = updated_solution[row]
            first_element = index_removed_requests[row, c][0]

            removed_counter = 0
            new_dict = {k: v for k, v in index_removed_requests.items()
                        if k[0] == row and k[1] < c}
            for item in new_dict.items():
                removed_counter += len(item[1])
            left_idx = first_element[2] - \
                1 if c == 0 else first_element[2] - 1 - removed_counter

            right_idx = first_element[2] if c == 0 else first_element[2] - \
                removed_counter

            left_node = vehicle_route[left_idx] if vehicle_route[left_idx][0] != 0 else None
            right_node = vehicle_route[right_idx] if right_idx != len(
                vehicle_route) else None

            if left_node and right_node:

                left_dev = left_node[2] if left_node[2] < timedelta(0) else 0
                right_dev = right_node[2] if right_node[2] > timedelta(
                    0) else 0

                if not right_dev and not left_dev:
                    continue

                l_node = left_node[0] % int(left_node[0])
                r_node = right_node[0] % int(right_node[0])
                left_node_id = int(
                    left_node[0] - 0.5 - 1 + self.heuristic.n if l_node else left_node[0] - 1)
                right_node_id = int(
                    right_node[0] - 0.5 - 1 + self.heuristic.n if r_node else right_node[0] - 1)
                travel_time = self.heuristic.travel_time(
                    left_node_id, right_node_id, True)

                reduction_dev = right_node[1] - timedelta(minutes=S) - \
                    left_node[1] - travel_time if right_node[1] - timedelta(minutes=S) - \
                    left_node[1] - travel_time > timedelta(0) else 0

                if not reduction_dev:
                    continue

                push_backward = None
                push_forward = None

                if left_dev and not right_dev:
                    push_forward = reduction_dev if reduction_dev < -left_dev else -left_dev

                elif right_dev and not left_dev:
                    push_backward = reduction_dev if reduction_dev < right_dev else right_dev

                else:
                    push_forward = left_dev.total_seconds()/(left_dev.total_seconds() - right_dev.total_seconds()) * reduction_dev if left_dev.total_seconds() / \
                        (left_dev.total_seconds() - right_dev.total_seconds()) * reduction_dev < -left_dev else -left_dev
                    push_backward = right_dev.total_seconds()/(left_dev.total_seconds() - right_dev.total_seconds()) * reduction_dev if right_dev.total_seconds()/(
                        left_dev.total_seconds() - right_dev.total_seconds()) * reduction_dev < right_dev else right_dev

                if push_backward:
                    vehicle_route = self.update_backward(
                        vehicle_route, left_idx, push_backward)
                if push_forward:
                    vehicle_route = self.update_forward(
                        vehicle_route, right_idx, push_forward, row)

                updated_solution[row] = vehicle_route

            elif left_node:
                left_dev = left_node[2] if left_node[2] < timedelta(0) else 0

                if not left_dev:
                    continue

                n, t, d, p, w, r = vehicle_route[left_idx]
                t = t - left_dev
                d = d - left_dev
                vehicle_route[left_idx] = n, t, d, p, w, r

                updated_solution[row] = vehicle_route

            elif right_node:
                right_dev = right_node[2] if right_node[2] > timedelta(
                    0) else 0

                if not right_dev:
                    continue

                n, t, d, p, w, r = vehicle_route[right_idx]
                t = t - right_dev
                d = d - right_dev
                vehicle_route[right_idx] = n, t, d, p, w, r

                updated_solution[row] = vehicle_route

        return updated_solution

    def update_backward(self, vehicle_route, start_idx, push_backward, activated_checks, rid, request, introduced_vehicle):
        idx = start_idx
        for n, t, d, p, w, r in vehicle_route[start_idx:]:
            # since updating happens at start_idx + 1, there is no need to check for depot
            if idx > start_idx:
                n_prev, t_prev, d_prev, p_prev, w_prev, r_prev = vehicle_route[idx-1]
                n_node = n % int(n)
                n_prev_node = n_prev % int(n_prev)
                n_node_id = int(
                    n - 0.5 - 1 + self.heuristic.n if n_node else n - 1)
                n_prev_node_id = int(
                    n_prev - 0.5 - 1 + self.heuristic.n if n_prev_node else n_prev - 1)
                travel_time = self.heuristic.travel_time(
                    n_node_id, n_prev_node_id, True)
                push_backward = t - travel_time - t_prev - \
                    timedelta(minutes=S) if t - travel_time - t_prev - \
                    timedelta(minutes=S) > timedelta(0) else timedelta(0)

            if d is not None and push_backward == timedelta(0):
                break

            t = t - push_backward if d > timedelta(0) else t
            d = d - push_backward if d > timedelta(0) else d
            vehicle_route[idx] = (n, t, d, p, w, r)
            idx += 1

        return vehicle_route

    def update_forward(self, vehicle_route, start_idx, push_forward, introduced_vehicle):
        for idx in range(start_idx, -1, -1):
            n, t, d, p, w, r = vehicle_route[idx]
            if idx < start_idx:
                n_next, t_next, d_next, p_next, w_next, r_next = vehicle_route[idx+1]
                n_node = n % int(n) if n > 0 else 0
                n_next_node = n_next % int(n_next)
                n_node_id = int(
                    n - 0.5 - 1 + self.heuristic.n if n_node else n - 1)
                n_node_id = 2*self.heuristic.n + introduced_vehicle if n == 0 else n_node_id
                n_next_node_id = int(
                    n_next - 0.5 - 1 + self.heuristic.n if n_next_node else n_next - 1)
                travel_time = self.heuristic.travel_time(
                    n_node_id, n_next_node_id, True)
                push_forward = t_next - travel_time - t - \
                    timedelta(minutes=S) if t_next - t - travel_time - \
                    timedelta(minutes=S) > timedelta(0) else timedelta(0)

            if push_forward == timedelta(0):
                break

            if d is not None:
                t = t + push_forward if d < timedelta(0) else t
                d = d + push_forward if d < timedelta(0) else d
                vehicle_route[idx] = (n, t, d, p, w, r)
            else:
                t = t + push_forward
                vehicle_route[idx] = (n, t, d, p, w, r)

        return vehicle_route

    def filter_indexes(self, index_removed_requests):
        bundles = dict()  # (row, counter) --> sequences([(node,row,col),...])
        rows = set([i[1] for i in index_removed_requests])
        for row in rows:
            values = [x for x in index_removed_requests if x[1] == row]
            values.sort(key=lambda x: x[2])
            spl = [0]+[i for i in range(1, len(values))
                       if values[i][2]-values[i-1][2] > 1]+[None]
            seqs = [values[b:e] for (b, e) in [(spl[i-1], spl[i])
                                               for i in range(1, len(spl))]]
            c = 0
            for seq in seqs:
                bundles[(row, c)] = seq
                c += 1
        return bundles
