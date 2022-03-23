import copy
import math
import sys

import numpy.random as rnd
from datetime import datetime
import pandas as pd
from datetime import timedelta
import traceback
from heuristic.construction.construction import ConstructionHeuristic
from config.construction_config import *
from heuristic.improvement.reopt.reopt_repair_generator import ReOptRepairGenerator


class ReOptOperators:
    def __init__(self, alns, sim_clock):
        self.destruction_degree = alns.destruction_degree
        self.constructor = alns.constructor
        self.T_ij = self.constructor.T_ij
        self.reopt_repair_generator = ReOptRepairGenerator(self.constructor)
        self.sim_clock = sim_clock

    # Find number of requests to remove based on degree of destruction
    def nodes_to_remove(self, route_plan):

        # Count number of requests in route_plan
        total_requests = 0
        for row in route_plan:
            for col in row:
                if col[0]:
                    total_requests += 0.5

        # Calculate number of requests to remove
        num_remove = math.ceil(total_requests * self.destruction_degree)
        return num_remove

    def random_removal(self, current_route_plan, current_infeasible_set):
        destroyed_route_plan = copy.deepcopy(current_route_plan)
        to_remove = []
        removed_requests = []
        index_removed_requests = []
        possible_removals = self.find_possible_removals(destroyed_route_plan)
        empty = 0
        for vehicle in possible_removals:
            empty += len(vehicle)

        if not empty:
            return current_route_plan, removed_requests, index_removed_requests, False

        # Number of requests to remove
        num_remove = self.nodes_to_remove(possible_removals)

        # Find the requests to remove
        while len(to_remove)/2 < num_remove:

            # Pick random node in route plan to remove and to compare other nodes to
            rows = [i for i in range(0, len(possible_removals))]
            rnd.shuffle(rows)

            for row in rows:
                if len(possible_removals[row]) < 3:
                    continue
                elif len(possible_removals[row]) == 3:
                    col = 1
                    break
                else:
                    col = rnd.randint(
                        1, len(possible_removals[row]))
                    break
            node = possible_removals[row][col]
            destroy_node = destroyed_route_plan[row][node[6]]

            # Find col-index of associated pickup/drop-off node
            index, pickup = self.find_associated_node(
                row, col, possible_removals)
            associated_node = possible_removals[row][index]
            destroy_associated_node = destroyed_route_plan[row][associated_node[6]]

            # Skip already added nodes
            if [node, row, destroy_node] in to_remove or [associated_node, row, destroy_associated_node] in to_remove:
                continue

            # Add both pickup and drop-off node to to_remove
            to_remove.append([node, row, destroy_node])
            to_remove.append([associated_node, row, destroy_associated_node])

        # Remove nearest nodes from destroyed route plan and from possible_removals
        for n in to_remove:
            index_removed_requests.append(
                (n[0][0], n[1], n[0][6]))
        for n in to_remove:
            possible_removals[n[1]].remove(n[0])
            destroyed_route_plan[n[1]].remove(n[2])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    def worst_deviation_removal(self, current_route_plan, current_infeasible_set):
        destroyed_route_plan = copy.deepcopy(current_route_plan)
        to_remove = []
        removed_requests = []
        index_removed_requests = []
        possible_removals = self.find_possible_removals(destroyed_route_plan)
        empty = 0
        for vehicle in possible_removals:
            empty += len(vehicle)

        if not empty:
            return current_route_plan, removed_requests, index_removed_requests, False

        # Number of requests to remove
        num_remove = self.nodes_to_remove(possible_removals)

        # Find the requests to remove
        for j in range(num_remove):
            worst_deviation = timedelta(0)
            worst_node = None

            for row in range(len(possible_removals)):
                for col in range(1, len(possible_removals[row])):

                    temp = possible_removals[row][col]
                    destroyed_temp = destroyed_route_plan[row][temp[6]]

                    # Skip already added nodes
                    if [temp, row, destroyed_temp] in to_remove:
                        continue

                    # Find associated drop off/pickup node
                    index, pickup = self.find_associated_node(
                        row, col, possible_removals)
                    associated_temp = possible_removals[row][index]
                    destroyed_associated_temp = destroyed_route_plan[row][associated_temp[6]]

                    temp_deviation = temp[2]
                    associated_temp_deviation = associated_temp[2]

                    if temp_deviation < timedelta(0):
                        temp_deviation = timedelta(
                            seconds=-temp_deviation.total_seconds())

                    if associated_temp_deviation < timedelta(0):
                        associated_temp_deviation = timedelta(
                            seconds=-associated_temp_deviation.total_seconds())

                    # Calculate total deviation for request
                    deviation = temp_deviation + associated_temp_deviation

                    # Update worst deviation so far
                    if deviation > worst_deviation and deviation > timedelta(0):
                        worst_deviation = deviation
                        worst_node = [temp, row, destroyed_temp]
                        worst_associated_node = [
                            associated_temp, row, destroyed_associated_temp]

            # Add node with worst deviation to list of nodes to remove

            if worst_node is not None and worst_node in to_remove:
                continue
            if worst_node is not None:
                to_remove.append(worst_node)
                to_remove.append(worst_associated_node)

        # If not enough nodes have deviation > 0, remove the rest randomly
        if len(to_remove)/2 < num_remove:
            to_remove = self.worst_deviation_random_removal(
                destroyed_route_plan, possible_removals, num_remove, to_remove)

        # Remove nearest nodes from destroyed route plan and from possible_removals
        for n in to_remove:
            index_removed_requests.append(
                (n[0][0], n[1], n[0][6]))
        for n in to_remove:
            possible_removals[n[1]].remove(n[0])
            destroyed_route_plan[n[1]].remove(n[2])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    # Related in travel time
    def distance_related_removal(self, current_route_plan, current_infeasible_set):
        destroyed_route_plan = copy.deepcopy(current_route_plan)
        removed_requests = []
        index_removed_requests = []
        possible_removals = self.find_possible_removals(destroyed_route_plan)
        empty = 0
        for vehicle in possible_removals:
            empty += len(vehicle)

        if not empty:
            return current_route_plan, removed_requests, index_removed_requests, False

        # Number of requests to remove
        num_remove = self.nodes_to_remove(possible_removals)

        if len(current_infeasible_set) != 0:
            # Pick random node in infeasible_set to compare other nodes to - always pickup nodes
            initial_node = current_infeasible_set[rnd.randint(
                0, len(current_infeasible_set))]
            node = self.get_pickup(initial_node)
            pickup = True

            # Find associated node - dropoff node
            associated_node = self.get_dropoff(initial_node)

            to_remove = []

        else:
            # Pick random node in route plan to remove and to compare other nodes to
            rows = [i for i in range(0, len(possible_removals))]
            rnd.shuffle(rows)

            for row_index in rows:
                if len(possible_removals[row_index]) < 3:
                    continue
                elif len(possible_removals[row_index]) == 3:
                    col_index = 1
                    break
                else:
                    col_index = rnd.randint(
                        1, len(possible_removals[row_index]))
                    break
            node = possible_removals[row_index][col_index]
            destroy_node = destroyed_route_plan[row_index][node[6]]

            # Find associated node
            index, pickup = self.find_associated_node(
                row_index, col_index, possible_removals)
            associated_node = possible_removals[row_index][index]
            destroy_associated_node = destroyed_route_plan[row_index][associated_node[6]]

            # List of nodes to remove
            to_remove = [[node, row_index, destroy_node], [
                associated_node, row_index, destroy_associated_node]]

            # Remaining number of nodes to remove
            num_remove -= 1

        # Find the requests to remove
        for j in range(num_remove):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(possible_removals)):
                for col in range(1, len(possible_removals[row])):

                    # Drop off/pickup of request to compare
                    temp = possible_removals[row][col]
                    destroyed_temp = destroyed_route_plan[row][temp[6]]

                    # Skip already added nodes
                    if [temp, row, destroyed_temp] in to_remove:
                        continue

                    # Find associated drop off/pickup node of request to compare
                    temp_index, temp_pickup = self.find_associated_node(
                        row, col, possible_removals)
                    associated_temp = possible_removals[row][temp_index]
                    destroyed_associated_temp = destroyed_route_plan[row][associated_temp[6]]

                    # Find difference in distance between pickup and drop-off of requests
                    if (temp_pickup == pickup) & pickup:
                        diff = self.travel_time_difference(temp[0], node[0])

                    elif (temp_pickup == pickup) & (not pickup):
                        diff = self.travel_time_difference(
                            associated_temp[0], associated_node[0])

                    elif (temp_pickup != pickup) & pickup:
                        diff = self.travel_time_difference(
                            associated_temp[0], node[0])

                    else:
                        diff = self.travel_time_difference(
                            temp[0], associated_node[0])

                    # Compare with smallest difference in current iteration
                    if diff < best_diff:
                        best_diff = diff
                        nearest_node = [temp, row, destroyed_temp]
                        nearest_associated_node = [
                            associated_temp, row, destroyed_associated_temp]

            to_remove.append(nearest_node)
            to_remove.append(nearest_associated_node)

        # Remove nearest nodes from destroyed route plan and from possible_removals
        for n in to_remove:
            index_removed_requests.append(
                (n[0][0], n[1], n[0][6]))
        for n in to_remove:
            possible_removals[n[1]].remove(n[0])
            destroyed_route_plan[n[1]].remove(n[2])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    # Related in service time
    def time_related_removal(self, current_route_plan, current_infeasible_set):
        destroyed_route_plan = copy.deepcopy(current_route_plan)
        removed_requests = []
        index_removed_requests = []
        possible_removals = self.find_possible_removals(destroyed_route_plan)
        empty = 0
        for vehicle in possible_removals:
            empty += len(vehicle)

        if not empty:
            return current_route_plan, removed_requests, index_removed_requests, False

        # Number of requests to remove
        num_remove = self.nodes_to_remove(possible_removals)

        if len(current_infeasible_set) != 0:
            # Pick random node in infeasible_set to compare other nodes to - always pickup nodes
            initial_node = current_infeasible_set[rnd.randint(
                0, len(current_infeasible_set))]
            node = self.get_pickup(initial_node)
            pickup = True

            # Find associated node - dropoff node
            associated_node = self.get_dropoff(initial_node)

            to_remove = []

        else:
            # Pick random node in route plan to remove and to compare other nodes to
            rows = [i for i in range(0, len(possible_removals))]
            rnd.shuffle(rows)

            for row_index in rows:
                if len(possible_removals[row_index]) < 3:
                    continue
                elif len(possible_removals[row_index]) == 3:
                    col_index = 1
                    break
                else:
                    col_index = rnd.randint(
                        1, len(possible_removals[row_index]))
                    break
            node = possible_removals[row_index][col_index]
            destroy_node = destroyed_route_plan[row_index][node[6]]

            # Find associated node
            index, pickup = self.find_associated_node(
                row_index, col_index, possible_removals)
            associated_node = possible_removals[row_index][index]
            destroy_associated_node = destroyed_route_plan[row_index][associated_node[6]]

            # List of nodes to remove
            to_remove = [[node, row_index, destroy_node], [
                associated_node, row_index, destroy_associated_node]]

            # Remaining number of nodes to remove
            num_remove -= 1

        # Find the requests to remove
        for j in range(num_remove):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(possible_removals)):
                for col in range(1, len(possible_removals[row])):

                    temp = possible_removals[row][col]
                    destroyed_temp = destroyed_route_plan[row][temp[6]]

                    # Skip already added nodes
                    if [temp, row, destroyed_temp] in to_remove:
                        continue

                    # Find associated drop off/pickup node
                    temp_index, temp_pickup = self.find_associated_node(
                        row, col, possible_removals)
                    associated_temp = possible_removals[row][temp_index]
                    destroyed_associated_temp = destroyed_route_plan[row][associated_temp[6]]

                    # Find difference between pickup-times and drop off-times of requests
                    if temp_pickup == pickup:
                        diff = self.time_difference(
                            temp, node, associated_temp, associated_node)

                    else:
                        diff = self.time_difference(
                            temp, associated_node, associated_temp, node)

                    # Compare with smallest difference in current iteration
                    if diff < best_diff:
                        best_diff = diff
                        nearest_node = [temp, row, destroyed_temp]
                        nearest_associated_node = [
                            associated_temp, row, destroyed_associated_temp]

            to_remove.append(nearest_node)
            to_remove.append(nearest_associated_node)

        # Remove nearest nodes from destroyed route plan and from possible_removals
        for n in to_remove:
            index_removed_requests.append(
                (n[0][0], n[1], n[0][6]))
        for n in to_remove:
            possible_removals[n[1]].remove(n[0])
            destroyed_route_plan[n[1]].remove(n[2])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    # Related in both service time and travel time
    def related_removal(self, current_route_plan, current_infeasible_set):
        destroyed_route_plan = copy.deepcopy(current_route_plan)
        removed_requests = []
        index_removed_requests = []
        possible_removals = self.find_possible_removals(destroyed_route_plan)
        empty = 0
        for vehicle in possible_removals:
            empty += len(vehicle)

        if not empty:
            return current_route_plan, removed_requests, index_removed_requests, False

        # Number of requests to remove
        num_remove = self.nodes_to_remove(possible_removals)

        if len(current_infeasible_set) != 0:
            # Pick random node in infeasible_set to compare other nodes to - always pickup nodes
            initial_node = current_infeasible_set[rnd.randint(
                0, len(current_infeasible_set))]
            node = self.get_pickup(initial_node)
            pickup = True

            # Find associated node - dropoff node
            associated_node = self.get_dropoff(initial_node)

            to_remove = []

        else:
            # Pick random node in route plan to remove and to compare other nodes to
            rows = [i for i in range(0, len(possible_removals))]
            rnd.shuffle(rows)

            for row_index in rows:
                if len(possible_removals[row_index]) < 3:
                    continue
                elif len(possible_removals[row_index]) == 3:
                    col_index = 1
                    break
                else:
                    col_index = rnd.randint(
                        1, len(possible_removals[row_index]))
                    break
            node = possible_removals[row_index][col_index]
            destroy_node = destroyed_route_plan[row_index][node[6]]

            # Find associated node
            index, pickup = self.find_associated_node(
                row_index, col_index, possible_removals)
            associated_node = possible_removals[row_index][index]
            destroy_associated_node = destroyed_route_plan[row_index][associated_node[6]]

            # List of nodes to remove
            to_remove = [[node, row_index, destroy_node], [
                associated_node, row_index, destroy_associated_node]]

            # Remaining number of nodes to remove
            num_remove -= 1

        # Find the requests to remove
        for j in range(num_remove):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(possible_removals)):
                for col in range(1, len(possible_removals[row])):

                    temp = possible_removals[row][col]
                    destroyed_temp = destroyed_route_plan[row][temp[6]]

                    # Skip already added nodes
                    if [temp, row, destroyed_temp] in to_remove:
                        continue

                    # Find associated drop off/pickup node
                    temp_index, temp_pickup = self.find_associated_node(
                        row, col, possible_removals)
                    associated_temp = possible_removals[row][temp_index]
                    destroyed_associated_temp = destroyed_route_plan[row][associated_temp[6]]

                    # Find difference between requests
                    if (temp_pickup == pickup) & pickup:
                        diff_distance = self.travel_time_difference(
                            temp[0], node[0])
                        diff_time = self.time_difference(
                            temp, node, associated_temp, associated_node)

                    elif (temp_pickup == pickup) & (not pickup):
                        diff_distance = self.travel_time_difference(
                            associated_temp[0], associated_node[0])
                        diff_time = self.time_difference(
                            temp, node, associated_temp, associated_node)

                    elif (temp_pickup != pickup) & pickup:
                        diff_distance = self.travel_time_difference(
                            associated_temp[0], node[0])
                        diff_time = self.time_difference(
                            temp, associated_node, associated_temp, node)

                    else:
                        diff_distance = self.travel_time_difference(
                            temp[0], associated_node[0])
                        diff_time = self.time_difference(
                            temp, associated_node, associated_temp, node)

                    diff = diff_distance + diff_time

                    # Compare with smallest difference in current iteration
                    if diff < best_diff:
                        best_diff = diff
                        nearest_node = [temp, row, destroyed_temp]
                        nearest_associated_node = [
                            associated_temp, row, destroyed_associated_temp]

            to_remove.append(nearest_node)
            to_remove.append(nearest_associated_node)

        # Remove nearest nodes from destroyed route plan and from possible_removals
        for n in to_remove:
            index_removed_requests.append(
                (n[0][0], n[1], n[0][6]))
        for n in to_remove:
            possible_removals[n[1]].remove(n[0])
            destroyed_route_plan[n[1]].remove(n[2])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    # Repair operators
    def greedy_repair(self, destroyed_route_plan, removed_requests, initial_infeasible_set, current_route_plan, index_removed_requests):
        unassigned_requests = removed_requests.copy() + initial_infeasible_set.copy()
        unassigned_requests.sort(key=lambda x: x[0])
        route_plan = copy.deepcopy(destroyed_route_plan)
        current_objective = timedelta(0)
        infeasible_set = []
        unassigned_requests = pd.DataFrame(unassigned_requests)
        for i in range(unassigned_requests.shape[0]):
            # while not unassigned_requests.empty:
            rid = unassigned_requests.iloc[i][0]
            request = unassigned_requests.iloc[i][1]
            index_removal = [
                i for i in index_removed_requests if i[0] == rid or i[0] == rid+0.5]

            route_plan, new_objective, infeasible_set = self.reopt_repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=infeasible_set,
                initial_route_plan=current_route_plan, index_removed=index_removal, sim_clock=self.sim_clock, objectives=False)

            # update current objective
            current_objective = new_objective

        return route_plan, current_objective, infeasible_set

    def regret_2_repair(self, destroyed_route_plan, removed_requests, initial_infeasible_set, current_route_plan, index_removed_requests):
        unassigned_requests = removed_requests.copy() + initial_infeasible_set.copy()
        unassigned_requests.sort(key=lambda x: x[0])
        route_plan = copy.deepcopy(destroyed_route_plan)
        current_objective = timedelta(0)
        infeasible_set = []
        unassigned_requests = pd.DataFrame(unassigned_requests)
        regret_values = []
        for i in range(unassigned_requests.shape[0]):
            rid = unassigned_requests.iloc[i][0]
            request = unassigned_requests.iloc[i][1]
            index_removal = [
                i for i in index_removed_requests if i[0] == rid or i[0] == rid+0.5]

            first_objective, second_objective = self.reopt_repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=infeasible_set,
                initial_route_plan=current_route_plan, index_removed=index_removal, sim_clock=self.sim_clock, objectives=True)

            regret_values.append(
                (rid, request, second_objective-first_objective))

        regret_values.sort(key=lambda x: x[2])

        # iterate through requests in order of regret k value
        for i in reversed(regret_values):
            rid = i[0]
            request = i[1]
            index_removal = [
                i for i in index_removed_requests if i[0] == rid or i[0] == rid+0.5]

            route_plan, new_objective, infeasible_set = self.reopt_repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=infeasible_set,
                initial_route_plan=current_route_plan, index_removed=index_removal, sim_clock=self.sim_clock, objectives=False)

            # update current objective
            current_objective = new_objective

        return route_plan, current_objective, infeasible_set

    # Function to find random requests to remove if worst deviation removal does not remove enough
    def worst_deviation_random_removal(self, destroyed_route_plan, possible_removals, num_remove, to_remove):

        # Find the requests to remove
        while len(to_remove)/2 < num_remove:

            # Pick random node in route plan to remove and to compare other nodes to
            rows = [i for i in range(0, len(possible_removals))]
            rnd.shuffle(rows)

            for row in rows:
                if len(possible_removals[row]) < 3:
                    continue
                elif len(possible_removals[row]) == 3:
                    col = 1
                    break
                else:
                    col = rnd.randint(
                        1, len(possible_removals[row]))
                    break
            node = possible_removals[row][col]
            destroy_node = destroyed_route_plan[row][node[6]]

            # Find col-index of associated pickup/drop-off node
            index, pickup = self.find_associated_node(
                row, col, possible_removals)
            associated_node = possible_removals[row][index]
            destroy_associated_node = destroyed_route_plan[row][associated_node[6]]

            # Skip already added nodes
            if [node, row, destroy_node] in to_remove or [associated_node, row, destroy_associated_node] in to_remove:
                continue

            # Add both pickup and drop-off node to to_remove
            to_remove.append([node, row, destroy_node])
            to_remove.append([associated_node, row, destroy_associated_node])

        return to_remove

    # Function to calculate total travel time differences between requests
    def travel_time_difference(self, request_1, request_2):
        num_requests = int(len(self.T_ij) / 2)
        idx_1 = request_1 - 1
        idx_2 = request_2 - 1
        return self.T_ij[idx_1][idx_2] + \
            self.T_ij[idx_1 + num_requests][idx_2 + num_requests] + \
            self.T_ij[idx_1 + num_requests][idx_2] + \
            self.T_ij[idx_1][idx_2 + num_requests]

    # Function to calculate service time differences between requests
    @ staticmethod
    def time_difference(pickup_1, pickup_2, dropoff_1, dropoff_2):
        return abs((pickup_1[1] - pickup_2[1]).total_seconds()) + abs((dropoff_1[1] - dropoff_2[1]).total_seconds())

    # Function to find associated pickup/drop-off of a node.
    @ staticmethod
    def find_associated_node(row, col, route_plan):
        node = route_plan[row][col]

        if node[0] % int(node[0]):
            # Node is drop-off, must find pickup
            pickup = False
            request = node[0] - 0.5
            for index in range(col):
                temp = route_plan[row][index]
                if temp[0] == request:
                    return index, pickup

        else:
            # Node is pickup, must find drop-off
            pickup = True
            request = node[0] + 0.5
            for index in range(len(route_plan[row])):
                temp = route_plan[row][index]
                if temp[0] == request:
                    return index, pickup

    @ staticmethod
    def find_associated_node_infeasible(infeasible_set, node):
        if node[0] % int(node[0]):
            # Node is drop-off, must find pickup
            pickup = False
            request = node[0] - 0.5
            for index in range(len(infeasible_set)):
                temp = infeasible_set[index]
                if temp[0] == request:
                    return index, pickup

        else:
            # Node is pickup, must find drop-off
            pickup = True
            request = node[0] + 0.5
            for index in range(len(infeasible_set)):
                temp = infeasible_set[index]
                if temp[0] == request:
                    return index, pickup

    def get_pickup(self, node):
        # Node is pickup, find requested pickup time or calculated pickup time
        rid = node[0]
        if not pd.isnull(node[1]["Requested Pickup Time"]):
            time = node[1]["Requested Pickup Time"]
        else:
            time = node[1]["Requested Dropoff Time"] - self.constructor.travel_time(
                rid - 1, self.constructor.n + rid - 1, True)

        node = (rid, time)
        return node

    def get_dropoff(self, node):
        # Node is dropoff, find requested dropoff time or calculated dropoff time
        rid = node[0]
        d_rid = rid + 0.5
        if not pd.isnull(node[1]["Requested Dropoff Time"]):
            time = node[1]["Requested Dropoff Time"]
        else:
            time = node[1]["Requested Pickup Time"] + self.constructor.travel_time(
                rid - 1, self.constructor.n + rid - 1, True)

        node = (d_rid, time)
        return node

    def find_possible_removals(self, route_plan):

        possible_removals = [[(rid, t, d, p, w, request, idx + 1) for idx, (rid, t, d, p, w, request) in
                              enumerate(route_plan[vehicle][1:]) if t >= self.sim_clock] for vehicle in
                             range(0, len(route_plan))]

        for vehicle in possible_removals:
            rids = [rid for (rid, t, d, p, w, request, idx) in vehicle]
            for node in vehicle:
                if node[0] % int(node[0]):
                    if not node[0] - 0.5 in rids:
                        possible_removals[possible_removals.index(
                            vehicle)].remove(node)
                else:
                    if not node[0] + 0.5 in rids:
                        possible_removals[possible_removals.index(
                            vehicle)].remove(node)

        return possible_removals
