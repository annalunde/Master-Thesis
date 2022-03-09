import math
import numpy.random as rnd
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from datetime import timedelta
from heuristic.construction.heuristic_config import *
from heuristic.improvement.repair_generator import RepairGenerator


class Operators:

    def __init__(self, alns):
        self.destruction_degree = alns.destruction_degree
        self.T_ij = alns.constructor.T_ij
        self.repair_generator = RepairGenerator(alns.constructor)

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

    def random_removal(self, current_route_plan):
        destroyed_route_plan = current_route_plan.copy()
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_route_plan)

        # Find the requests to remove
        for j in range(num_remove):

            # Pick random node to remove
            row = rnd.randint(0, len(destroyed_route_plan))
            while len(destroyed_route_plan[row]) == 1:
                row = rnd.randint(0, len(destroyed_route_plan))
            if len(destroyed_route_plan[row]) == 3:
                col = 1
            else:
                col = rnd.randint(1, len(destroyed_route_plan[row]) - 1)
            node = destroyed_route_plan[row][col]

            # Find col-index of associated pickup/drop-off node
            index, pickup = self.find_associated_node(row, col, destroyed_route_plan)
            associated_node = destroyed_route_plan[row][index]

            # Remove both pickup and drop-off node and add request-index to removed_requests
            if pickup:
                removed_requests.append([node[0], node[5]])
                del destroyed_route_plan[row][index]
                del destroyed_route_plan[row][col]
            else:
                removed_requests.append([associated_node[0], associated_node[5]])
                del destroyed_route_plan[row][col]
                del destroyed_route_plan[row][index]

        return destroyed_route_plan, removed_requests

    def worst_deviation_removal(self, current_route_plan):
        destroyed_route_plan = current_route_plan.copy()
        to_remove = []
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_route_plan)

        # Find the requests to remove
        for j in range(num_remove):
            worst_deviation = timedelta(0)

            for row in range(len(destroyed_route_plan)):
                for col in range(1, len(destroyed_route_plan[row])):

                    temp = destroyed_route_plan[row][col]

                    # Skip already added nodes
                    if [temp, row] in to_remove:
                        continue

                    # Find associated drop off/pickup node
                    index, pickup = self.find_associated_node(row, col, destroyed_route_plan)
                    associated_temp = destroyed_route_plan[row][index]

                    temp_deviation = temp[2]
                    associated_temp_deviation = associated_temp[2]

                    if temp_deviation < timedelta(0):
                        temp_deviation = timedelta(seconds=-temp_deviation.total_seconds())

                    if associated_temp_deviation < timedelta(0):
                        associated_temp_deviation = timedelta(seconds=-associated_temp_deviation.total_seconds())

                    # Calculate total deviation for request
                    deviation = temp_deviation + associated_temp_deviation

                    # Update worst deviation so far
                    if deviation > worst_deviation and deviation > timedelta(0):
                        worst_deviation = deviation
                        worst_node = [temp, row]
                        worst_associated_node = [associated_temp, row]

            # Add node with worst deviation to list of nodes to remove
            w_n = worst_node
            dev = deviation
            if worst_node in to_remove:
                continue
            to_remove.append(worst_node)
            to_remove.append(worst_associated_node)

        # Remove nearest nodes from destroyed route plan
        for n in to_remove:
            destroyed_route_plan[n[1]].remove(n[0])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))
        return destroyed_route_plan, removed_requests

    # Related in travel time
    def distance_related_removal(self, current_route_plan):
        destroyed_route_plan = current_route_plan.copy()
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_route_plan)

        # Pick random node
        row_index = rnd.randint(0, len(destroyed_route_plan))
        col_index = rnd.randint(1, len(destroyed_route_plan[row_index]) - 1)
        node = destroyed_route_plan[row_index][col_index]

        # Find associated node
        index, pickup = self.find_associated_node(row_index, col_index, destroyed_route_plan)
        associated_node = destroyed_route_plan[row_index][index]

        # List of nodes to remove
        to_remove = [[node, row_index], [associated_node, row_index]]

        # Find the requests to remove
        for j in range(num_remove - 1):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(destroyed_route_plan)):
                for col in range(1, len(destroyed_route_plan[row])):

                    # Drop off/pickup of request to compare
                    temp = destroyed_route_plan[row][col]

                    # Skip already added nodes
                    if [temp, row] in to_remove:
                        continue

                    # Find associated drop off/pickup node of request to compare
                    temp_index, temp_pickup = self.find_associated_node(row, col, destroyed_route_plan)
                    associated_temp = destroyed_route_plan[row][temp_index]

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
                        nearest_node = [temp, row]
                        nearest_associated_node = [associated_temp, row]

            to_remove.append(nearest_node)
            to_remove.append(nearest_associated_node)

        # Remove nearest nodes from destroyed route plan
        for n in to_remove:
            destroyed_route_plan[n[1]].remove(n[0])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))

        return destroyed_route_plan, removed_requests

    # Related in service time
    def time_related_removal(self, current_route_plan):
        destroyed_route_plan = current_route_plan.copy()
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_route_plan)

        # Pick random node
        row_index = rnd.randint(0, len(destroyed_route_plan))
        col_index = rnd.randint(1, len(destroyed_route_plan[row_index]) - 1)
        node = destroyed_route_plan[row_index][col_index]

        # Find associated node
        index, pickup = self.find_associated_node(row_index, col_index, destroyed_route_plan)
        associated_node = destroyed_route_plan[row_index][index]

        # List of nodes to remove
        nodes_to_remove = [[node, row_index], [associated_node, row_index]]

        # Find the requests to remove
        for j in range(num_remove - 1):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(destroyed_route_plan)):
                for col in range(1, len(destroyed_route_plan[row])):

                    temp = destroyed_route_plan[row][col]

                    # Skip already added nodes
                    if [temp, row] in nodes_to_remove:
                        continue

                    # Find associated drop off/pickup node
                    temp_index, temp_pickup = self.find_associated_node(row, col, destroyed_route_plan)
                    associated_temp = destroyed_route_plan[row][temp_index]

                    # Find difference between pickup-times and drop off-times of requests
                    if temp_pickup == pickup:
                        diff = self.time_difference(temp, node, associated_temp, associated_node)

                    else:
                        diff = self.time_difference(temp, associated_node, associated_temp, node)

                    # Compare with smallest difference in current iteration
                    if diff < best_diff:
                        best_diff = diff
                        nearest_node = [temp, row]
                        nearest_associated_node = [associated_temp, row]

            nodes_to_remove.append(nearest_node)
            nodes_to_remove.append(nearest_associated_node)

        # Remove nearest nodes from destroyed route plan
        for n in nodes_to_remove:
            destroyed_route_plan[n[1]].remove(n[0])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))

        return destroyed_route_plan, removed_requests

    # Related in both service time and travel time
    def related_removal(self, current_route_plan):
        destroyed_route_plan = current_route_plan.copy()
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_route_plan)

        # Pick random node
        row_index = rnd.randint(0, len(destroyed_route_plan))
        col_index = rnd.randint(1, len(destroyed_route_plan[row_index]) - 1)
        node = destroyed_route_plan[row_index][col_index]

        # Find associated node
        index, pickup = self.find_associated_node(row_index, col_index, destroyed_route_plan)
        associated_node = destroyed_route_plan[row_index][index]

        # List of nodes to remove
        nodes_to_remove = [[node, row_index], [associated_node, row_index]]

        # Find the requests to remove
        for j in range(num_remove - 1):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(destroyed_route_plan)):
                for col in range(1, len(destroyed_route_plan[row])):

                    temp = destroyed_route_plan[row][col]

                    # Skip already added nodes
                    if [temp, row] in nodes_to_remove:
                        continue

                    # Find associated drop off/pickup node
                    temp_index, temp_pickup = self.find_associated_node(row, col, destroyed_route_plan)
                    associated_temp = destroyed_route_plan[row][temp_index]

                    # Find difference between requests
                    if (temp_pickup == pickup) & pickup:
                        diff_distance = self.travel_time_difference(temp[0], node[0])
                        diff_time = self.time_difference(temp, node, associated_temp, associated_node)

                    elif (temp_pickup == pickup) & (not pickup):
                        diff_distance = self.travel_time_difference(associated_temp[0], associated_node[0])
                        diff_time = self.time_difference(temp, node, associated_temp, associated_node)

                    elif (temp_pickup != pickup) & pickup:
                        diff_distance = self.travel_time_difference(associated_temp[0], node[0])
                        diff_time = self.time_difference(temp, associated_node, associated_temp, node)

                    else:
                        diff_distance = self.travel_time_difference(temp[0], associated_node[0])
                        diff_time = self.time_difference(temp, associated_node, associated_temp, node)

                    diff = diff_distance + diff_time

                    # Compare with smallest difference in current iteration
                    if diff < best_diff:
                        best_diff = diff
                        nearest_node = [temp, row]
                        nearest_associated_node = [associated_temp, row]

            nodes_to_remove.append(nearest_node)
            nodes_to_remove.append(nearest_associated_node)


        # Remove nearest nodes from destroyed route plan
        for n in nodes_to_remove:
            destroyed_route_plan[n[1]].remove(n[0])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append((n[0][0], n[0][5]))

        return destroyed_route_plan, removed_requests

    # Repair operators
    def greedy_repair(self, destroyed_route_plan, removed_requests, initial_infeasible_set):
        unassigned_requests = removed_requests + initial_infeasible_set
        unassigned_requests.sort(key=lambda x:x[1]["Requested Pickup Time"])
        route_plan = destroyed_route_plan.copy()
        current_objective = timedelta(0)
        infeasible_set = []
        unassigned_requests = pd.DataFrame(unassigned_requests)
        for i in tqdm(range(unassigned_requests.shape[0]), colour='#39ff14'):
            # while not unassigned_requests.empty:
            rid = unassigned_requests.iloc[i][0]
            request = unassigned_requests.iloc[i][1]

            route_plan, delta_objective, infeasible_set = self.repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=infeasible_set)

            # update current objective
            current_objective = delta_objective

        return route_plan, current_objective, infeasible_set

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
    @staticmethod
    def time_difference(pickup_1, pickup_2, dropoff_1, dropoff_2):
        return abs((pickup_1[1] - pickup_2[1]).total_seconds()) + abs((dropoff_1[1] - dropoff_2[1]).total_seconds())

    # Function to find associated pickup/drop-off of a node.
    @staticmethod
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
