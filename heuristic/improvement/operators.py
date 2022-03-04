import math
import numpy.random as rnd
from datetime import datetime


class Operators:

    def __init__(self, alns):
        self.destruction_degree = alns.destruction_degree
        self.T_ij = alns.T_ij

    # Find number of requests to remove based on degree of destruction
    def nodes_to_remove(self, solution):

        # Count number of requests in solution
        total_requests = 0
        for row in solution:
            for col in row:
                total_requests += 0.5

        # Calculate number of requests to remove
        num_remove = math.ceil(total_requests * self.destruction_degree)
        return num_remove

    def random_removal(self, current_solution):
        destroyed_solution = current_solution.copy()
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_solution)

        # Find the requests to remove
        for j in range(num_remove):

            # Pick random node to remove
            row = rnd.randint(0, len(destroyed_solution))
            col = rnd.randint(0, len(destroyed_solution[row]) - 1)
            node = destroyed_solution[row][col]

            # Find col-index of associated pickup/drop-off node
            index, pickup = self.find_associated_node(row, col, destroyed_solution)
            associated_node = destroyed_solution[row][index]

            # Remove both pickup and drop-off node and add request-index to removed_requests
            if pickup:
                removed_requests.append(node[0])
                del destroyed_solution[row][index]
                del destroyed_solution[row][col]
            else:
                removed_requests.append(associated_node[0])
                del destroyed_solution[row][col]
                del destroyed_solution[row][index]

        return destroyed_solution, removed_requests

    def worst_deviation_removal(self, current_solution):
        destroyed_solution = current_solution.copy()
        to_remove = []
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_solution)

        # Find the requests to remove
        for j in range(num_remove):
            worst_dev = 0

            for row in range(len(destroyed_solution)):
                for col in range(len(destroyed_solution[row])):

                    temp = destroyed_solution[row][col]

                    # Skip already added nodes
                    if [temp, row] in to_remove:
                        continue

                    # Find associated drop off/pickup node
                    index, pickup = self.find_associated_node(row, col, destroyed_solution)
                    associated_temp = destroyed_solution[row][index]

                    # Calculate total deviation for request
                    dev = temp[2] + associated_temp[2]

                    # Update worst deviation so far
                    if dev > worst_dev:
                        worst_dev = dev
                        worst_node = [temp, row]
                        worst_associated_node = [associated_temp, row]

            # Add node with worst deviation to list of nodes to remove
            to_remove.append(worst_node)
            to_remove.append(worst_associated_node)

        # Remove nearest nodes from destroyed solution
        for n in to_remove:
            destroyed_solution[n[1]].remove(n[0])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append(n[0][0])

        return destroyed_solution, removed_requests

    # Related in travel time
    def distance_related_removal(self, current_solution):
        destroyed_solution = current_solution.copy()
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_solution)

        # Pick random node
        row_index = rnd.randint(0, len(destroyed_solution))
        col_index = rnd.randint(0, len(destroyed_solution[row_index]) - 1)
        node = destroyed_solution[row_index][col_index]

        # Find associated node
        index, pickup = self.find_associated_node(row_index, col_index, destroyed_solution)
        associated_node = destroyed_solution[row_index][index]

        # List of nodes to remove
        to_remove = [[node, row_index], [associated_node, row_index]]

        # Find the requests to remove
        for j in range(num_remove - 1):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(destroyed_solution)):
                for col in range(len(destroyed_solution[row])):

                    # Drop off/pickup of request to compare
                    temp = destroyed_solution[row][col]

                    # Skip already added nodes
                    if [temp, row] in to_remove:
                        continue

                    # Find associated drop off/pickup node of request to compare
                    temp_index, temp_pickup = self.find_associated_node(row, col, destroyed_solution)
                    associated_temp = destroyed_solution[row][temp_index]

                    # Find difference in distance between pickup and drop-off of requests
                    if (temp_pickup == pickup) & pickup:
                        diff = self.travel_time_difference(temp[0], node[0])

                    elif (temp_pickup == pickup) & (not pickup):
                        diff = self.travel_time_difference(associated_temp[0], associated_node[0])

                    elif (temp_pickup != pickup) & pickup:
                        diff = self.travel_time_difference(associated_temp[0], node[0])

                    else:
                        diff = self.travel_time_difference(temp[0], associated_node[0])

                    # Compare with smallest difference in current iteration
                    if diff < best_diff:
                        best_diff = diff
                        nearest_node = [temp, row]
                        nearest_associated_node = [associated_temp, row]

            to_remove.append(nearest_node)
            to_remove.append(nearest_associated_node)

        # Remove nearest nodes from destroyed solution
        for n in to_remove:
            destroyed_solution[n[1]].remove(n[0])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append(n[0][0])

        return destroyed_solution, removed_requests

    # Related in service time
    def time_related_removal(self, current_solution):
        destroyed_solution = current_solution.copy()
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_solution)

        # Pick random node
        row_index = rnd.randint(0, len(destroyed_solution))
        col_index = rnd.randint(0, len(destroyed_solution[row_index]) - 1)
        node = destroyed_solution[row_index][col_index]

        # Find associated node
        index, pickup = self.find_associated_node(row_index, col_index, destroyed_solution)
        associated_node = destroyed_solution[row_index][index]

        # List of nodes to remove
        nodes_to_remove = [[node, row_index], [associated_node, row_index]]

        # Find the requests to remove
        for j in range(num_remove - 1):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(destroyed_solution)):
                for col in range(len(destroyed_solution[row])):

                    temp = destroyed_solution[row][col]

                    # Skip already added nodes
                    if [temp, row] in nodes_to_remove:
                        continue

                    # Find associated drop off/pickup node
                    temp_index, temp_pickup = self.find_associated_node(row, col, destroyed_solution)
                    associated_temp = destroyed_solution[row][temp_index]

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

        # Remove nearest nodes from destroyed solution
        for n in nodes_to_remove:
            destroyed_solution[n[1]].remove(n[0])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append(n[0][0])

        return destroyed_solution, removed_requests

    # Related in both service time and travel time
    def related_removal(self, current_solution):
        destroyed_solution = current_solution.copy()
        removed_requests = []

        # Number of requests to remove
        num_remove = self.nodes_to_remove(destroyed_solution)

        # Pick random node
        row_index = rnd.randint(0, len(destroyed_solution))
        col_index = rnd.randint(0, len(destroyed_solution[row_index]) - 1)
        node = destroyed_solution[row_index][col_index]

        # Find associated node
        index, pickup = self.find_associated_node(row_index, col_index, destroyed_solution)
        associated_node = destroyed_solution[row_index][index]

        # List of nodes to remove
        nodes_to_remove = [[node, row_index], [associated_node, row_index]]

        # Find the requests to remove
        for j in range(num_remove - 1):

            # To do: finne ut hva denne initielt skal settes som
            best_diff = 48 * 60 * 60

            for row in range(len(destroyed_solution)):
                for col in range(len(destroyed_solution[row])):

                    temp = destroyed_solution[row][col]

                    # Skip already added nodes
                    if [temp, row] in nodes_to_remove:
                        continue

                    # Find associated drop off/pickup node
                    temp_index, temp_pickup = self.find_associated_node(row, col, destroyed_solution)
                    associated_temp = destroyed_solution[row][temp_index]

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

        # Remove nearest nodes from destroyed solution
        for n in nodes_to_remove:
            destroyed_solution[n[1]].remove(n[0])

            # Add request id to removed_requests
            if not n[0][0] % int(n[0][0]):
                removed_requests.append(n[0][0])

        return destroyed_solution, removed_requests

    # Repair operators
    def greedy_repair(self, destroyed_solution, removed_requests):
        return new_solution, new_objective

    def regret_repair(self, destroyed_solution, removed_requests):
        return new_solution, new_objective

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
    def find_associated_node(row, col, solution):
        node = solution[row][col]

        if node[0] % int(node[0]):
            # Node is drop-off, must find pickup
            pickup = False
            request = node[0] - 0.5
            for index in range(col):
                temp = solution[row][index]
                if temp[0] == request:
                    return index, pickup

        else:
            # Node is pickup, must find drop-off
            pickup = True
            request = node[0] + 0.5
            for index in range(len(solution[row])):
                temp = solution[row][index]
                if temp[0] == request:
                    return index, pickup

