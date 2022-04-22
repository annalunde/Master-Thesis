from functools import reduce
from math import ceil
from numpy.random import randint, choice
import pandas as pd
from config.main_config import *
from heuristic.improvement.initial.initial_repair_generator import RepairGenerator


class Operators:
    def __init__(self, alns):
        self.destruction_degree = alns.destruction_degree
        self.constructor = alns.constructor
        self.T_ij = self.constructor.T_ij
        self.repair_generator = RepairGenerator(self.constructor)

    # Find number of requests to remove based on degree of destruction
    def nodes_to_remove(self, route_plan):

        # Count number of requests in route_plan
        total_requests = reduce(lambda count, l: count + (len(l) - 1)/2, route_plan, 0)

        # Calculate number of requests to remove
        return ceil(total_requests * self.destruction_degree)

    def random_removal(self, current_route_plan, current_infeasible_set):

        # Number of requests to remove
        num_remove = self.nodes_to_remove(current_route_plan)

        # get ((pickup indices), (dropoff indices), pickup node, dropoff node) for all requests
        possible_removals = [((p_node[0], vehicle, p_idx + 1), (d_node[0], vehicle, d_idx + 1), p_node, d_node)
                             for vehicle in range(0, len(current_route_plan))
                             for p_idx, p_node in enumerate(current_route_plan[vehicle][1:])
                             if not p_node[0] % int(p_node[0])
                             for d_idx, d_node in enumerate(current_route_plan[vehicle][1:])
                             if d_node[0] == p_node[0] + 0.5]

        # get requests to destroy
        removal_idx = choice(len(possible_removals), num_remove, False)
        removal = [possible_removals[i] for i in removal_idx]
        removal_nodes = [node for request in removal for node in request[2:]]

        # remove destroyed requests from route plan
        destroyed_route_plan = [[node for node in vehicle if node not in removal_nodes] for vehicle in current_route_plan]
        removed_requests = [(node[2][0], node[2][5]) for node in removal]
        index_removed_requests = [ind for request in removal for ind in request[:2]]

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    def worst_deviation_removal(self, current_route_plan, current_infeasible_set):

        # Number of requests to remove
        num_remove = self.nodes_to_remove(current_route_plan)

        # get ((pickup indices), (dropoff indices), pickup node, dropoff node, total deviation) for all requests
        possible_removals = [((p_node[0], vehicle, p_idx + 1), (d_node[0], vehicle, d_idx + 1), p_node, d_node,
                              p_node[2] + d_node[2] if p_node[2] >= timedelta(0) and d_node[2] >= timedelta(0)
                              else p_node[2] + timedelta(seconds=-d_node[2].total_seconds())
                              if d_node[2] < timedelta(0) <= p_node[2]
                              else d_node[2] + timedelta(seconds=-p_node[2].total_seconds())
                              if p_node[2] < timedelta(0) <= d_node[2]
                              else timedelta(seconds=-p_node[2].total_seconds()) + timedelta(seconds=-d_node[2].total_seconds()))
                             for vehicle in range(0, len(current_route_plan))
                             for p_idx, p_node in enumerate(current_route_plan[vehicle][1:])
                             if not p_node[0] % int(p_node[0])
                             for d_idx, d_node in enumerate(current_route_plan[vehicle][1:])
                             if d_node[0] == p_node[0] + 0.5]

        # get requests to destroy
        possible_removals.sort(reverse=True, key=lambda x: x[4])
        removal = [request for request in possible_removals[:num_remove] if request[4] > timedelta(0)]
        removal_nodes = [node for request in removal for node in request[2:4]]

        # If not enough nodes have deviation > 0, remove the rest randomly
        if len(removal) < num_remove:
            possible_removals_no_dev = [request for request in possible_removals if request[4] <= timedelta(0)]
            random_removal, random_removal_nodes = self.worst_deviation_random_removal(
                possible_removals_no_dev, num_remove - len(removal))
            removal = removal + random_removal
            removal_nodes = removal_nodes + random_removal_nodes

        # remove destroyed requests from route plan
        destroyed_route_plan = [[node for node in vehicle if node not in removal_nodes] for vehicle in current_route_plan]
        removed_requests = [(node[2][0], node[2][5]) for node in removal]
        index_removed_requests = [ind for request in removal for ind in request[:2]]

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    # Related in travel time
    def distance_related_removal(self, current_route_plan, current_infeasible_set):

        # Number of requests to remove
        num_remove = self.nodes_to_remove(current_route_plan)

        # get ((pickup indices), (dropoff indices), pickup node, dropoff node) for all requests
        possible_removals = [((p_node[0], vehicle, p_idx + 1), (d_node[0], vehicle, d_idx + 1), p_node, d_node)
                             for vehicle in range(0, len(current_route_plan))
                             for p_idx, p_node in enumerate(current_route_plan[vehicle][1:])
                             if not p_node[0] % int(p_node[0])
                             for d_idx, d_node in enumerate(current_route_plan[vehicle][1:])
                             if d_node[0] == p_node[0] + 0.5]

        if len(current_infeasible_set) != 0:
            # Pick random request in infeasible_set to compare other requests to
            initial_node = current_infeasible_set[randint(0, len(current_infeasible_set))]
            initial_rid, initial_removal = initial_node[0], []

        else:
            # Pick random request in route plan to remove and to compare other requests to
            init_node_idx = randint(len(possible_removals))
            initial_node = possible_removals[init_node_idx]
            possible_removals.pop(init_node_idx)
            initial_rid, initial_removal = initial_node[2][0], [initial_node]
            num_remove -= 1

        diff_removals = [(request, self.travel_time_difference(initial_rid, request[2][0]))
                         for request in possible_removals]

        # get requests to destroy
        diff_removals.sort(key=lambda x: x[1])
        removal = [request[0] for request in diff_removals[:num_remove]]
        removal = removal + initial_removal
        removal_nodes = [node for request in removal for node in request[2:4]]

        # remove destroyed requests from route plan
        destroyed_route_plan = [[node for node in vehicle if node not in removal_nodes] for vehicle in current_route_plan]
        removed_requests = [(node[2][0], node[2][5]) for node in removal]
        index_removed_requests = [ind for request in removal for ind in request[:2]]

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    # Related in service time
    def time_related_removal(self, current_route_plan, current_infeasible_set):

        # Number of requests to remove
        num_remove = self.nodes_to_remove(current_route_plan)

        # get ((pickup indices), (dropoff indices), pickup node, dropoff node) for all requests
        possible_removals = [((p_node[0], vehicle, p_idx + 1), (d_node[0], vehicle, d_idx + 1), p_node, d_node)
                             for vehicle in range(0, len(current_route_plan))
                             for p_idx, p_node in enumerate(current_route_plan[vehicle][1:])
                             if not p_node[0] % int(p_node[0])
                             for d_idx, d_node in enumerate(current_route_plan[vehicle][1:])
                             if d_node[0] == p_node[0] + 0.5]

        if len(current_infeasible_set) != 0:
            # Pick random request in infeasible_set to compare other requests to
            initial_node = current_infeasible_set[randint(0, len(current_infeasible_set))]
            initial_p_time, initial_d_time, initial_removal = self.get_pickup(initial_node), \
                                                              self.get_dropoff(initial_node), []

        else:
            # Pick random request in route plan to remove and to compare other requests to
            init_node_idx = randint(len(possible_removals))
            initial_node = possible_removals[init_node_idx]
            possible_removals.pop(init_node_idx)
            initial_p_time, initial_d_time, initial_removal = initial_node[2][1], initial_node[3][1], [initial_node]
            num_remove -= 1

        diff_removals = [(request,
                          abs((initial_p_time - request[2][1]).total_seconds()) +
                          abs((initial_d_time - request[3][1]).total_seconds()))
                         for request in possible_removals]

        # get requests to destroy
        diff_removals.sort(key=lambda x: x[1])
        removal = [request[0] for request in diff_removals[:num_remove]]
        removal = removal + initial_removal
        removal_nodes = [node for request in removal for node in request[2:4]]

        # remove destroyed requests from route plan
        destroyed_route_plan = [[node for node in vehicle if node not in removal_nodes] for vehicle in
                                current_route_plan]
        removed_requests = [(node[2][0], node[2][5]) for node in removal]
        index_removed_requests = [ind for request in removal for ind in request[:2]]

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    # Related in both service time and travel time
    def related_removal(self, current_route_plan, current_infeasible_set):

        # Number of requests to remove
        num_remove = self.nodes_to_remove(current_route_plan)

        # get ((pickup indices), (dropoff indices), pickup node, dropoff node) for all requests
        possible_removals = [((p_node[0], vehicle, p_idx + 1), (d_node[0], vehicle, d_idx + 1), p_node, d_node)
                             for vehicle in range(0, len(current_route_plan))
                             for p_idx, p_node in enumerate(current_route_plan[vehicle][1:])
                             if not p_node[0] % int(p_node[0])
                             for d_idx, d_node in enumerate(current_route_plan[vehicle][1:])
                             if d_node[0] == p_node[0] + 0.5]

        if len(current_infeasible_set) != 0:
            # Pick random request in infeasible_set to compare other requests to
            initial_node = current_infeasible_set[randint(0, len(current_infeasible_set))]
            initial_rid, initial_p_time, initial_d_time, initial_removal = initial_node[0], \
                                                                           self.get_pickup(initial_node), \
                                                                           self.get_dropoff(initial_node), []

        else:
            # Pick random request in route plan to remove and to compare other requests to
            init_node_idx = randint(len(possible_removals))
            initial_node = possible_removals[init_node_idx]
            possible_removals.pop(init_node_idx)
            initial_rid, initial_p_time, initial_d_time, initial_removal = initial_node[2][0], \
                                                                           initial_node[2][1], \
                                                                           initial_node[3][1], [initial_node]
            num_remove -= 1

        diff_removals = [(request,
                          abs((initial_p_time - request[2][1]).total_seconds()) +
                          abs((initial_d_time - request[3][1]).total_seconds()) +
                          self.travel_time_difference(initial_rid, request[2][0]))
                         for request in possible_removals]

        # get requests to destroy
        diff_removals.sort(key=lambda x: x[1])
        removal = [request[0] for request in diff_removals[:num_remove]]
        removal = removal + initial_removal
        removal_nodes = [node for request in removal for node in request[2:4]]

        # remove destroyed requests from route plan
        destroyed_route_plan = [[node for node in vehicle if node not in removal_nodes] for vehicle in
                                current_route_plan]
        removed_requests = [(node[2][0], node[2][5]) for node in removal]
        index_removed_requests = [ind for request in removal for ind in request[:2]]

        return destroyed_route_plan, removed_requests, index_removed_requests, True

    # Repair operators
    def greedy_repair(self, destroyed_route_plan, removed_requests, initial_infeasible_set, current_route_plan, index_removed_requests, delayed, still_delayed):
        unassigned_requests = removed_requests.copy() + initial_infeasible_set.copy()
        unassigned_requests.sort(key=lambda x: x[0])
        route_plan = list(map(list, destroyed_route_plan))
        current_objective = timedelta(0)
        self.constructor.infeasible_set = []
        unassigned_requests = pd.DataFrame(unassigned_requests)
        for i in range(unassigned_requests.shape[0]):
            # while not unassigned_requests.empty:
            rid = unassigned_requests.iloc[i][0]
            request = unassigned_requests.iloc[i][1]
            index_removal = [
                i for i in index_removed_requests if i[0] == rid or i[0] == rid+0.5]

            route_plan, new_objective, infeasible_set = self.repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=self.constructor.infeasible_set,
                initial_route_plan=current_route_plan, index_removed=index_removal, objectives=0,
                prev_objective=current_objective)

            # update current objective
            current_objective = new_objective

        if len(self.constructor.infeasible_set) == len(unassigned_requests):
            current_objective = self.constructor.new_objective(destroyed_route_plan, self.constructor.infeasible_set)

        return route_plan, current_objective, infeasible_set

    def regret_2_repair(self, destroyed_route_plan, removed_requests, initial_infeasible_set, current_route_plan, index_removed_requests, delayed, still_delayed):
        unassigned_requests = removed_requests.copy() + initial_infeasible_set.copy()
        unassigned_requests.sort(key=lambda x: x[0])
        route_plan = list(map(list, destroyed_route_plan))
        current_objective = timedelta(0)
        self.constructor.infeasible_set, regret_values = [], []
        unassigned_requests = pd.DataFrame(unassigned_requests)
        for i in range(unassigned_requests.shape[0]):
            rid = unassigned_requests.iloc[i][0]
            request = unassigned_requests.iloc[i][1]
            index_removal = [
                i for i in index_removed_requests if i[0] == rid or i[0] == rid+0.5]

            first_objective, second_objective = self.repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=self.constructor.infeasible_set,
                initial_route_plan=current_route_plan, index_removed=index_removal, objectives=2,
                prev_objective=self.constructor.new_objective(destroyed_route_plan, self.constructor.infeasible_set))

            self.constructor.infeasible_set = []

            regret_values.append(
                (rid, request, second_objective-first_objective))

        regret_values.sort(key=lambda x: x[2], reverse=True)

        # iterate through requests in order of regret k value
        for i in regret_values:
            rid = i[0]
            request = i[1]
            index_removal = [
                i for i in index_removed_requests if i[0] == rid or i[0] == rid+0.5]

            route_plan, new_objective, infeasible_set = self.repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=self.constructor.infeasible_set,
                initial_route_plan=current_route_plan, index_removed=index_removal, objectives=0,
                prev_objective=current_objective)

            # update current objective
            current_objective = new_objective

        if len(self.constructor.infeasible_set) == len(unassigned_requests):
            current_objective = self.constructor.new_objective(destroyed_route_plan, self.constructor.infeasible_set)
        return route_plan, current_objective, infeasible_set

    def regret_3_repair(self, destroyed_route_plan, removed_requests, initial_infeasible_set, current_route_plan, index_removed_requests, delayed, still_delayed):
        unassigned_requests = removed_requests.copy() + initial_infeasible_set.copy()
        unassigned_requests.sort(key=lambda x: x[0])
        route_plan = list(map(list, destroyed_route_plan))
        current_objective = timedelta(0)
        self.constructor.infeasible_set, regret_values = [], []
        unassigned_requests = pd.DataFrame(unassigned_requests)

        for i in range(unassigned_requests.shape[0]):
            rid = unassigned_requests.iloc[i][0]
            request = unassigned_requests.iloc[i][1]
            index_removal = [
                i for i in index_removed_requests if i[0] == rid or i[0] == rid+0.5]

            first_objective, third_objective = self.repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=self.constructor.infeasible_set,
                initial_route_plan=current_route_plan, index_removed=index_removal, objectives=3,
                prev_objective=self.constructor.new_objective(destroyed_route_plan, self.constructor.infeasible_set))

            self.constructor.infeasible_set = []

            regret_values.append(
                (rid, request, third_objective-first_objective))

        regret_values.sort(key=lambda x: x[2], reverse=True)

        # iterate through requests in order of regret k value
        for i in regret_values:
            rid = i[0]
            request = i[1]
            index_removal = [
                i for i in index_removed_requests if i[0] == rid or i[0] == rid+0.5]

            route_plan, new_objective, infeasible_set = self.repair_generator.generate_insertions(
                route_plan=route_plan, request=request, rid=rid, infeasible_set=self.constructor.infeasible_set,
                initial_route_plan=current_route_plan, index_removed=index_removal, objectives=0,
                prev_objective=current_objective)

            # update current objective
            current_objective = new_objective

        if len(self.constructor.infeasible_set) == len(unassigned_requests):
            current_objective = self.constructor.new_objective(destroyed_route_plan, self.constructor.infeasible_set)

        return route_plan, current_objective, infeasible_set

    # Function to find random requests to remove if worst deviation removal does not remove enough

    def worst_deviation_random_removal(self, possible_removals, num_remove):

        # get requests to destroy
        removal_idx = choice(len(possible_removals), num_remove, False)
        removal = [possible_removals[i] for i in removal_idx]
        removal_nodes = [node for request in removal for node in request[2:4]]

        return removal, removal_nodes

    # Function to calculate total travel time differences between requests
    def travel_time_difference(self, rid1, rid2):
        num_requests = self.constructor.n
        idx_1 = rid1 - 1
        idx_2 = rid2 - 1
        return self.T_ij[idx_1][idx_2] + \
            self.T_ij[idx_1 + num_requests][idx_2 + num_requests] + \
            self.T_ij[idx_1 + num_requests][idx_2] + \
            self.T_ij[idx_1][idx_2 + num_requests]

    def get_pickup(self, node):
        # Node is pickup, find requested pickup time or calculated pickup time
        rid, s = node[0], S_W if node[1]["Wheelchair"] else S_P

        time = node[1]["Requested Dropoff Time"] - self.constructor.travel_time(
            rid - 1, self.constructor.n + rid - 1, True) - timedelta(minutes=s) \
            if pd.isnull(node[1]["Requested Pickup Time"]) \
            else node[1]["Requested Pickup Time"] + timedelta(minutes=s)

        return time

    def get_dropoff(self, node):
        # Node is dropoff, find requested dropoff time or calculated dropoff time
        rid, s = node[0], S_W if node[1]["Wheelchair"] else S_P

        time = node[1]["Requested Pickup Time"] + self.constructor.travel_time(
                rid - 1, self.constructor.n + rid - 1, True) + 2*timedelta(minutes=s) \
            if pd.isnull(node[1]["Requested Dropoff Time"]) \
            else node[1]["Requested Dropoff Time"] + 2*timedelta(minutes=s)

        return time