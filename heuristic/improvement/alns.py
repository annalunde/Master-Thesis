import numpy as np
import pandas as pd
import numpy.random as rnd
from collections import OrderedDict

from heuristic.improvement.operators import Operators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing
from heuristic.improvement.improvement_config import *


class ALNS:
    def __init__(self, weights, reaction_factor, current_route_plan, current_objective, infeasible_set, criterion, destruction_degree, T_ij, df, rnd_state=rnd.RandomState()):
        # Reaction_factor (r) is a parameter that controls how fast weights adjust.
        # Weights is array of four elements, index 0 highest, 3 lowest
        self.destroy_operators = OrderedDict()
        self.repair_operators = OrderedDict()
        self.rnd_state = rnd_state
        self.reaction_factor = reaction_factor
        self.infeasible_set = infeasible_set
        self.weights = weights
        self.route_plan = current_route_plan
        self.destruction_degree = destruction_degree
        self.objective = current_objective
        self.T_ij = self.travel_matrix(self.route_plan)
        self.T_ij = T_ij
        self.criterion = criterion

    # Travel time matrix
    def travel_matrix(self, route_plan):
        return T_ij

    # Add operator to the heuristic instance
    def add_destroy_operator(self, operator):
        name = operator.__name__
        self.destroy_operators[name] = operator

    def add_repair_operator(self, operator):
        name = operator.__name__
        self.repair_operators[name] = operator

    # Return elements in repair/destroy operators
    def get_repair_operators(self):
        return list(self.repair_operators.items())

    def get_destroy_operators(self):
        return list(self.destroy_operators.items())

    # Select destroy/repair operator
    @staticmethod
    def select_operator(operator, weights, rnd_state):
        return rnd_state.choice(np.arrange(0, len(operator)), p=weights / np.sum(weights))

    # Evaluate candidate
    def evaluate_candidate(self, best, best_objective, current, current_objective, candidate, candidate_objective,
                           criterion):
        # If solution is accepted by criterion (simulated annealing)
        if criterion.accept_criterion(self.rnd_state, current_objective, candidate_objective):
            if candidate_objective < current_objective:
                # Solution is better
                weight_score = 1
            else:
                # Solution is not better, but accepted
                weight_score = 2
            current = candidate
            current_objective = candidate_objective
        else:
            # Solution is rejected
            weight_score = 3

        if candidate_objective < best_objective:
            # Solution is new global best
            current = candidate
            current_objective = candidate_objective
            best = candidate
            best_objective = candidate_objective
            weight_score = 0

        return best, best_objective, current, current_objective, weight_score

    # Run ALNS algorithm
    def iterate(self, num_iterations):

        weights = np.asarray(self.weights, dtype=np.float16)
        current = best = self.route_plan
        current_objective = best_objective = self.objective

        d_weights = np.ones(len(self.destroy_operators), dtype=np.float16)
        r_weights = np.ones(len(self.repair_operators), dtype=np.float16)

        for i in range(num_iterations):
            # Select destroy method
            destroy = self.select_operator(self.get_destroy_operators(), weights, self.rnd_state)

            # Select repair method
            repair = self.select_operator(self.get_repair_operators(), weights, self.rnd_state)

            # Destroy solution
            d_name, d_operator = self.destroy_operators[destroy]
            destroyed_sol, removed_nodes = d_operator(current)

            # Fix solution
            r_name, r_operator = self.repair_operators[repair]
            candidate, candidate_objective = r_operator(destroyed_sol, removed_nodes)

            # Compare solutions
            best, best_objective, current, current_objective, weight_score = self.evaluate_candidate(best,
                                                                                                     best_objective,
                                                                                                     current,
                                                                                                     current_objective,
                                                                                                     candidate,
                                                                                                     candidate_objective,
                                                                                                     self.criterion)

            # Update weights
            d_weights = d_weights * (1 - self.reaction_factor) + (self.reaction_factor * weights[weight_score])
            r_weights = r_weights * (1 - self.reaction_factor) + (self.reaction_factor * weights[weight_score])

        return best, best_objective

def main():

    random_state = rnd.RandomState(seed)

    criterion = SimulatedAnnealing(start_temperature, end_temperature, step)

    alns = ALNS(weights, reaction_factor, current_route_plan, current_objective, infeasible_set, criterion, destruction_degree, T_ij, df, random_state)

    operators = Operators(alns)

    # Add destroy operators
    alns.add_destroy_operator(operators.random_removal)
    alns.add_destroy_operator(operators.time_related_removal)
    alns.add_destroy_operator(operators.distance_related_removal)
    alns.add_destroy_operator(operators.related_removal)
    alns.add_destroy_operator(operators.worst_deviation_removal)

    # Add repair operators
    alns.add_repair_operator(operators.greedy_repair)
    alns.add_repair_operator(operators.regret_repair)

    # Run algorithm
    #result = alns.iterate(iterations)

    print("route_plan: " + "\n", current_route_plan)

    route_plan, removed_requests = operators.distance_related_removal(alns.route_plan)

    print("modified:" + "\n", route_plan)
    print("removed_requests: ", removed_requests)


if __name__ == "__main__":
    main()
