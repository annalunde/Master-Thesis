import numpy as np
import numpy.random as rnd
from collections import OrderedDict

from heuristic.improvement.operators import Operators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing
from heuristic.improvement.improvement_config import *


class ALNS:
    def __init__(self, weights, reaction_factor, current_route_plan, current_objective, infeasible_set, criterion, destruction_degree, constructor, rnd_state=rnd.RandomState()):
        # Reaction_factor (r) is a parameter that controls how fast weights adjust.
        # Weights is array of four elements, index 0 highest, 3 lowest
        self.destroy_operators = []
        self.repair_operators = []
        self.rnd_state = rnd_state
        self.reaction_factor = reaction_factor
        self.infeasible_set = infeasible_set
        self.weights = weights
        self.route_plan = current_route_plan
        self.destruction_degree = destruction_degree
        self.objective = current_objective
        self.criterion = criterion
        self.constructor = constructor

    # Add operator to the heuristic instance
    def add_destroy_operator(self, operator):
        self.destroy_operators.append(operator)

    def add_repair_operator(self, operator):
        self.repair_operators.append(operator)

    # Select destroy/repair operator
    @staticmethod
    def select_operator(operator, weights, rnd_state):
        w = weights / np.sum(weights)
        a = [i for i in range(len(operator))]
        return rnd_state.choice(a=a, p=w)

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
        d_count = np.zeros(len(self.destroy_operators), dtype=np.float16)
        r_count = np.zeros(len(self.repair_operators), dtype=np.float16)

        for i in range(num_iterations):
            # Select destroy method
            destroy = self.select_operator(
                self.destroy_operators, d_weights, self.rnd_state)

            # Select repair method
            repair = self.select_operator(
                self.repair_operators, r_weights, self.rnd_state)

            # Destroy solution
            d_operator = self.destroy_operators[destroy]
            print(d_operator)
            print("kult")
            destroyed_route, removed_requests = d_operator(
                current)
            d_count[destroy] += 1
            for n in removed_requests:
                print(n[0])
            # Fix solution
            r_operator = self.repair_operators[repair]
            candidate, candidate_objective, infeasible_set = r_operator(
                destroyed_route, removed_requests, self.infeasible_set)

            r_count[repair] += 1

            # Compare solutions
            best, best_objective, current, current_objective, weight_score = self.evaluate_candidate(
                best, best_objective, current, current_objective, candidate, candidate_objective, self.criterion)

            # Update weights
            d_weights[destroy] = d_weights[destroy] * \
                (1 - self.reaction_factor) + \
                (self.reaction_factor * weights[weight_score]/d_count[destroy])
            r_weights[repair] = r_weights[repair] * \
                (1 - self.reaction_factor) + \
                (self.reaction_factor * weights[weight_score]/r_count[repair])
        return best, best_objective

def main():

    random_state = rnd.RandomState(seed)

    criterion = SimulatedAnnealing(start_temperature, end_temperature, step)

    alns = ALNS(weights, reaction_factor, current_route_plan, current_objective, infeasible_set, criterion, destruction_degree, random_state)

    operators = Operators(alns)

    # Add destroy operators
    alns.add_destroy_operator(operators.random_removal)
    alns.add_destroy_operator(operators.time_related_removal)
    alns.add_destroy_operator(operators.distance_related_removal)
    alns.add_destroy_operator(operators.related_removal)
    alns.add_destroy_operator(operators.worst_deviation_removal)

    # Add repair operators
    alns.add_repair_operator(operators.greedy_repair)
    #alns.add_repair_operator(operators.regret_repair)

    # Run algorithm
    #result = alns.iterate(iterations)

    print("route_plan: " + "\n", current_route_plan)

    route_plan, removed_requests = operators.distance_related_removal(alns.route_plan)

    print("modified:" + "\n", route_plan)
    print("removed_requests: ", removed_requests)


if __name__ == "__main__":
    main()

