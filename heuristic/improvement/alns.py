import numpy as np
import numpy.random as rnd
from collections import OrderedDict

from heuristic.improvement.operators import Operators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing
from heuristic.improvement.improvement_config import *


class ALNS:
    def __init__(self, weights, reaction_factor, rnd_state=rnd.RandomState()):
        # Reaction_factor (r) is a parameter that controls how fast weights adjust.
        # Weights is array of four elements, index 0 highest, 3 lowest
        self.destroy_operators = OrderedDict()
        self.repair_operators = OrderedDict()
        self.rnd_state = rnd_state
        self.reaction_factor = reaction_factor
        self.weights = weights

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
    def iterate(self, initial_solution, initial_objective, num_iterations, criterion):

        weights = np.asarray(self.weights, dtype=np.float16)
        current = best = initial_solution
        current_objective = best_objective = initial_objective

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
                                                                                                     criterion)

            # Update weights
            d_weights = d_weights * (1 - self.reaction_factor) + (self.reaction_factor * weights[weight_score])
            r_weights = r_weights * (1 - self.reaction_factor) + (self.reaction_factor * weights[weight_score])

        return best, best_objective

def main():

    random_state = rnd.RandomState(seed)

    alns = ALNS(weights, reaction_factor, random_state)

    operators = Operators(destruction_degree, travel_times)

    criterion = SimulatedAnnealing(start_temperature, end_temperature, step)

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
    result = alns.iterate(initial_solution, initial_objective, iterations, criterion)


if __name__ == "__main__":
    main()
