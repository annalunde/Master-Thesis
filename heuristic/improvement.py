import numpy as np
import numpy.random as rnd
from collections import OrderedDict


class ImprovementHeuristic:

    def __init__(self, weights, reaction_factor, start_temperature, end_temperature, step):
        """
        weights: array_like
            A list of four non-negative elements, representing the weight
            updates when the candidate solution results in a new global best
            (index 0), is better than the current solution (index 1), the solution
            is accepted (index 2), or rejected (index 3).
        """

        self._destroy_operators = OrderedDict()
        self._repair_operators = OrderedDict()
        self.weights = weights
        self.reaction_factor = reaction_factor
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.step = step
        self.temperature = start_temperature
        self._rnd_state = rnd.RandomState()

    # Operators
    # Destroy operators
    def random_removal(self, current_solution):
        destroyed_solution = current_solution.copy()
        return destroyed_solution

    def worst_removal(self, current_solution):
        destroyed_solution = current_solution.copy()
        return destroyed_solution

    def shaw_removal(self, current_solution):
        destroyed_solution = current_solution.copy()
        return destroyed_solution

    # Repair operators
    def greedy_insertion(self, destroyed_solution):
        return new_solution, new_objective

    def regret_insertion(self, destroyed_solution):
        return new_solution, new_objective

    # Add operator to the heuristic instance
    def add_destroy_operator(self, operator):
        name = operator.__name__
        self._destroy_operators[name] = operator

    def add_repair_operator(self, operator):
        name = operator.__name__
        self._repair_operators[name] = operator

    # return elements in repair/destroy operators
    def repair_operators(self):
        return list(self._repair_operators.items())

    def destroy_operators(self):
        return list(self._destroy_operators.items())

    # Simulated annealing accept criterion
    def accept_criterion(self, rnd, current_objective, candidate_objective):
        probability = np.exp((current_objective - candidate_objective) / self.temperature)

        # Should not set a temperature that is lower than the end temperature.
        self.temperature = max(self.end_temperature, self.temperature - self.step)

        return probability >= rnd.random()

    # Select destroy/repair operator
    @staticmethod
    def select_operator(operator, weights, rnd_state):
        return rnd_state.choice(np.arrange(0, len(operator)), p=weights / np.sum(weights))

    # Evaluate candidate
    def evaluate_candidate(self, best, best_objective, current, current_objective, candidate, candidate_objective):
        if accept_criterion(self._rnd_state, current_objective, candidate_objective):
            if candidate_objective < current_objective:
                # Solution is better
                weight_score = 1
            else:
                # solution is accepted
                weight_score = 2
            current = candidate
            current_objective = candidate_objective
        else:
            # solution is rejected
            weight_score = 3

        if candidate_objective < best_objective:
            # solution is new global best
            current = candidate
            current_objective = candidate_objective
            best = candidate
            best_objective = candidate_objective
            weight_score = 0

        return best, best_objective, current, current_objective, weight_score

    def iterate(self, initial_solution, initial_objective, num_iterations):
        # Reaction_factor (r) is a parameter that controls how fast weights adjust.

        weights = np.asarray(self.weights, dtype=np.float16)
        current = best = initial_solution
        current_objective = best_objective = initial_objective

        d_weights = np.ones(len(self._destroy_operators), dtype=np.float16)
        r_weights = np.ones(len(self._repair_operators), dtype=np.float16)

        i = 0
        while i < num_iterations:
            # Select destroy method
            destroy = self.select_operator(self.destroy_operators(), weights, self._rnd_state)

            # Select repair method
            repair = self.select_operator(self.repair_operators(), weights, self._rnd_state)

            # Destroy solution
            d_name, d_operator = self.destroy_operators[destroy]
            destroyed = d_operator(current, self._rnd_state)

            # Fix solution
            r_name, r_operator = self.repair_operators[repair]
            candidate, candidate_objective = r_operator(destroyed, self._rnd_state)

            # Compare solutions
            best, best_objective, current, current_objective, weight_score = self.evaluate_candidate(best,
                                                                                                     best_objective,
                                                                                                     current,
                                                                                                     current_objective,
                                                                                                     candidate,
                                                                                                     candidate_objective)

            # Update weights
            d_weights = d_weights * (1 - self.reaction_factor) + (self.reaction_factor * weights[weight_score])
            r_weights = r_weights * (1 - self.reaction_factor) + (self.reaction_factor * weights[weight_score])

            i += 1

        return best, best_objective
