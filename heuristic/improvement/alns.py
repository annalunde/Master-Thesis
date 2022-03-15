import copy
import numpy as np
import numpy.random as rnd
from tqdm import tqdm
from collections import OrderedDict
from heuristic.improvement.operators import Operators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing
from heuristic.improvement.improvement_config import *


class ALNS:
    def __init__(self, weights, reaction_factor, current_route_plan, current_objective, initial_infeasible_set, criterion, destruction_degree, constructor, rnd_state=rnd.RandomState()):
        # Reaction_factor (r) is a parameter that controls how fast weights adjust.
        # Weights is array of four elements, index 0 highest, 3 lowest
        self.destroy_operators = []
        self.repair_operators = []
        self.rnd_state = rnd_state
        self.reaction_factor = reaction_factor
        self.initial_infeasible_set = initial_infeasible_set
        self.weights = weights
        self.route_plan = current_route_plan
        self.destruction_degree = destruction_degree
        self.objective = current_objective
        self.criterion = criterion
        self.constructor = constructor

    # Run ALNS algorithm

    def iterate(self, num_iterations):
        weights = np.asarray(self.weights, dtype=np.float16)
        current = copy.deepcopy(self.route_plan)
        best = copy.deepcopy(self.route_plan)
        current_objective = copy.deepcopy(self.objective)
        best_objective = copy.deepcopy(self.objective)
        current_infeasible_set = copy.deepcopy(self.initial_infeasible_set)
        best_infeasible_set = copy.deepcopy(self.initial_infeasible_set)

        d_weights = np.ones(len(self.destroy_operators), dtype=np.float16)
        r_weights = np.ones(len(self.repair_operators), dtype=np.float16)
        d_count = np.zeros(len(self.destroy_operators), dtype=np.float16)
        r_count = np.zeros(len(self.repair_operators), dtype=np.float16)

        for i in tqdm(range(num_iterations), colour='#39ff14'):
            # Select destroy method
            destroy = self.select_operator(
                self.destroy_operators, d_weights, self.rnd_state)

            # Select repair method
            repair = self.select_operator(
                self.repair_operators, r_weights, self.rnd_state)

            # Destroy solution
            d_operator = self.destroy_operators[destroy]

            destroyed_route, removed_requests = d_operator(
                current)
            d_count[destroy] += 1

            # Fix solution
            r_operator = self.repair_operators[repair]
            candidate, candidate_objective, candidate_infeasible_set = r_operator(
                destroyed_route, removed_requests, current_infeasible_set)
            if current_infeasible_set:
                print(
                    "ERROR: You cannot serve all obligatory requests with current fleet.")
                break

            r_count[repair] += 1

            # Compare solutions
            best, best_objective, best_infeasible_set, current, current_objective, current_infeasible_set, weight_score = self.evaluate_candidate(
                best, best_objective, best_infeasible_set,
                current, current_objective, current_infeasible_set,
                candidate, candidate_objective, candidate_infeasible_set, self.criterion)

            # Update weights
            d_weights[destroy] = d_weights[destroy] * \
                (1 - self.reaction_factor) + \
                (self.reaction_factor * weights[weight_score]/d_count[destroy])
            r_weights[repair] = r_weights[repair] * \
                (1 - self.reaction_factor) + \
                (self.reaction_factor * weights[weight_score]/r_count[repair])

        return best, best_objective, best_infeasible_set

    def set_operators(self, operators):
        # Add destroy operators
        self.add_destroy_operator(operators.random_removal)
        self.add_destroy_operator(operators.time_related_removal)
        self.add_destroy_operator(operators.distance_related_removal)
        self.add_destroy_operator(operators.related_removal)
        self.add_destroy_operator(operators.worst_deviation_removal)

        # Add repair operators
        self.add_repair_operator(operators.greedy_repair)
        # alns.add_repair_operator(operators.regret_repair)

    # Add operator to the heuristic instance

    def add_destroy_operator(self, operator):
        self.destroy_operators.append(operator)

    def add_repair_operator(self, operator):
        self.repair_operators.append(operator)

    # Select destroy/repair operator
    @staticmethod
    def select_operator(operators, weights, rnd_state):
        w = weights / np.sum(weights)
        a = [i for i in range(len(operators))]
        return rnd_state.choice(a=a, p=w)

    # Evaluate candidate
    def evaluate_candidate(self, best, best_objective, best_infeasible_set, current, current_objective,
                           current_infeasible_set, candidate, candidate_objective, candidate_infeasible_set,
                           criterion):
        # If solution is accepted by criterion (simulated annealing)
        if criterion.accept_criterion(self.rnd_state, current_objective, candidate_objective):
            if candidate_objective <= current_objective:
                # Solution is better
                weight_score = 1
            else:
                # Solution is not better, but accepted
                weight_score = 2
            current = copy.deepcopy(candidate)
            current_objective = copy.deepcopy(candidate_objective)
            current_infeasible_set = copy.deepcopy(candidate_infeasible_set)
        else:
            # Solution is rejected
            weight_score = 3

        if candidate_objective <= best_objective:
            # Solution is new global best
            current = copy.deepcopy(candidate)
            current_objective = copy.deepcopy(candidate_objective)
            current_infeasible_set = copy.deepcopy(candidate_infeasible_set)
            best = copy.deepcopy(candidate)
            best_objective = copy.deepcopy(candidate_objective)
            best_infeasible_set = copy.deepcopy(candidate_infeasible_set)
            weight_score = 0

        return best, best_objective, best_infeasible_set, current, current_objective, current_infeasible_set, weight_score