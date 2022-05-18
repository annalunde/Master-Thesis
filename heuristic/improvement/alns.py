from copy import copy
import numpy as np
import numpy.random as rnd
from math import ceil
from numpy import log
from config.main_config import *
from datetime import datetime, timedelta
from heuristic.improvement.destroy_repair_updater import Destroy_Repair_Updater


class ALNS:
    def __init__(self, weights, reaction_factor, current_route_plan, current_objective, initial_infeasible_set,
                 criterion, destruction_degree, constructor, rnd_state=rnd.RandomState()):
        # Reaction_factor (r) is a parameter that controls how fast weights adjust.
        # Weights is array of four elements, index 0 highest, 3 lowest
        self.destroy_operators, self.repair_operators = [], []
        self.rnd_state = rnd_state
        self.reaction_factor = reaction_factor
        self.initial_infeasible_set = initial_infeasible_set
        self.weights = weights
        self.route_plan = current_route_plan
        self.destruction_degree = destruction_degree
        self.objective = current_objective
        self.criterion = criterion
        self.constructor = constructor
        self.destroy_repair_updater = Destroy_Repair_Updater(constructor)

    # Run ALNS algorithm
    def iterate(self, num_iterations, z, disrupted, index_removed, disruption_time, delayed, reopt, df_operators, run):

        N_U = N_U_reopt if reopt else N_U_init
        weights = np.asarray(self.weights, dtype=np.float16)
        best, current_route_plan, initial_route_plan = list(map(list, self.route_plan)), list(
            map(list, self.route_plan)), list(map(list, self.route_plan))
        current_objective, best_objective = copy(
            self.objective), copy(self.objective)
        current_infeasible_set, best_infeasible_set = copy(
            self.initial_infeasible_set), copy(self.initial_infeasible_set)
        found_solutions = {}

        d_weights, r_weights = np.ones(len(self.destroy_operators), dtype=np.float16), np.ones(
            len(self.repair_operators), dtype=np.float16)
        d_scores, r_scores = np.ones(len(self.destroy_operators), dtype=np.float16), np.ones(
            len(self.repair_operators), dtype=np.float16)
        d_count, r_count = np.zeros(len(self.destroy_operators), dtype=np.float16), np.zeros(
            len(self.repair_operators), dtype=np.float16)

        if disrupted:
            # Update disrupted solution
            current_route_plan = self.destroy_repair_updater.update_solution(
                current_route_plan, index_removed, disruption_time)

        for i in range(num_iterations):
            start_time = datetime.now()
            updated_now = False
            already_found = False
            still_delayed_nodes = []

            # Select destroy method
            destroy = self.select_operator(
                self.destroy_operators, d_weights, self.rnd_state)

            # Select repair method
            repair = self.select_operator(
                self.repair_operators, r_weights, self.rnd_state)

            # Destroy solution
            d_operator = self.destroy_operators[destroy]
            destroyed_route_plan, removed_requests, index_removed, destroyed = d_operator(
                current_route_plan, current_infeasible_set)

            if not destroyed:
                break

            d_count[destroy] += 1

            if delayed[0]:
                still_delayed_nodes = self.filter_still_delayed(
                    delayed, current_route_plan=destroyed_route_plan, initial_route_plan=initial_route_plan)

            # Update solution
            trunc_route_plan = self.destroy_repair_updater.update_solution(
                destroyed_route_plan, index_removed, disruption_time)
            updated_route_plan = self.destroy_repair_updater.update_capacities(
                trunc_route_plan, index_removed)

            # Fix solution
            r_operator = self.repair_operators[repair]
            candidate, candidate_objective, candidate_infeasible_set = r_operator(
                updated_route_plan, removed_requests, current_infeasible_set, current_route_plan, index_removed,
                delayed, still_delayed_nodes)

            r_count[repair] += 1

            if i == 0:
                self.criterion.temperature = -((current_objective * (1 + z) - current_objective).total_seconds()/60) /\
                    (log(0.5))

            # Compare solutions
            best, best_objective, best_infeasible_set, current_route_plan, current_objective, current_infeasible_set, weight_score = self.evaluate_candidate(
                best, best_objective, best_infeasible_set,
                current_route_plan, current_objective, current_infeasible_set,
                candidate, candidate_objective, candidate_infeasible_set, self.criterion)

            if hash(str(candidate)) == hash(str(current_route_plan)) and hash(str(candidate)) in found_solutions.keys():
                already_found = True
            else:
                found_solutions[hash(str(current_route_plan))] = 1

            if not already_found:
                # Update scores
                d_scores[destroy] += weights[weight_score]
                r_scores[repair] += weights[weight_score]

            # After a certain number of iterations, update weight
            if (i+1) % ceil(num_iterations * N_U) == 0:
                # Update weights with scores
                updated_now = True
                for destroy in range(len(d_weights)):
                    if d_count[destroy] == 0:
                        continue
                    d_weights[destroy] = d_weights[destroy] * \
                        (1 - self.reaction_factor) + \
                        (self.reaction_factor *
                            d_scores[destroy] / d_count[destroy])
                for repair in range(len(r_weights)):
                    if r_count[repair] == 0:
                        continue
                    r_weights[repair] = r_weights[repair] * \
                        (1 - self.reaction_factor) + \
                        (self.reaction_factor *
                            r_scores[repair] / r_count[repair])

                # Reset scores
                d_scores, r_scores = np.ones(
                    len(self.destroy_operators), dtype=np.float16), np.ones(
                    len(self.repair_operators), dtype=np.float16)
                d_count, r_count = np.zeros(
                    len(self.destroy_operators), dtype=np.float16), np.zeros(
                    len(self.repair_operators), dtype=np.float16)

        df_operators.append([run, not reopt, i, d_operator, r_operator, d_weights[destroy], r_weights[repair], d_scores[destroy],
                             r_scores[repair], d_count[destroy], r_count[repair], (datetime.now() - start_time).total_seconds(), updated_now, best_objective])

        if delayed[0]:
            still_delayed_nodes = self.filter_still_delayed(
                delayed, best, initial_route_plan)

        return df_operators, best, best_objective, best_infeasible_set, still_delayed_nodes

    def set_operators(self, operators, repair_removed, destroy_removed):
        # Add destroy operators
        self.add_destroy_operator(operators.random_removal)
        self.add_destroy_operator(operators.time_related_removal)
        self.add_destroy_operator(operators.distance_related_removal)
        self.add_destroy_operator(operators.related_removal)
        self.add_destroy_operator(operators.worst_deviation_removal)
        if destroy_removed is not None:
            destroy_removed.sort(key=lambda x: x, reverse=True)
            for idx in destroy_removed:
                self.destroy_operators.pop(idx)

        # Add repair operators
        self.add_repair_operator(operators.greedy_repair)
        self.add_repair_operator(operators.regret_2_repair)
        self.add_repair_operator(operators.regret_3_repair)
        if repair_removed is not None:
            repair_removed.sort(key=lambda x: x, reverse=True)
            for idx in repair_removed:
                self.repair_operators.pop(idx)
    # Add operator to the heuristic instance

    def add_destroy_operator(self, operator):
        self.destroy_operators.append(operator)

    def add_repair_operator(self, operator):
        self.repair_operators.append(operator)

    # Select destroy/repair operator
    @ staticmethod
    def select_operator(operators, weights, rnd_state):
        w = weights / np.sum(weights)
        a = [i for i in range(len(operators))]
        return rnd_state.choice(a=a, p=w)

    @ staticmethod
    def filter_still_delayed(delayed, current_route_plan, initial_route_plan):
        initial_delayed_nodes = [i[0]
                                 for i in initial_route_plan[delayed[1]][delayed[2]:]]
        return [j[0] for j in current_route_plan[delayed[1]]
                if j[0] in initial_delayed_nodes]

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
            current = copy(candidate)
            current_objective = copy(candidate_objective)
            current_infeasible_set = copy(candidate_infeasible_set)
        else:
            # Solution is rejected
            weight_score = 3

        if candidate_objective <= best_objective and len(candidate_infeasible_set) <= len(best_infeasible_set):
            # Solution is new global best
            current, best = copy(candidate), copy(candidate)
            current_objective, best_objective = copy(
                candidate_objective), copy(candidate_objective)
            current_infeasible_set, best_infeasible_set = copy(
                candidate_infeasible_set), copy(candidate_infeasible_set)
            weight_score = 0

        return best, best_objective, best_infeasible_set, current, current_objective, current_infeasible_set, weight_score
