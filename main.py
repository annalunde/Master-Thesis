import copy

import pandas as pd
from decouple import config
import sys
import numpy.random as rnd
import numpy as np
import traceback
from heuristic.construction.construction import ConstructionHeuristic
from config.construction_config import *
from heuristic.improvement.alns import ALNS
from config.main_config import *
from heuristic.improvement.initial.initial_operators import Operators
from heuristic.improvement.reopt.reopt_operators import ReOptOperators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing
from simulation.simulator import Simulator
from heuristic.improvement.reopt.disruption_updater import DisruptionUpdater
from heuristic.improvement.reopt.new_request_updater import NewRequestUpdater


def main():
    constructor = None
    simulator = None

    try:
        # CONSTRUCTION OF INITIAL SOLUTION
        df = pd.read_csv(config("test_data_construction"))
        constructor = ConstructionHeuristic(requests=df.head(R), vehicles=V)
        print("Constructing initial solution")
        initial_route_plan, initial_objective, initial_infeasible_set = constructor.construct_initial()
        constructor.print_new_objective(
            initial_route_plan, initial_infeasible_set)
        delayed = (False, None, None)

        # IMPROVEMENT OF INITIAL SOLUTION
        random_state = rnd.RandomState()

        criterion = SimulatedAnnealing(
            start_temperature, end_temperature, step)

        alns = ALNS(weights, reaction_factor, initial_route_plan, initial_objective, initial_infeasible_set, criterion,
                    destruction_degree, constructor, num_update, random_state)

        operators = Operators(alns)

        alns.set_operators(operators)

        # Run ALNS
        current_route_plan, current_objective, current_infeasible_set, _ = alns.iterate(
            iterations, None, None, None, delayed)

        constructor.print_new_objective(
            current_route_plan, current_infeasible_set)

        # SIMULATION
        print("Start simulation")
        sim_clock = datetime.strptime(
            "2021-05-10 10:00:00", "%Y-%m-%d %H:%M:%S")
        simulator = Simulator(sim_clock)
        new_request_updater = NewRequestUpdater(
            df.head(R), V, initial_infeasible_set)
        disruption_updater = DisruptionUpdater(new_request_updater)
        first_iteration = True

        while len(simulator.disruptions_stack) > 0:
            prev_inf_len = len(current_infeasible_set)
            delayed = (False, None, None)
            delay_deltas = []

            # use correct data path
            if not first_iteration:
                disruption_type, disruption_time, disruption_info = simulator.get_disruption(current_route_plan, config(
                    "data_simulator_path"))
            else:
                disruption_type, disruption_time, disruption_info = simulator.get_disruption(current_route_plan, config(
                    "data_processed_path"))
                first_iteration = False

            print("Disruption type", disruption_type)
            print("Disruption time:", disruption_time)
            print()
            # updates before heuristic
            disrupt = (False, None)
            if disruption_type == 'request':
                current_route_plan = disruption_updater.update_route_plan(
                    current_route_plan, disruption_type, disruption_info)
                current_route_plan, current_objective, current_infeasible_set = new_request_updater.\
                    greedy_insertion_new_request(
                        current_route_plan, current_infeasible_set, disruption_info, disruption_time)
            elif disruption_type == 'no disruption':
                continue
            else:
                if disruption_type == 'delay':
                    print("hey")

                current_route_plan = disruption_updater.update_route_plan(
                    current_route_plan, disruption_type, disruption_info)
                current_objective = new_request_updater.new_objective(
                    current_route_plan, current_infeasible_set)

                if disruption_type == 'cancel' or disruption_type == 'no show':
                    index_removed = [(disruption_info[3], disruption_info[0], disruption_info[1]),
                                     (disruption_info[4], disruption_info[0], disruption_info[2])]
                    disrupt = (True, index_removed)
                elif disruption_type == 'delay':
                    delayed = (True, disruption_info[0], disruption_info[1])
                    delay_deltas.append(current_objective)

            # heuristic
            alns = ALNS(weights, reaction_factor, current_route_plan, current_objective, current_infeasible_set,
                        criterion,
                        destruction_degree, new_request_updater, random_state)

            operators = ReOptOperators(alns, disruption_time)

            alns.set_operators(operators)

            # Run ALNS
            current_route_plan, current_objective, current_infeasible_set, still_delayed_nodes = alns.iterate(
                iterations, disrupt[0], disrupt[1], disruption_time, delayed)
            if delayed[0]:
                delay_deltas[-1] = delay_deltas[-1] - current_objective
                print("Reduction in objective of delay: ", delay_deltas[-1])
                current_route_plan = disruption_updater.recalibrate_solution(
                    current_route_plan, disruption_info, still_delayed_nodes)

            if disruption_type == 'request' and not(len(current_infeasible_set) > prev_inf_len):
                print("New request inserted")

    except Exception as e:
        print("ERROR:", e)
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        full_traceback = traceback.format_exc()
        print("FULL TRACEBACK: ", full_traceback)

        print("Exception type: ", exception_type)
        print("File name: ", filename)
        print("Line number: ", line_number)


if __name__ == "__main__":
    main()
