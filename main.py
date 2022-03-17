import copy

import pandas as pd
from decouple import config
import sys
import numpy.random as rnd
import numpy as np
import traceback
from heuristic.construction.construction import ConstructionHeuristic
from heuristic.construction.heuristic_config import *
from heuristic.improvement.alns import ALNS
from heuristic.improvement.improvement_config import *
from heuristic.improvement.operators import Operators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing
from simulation.simulator import Simulator
from updater.updater import Updater


def main():
    constructor = None
    simulator = None

    try:
        # CONSTRUCTION OF INITIAL SOLUTION
        df = pd.read_csv(config("test_data_construction"))
        constructor = ConstructionHeuristic(requests=df.head(
            60), vehicles=V)  # husk å oppdatere counter_rid også
        print("Constructing initial solution")
        initial_route_plan, initial_objective, initial_infeasible_set = constructor.construct_initial()
        print("Initial objective: ", initial_objective)

        # IMPROVEMENT OF INITIAL SOLUTION
        random_state = rnd.RandomState()

        criterion = SimulatedAnnealing(
            start_temperature, end_temperature, step)

        alns = ALNS(weights, reaction_factor, initial_route_plan, initial_objective, initial_infeasible_set, criterion,
                    destruction_degree, constructor, random_state)

        operators = Operators(alns)

        alns.set_operators(operators)

        # Run ALNS
        current_route_plan, current_objective, current_infeasible_set = alns.iterate(
            iterations)
        # print(current_route_plan)
        print("Objective", current_objective)
        print("Num vehicles:", len(current_route_plan))
        print(current_infeasible_set)
        constructor.print_new_objective(
            initial_route_plan, initial_infeasible_set)
        constructor.print_new_objective(
            current_route_plan, current_infeasible_set)

        # SIMULATION
        print("Start simulation")
        sim_clock = datetime.strptime(
            "2021-05-10 10:00:00", "%Y-%m-%d %H:%M:%S")
        simulator = Simulator(sim_clock)
        updater = Updater()
        first_iteration = True
        counter_rid = 20

        while len(simulator.disruptions_stack) > 0:
            # use correct data path
            if not first_iteration:
                disruption_type, disruption_time, disruption_info = simulator.get_disruption(current_route_plan, config(
                    "data_simulator_path"))
            else:
                disruption_type, disruption_time, disruption_info = simulator.get_disruption(current_route_plan, config(
                    "data_processed_path"))
                first_iteration = False

            # updates before heuristic
            if disruption_type == 'request':
                # add new request to infeasible set - diskutere om dette skal gjøres
                # evt greedy insertion her
                counter_rid += 1
                #(counter_rid, disruption_info)
            elif disruption_type == 'no disruption':
                continue
            else:
                current_route_plan = updater.update_route_plan(
                    current_route_plan, disruption_type, disruption_info)
                current_objective = constructor.new_objective(
                    current_route_plan, current_infeasible_set)  # update objective?

            # heuristic
            alns = ALNS(weights, reaction_factor, current_route_plan, current_objective, current_infeasible_set,
                        criterion,
                        destruction_degree, constructor, random_state)

            operators = Operators(alns)

            alns.set_operators(operators)

            # Run ALNS
            current_route_plan, current_objective, current_infeasible_set = alns.iterate(
                iterations)

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
