# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
# ---

import pandas as pd
from decouple import config
import sys
import numpy.random as rnd
import traceback
import cProfile
from profiling.profiler import Profile
from heuristic.construction.construction import ConstructionHeuristic
from heuristic.improvement.alns import ALNS
from config.main_config import *
from heuristic.improvement.initial.initial_operators import Operators
from heuristic.improvement.reopt.reopt_operators import ReOptOperators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing
from simulation.simulator import Simulator
from heuristic.improvement.reopt.disruption_updater import DisruptionUpdater
from heuristic.improvement.reopt.new_request_updater import NewRequestUpdater


def main(test_instance, test_instance_date,
         run, iterations):
    constructor, simulator = None, None

    try:
        # TRACKING
        tracking = []

        # CUMULATIVE OBJECTIVE
        cumulative_infeasible, cumulative_recalibration, cumulative_objective = 0, timedelta(
            0), timedelta(0)

        # CONSTRUCTION OF INITIAL SOLUTION
        df = pd.read_csv(config(test_instance))
        constructor = ConstructionHeuristic(requests=df, vehicles=V)
        print("Constructing initial solution")
        initial_route_plan, initial_objective, initial_infeasible_set = constructor.construct_initial()
        cumulative_infeasible = len(initial_infeasible_set)

        # IMPROVEMENT OF INITIAL SOLUTION
        random_state = rnd.RandomState()

        criterion = SimulatedAnnealing(cooling_rate)

        alns = ALNS(weights, reaction_factor, initial_route_plan, initial_objective, initial_infeasible_set, criterion,
                    destruction_degree, constructor, rnd_state=rnd.RandomState())

        operators = Operators(alns)

        alns.set_operators(operators)

        # Run ALNS
        delayed = (False, None, None)

        current_route_plan, current_objective, current_infeasible_set, _ = alns.iterate(
            iterations, None, None, None, delayed)

        if current_infeasible_set:
            cumulative_infeasible = len(current_infeasible_set)
            print(
                "Error: The service cannot serve the number of initial requests required")
            current_infeasible_set = []

        # Recalibrate current solution
        current_route_plan = constructor.recalibrate_solution(
            current_route_plan)

        delta_dev_objective = constructor.get_delta_objective(
            current_route_plan, [], current_objective)
        cumulative_recalibration += delta_dev_objective
        current_objective -= delta_dev_objective

        tracking.append([cumulative_objective,
                        (datetime.now() - start_time).total_seconds()])

        df_tracking = pd.DataFrame(
            tracking, columns=["Current Objective", "Solution Time"])
        df_tracking.to_csv(config("tuning_path") + "param_tuning" +
                           str(iterations) + "_" + str(run) + "_" + test_instance + ".csv")  # Path:  param_tuning_numIterations_run_instance

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
    """
    # Profiling
    profile = Profile()
    cProfile.run('main()', 'profiling/restats')
    profile.display()

    # Generate test instance datetime from filename
    test_instance_d = test_instance.split(
                "/")[-1].split("_")[-1].split(".")[0]
            test_instance_date = test_instance_d[0:4] + "-" + \
                test_instance_d[4:6] + "-" + \
                test_instance_d[6:8] + " 10:00:00"
    """
    runs = 10
    iteration_tests = [10, 20, 30]

    for num_iterations in iteration_tests:
        for test_instance in test_instances:
            for run in range(runs):
                main(test_instance, "2021-05-10 10:00:00",
                     run, num_iterations)
