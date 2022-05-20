from copy import copy
import pandas as pd
from decouple import config
import sys
import numpy.random as rnd
import traceback
from heuristic.construction.construction import ConstructionHeuristic
from heuristic.improvement.alns import ALNS
from config.main_config import *
from heuristic.improvement.initial.initial_operators import Operators
from heuristic.improvement.reopt.reopt_operators import ReOptOperators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing
from simulation.simulator import Simulator
from heuristic.improvement.reopt.disruption_updater import DisruptionUpdater
from heuristic.improvement.reopt.new_request_updater import NewRequestUpdater


def main(test_instance, test_instance_date, run, repair_removed, destroy_removed):
    constructor, simulator = None, None

    try:
        # TRACKING
        start_time = datetime.now()
        df_operators, df_runtime, df_reqs = [], [], []

        # CUMULATIVE OBJECTIVE
        cumulative_rejected, cumulative_recalibration, cumulative_objective = 0, timedelta(
            0), timedelta(0)

        # CONSTRUCTION OF INITIAL SOLUTION
        df = pd.read_csv(config(test_instance))
        constructor = ConstructionHeuristic(
            requests=df, vehicles=V, alpha=alpha, beta=beta)
        print("Constructing initial solution")
        initial_route_plan, initial_objective, initial_infeasible_set = constructor.construct_initial()

        # IMPROVEMENT OF INITIAL SOLUTION
        criterion = SimulatedAnnealing(cooling_rate)

        alns = ALNS(weights, reaction_factor, initial_route_plan, initial_objective, initial_infeasible_set, criterion,
                    destruction_degree, constructor, rnd_state=rnd.RandomState())

        operators = Operators(alns)

        alns.set_operators(operators, repair_removed, destroy_removed)

        # Run ALNS
        delayed = (False, None, None)

        df_operators, current_route_plan, current_objective, current_infeasible_set, _ = alns.iterate(
            initial_iterations, initial_Z, None, None, None, delayed, False, df_operators, run)

        if current_infeasible_set:
            cumulative_rejected = len(current_infeasible_set)
            print(
                "Error: The service cannot serve the number of initial requests required")
            print("Number of rejected", cumulative_rejected)
            current_infeasible_set = []

        # Recalibrate current solution
        current_route_plan = constructor.recalibrate_solution(
            current_route_plan)

        delta_dev_objective = constructor.get_delta_objective(
            current_route_plan, [i for i in range(cumulative_rejected)], current_objective)
        cumulative_recalibration += delta_dev_objective
        current_objective -= delta_dev_objective

        df_runtime.append(
            [run, "Initial", (datetime.now() - start_time).total_seconds(), current_objective.total_seconds()])

        print("Initial objective", current_objective.total_seconds())
        print("Initial rejected", cumulative_rejected)

        # SIMULATION
        print("Start simulation")
        sim_clock = datetime.strptime(
            test_instance_date, "%Y-%m-%d %H:%M:%S")
        simulator = Simulator(sim_clock)
        new_request_updater = NewRequestUpdater(
            constructor)
        disruption_updater = DisruptionUpdater(new_request_updater)
        rejected = []
        print("Length of disruption stack", len(simulator.disruptions_stack))
        while len(simulator.disruptions_stack) > 0:
            start_time = datetime.now()
            prev_inf_len = cumulative_rejected
            delayed, delay_deltas = (False, None, None), []
            i = 0
            prev_objective = current_objective
            rejection = False

            # use correct data path
            disruption_type, disruption_time, disruption_info = simulator.get_disruption(
                current_route_plan, config("data_simulator_path"))

            # updates before heuristic
            disrupt = (False, None)
            if disruption_type == 4:  # No disruption
                continue
            elif disruption_type == 0:  # Disruption: new request
                current_route_plan, vehicle_clocks, artificial_depot = disruption_updater.update_route_plan(
                    current_route_plan, disruption_type, disruption_info, disruption_time)
                updated_objective = new_request_updater.new_objective(
                    current_route_plan, [], False)
                current_route_plan, removed_filtering = disruption_updater.filter_route_plan(
                    current_route_plan, vehicle_clocks, None, disruption_type, False)  # Filter route plan
                filter_objective = new_request_updater.new_objective(
                    current_route_plan, [], False)
                cumulative_objective = copy(
                    cumulative_objective) + copy(updated_objective) - copy(filter_objective)
                current_route_plan, current_objective, current_infeasible_set, vehicle_clocks, rejection, rid = new_request_updater.\
                    greedy_insertion_new_request(
                        current_route_plan, current_infeasible_set, disruption_info, disruption_time, vehicle_clocks, i, filter_objective)
                disruption_type = str(disruption_type) + "_False"
                if rejection:
                    disruption_type = str(disruption_type) + "_True"
                    rejected.append(rid)
                    cumulative_rejected += 1
                    for i in range(1, N_R+1):
                        current_route_plan, current_objective, current_infeasible_set, vehicle_clocks, rejection, rid = new_request_updater.\
                            greedy_insertion_new_request(
                                current_route_plan, current_infeasible_set, disruption_info, disruption_time, vehicle_clocks, i, filter_objective)
                        if not rejection:
                            rejected.remove(rid)
                            cumulative_rejected -= 1
                            break
                current_infeasible_set = []
                df_reqs.append(
                    [run, rid, (datetime.now() - start_time).total_seconds()])

            else:
                current_route_plan, vehicle_clocks, artificial_depot = disruption_updater.update_route_plan(
                    current_route_plan, disruption_type, disruption_info, disruption_time)
                before_route_plan = list(map(list, current_route_plan))
                updated_objective = new_request_updater.new_objective(
                    current_route_plan, [], False)
                current_route_plan, removed_filtering = disruption_updater.filter_route_plan(
                    current_route_plan, vehicle_clocks, disruption_info, disruption_type, artificial_depot)  # Filter route plan
                filter_objective = new_request_updater.new_objective(
                    current_route_plan, [], False)
                cumulative_objective = copy(
                    cumulative_objective) + copy(updated_objective) - copy(filter_objective)
                current_objective = new_request_updater.new_objective(
                    current_route_plan, current_infeasible_set, False)
                if disruption_type == 2 or disruption_type == 3:  # Disruption: cancel or no show
                    index_removed = [(disruption_info[3], disruption_info[0], disruption_info[1] - removed_filtering),
                                     (disruption_info[4], disruption_info[0], disruption_info[2] - removed_filtering)]
                    index_removed[0] = (
                        None, None, None) if artificial_depot or disruption_type == 3 else index_removed[0]
                    disrupt = (True, index_removed)
                elif disruption_type == 1:  # Disruption: delay
                    node_idx = next(i for i, (node, *_) in enumerate(current_route_plan[disruption_info[0]]) if
                                    node == disruption_info[3])
                    delayed = (True, disruption_info[0], node_idx)
                    delay_deltas.append(current_objective)
            if not rejection:
                # Heuristic
                criterion = SimulatedAnnealing(cooling_rate)

                alns = ALNS(weights, reaction_factor, current_route_plan, current_objective, current_infeasible_set,
                            criterion,
                            destruction_degree, new_request_updater, rnd_state=rnd.RandomState())

                operators = ReOptOperators(
                    alns, disruption_time, vehicle_clocks)

                alns.set_operators(operators, repair_removed, destroy_removed)

                # Run ALNS
                df_operators, current_route_plan, current_objective, current_infeasible_set, still_delayed_nodes = alns.iterate(
                    reopt_iterations, reopt_Z, disrupt[0], disrupt[1], disruption_time, delayed, True, df_operators, run)

                if delayed[0]:
                    delay_deltas[-1] = delay_deltas[-1] - current_objective
                    current_route_plan = disruption_updater.recalibrate_solution(
                        current_route_plan, disruption_info, still_delayed_nodes)

                    delta_dev_objective = new_request_updater.get_delta_objective(
                        current_route_plan, current_infeasible_set, current_objective)
                    cumulative_recalibration += delta_dev_objective
                    current_objective -= delta_dev_objective

            total_objective = new_request_updater.total_objective(current_objective,
                                                                  cumulative_objective,
                                                                  cumulative_recalibration, cumulative_rejected, rejection)

            df_runtime.append(
                [run, str(disruption_type), (datetime.now() - start_time).total_seconds(), total_objective.total_seconds()])

        print("End simulation")
        print("Rejected rids", rejected)

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

    return df_operators, df_runtime, df_reqs


if __name__ == "__main__":
    """
    # Profilinga
    profile = Profile()
    cProfile.run('main()', 'profiling/restats')
    profile.display()
    """

    # Generate test instance datetime from filename
    test_instance_d = test_instance.split(
        "/")[-1].split("_")[-1].split(".")[0]
    test_instance_date = test_instance_d[0:4] + "-" + \
        test_instance_d[4:6] + "-" + \
        test_instance_d[6:8] + " 10:00:00"
    print("Test instance:", test_instance)

    repair_removed = None
    destroy_removed = None
    runs = 5
    df_requests_runs, df_runtime_runs, df_operators_runs = [], [], []
    for run in range(runs):
        df_operators, df_runtime, df_req_runtime = main(
            test_instance, test_instance_date, run, repair_removed, destroy_removed)
        df_requests_runs.append(pd.DataFrame(df_req_runtime, columns=[
            "Run", "Request", "Response Time"]))
        df_runtime_runs.append(pd.DataFrame(df_runtime, columns=[
            "Run", "Disruption Type", "Solution Time", "Objective"]))
        df_operators_runs.append(pd.DataFrame(df_operators, columns=[
            "Run", "Initial", "Iteration", "Destroy Operator", "Repair Operator", "Destroy Weight", "Repair Weight", "Update destroy weight score", "Update repair weight score", "Destroy Used", "Repair Used", "Runtime", "Updated this round", "Best Objective"]))

    df_track_req_runtime = pd.concat(df_requests_runs)
    df_track_req_runtime.to_csv(
        config("run_path") + "value_of_adapt" + test_instance + "runtime_reqs" + ".csv")

    df_track_runtime = pd.concat(df_runtime_runs)
    df_track_runtime.to_csv(config("run_path") + "value_of_adapt" +
                            test_instance + "computational_time" + ".csv")

    df_operators_total = pd.concat(df_operators_runs)
    df_operators_total.to_csv(
        config("run_path") + "value_of_adapt"+test_instance + "impact_operators" + ".csv")

    print("DONE WITH ALL RUNS")

"""
TODO:
    - Add index to removed repair operator:
        - None = none removed 
        - 0 = greedy_repair
        - 1 = regret_2_repair
        - 2 = regret_3_repair
    - Add index to removed destroy operator:
        - None = none removed
        - 0 = random_removal
        - 1 = time_related_removal
        - 2 = distance_related_removal
        - 3 = related_removal
        - 4 = worst_deviation_removal
"""
