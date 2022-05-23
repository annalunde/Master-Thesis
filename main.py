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
from measures import Measures
import argparse


def main(test_instance, test_instance_date, run, repair_removed, destroy_removed, breakpoint_hour_date):
    constructor, simulator = None, None

    try:
        # TRACKING
        start_time = datetime.now()
        df_run_b2, df_run_a2 = [], []
        cost_per_trip_b2, cost_per_trip_a2, ride_sharing_passengers_b2, ride_sharing_passengers_a2, ride_sharing_arcs_b2, ride_sharing_arcs_a2, processed_nodes_b2, processed_nodes_a2, cpt_b2, cpt_a2, ride_sharing_b2, ride_sharing_a2 = {
            idx: (0, None, None) for idx in range(V_before2+standby)}, {idx: (0, None, None) for idx in range(V_after2+standby)}, 0, 0, set(), 0, 0

        # CUMULATIVE OBJECTIVE
        cumulative_rejected_b2, cumulative_rejected_a2, cumulative_recalibration_b2, cumulative_recalibration_a2, cumulative_objective_b2, cumulative_objective_a2, cumulative_travel_time_b2, \
            cumulative_travel_time_a2, cumulative_deviation_a2, cumulative_deviation_b2 = 0, 0, timedelta(
                0), timedelta(0), timedelta(0), timedelta(0), timedelta(
                0), timedelta(0), timedelta(0), timedelta(0)

        # CONSTRUCTION OF INITIAL SOLUTION
        df = pd.read_csv(config(test_instance))
        constructor = ConstructionHeuristic(
            requests=df, vehicles_before2=V_before2, vehicles_after2=V_after2, alpha=alpha, beta=beta)
        print("Constructing initial solution")
        initial_route_plan_b2, initial_objective_b2, initial_infeasible_set_b2, initial_route_plan_a2, initial_objective_a2, initial_infeasible_set_a2 = constructor.construct_initial()

        # IMPROVEMENT OF INITIAL SOLUTION
        criterion = SimulatedAnnealing(cooling_rate)

        alns = ALNS(weights, reaction_factor, initial_route_plan_b2, initial_objective_b2, initial_infeasible_set_b2, initial_route_plan_a2, initial_objective_a2, initial_infeasible_set_a2, criterion,
                    destruction_degree, constructor, rnd_state=rnd.RandomState())

        operators = Operators(alns)

        alns.set_operators(operators, repair_removed, destroy_removed)

        # Run ALNS
        delayed_b2 = (False, None, None)
        delayed_a2 = (False, None, None)

        current_route_plan_b2, current_objective_b2, current_infeasible_set_b2, current_route_plan_a2, current_objective_a2, current_infeasible_set_a2, _ = alns.iterate(
            initial_iterations, initial_Z, None, None, None, delayed_b2, delayed_a2, False, run)

        if current_infeasible_set_b2 or current_infeasible_set_a2:
            cumulative_rejected_b2 = len(current_infeasible_set_b2)
            cumulative_rejected_a2 = len(current_infeasible_set_a2)
            print(
                "Error: The service cannot serve the number of initial requests required")
            print("Number of rejected before 2", cumulative_rejected_b2)
            print("Number of rejected after 2", cumulative_rejected_a2)
            current_infeasible_set_b2 = []
            current_infeasible_set_a2 = []

        # Recalibrate current solution
        current_route_plan_b2 = constructor.recalibrate_solution(
            current_route_plan_b2)
        current_route_plan_a2 = constructor.recalibrate_solution(
            current_route_plan_a2)

        delta_dev_objective_b2 = constructor.get_delta_objective(
            current_route_plan_b2, [i for i in range(cumulative_rejected_b2)], current_objective_b2)
        delta_dev_objective_a2 = constructor.get_delta_objective(
            current_route_plan_a2, [i for i in range(cumulative_rejected_a2)], current_objective_a2)
        cumulative_recalibration_b2 += delta_dev_objective_b2
        cumulative_recalibration_a2 += delta_dev_objective_a2
        current_objective_b2 -= delta_dev_objective_b2
        current_objective_a2 -= delta_dev_objective_a2

        ride_time_objective_b2, deviation_objective_b2, rejected_objective_b2 = constructor.print_objective(
            current_route_plan_b2, [i for i in range(cumulative_rejected_b2)])
        ride_time_objective_a2, deviation_objective_a2, rejected_objective_a2 = constructor.print_objective(
            current_route_plan_a2, [i for i in range(cumulative_rejected_a2)])

        total_objective_b2 = constructor.total_objective(current_objective_b2, cumulative_objective_b2,
                                                         cumulative_recalibration_b2)
        total_objective_a2 = constructor.total_objective(current_objective_a2, cumulative_objective_a2,
                                                         cumulative_recalibration_a2)

        df_run_b2.append([run, "Initial", total_objective_b2.total_seconds(), (datetime.now() - start_time).total_seconds(), cumulative_rejected_b2, rejected_objective_b2.total_seconds(),
                          cumulative_recalibration_b2.total_seconds()/beta, ride_time_objective_b2.total_seconds(), ride_sharing_b2, cpt_b2, len(current_route_plan_b2), "10:00"])
        df_run_a2.append([run, "Initial", total_objective.total_seconds(), (datetime.now() - start_time).total_seconds(), cumulative_rejected, rejected_objective.total_seconds(),
                          cumulative_recalibration.total_seconds()/beta, ride_time_objective.total_seconds(), ride_sharing, cpt, len(current_route_plan), "10:00"])

        cumulative_deviation_b2 = copy(cumulative_recalibration_b2)/beta
        cumulative_deviation_a2 = copy(cumulative_recalibration_a2)/beta

        # SIMULATION
        print("Start simulation")
        sim_clock = datetime.strptime(
            test_instance_date, "%Y-%m-%d %H:%M:%S")
        simulator = Simulator(sim_clock)
        new_request_updater = NewRequestUpdater(
            constructor)
        disruption_updater = DisruptionUpdater(
            new_request_updater, breakpoint_hour_date)
        rejected_b2, rejected_a2 = [], []
        print("Length of disruption stack", len(simulator.disruptions_stack))
        while len(simulator.disruptions_stack) > 0:
            start_time = datetime.now()
            prev_inf_len_b2, prev_inf_len_a2 = cumulative_rejected_b2, cumulative_rejected_a2
            delayed_b2, delayed_a2, delay_deltas_b2, delay_deltas_a2 = (
                False, None, None), (False, None, None), [], []
            i = 0
            prev_objective_b2, prev_objective_a2 = current_objective_b2, current_objective_a2
            rejection = False

            # use correct data path
            disruption_type, disruption_time, disruption_info = simulator.get_disruption(
                current_route_plan, config("data_simulator_path"))
            # updates before heuristic
            disrupt = (False, None)
            if disruption_type == 4:  # No disruption
                continue
            elif disruption_type == 0:  # Disruption: new request
                # Check if new request affects before 2 or after 2 route plan
                before2 = disruption_updater.before2(
                    disruption_type, disruption_info, disruption_time)
                current_route_plan = list(map(list, current_route_plan_b2)) if before2 else list(
                    map(list, current_route_plan_a2))
                current_route_plan, vehicle_clocks, artificial_depot = disruption_updater.update_route_plan(
                    current_route_plan, disruption_type, disruption_info, disruption_time)
                current_route_plan, removed_filtering, filtered_away, middle, filtered_size = disruption_updater.\
                    filter_route_plan(current_route_plan,
                                      vehicle_clocks, None, disruption_type, False)  # Filter route plan
                new_request_updater.middle = middle
                filter_objective = new_request_updater.new_objective(
                    current_route_plan, [], False)
                filter_away_objective, filter_away_travel_time, filter_away_deviation = \
                    new_request_updater.norm_objective(
                        filtered_away, [], True, filtered_size)
                if before2:
                    cumulative_objective_b2 = copy(
                        cumulative_objective_b2) + copy(filter_away_objective_b2)
                    cumulative_travel_time_b2 = copy(
                        cumulative_travel_time_b2) + copy(filter_away_travel_time_b2)
                    cumulative_deviation_b2 = copy(
                        cumulative_deviation_b2) + copy(filter_away_deviation_b2)

                else:
                    cumulative_objective_a2 = copy(
                        cumulative_objective_a2) + copy(filter_away_objective_a2)
                    cumulative_travel_time_a2 = copy(
                        cumulative_travel_time_a2) + copy(filter_away_travel_time_a2)
                    cumulative_deviation_a2 = copy(
                        cumulative_deviation_a2) + copy(filter_away_deviation_a2)

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

            else:
                removed_time = None
                if disruption_type == 2:
                    removed_time = current_route_plan[disruption_info[0]
                                                      ][disruption_info[1]][1]
                current_route_plan, vehicle_clocks, artificial_depot = disruption_updater.update_route_plan(
                    current_route_plan, disruption_type, disruption_info, disruption_time)

                current_route_plan, removed_filtering, filtered_away, middle, filtered_size = disruption_updater.\
                    filter_route_plan(current_route_plan,
                                      vehicle_clocks, disruption_info, disruption_type, artificial_depot)  # Filter route plan
                new_request_updater.middle = middle
                filter_away_objective, filter_away_travel_time, filter_away_deviation = \
                    new_request_updater.norm_objective(
                        filtered_away, [], True, filtered_size)

                cumulative_objective = copy(
                    cumulative_objective) + copy(filter_away_objective)
                cumulative_travel_time = copy(
                    cumulative_travel_time) + copy(filter_away_travel_time)
                cumulative_deviation = copy(
                    cumulative_deviation) + copy(filter_away_deviation)

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
                current_route_plan, current_objective, current_infeasible_set, still_delayed_nodes = alns.iterate(
                    reopt_iterations, reopt_Z, disrupt[0], disrupt[1], disruption_time, delayed, True,  run)

                if delayed[0]:
                    delay_deltas[-1] = delay_deltas[-1] - current_objective
                    current_route_plan = disruption_updater.recalibrate_solution(
                        current_route_plan, disruption_info, still_delayed_nodes)

                    delta_dev_objective = new_request_updater.get_delta_objective(
                        current_route_plan, current_infeasible_set, current_objective)
                    cumulative_recalibration += delta_dev_objective
                    current_objective -= delta_dev_objective
                    cumulative_deviation += delta_dev_objective/beta

            total_objective, rejected_objective = new_request_updater.total_objective(current_objective, cumulative_objective,
                                                                                      cumulative_recalibration, cumulative_rejected, rejection)

            _, current_travel_time, current_deviation = new_request_updater.norm_objective(
                current_route_plan, [], False, filtered_size)

            ride_time_objective = copy(
                cumulative_travel_time) + copy(current_travel_time)

            deviation_objective = copy(
                cumulative_deviation) + copy(current_deviation)

            ride_sharing = ride_sharing_passengers / \
                ride_sharing_arcs if ride_sharing_arcs > 0 else 0
            cost_per_trip_filtered = dict(
                filter(lambda elem: elem[1][0] > 0, cost_per_trip.items()))
            if len(cost_per_trip_filtered) > 0:
                cpt = sum(elem[1][0]/8
                          for elem in cost_per_trip_filtered.items())/len(cost_per_trip_filtered)
            introduced_vehicles = len(current_route_plan)
            df_run.append([run, str(disruption_type), total_objective.total_seconds(), (datetime.now() - start_time).total_seconds(), cumulative_rejected, rejected_objective.total_seconds(),
                          deviation_objective.total_seconds(), ride_time_objective.total_seconds(), ride_sharing, cpt, introduced_vehicles, str(simulator.sim_clock)])

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

    return df_run


if __name__ == "__main__":
    """
    # Profilinga
    profile = Profile()
    cProfile.run('main()', 'profiling/restats')
    profile.display()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int)
    parser.add_argument('--branch', type=str)
    parser.add_argument('--instance', type=str)
    args = parser.parse_args()

    run = args.run
    branch = args.branch
    print(f'Config says instance {test_instance}')
    test_instance = args.instance
    print(f'Replaced to argument instance {args.instance}')

    # Generate test instance datetime from filename
    test_instance_d = test_instance.split(
        "/")[-1].split("_")[-1].split(".")[0]
    test_instance_date = test_instance_d[0:4] + "-" + \
        test_instance_d[4:6] + "-" + \
        test_instance_d[6:8] + " 10:00:00"
    breakpoint_hour_date = test_instance_d[0:4] + "-" + \
        test_instance_d[4:6] + "-" + \
        test_instance_d[6:8] + breakpoint_hour

    naive = False
    adaptive = True
    repair_removed = None
    destroy_removed = [0, 2]
    standby = 0

    print("Test instance:", test_instance)

    df_runs = []
    # for run in range(runs):
    df_run = main(
        test_instance, test_instance_date, run, repair_removed, destroy_removed, naive, adaptive, standby, breakpoint_hour_date)
    df_runs.append(pd.DataFrame(df_run, columns=[
        "Run", "Initial/Disruption", "Current Objective", "Solution Time", "Norm Rejected", "Gamma Rejected",  "Norm Deviation Objective", "Norm Ride Time Objective", "Ride Sharing", "Cost Per Trip", "Introduced Vehicles", "Sim_Clock"]))

    df_track_run = pd.concat(df_runs)
    df_track_run.to_csv(
        config("run_path") + "Before2_After2" + "Run:" + str(run) + test_instance + "analysis" + ".csv")

    print("DONE WITH ALL RUNS")

"""

- For initial: Check if requested pickup time is before or after 14:
    - current route plan is updated accordingly
    - introduced vehicles and vehicle sets must be updated
    - ALNS only on before 14 route plan

- For reopt: Check if disruption time is after or before 14: 
    - for cancel, no show & delay: affected route plan according to where to node lies
    - for new request: check if requested pickup time is before or after 14:
    - current route plan is updated accordingly
    - introduced vehicles and vehicle sets must be updated
    - global sets (might) need to be updated (only if depot is added) 
    - two sets of vehicle clocks and sim clocks
    - ALNS alternating on the two according to if disruption time is before or after
    - Simulator must know where to find the disrupted node

- Implement two different V params
"""
