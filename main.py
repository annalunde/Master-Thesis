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


def main(test_instance, test_instance_date, run, repair_removed, destroy_removed, naive, adaptive, standby):
    constructor, simulator = None, None

    try:
        # TRACKING
        start_time = datetime.now()
        df_run, df_cancel, df_req_runtime = [], [], []
        passengers_total, ride_sharing_passengers, ride_sharing_arcs, processed_nodes, ride_sharing = 0, 0, 0, set(), 0

        # CUMULATIVE OBJECTIVE
        cumulative_rejected, cumulative_recalibration, cumulative_objective, cumulative_travel_time, \
            cumulative_deviation = 0, timedelta(
                0), timedelta(0), timedelta(0), timedelta(0)

        # CONSTRUCTION OF INITIAL SOLUTION
        df = pd.read_csv(config(test_instance))
        constructor = ConstructionHeuristic(
            requests=df, vehicles=V+standby, alpha=alpha, beta=beta)
        print("Constructing initial solution")
        initial_route_plan, initial_objective, initial_infeasible_set = constructor.construct_initial()
        measures = Measures()

        # IMPROVEMENT OF INITIAL SOLUTION
        # if not naive:
        criterion = SimulatedAnnealing(cooling_rate)

        alns = ALNS(weights, reaction_factor, initial_route_plan, initial_objective, initial_infeasible_set, criterion,
                    destruction_degree, constructor, rnd_state=rnd.RandomState())

        operators = Operators(alns, standby)

        alns.set_operators(operators, repair_removed, destroy_removed)

        # Run ALNS
        delayed = (False, None, None)

        current_route_plan, current_objective, current_infeasible_set, _ = alns.iterate(
            initial_iterations, initial_Z, None, None, None, delayed, False, run, adaptive)

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

        ride_time_objective, deviation_objective, rejected_objective = constructor.print_objective(
            current_route_plan, [i for i in range(cumulative_rejected)])

        total_objective = constructor.total_objective(current_objective, cumulative_objective,
                                                      cumulative_recalibration)

        df_run.append([run, "Initial", total_objective.total_seconds(), (datetime.now() - start_time).total_seconds(), cumulative_rejected, rejected_objective.total_seconds(),
                       cumulative_recalibration.total_seconds()/beta, ride_time_objective.total_seconds(), ride_sharing, ride_sharing_arcs, ride_sharing_passengers, passengers_total, len(current_route_plan), "10:00"])
        print("Initial objective", current_objective.total_seconds())
        print("Initial rejected", cumulative_rejected)
        cumulative_deviation = copy(cumulative_recalibration)/beta

        # SIMULATION
        print("Start simulation")
        sim_clock = datetime.strptime(
            test_instance_date, "%Y-%m-%d %H:%M:%S")
        simulator = Simulator(sim_clock)
        new_request_updater = NewRequestUpdater(
            constructor, standby)
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
                current_route_plan, removed_filtering, filtered_away, middle, filtered_size = disruption_updater.\
                    filter_route_plan(current_route_plan,
                                      vehicle_clocks, None, disruption_type, False)  # Filter route plan
                new_request_updater.middle = middle
                filter_objective = new_request_updater.new_objective(
                    current_route_plan, [], False)
                filter_away_objective, filter_away_travel_time, filter_away_deviation = \
                    new_request_updater.norm_objective(
                        filtered_away, [], True, filtered_size)

                cumulative_objective = copy(
                    cumulative_objective) + copy(filter_away_objective)
                cumulative_travel_time = copy(
                    cumulative_travel_time) + copy(filter_away_travel_time)
                cumulative_deviation = copy(
                    cumulative_deviation) + copy(filter_away_deviation)

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
                df_req_runtime.append([rid, disruption_time,
                                       disruption_info.iloc[0]['Requested Pickup Time'],
                                       disruption_info.iloc[0]['Requested Dropoff Time'],
                                       disruption_info.iloc[0]['Wheelchair'],
                                       disruption_info.iloc[0]['Number of Passengers'],
                                       disruption_info.iloc[0]['Origin Lat'],
                                       disruption_info.iloc[0]['Origin Lng'],
                                       disruption_info.iloc[0]['Destination Lat'],
                                       disruption_info.iloc[0]['Destination Lng']])

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
                    if disruption_type == 2:
                        df_cancel.append(
                            [disruption_time, disruption_info[3], disruption_info[4], str(removed_time)])
                elif disruption_type == 1:  # Disruption: delay
                    node_idx = next(i for i, (node, *_) in enumerate(current_route_plan[disruption_info[0]]) if
                                    node == disruption_info[3])
                    delayed = (True, disruption_info[0], node_idx)
                    delay_deltas.append(current_objective)

            passengers_total = measures.cpt_calc(
                filtered_away, passengers_total, processed_nodes)
            ride_sharing_passengers, ride_sharing_arcs, processed_nodes = measures.ride_sharing(
                filtered_away, ride_sharing_passengers, ride_sharing_arcs, processed_nodes)

            if len(simulator.disruptions_stack) == 0:
                passengers_total = measures.cpt_calc(
                    current_route_plan, passengers_total, processed_nodes)
                ride_sharing_passengers, ride_sharing_arcs, processed_nodes = measures.ride_sharing(
                    current_route_plan, ride_sharing_passengers, ride_sharing_arcs, processed_nodes)

            if not naive and not rejection:
                # Heuristic
                criterion = SimulatedAnnealing(cooling_rate)

                alns = ALNS(weights, reaction_factor, current_route_plan, current_objective, current_infeasible_set,
                            criterion,
                            destruction_degree, new_request_updater, rnd_state=rnd.RandomState())

                operators = ReOptOperators(
                    alns, disruption_time, vehicle_clocks, standby)

                alns.set_operators(operators, repair_removed, destroy_removed)

                # Run ALNS
                current_route_plan, current_objective, current_infeasible_set, still_delayed_nodes = alns.iterate(
                    reopt_iterations, reopt_Z, disrupt[0], disrupt[1], disruption_time, delayed, True, run, adaptive)

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

            introduced_vehicles = len(current_route_plan)
            df_run.append([run, str(disruption_type), total_objective.total_seconds(), (datetime.now() - start_time).total_seconds(), cumulative_rejected, rejected_objective.total_seconds(),
                          deviation_objective.total_seconds(), ride_time_objective.total_seconds(), ride_sharing, ride_sharing_arcs, ride_sharing_passengers, passengers_total, introduced_vehicles, str(simulator.sim_clock)])

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

    return df_run, df_cancel, df_req_runtime


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

    naive = True
    adaptive = False
    repair_removed = None
    destroy_removed = [0, 2]
    standby = 0

    print("Test instance:", test_instance)
    print("Naive:", naive)
    print("Adaptive:", adaptive)

    df_runs, df_requests_runs, df_cancel_runs = [], [], []
    df_run, df_cancel, df_req_runtime = main(
        test_instance, test_instance_date, run, repair_removed, destroy_removed, naive, adaptive, standby)
    df_runs.append(pd.DataFrame(df_run, columns=[
        "Run", "Initial/Disruption", "Current Objective", "Solution Time", "Norm Rejected", "Gamma Rejected",  "Norm Deviation Objective", "Norm Ride Time Objective", "Ride Sharing", "Ride Sharing Arcs", "Ride Sharing Passengers", "Total Served Passengers", "Introduced Vehicles", "Sim_Clock"]))

    df_track_run = pd.concat(df_runs)
    df_track_run.to_csv(
        config("run_path") + "Semi-Naive" + "Run:" + str(run) + test_instance + "analysis" + ".csv")

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
