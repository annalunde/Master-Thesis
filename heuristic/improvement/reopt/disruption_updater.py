import copy
import pandas as pd
from decouple import config
from heuristic.construction.construction import ConstructionHeuristic
from config.construction_config import *
from simulation.simulator import Simulator
from heuristic.improvement.reopt.new_request_updater import NewRequestUpdater


class DisruptionUpdater:
    def __init__(self, new_request_updater):
        self.new_request_updater = new_request_updater

    def update_route_plan(self, current_route_plan, disruption_type, disruption_info, sim_clock):

        # adding current position for each vehicle
        vehicle_clocks, artificial_depot = self.update_vehicle_clocks(
            current_route_plan, sim_clock, disruption_type, disruption_info)

        updated_route_plan = copy.deepcopy(current_route_plan)

        if disruption_type == 'request':
            self.new_request_updater.set_parameters(disruption_info)

        elif disruption_type == 'delay':
            updated_route_plan = self.update_with_delay(
                current_route_plan, disruption_info)

        elif disruption_type == 'cancel':
            # update capacities
            updated_vehicle_route = self.update_capacities(
                updated_route_plan[disruption_info[0]], disruption_info[1], disruption_info[2],
                updated_route_plan[disruption_info[0]][disruption_info[1]][5])

            updated_route_plan[disruption_info[0]] = updated_vehicle_route

            if artificial_depot:
                # remove dropoff node
                del updated_route_plan[disruption_info[0]][disruption_info[2]]
            else:
                # remove dropoff node
                del updated_route_plan[disruption_info[0]][disruption_info[2]]
                # remove pickup node
                del updated_route_plan[disruption_info[0]][disruption_info[1]]

        else:
            # no show
            # update capacities
            updated_vehicle_route = self.update_capacities(
                updated_route_plan[disruption_info[0]], disruption_info[1], disruption_info[2],
                updated_route_plan[disruption_info[0]][disruption_info[1]][5])

            updated_route_plan[disruption_info[0]] = updated_vehicle_route

            # remove dropoff node
            del updated_route_plan[disruption_info[0]][disruption_info[2]]

        return updated_route_plan, vehicle_clocks

    def update_with_delay(self, current_route_plan, disruption_info):
        delay_duration = disruption_info[2]
        route_plan = copy.deepcopy(current_route_plan)

        start_idx = disruption_info[1]
        for node in route_plan[disruption_info[0]][disruption_info[1]:]:
            t = node[1] + delay_duration
            d = node[2] + delay_duration
            node = (node[0], t, d, node[3], node[4], node[5])
            route_plan[disruption_info[0]][start_idx] = node
            start_idx += 1

        return route_plan

    @staticmethod
    def recalibrate_solution(current_route_plan, disruption_info):
        delay_duration = disruption_info[2]
        route_plan = copy.deepcopy(current_route_plan)

        start_idx = disruption_info[1]
        for node in route_plan[disruption_info[0]][disruption_info[1]:]:
            d = node[2] - delay_duration
            node = (node[0], node[1], d, node[3], node[4], node[5])
            route_plan[disruption_info[0]][start_idx] = node
            start_idx += 1

        return route_plan

    def update_vehicle_clocks(self, current_route_plan, sim_clock, disruption_type, disruption_info):
        artificial_depot = False

        # find index for next node after sim_clock and corresponding time of service
        vehicle_clocks = []
        for vehicle_route in current_route_plan:
            if len(vehicle_route) > 1:
                if vehicle_route[0][1] < sim_clock:
                    prev_idx = 0
                    for idx, (node, time, deviation, passenger, wheelchair, _) in enumerate(vehicle_route):
                        if time <= sim_clock:
                            prev_idx = idx

                    if prev_idx == len(vehicle_route) - 1:
                        vehicle_clocks.append(sim_clock)

                    else:
                        next_idx = prev_idx + 1
                        vehicle_clocks.append(vehicle_route[next_idx][1])

                        if disruption_type == 'cancel':
                            # check whether next node after sim_clock is the request that is cancelled
                            if current_route_plan[disruption_info[0]][disruption_info[1]] == vehicle_route[next_idx]:
                                artificial_depot = True

                else:
                    vehicle_clocks.append(sim_clock)

            else:
                vehicle_clocks.append(sim_clock)

        return vehicle_clocks, artificial_depot

    def update_capacities(self, vehicle_route, start_id, dropoff_id, request):
        idx = start_id
        for n, t, d, p, w, _ in vehicle_route[start_id:dropoff_id]:
            p -= request["Number of Passengers"]
            w -= request["Wheelchair"]
            vehicle_route[idx] = (n, t, d, p, w, _)
            idx += 1
        return vehicle_route

