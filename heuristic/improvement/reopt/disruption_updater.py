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

    def update_route_plan(self, current_route_plan, disruption_type, disruption_info):

        # adding current position for each vehicle

        updated_route_plan = copy.deepcopy(current_route_plan)

        if disruption_type == 'request':
            self.new_request_updater.set_parameters(disruption_info)

        elif disruption_type == 'delay':
            updated_route_plan = self.update_with_delay(
                current_route_plan, disruption_info)

        elif disruption_type == 'cancel':
            # remove dropoff node
            del updated_route_plan[disruption_info[0]][disruption_info[2]]
            # remove pickup node
            del updated_route_plan[disruption_info[0]][disruption_info[1]]

        else:
            # remove dropoff node
            del updated_route_plan[disruption_info[0]][disruption_info[2]]

        return updated_route_plan

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
    def recalibrate_solution(current_route_plan, disruption_info, still_delayed_nodes):
        delay_duration = disruption_info[2]
        route_plan = copy.deepcopy(current_route_plan)

        for node in still_delayed_nodes:
            idx = next(i for i, (node_test, *_)
                       in enumerate(route_plan[disruption_info[0]]) if node_test == node)
            node_route = route_plan[disruption_info[0]][idx]
            d = node_route[2] - delay_duration
            node_route = (node_route[0], node_route[1], d,
                          node_route[3], node_route[4], node_route[5])
            route_plan[disruption_info[0]][idx] = node_route

        return route_plan
