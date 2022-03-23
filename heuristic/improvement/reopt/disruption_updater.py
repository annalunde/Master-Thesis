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
        updated_route_plan = copy.deepcopy(current_route_plan)

        if disruption_type == 'delay':
            pass
        elif disruption_type == 'cancel':
            # remove dropoff node
            del updated_route_plan[disruption_info[0]][disruption_info[2]]
            # remove pickup node
            del updated_route_plan[disruption_info[0]][disruption_info[1]]

        else:
            # remove dropoff node
            del updated_route_plan[disruption_info[0]][disruption_info[2]]

        # updating times of the disrupted route plan

        return updated_route_plan

    def update_new_request(self, new_request):
        self.new_request_updater.set_parameters(new_request)
