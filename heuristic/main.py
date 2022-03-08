import pandas as pd
from decouple import config
import sys
import numpy.random as rnd
from heuristic.construction.construction import ConstructionHeuristic
from heuristic.construction.heuristic_config import *
from heuristic.improvement.alns import ALNS
from heuristic.improvement.improvement_config import *
from heuristic.improvement.operators import Operators
from heuristic.improvement.simulated_annealing import SimulatedAnnealing


def main():
    constructor = None

    try:
        # CONSTRUCTION OF INITIAL SOLUTION
        df = pd.read_csv(config("test_data_construction"))
        constructor = ConstructionHeuristic(requests=df.head(20), vehicles=V)
        print("Constructing initial solution")
        initial_route_plan, initial_objective, infeasible_set = constructor.construct_initial()

        # IMPROVEMENT OF INITIAL SOLUTION
        random_state = rnd.RandomState(seed)

        criterion = SimulatedAnnealing(start_temperature, end_temperature, step)

        alns = ALNS(weights, reaction_factor, initial_route_plan, initial_objective, infeasible_set, criterion,
                    destruction_degree, constructor.T_ij, constructor.preprocessed, random_state)

        operators = Operators(alns)

        # Add destroy operators
        alns.add_destroy_operator(operators.random_removal)
        alns.add_destroy_operator(operators.time_related_removal)
        alns.add_destroy_operator(operators.distance_related_removal)
        alns.add_destroy_operator(operators.related_removal)
        alns.add_destroy_operator(operators.worst_deviation_removal)

        # Add repair operators
        alns.add_repair_operator(operators.greedy_repair)
        alns.add_repair_operator(operators.regret_repair)

        # Run ALNS
        result = alns.iterate(iterations)

    except Exception as e:
        print("ERROR:", e)
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno

        print("Exception type: ", exception_type)
        print("File name: ", filename)
        print("Line number: ", line_number)

if __name__ == "__main__":
    main()