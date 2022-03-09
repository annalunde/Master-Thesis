import pandas as pd
import numpy as np
import copy
import math

""


class MonteCarloSimulator:
    def __init__(self, standby, route_plan):
        self.standby_vehicles = standby

    def main():
    mcs = None

    try:
        mcs = MonteCarloSimulator(standby=2, route_plan=None)
        print("Starting Monte Carlo Simulations")

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
