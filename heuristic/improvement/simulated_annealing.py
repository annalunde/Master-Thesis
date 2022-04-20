import numpy as np
import numpy.random as rnd
from datetime import timedelta


class SimulatedAnnealing:

    def __init__(self, cooling_rate):
        self.cooling_rate = cooling_rate
        self.temperature = 0

    # Simulated annealing acceptance criterion
    def accept_criterion(self, random_state, current_objective, candidate_objective):

        # Always accept better solution
        if candidate_objective <= current_objective:
            #print("Found better solution")
            accept = True

        # Sometimes accept worse
        else:
            #print("Did not find better solution")

            diff = (candidate_objective.total_seconds() -
                    current_objective.total_seconds())/60
            #print("DIFF", diff)
            probability = np.exp(-diff / self.temperature)
            #print("Probability ", probability)
            accept = (probability >= random_state.random())
            #print("Did we still go with worse solution: ", accept)

        # Should not set a temperature that is lower than the end temperature.
        self.temperature = self.temperature*self.cooling_rate

        return accept
