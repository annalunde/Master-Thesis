import numpy as np
import numpy.random as rnd


class SimulatedAnnealing:

    def __init__(self, start_temperature, end_temperature, step):
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.step = step
        self.temperature = start_temperature

    # Simulated annealing acceptance criterion
    def accept_criterion(self, random_state, current_objective, candidate_objective):

        # Always accept better solution
        if candidate_objective > current_objective:
            print("better")
            accept = True

        # Sometimes accept worse
        else:
            print(" not better")
            probability = np.exp(-(current_objective - candidate_objective) / self.temperature)
            print("probability ", probability)
            accept = (probability >= random_state.random())

        # Should not set a temperature that is lower than the end temperature.
        self.temperature = max(self.end_temperature, self.temperature - self.step)

        return accept

def main():
    test = SimulatedAnnealing(200, 10, 10)

    accept = test.accept_criterion(rnd.RandomState(), 500, 450)
    print(accept)

    accept = test.accept_criterion(rnd.RandomState(), 500, 450)
    print(accept)

    accept = test.accept_criterion(rnd.RandomState(), 500, 450)
    print(accept)


if __name__ == "__main__":
    main()

