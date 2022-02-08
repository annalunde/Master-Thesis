"""
The implementation of the GRASP heuristic
"""


class ConstructionHeuristic:
    def __init__(self, requests):
        self.requests = requests

    def grasp(self, iterations):
        current_best = None
        while iterations != 0:
            solution = construct_solution()
            new_solution = local_search(solution)
            if better(new_solution, current_best):
                current_best = new_solution
            iterations = iterations - 1
        return current_best

    def construct_solution(self):
        solution = None
        finished = False
        while not finished:
            candidates = make_candidates()
            s = select_candidate(candidates)
            solution = union(solution, s)
            change_greedy_function()
        return solution

    def local_search(self, solution):
        return None

    def better(self, new, current):
        return True if new > current else False

    def make_candidates(self):
        return []

    def select_candidate(self, candidates):
        return []

    def change_greedy_function(self):
        return None

    def union(self):
        return None