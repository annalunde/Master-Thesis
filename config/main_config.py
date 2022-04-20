from datetime import datetime, timedelta


reaction_factor = 0.7

weights = [15, 10, 5, 0]

iterations = 1000

destruction_degree = 0.4

# Number of requests
R = 105

# Number of vehicles
V = 16

# Number of tries for + 15 minutes for a rejected request:
N_R = 0

# Simulated annealing temperatures -- NOTE: these must be tuned
cooling_rate = 0.999
Z = 1.5
