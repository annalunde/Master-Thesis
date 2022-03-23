from datetime import datetime, timedelta


reaction_factor = 0.2

weights = [3, 2, 1, 0.5]

iterations = 10

destruction_degree = 0.5

# Number of requests
R = 50

# Number of vehicles
V = 16

# Simulated annealing temperatures -- NOTE: these must be tuned
start_temperature = 50
end_temperature = 10
step = 5
