from datetime import datetime, timedelta

reaction_factor = 0.2

weights = [3, 2, 1, 0.5]

iterations = 1000

destruction_degree = 0.5

# Allowed excess ride time
F = 1

# Number of vehicles
V = 16

# Allowed deviaiton from Requested service time (either pickup or dropoff)
U_D_N = timedelta(minutes=5)
L_D_N = timedelta(minutes=-5)
U_D_D = timedelta(minutes=30)
L_D_D = timedelta(minutes=-30)

# Estimated time to serve a node
S = 2

# Vehicle standard seats capacity
P = 15

# Vehicle wheelchair seats capacity
W = 1

# Weight of ride time in objective function
alpha = 1

# Weight of deviation in objective function
beta = 5

# Weight of infeasible set in objective function
gamma = 10000


# Simulated annealing temperatures -- NOTE: these must be tuned
start_temperature = 50
end_temperature = 10
step = 5
