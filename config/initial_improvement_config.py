from datetime import datetime, timedelta


# Allowed excess ride time
F = 1

# Number of destroy/repair pairs before updating weights in initial
N_U = 0.01


# Number of vehicles
V = 16

# Allowed deviaiton from Requested service time (either pickup or dropoff)
U_D_N = timedelta(minutes=15)
L_D_N = timedelta(minutes=-15)
U_D_D = timedelta(minutes=60)
L_D_D = timedelta(minutes=-60)
P_S = timedelta(minutes=7.5)  # penalized soft time windows

# Estimated time to serve a node
S_P = 2  # standard seats
S_W = 5  # wheelchair

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
