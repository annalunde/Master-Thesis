
from datetime import datetime, timedelta

'''
CONSTANT PARAMETERS
'''

# Allowed excess ride time
F = 0.5

# Estimated time to serve a node
S_P = 2  # standard seats
S_W = 5  # wheelchair

# Rush hour factor
R_F = 1.5

# Number of vehicles
V = 16

# Vehicle standard seats capacity
P = 15

# Vehicle wheelchair seats capacity
W = 1

# CONSTRUCTION - Allowed deviation from Requested service time (either pickup or dropoff)
U_D = timedelta(minutes=15)
L_D = timedelta(minutes=-15)
P_S_C = timedelta(minutes=7.5)  # penalized soft time windows

# INITIAL - Allowed deviation from Requested service time (either pickup or dropoff)
U_D_N_I = timedelta(minutes=15)
L_D_N_I = timedelta(minutes=-15)
U_D_D_I = timedelta(minutes=60)
L_D_D_I = timedelta(minutes=-60)
P_S_I = timedelta(minutes=7.5)  # penalized soft time windows

# REOPT - Allowed deviation from Requested service time (either pickup or dropoff)
U_D_N_R = timedelta(minutes=7.5)
L_D_N_R = timedelta(minutes=-7.5)
U_D_D_R = timedelta(minutes=60)
L_D_D_R = timedelta(minutes=-60)
P_S_R = timedelta(minutes=0)  # penalized soft time windows

'''
CHANGEABLE PARAMETERS
'''


# Number of tries for + 15 minutes for a rejected request:
N_R = 0

# Number of iterations in ALNS
initial_iterations = 100
reopt_iterations = 100

# Weight of ride time in objective function
alpha = 2
alpha_reopt = 2

# Weight of deviation in objective function
beta = 100
beta_reopt = 100

# Start temperature control parameter
initial_Z = 0.6
reopt_Z = 0.6

# Cooling rate
cooling_rate = 0.96

# Weight scores
weights = [10, 5, 1, 0]

# Reaction factor
reaction_factor = 0.7

# Number of destroy/repair pairs before updating weights in initial
N_U_init = 0.3
N_U_reopt = 0.3

# Destruction degree
destruction_degree = 0.4


# Tuning data
test_instances = ["test_instance_small_1_20210703", "test_instance_small_2_20210724", "test_instance_small_3_20210918", "test_instance_medium_1_20210706", "test_instance_medium_2_20210830", "test_instance_medium_3_20211015", "test_instance_large_1_20211005",
                  "test_instance_large_2_20211014", "test_instance_large_3_20220112"]