from datetime import datetime

reaction_factor = 0.2

weights = [3, 2, 1, 0.5]

seed = 9876

iterations = 1000

# Initial route plan
initial_solution = [
    [(1, datetime.strptime("2021-05-10 12:02:00", "%Y-%m-%d %H:%M:%S"), 2),
     (1.5, datetime.strptime("2021-05-10 12:10:00", "%Y-%m-%d %H:%M:%S"), 0),
     (3, datetime.strptime("2021-05-10 16:02:00", "%Y-%m-%d %H:%M:%S"), 0),
     (3.5, datetime.strptime("2021-05-10 17:02:00", "%Y-%m-%d %H:%M:%S"), 0)],
    [(2, datetime.strptime("2021-05-10 13:15:00", "%Y-%m-%d %H:%M:%S"), 1),
     (2.5, datetime.strptime("2021-05-10 14:12:00", "%Y-%m-%d %H:%M:%S"), 2),
     (4, datetime.strptime("2021-05-10 12:15:00", "%Y-%m-%d %H:%M:%S"), 0),
     (4.5, datetime.strptime("2021-05-10 12:30:00", "%Y-%m-%d %H:%M:%S"), 0)
     ]
]

initial_objective = 10000

destruction_degree = 0.5

# T_ij, travel time matrix
travel_times = [
    # 1  2  3  4  1.5  2.5  3.5  4.5
    [0, 1, 4, 5, 5, 5, 4, 5],  # 1
    [1, 0, 2, 3, 5, 5, 3, 5],  # 2
    [4, 2, 0, 5, 5, 5, 5, 5],  # 3
    [5, 5, 3, 0, 5, 5, 5, 5],  # 4
    [5, 5, 5, 5, 0, 1, 5, 1],  # 1.5
    [5, 5, 5, 5, 1, 0, 2, 2],  # 2.5
    [4, 5, 5, 5, 0, 2, 0, 5],  # 3.5
    [5, 5, 5, 5, 1, 2, 5, 0]  # 4.5
]

# Disse må tunes
start_temperature = 50
end_temperature = 10
step = 5

