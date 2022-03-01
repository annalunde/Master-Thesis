from decouple import config


# Allowed excess ride time
F = 1

# Number of vehicles
V = 16

# Allowed deviaiton from Requested service time (either pickup or dropoff)
D = 5

# Estimated time to serve a node
S = 2

# Vehicle standard seats capacity
P = 15

# Vehicle wheelchair seats capacity
W = 1

# Weight of deviation in objective function
alpha = 1

# Weight of ride time in objective function
beta = 1

chosen_columns = [
    "Request Creation Time",
    "Wheelchair",
    "Request ID",
    "Request Status",
    "Rider ID",
    "Ride ID",
    "Number of Passengers",
    "Requested Pickup Time",
    "Actual Pickup Time",
    "Requested Dropoff Time",
    "Actual Dropoff Time",
    "Cancellation Time",
    "No Show Time",
    "Origin Zone",
    "Origin Lat",
    "Origin Lng",
    "Destination Zone",
    "Destination Lat",
    "Destination Lng",
    "Reason For Travel",
    "Original Planned Pickup Time",
]
