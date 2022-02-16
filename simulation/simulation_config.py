import numpy as np

# estimated service time of a node
S = 2

# ARRIVAL RATES FOR DISRUPTION TYPES
arrival_rate_request = np.array([14.728110599078342, 8.193548387096774, 6.2949308755760365, 5.557603686635945,
                                  5.428571428571429, 4.331797235023042, 2.824884792626728, 0.2350230414746544,
                                  0.0001], dtype=float)

# med delay = 10 minutes
arrival_rate_delay = np.array([2.944186046511628, 2.6, 2.469767441860465, 3.167441860465116,
                               3.269767441860465, 3.609302325581395, 2.469767441860465, 1.1162790697674418,
                               0.06976744186046512], dtype=float)

arrival_rate_cancel = np.array([3.0462962962962963, 2.1805555555555554, 1.9490740740740742, 1.75,
                                1.3981481481481481, 0.7824074074074074, 0.33796296296296297, 0.06944444444444445,
                                0.0001], dtype=float)

arrival_rate_no_show = np.array([0.719047619047619, 0.8714285714285714, 0.8380952380952381, 0.8809523809523809,
                                 1.061904761904762, 0.9761904761904762, 0.5428571428571428, 0.34285714285714286,
                                 0.01904761904761905], dtype=float)

# percentage of requests with requested pickup time
percentage_dropoff = 1.549186676994578/100

# PARAMETERS FOR GAMMA DISTRIBUTION FOR REQUESTED PICKUP TIME
pickup_fit_shape = 1.155987842566804
pickup_fit_loc = 0.18057194538803223
pickup_fit_scale = 99.79981438192749

# PARAMETERS FOR GAMMA DISTRIBUTION FOR REQUESTED DROPOFF TIME
dropoff_fit_shape = 2.542159349839424
dropoff_fit_loc = 19.77706537447331
dropoff_fit_scale = 62.8899508584643

# PARAMETERS FOR GAMMA DISTRIBUTION FOR DELAYS
delay_fit_shape = 0.9226135117635452
delay_fit_loc = 9.999999999999998
delay_fit_scale = 7.218157066378124

