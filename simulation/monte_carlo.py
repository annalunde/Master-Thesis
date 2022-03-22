from datetime import timedelta
from poisson import *
from new_requests import *
from config.simulation_config import *
import random
from scipy.stats import gamma


class MonteCarlo:
    def __init__(self, sim_clock, sim_duration):
        self.sim_clock = sim_clock
        self.sim_duration = sim_duration
        self.data_path = config("data_simulator_path")

    def get_new_requests(self):

        sim_clock = self.sim_clock

        requests = []

        while sim_clock <= self.sim_clock + self.sim_duration:
            request_arrival = Poisson(
                arrival_rate_request, sim_clock).disruption_time()
            print("Sim clock: ", sim_clock)
            print("Arrival: ", request_arrival)
            if request_arrival <= self.sim_clock + self.sim_duration:
                add_request, request = self.new_request(request_arrival)
                if add_request > 0:
                    requests.append(request)
                sim_clock = request_arrival
                print("New request")
                print()
            else:
                sim_clock = self.sim_clock + \
                    self.sim_duration + timedelta(minutes=1)
                print("Simulation done")

        # read all new requests to file
        if len(requests) > 0:
            df = pd.concat(requests)
        else:
            df = pd.DataFrame(columns=["Wheelchair",
                                       "Request ID",
                                       "Request Status",
                                       "Rider ID",
                                       "Ride ID",
                                       "Number of Passengers",
                                       "Cancellation Time",
                                       "No Show Time",
                                       "Origin Zone",
                                       "Origin Lat",
                                       "Origin Lng",
                                       "Destination Zone",
                                       "Destination Lat",
                                       "Destination Lng",
                                       "Reason For Travel",
                                       "Request Creation Time",
                                       "Requested Pickup Time",
                                       "Requested Dropoff Time"])
        df.to_csv(config("data_monte_carlo_new_requests"))
        print(df['Requested Pickup Time'])

    def new_request(self, request_arrival):
        # return new request data
        random_number = np.random.rand()
        if random_number > percentage_dropoff:
            # request has requested pickup time - draw random time
            requested_pickup_time = request_arrival + \
                timedelta(minutes=gamma.rvs(pickup_fit_shape,
                          pickup_fit_loc, pickup_fit_scale))
            print("Requested pickup time: ", requested_pickup_time)

            # if request arrival is after 17.45 or the requested pickup time is after 17.45, the request is ignored
            if request_arrival > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00) \
                    or requested_pickup_time > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00):
                return -1, -1

            else:
                # get random request
                random_request = NewRequests(
                    self.data_path).get_and_drop_random_request()

                # update creation time to request disruption time
                random_request['Request Creation Time'] = request_arrival

                # update requested pickup time and set requested dropoff time to NaN
                random_request['Requested Pickup Time'] = requested_pickup_time
                random_request['Requested Dropoff Time'] = None

                return 1, random_request

        else:
            # request has requested dropoff time - draw random time
            requested_dropoff_time = request_arrival + \
                timedelta(minutes=gamma.rvs(dropoff_fit_shape,
                          dropoff_fit_loc, dropoff_fit_scale))

            # if request arrival is after 17.45 or requested dropoff time is after 18.00, the request is ignored
            if request_arrival > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 17, 45, 00)\
                    or requested_dropoff_time > datetime(request_arrival.year, request_arrival.month, request_arrival.day, 18, 00, 00):
                return -1, -1

            else:
                # get random request
                random_request = NewRequests(
                    self.data_path).get_and_drop_random_request()

                # update creation time to request disruption time
                random_request['Request Creation Time'] = request_arrival

                # update requested pickup time and set requested dropoff time to NaN
                random_request['Requested Pickup Time'] = None
                random_request['Requested Dropoff Time'] = requested_dropoff_time

                return 1, random_request
