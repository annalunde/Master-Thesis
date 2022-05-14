from datetime import timedelta, datetime


class Measures:
    def __init__(self):
        pass

    @staticmethod
    def ride_sharing(current_route_plan, ride_sharing_passengers, ride_sharing_arcs, processed_nodes):
        vehicles = len(current_route_plan)
        for vehicle in range(vehicles):
            vehicle_route = current_route_plan[vehicle][1:]
            for node in vehicle_route:
                if node[0] not in processed_nodes:
                    ride_sharing_passengers += node[3]
                    ride_sharing_passengers += node[4]
                    ride_sharing_arcs += 1
                    processed_nodes.add(node[0])
        return ride_sharing_passengers, ride_sharing_arcs, processed_nodes

    @staticmethod
    def cpt_calc(filtered_away_route_plan, cost_per_trip):
        vehicles = len(filtered_away_route_plan)
        for vehicle_idx in range(vehicles):
            passengers = cost_per_trip[vehicle_idx][0]
            init_node = cost_per_trip[vehicle_idx][1]
            last_node = cost_per_trip[vehicle_idx][2]
            vehicle_route = filtered_away_route_plan[vehicle_idx]
            for node in vehicle_route:
                passengers += node[3]
                passengers += node[4]
            if len(vehicle_route) > 0 and init_node == None:
                cost_per_trip[vehicle_idx] = (
                    passengers, vehicle_route[0][1], last_node)
            else:
                cost_per_trip[vehicle_idx] = (passengers, init_node, last_node)
            if len(vehicle_route) > 0:
                last_node = vehicle_route[-1][1]
                init_node = cost_per_trip[vehicle_idx][1]
                cost_per_trip[vehicle_idx] = (
                    passengers, init_node, last_node)
        return cost_per_trip
