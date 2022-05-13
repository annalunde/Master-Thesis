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
    def cpt_calc(filtered_away_route_plan, cost_per_trip, initial, last):
        vehicles = len(filtered_away_route_plan)
        for vehicle_idx in range(vehicles):
            if len(filtered_away_route_plan[vehicle_idx]) > 0:
                passengers = cost_per_trip[vehicle_idx][0]
                init_node = cost_per_trip[vehicle_idx][1]
                vehicle_route = filtered_away_route_plan[vehicle_idx]
                for node in vehicle_route:
                    passengers += node[3]
                    passengers += node[4]
                cost_per_trip[vehicle_idx] = (
                    passengers, vehicle_route[0][1]) if initial else (passengers, init_node)
                if last and len(vehicle_route) > 0:
                    hours = (vehicle_route[-1][1] - init_node).total_seconds()/3600
                    cost_per_trip[vehicle_idx] = (
                        passengers, hours)
        return cost_per_trip