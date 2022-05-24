from datetime import timedelta, datetime


class Measures:
    def __init__(self):
        pass

    @staticmethod
    def ride_sharing(current_route_plan, ride_sharing_passengers, ride_sharing_arcs, processed_nodes):
        vehicles = len(current_route_plan)
        for vehicle in range(vehicles):
            if len(current_route_plan[vehicle]) > 0:
                added = 0
                if current_route_plan[vehicle][0][0] == 0:
                    vehicle_route = current_route_plan[vehicle][1:]
                else:
                    vehicle_route = current_route_plan[vehicle]
                for node in vehicle_route:
                    if node[0] not in processed_nodes:
                        added += 1
                        ride_sharing_passengers += node[3]
                        ride_sharing_passengers += node[4]
                        processed_nodes.add(node[0])
                ride_sharing_arcs += added - 1 if added > 1 else added
        return ride_sharing_passengers, ride_sharing_arcs, processed_nodes

    @staticmethod
    def cpt_calc(filtered_away_route_plan, passengers_total, processed_nodes):
        for vehicle_route in filtered_away_route_plan:
            for node in vehicle_route:
                if node[0] not in processed_nodes:
                    passengers_total += node[3]
                    passengers_total += node[4]
        return passengers_total
