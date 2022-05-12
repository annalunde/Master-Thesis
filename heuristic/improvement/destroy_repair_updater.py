from datetime import timedelta
#from config.reopt_improvement_config import *
from config.main_config import *


class Destroy_Repair_Updater:
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def update_solution(self, route_plan, index_removed_requests, disruption_time):
        updated_solution = list(map(list, route_plan))
        # (row, counter) --> sequences([(node,row,col),...])
        index_removed_requests = self.filter_indexes(index_removed_requests)
        # remove rows where there is no deviation to remove unnecessary computations
        improve_indexes = self.improve_indexes(
            index_removed_requests, updated_solution)

        for row, c in improve_indexes.keys():
            new_dict = {}
            vehicle_route = updated_solution[row]
            first_element = index_removed_requests[row, c][0]
            removed_counter = 0
            new_dict = {k: v for k, v in index_removed_requests.items()
                        if k[0] == row and k[1] < c}
            removed_counter = sum([len(i[1]) for i in new_dict.items()])
            left_idx = None if first_element[2] == 0 else first_element[2] - \
                1
            if left_idx is not None:
                left_idx = left_idx if c == 0 else left_idx - removed_counter
            right_idx = first_element[2] if c == 0 else first_element[2] - \
                removed_counter

            left_node = None if left_idx == None else vehicle_route[left_idx]
            right_node = None if right_idx >= len(
                vehicle_route) or right_idx < 0 else vehicle_route[right_idx]

            if left_node and left_node[0] == 0 and disruption_time is None:
                if right_idx is not None:
                    s = S_W if right_node[5]["Wheelchair"] else S_P
                    service_time_depot = right_node[1] + timedelta(minutes=s) - \
                        self.heuristic.travel_time(
                            right_node[0] - 1, 2 * self.heuristic.n + row, True)

                    n, t, d, p, w, r = vehicle_route[left_idx]
                    t = service_time_depot
                    vehicle_route[left_idx] = n, t, d, p, w, r

                updated_solution[row] = vehicle_route

            if left_node and left_node[0] != 0 and right_node:

                left_dev = left_node[2] if left_node[2] < timedelta(0) else 0
                right_dev = right_node[2] if right_node[2] > timedelta(
                    0) else 0

                if not right_dev and not left_dev:
                    continue

                l_node = left_node[0] % int(left_node[0])
                r_node = right_node[0] % int(right_node[0])
                left_node_id = int(
                    left_node[0] - 0.5 - 1 + self.heuristic.n if l_node else left_node[0] - 1)
                right_node_id = int(
                    right_node[0] - 0.5 - 1 + self.heuristic.n if r_node else right_node[0] - 1)
                travel_time = self.heuristic.travel_time(
                    left_node_id, right_node_id, True)

                reduction_dev = right_node[1] - \
                    left_node[1] - travel_time if right_node[1] - \
                    left_node[1] - travel_time > timedelta(0) else 0

                if not reduction_dev:
                    continue

                push_backward = None
                push_forward = None

                if left_dev and not right_dev:
                    push_forward = reduction_dev if reduction_dev < -left_dev else -left_dev

                elif right_dev and not left_dev:
                    push_backward = reduction_dev if reduction_dev < right_dev else right_dev

                else:
                    push_forward = left_dev.total_seconds()/(left_dev.total_seconds() - right_dev.total_seconds()) * reduction_dev if left_dev.total_seconds() / \
                        (left_dev.total_seconds() - right_dev.total_seconds()) * reduction_dev < -left_dev else -left_dev
                    push_backward = right_dev.total_seconds()/(left_dev.total_seconds() - right_dev.total_seconds()) * reduction_dev if right_dev.total_seconds()/(
                        left_dev.total_seconds() - right_dev.total_seconds()) * reduction_dev < right_dev else right_dev

                if push_backward:
                    vehicle_route = self.update_backward(
                        vehicle_route, left_idx, push_backward, disruption_time)
                if push_forward:
                    vehicle_route = self.update_forward(
                        vehicle_route, right_idx, push_forward, row, disruption_time)

                updated_solution[row] = vehicle_route

            elif left_node and left_node[0] != 0:
                left_dev = left_node[2] if left_node[2] < timedelta(0) else 0

                if not left_dev:
                    continue

                n, t, d, p, w, r = vehicle_route[left_idx]
                t = t - left_dev
                d = d - left_dev
                vehicle_route[left_idx] = n, t, d, p, w, r

                updated_solution[row] = vehicle_route

            elif right_node:
                right_dev = right_node[2] if right_node[2] > timedelta(
                    0) else 0

                if not right_dev:
                    continue

                n, t, d, p, w, r = vehicle_route[right_idx]
                t = t - right_dev
                d = d - right_dev
                vehicle_route[right_idx] = n, t, d, p, w, r

                updated_solution[row] = vehicle_route

        tid = 2
        return updated_solution

    def update_backward(self, vehicle_route, start_idx, push_backward, disruption_time):
        idx = start_idx
        for n, t, d, p, w, r in vehicle_route[start_idx:]:
            # since updating happens at start_idx + 1, there is no need to check for depot
            if idx > start_idx:
                n_prev, t_prev, d_prev, p_prev, w_prev, r_prev = vehicle_route[idx-1]
                n_node = n % int(n)
                n_prev_node = n_prev % int(n_prev)
                n_node_id = int(
                    n - 0.5 - 1 + self.heuristic.n if n_node else n - 1)
                n_prev_node_id = int(
                    n_prev - 0.5 - 1 + self.heuristic.n if n_prev_node else n_prev - 1)
                travel_time = self.heuristic.travel_time(
                    n_node_id, n_prev_node_id, True)
                push_backward = t - travel_time - t_prev if t - \
                    travel_time - t_prev > timedelta(0) else timedelta(0)

            if d is not None and push_backward == timedelta(0):
                break

            if disruption_time:
                if t - push_backward <= disruption_time:
                    push_backward = t-disruption_time

            t = t - \
                push_backward if d > timedelta(
                    0) else t
            d = d - \
                push_backward if d > timedelta(
                    0) else d
            vehicle_route[idx] = (n, t, d, p, w, r)
            idx += 1

        return vehicle_route

    def update_forward(self, vehicle_route, start_idx, push_forward, introduced_vehicle, disruption_time):
        for idx in range(start_idx, -1, -1):
            n, t, d, p, w, r = vehicle_route[idx]
            if idx < start_idx:
                n_next, t_next, d_next, p_next, w_next, r_next = vehicle_route[idx+1]
                n_node = n % int(n) if n > 0 else 0
                n_next_node = n_next % int(n_next)
                n_node_id = int(
                    n - 0.5 - 1 + self.heuristic.n if n_node else n - 1)
                n_node_id = 2*self.heuristic.n + introduced_vehicle if n == 0 else n_node_id
                n_next_node_id = int(
                    n_next - 0.5 - 1 + self.heuristic.n if n_next_node else n_next - 1)
                travel_time = self.heuristic.travel_time(
                    n_node_id, n_next_node_id, True)
                push_forward = t_next - travel_time - t if t_next - \
                    t - travel_time > timedelta(0) else timedelta(0)

            if push_forward == timedelta(0):
                break

            if disruption_time:
                if t + push_forward <= disruption_time or t <= disruption_time:
                    break

            if d is not None:
                t = t + \
                    push_forward if d < timedelta(
                        0) else t
                d = d + \
                    push_forward if d < timedelta(
                        0) else d
                vehicle_route[idx] = (n, t, d, p, w, r)
            else:
                t = t + push_forward
                vehicle_route[idx] = (n, t, d, p, w, r)

        return vehicle_route

    def filter_indexes(self, index_removed_requests):
        bundles = dict()  # (row, counter) --> sequences([(node,row,col),...])
        rows = set([i[1] for i in index_removed_requests if i[1] is not None])
        for row in rows:
            values = [x for x in index_removed_requests if x[1] == row]
            values.sort(key=lambda x: x[2])
            spl = [0]+[i for i in range(1, len(values))
                       if values[i][2]-values[i-1][2] > 1]+[None]
            seqs = [values[b:e] for (b, e) in [(spl[i-1], spl[i])
                                               for i in range(1, len(spl))]]
            c = 0
            for seq in seqs:
                bundles[(row, c)] = seq
                c += 1
        return bundles

    def improve_indexes(self, index_removed_requests, route_plan):
        # rows with not only timedelta(0) in deviation
        not_zero_dev_rows = [idx for idx, row in enumerate(route_plan) if not all(
            r[2] == timedelta(0) or r[2] == None for r in row)]
        return dict(filter(lambda x: x[0][0] in not_zero_dev_rows, index_removed_requests.items()))

    def update_capacities(self, trunc_route_plan, index_removed):
        rows = set([i[1] for i in index_removed])
        for row in rows:
            vehicle_route = trunc_route_plan[row]
            start_idx = 1
            for n, t, d, p, w, r in vehicle_route[1:]:
                p_prev = vehicle_route[start_idx-1][3]
                w_prev = vehicle_route[start_idx-1][4]
                p = p_prev + \
                    r["Number of Passengers"] if not(
                        n % int(n)) else p_prev - r["Number of Passengers"]
                w = w_prev + \
                    r["Wheelchair"] if not(
                        n % int(n)) else w_prev - r["Wheelchair"]
                vehicle_route[start_idx] = (n, t, d, p, w, r)
                start_idx += 1
            trunc_route_plan[row] = vehicle_route
        return trunc_route_plan
