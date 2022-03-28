import pstats
from pstats import SortKey


class Profile:
    def display(self):
        path = r"profiling/restats"
        p = pstats.Stats(path).strip_dirs()
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
