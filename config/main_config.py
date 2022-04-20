# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
# ---

from datetime import datetime, timedelta


reaction_factor = 0.7

weights = [15, 10, 5, 0]


destruction_degree = 0.4


# Number of vehicles
V = 16

# Number of tries for + 15 minutes for a rejected request:
N_R = 0

# Simulated annealing temperatures -- NOTE: these must be tuned
start_temperature = 50
end_temperature = 10
cooling_rate = 0.999


# Running data
test_instances = ["test_instance_small_1_20210703", "test_instance_small_2_20210724", "test_instance_small_3_20210918", "test_instance_medium_1_20210706",
                  "test_instance_medium_2_20210830", "test_instance_medium_3_20211015", "test_instance_large_1_20211005", "test_instance_large_2_20211014", "test_instance_large_3_20220112"]
start_time = datetime.now()
