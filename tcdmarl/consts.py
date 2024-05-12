from typing import List

# For simulating the effect of cooperating agents' actions
SIMULATION_THRESH: float = 0.1

# For agent synchronization
SYNCHRONIZATION_THRESH: float = 0.3

EXPERIMENT_NAMES: List[str] = ["routing"]

ALL_EXPERIMENT_NAMES: List[str] = [
    name if i == 0 else "centralized_" + name
    for name in EXPERIMENT_NAMES
    for i in (0, 1)
]
