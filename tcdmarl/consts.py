# For agent synchronization
SYNCHRONIZATION_THRESH: float = 0.3

EXPERIMENT_NAMES: list[str] = ["routing"]

ALL_EXPERIMENT_NAMES: list[str] = [
    name if i == 0 else "centralized_" + name
    for name in EXPERIMENT_NAMES
    for i in (0, 1)
]
