EXPERIMENT_NAMES: list[str] = ["routing", "generator", "laboratory"]

ALL_EXPERIMENT_NAMES: list[str] = [
    name if i == 0 else "centralized_" + name
    for name in EXPERIMENT_NAMES
    for i in (0, 1)
]

### AGENT SYNCHRONIZATION
SYNCHRONIZATION_THRESH: float = 0.3

### RL

INITIAL_EPISLON: float = 1.0
FINAL_EPISLON: float = 0.1
