# TODO implement this throughout the project

from typing import TypedDict


class ExperimentConfig(TypedDict):
    # Environment name. E.g., "routing"
    environment_name: str

    # Use centralized or decentralized
    centralized: bool

    # Use the TL-CD to expedite learning
    use_tl_cd: bool


class Config(TypedDict):
    # Number of separate trials to run the algorithm for
    num_trials: int

    # Step unit factor
    step_unit_factor: int
